# Resume Screening Analyzer

A fully local pipeline that parses PDF/DOCX resumes, stores section-aware embeddings in PostgreSQL/pgvector, retrieves candidates for a job description (JD), evaluates selected evidence with a local Qwen model, verifies quoted claims, and ranks candidates with deterministic Python scoring. FastAPI provides the backend and Streamlit provides the UI.

This is the codebase map for humans and coding agents. It documents the actual runtime flow, module ownership, persistence, recovery, API contracts, and invariants.

## Non-negotiable design rules

- Inference is local; no resume content is sent to a hosted model API.
- BGE embeddings run on CPU. The GGUF LLM is intended to use the GPU.
- The local LLM judges proficiency and ambiguous GitHub project matches and writes summaries. Python computes scores, ranks, and recommendations.
- One model instance handles one generation at a time.
- Only selected, anonymized chunks reach the Evaluator prompt.
- Code verifies evidence quotes before they affect scoring.
- Every candidate is saved immediately; one failure does not stop the rest.
- Queued/running screenings resume from LangGraph PostgreSQL checkpoints after an API restart.
- Re-ranking uses stored evaluations and makes no LLM calls.
- Candidate email uses deterministic selection and templates. Recruiter approval and dry-run are enabled by default.

Target hardware is a GTX 1650 (4 GB VRAM), 24 GB RAM, and Intel i5. Python 3.12 is recommended for CUDA wheel compatibility.

## Architecture

```text
Resume PDF/DOCX
  -> parse -> preprocess -> metadata -> section chunks -> CPU embeddings
  -> source file on disk + resume/chunks in PostgreSQL

JD text/PDF/DOCX/TXT
  -> queued run -> JD Parser (LLM) -> query embeddings
  -> pgvector evidence for in-scope resumes -> conservative code filter
  -> each candidate, sequentially:
       anonymize -> Evaluator (local LLM) -> verify evidence -> save
  -> resume-based rank -> Report summary (LLM)
  -> queue GitHub evidence for the top ranked candidates
  -> select and draft candidate emails -> recruiter review -> delivery
```

`app/screening/graph.py` contains two LangGraph graphs:

1. Screening runs `parse_jd -> retrieve -> evaluate -> rank -> report`. Ranking also enqueues the GitHub shortlist. The graph then continues to email review, with an optional wait for GitHub.
2. Candidate screening routes to `filtered_out` or runs the Evaluator, verifier, and save nodes. `app/workers/github_worker.py` collects GitHub evidence separately and updates stored rankings as results arrive.

Evaluator and Report use the local LLM. The GitHub agent collects public evidence in Python and calls the local LLM only for ambiguous project matches.

## Repository map

```text
app/
  api.py                    FastAPI lifecycle and routes
  ui.py                     Streamlit client
  config.py                 all environment-backed settings
  db.py                     connection pool and schema checks
  models.py                 Pydantic contracts
  timing.py                 pipeline latency logging
  ingest/
    parse.py                PDF/DOCX extraction and OCR fallback
    preprocess.py           normalization, page cleanup, headings/dates
    metadata.py             skills, experience, education extraction
    chunk.py                section/role/project-aware chunking
    pipeline.py             upload transaction and deduplication
  services/
    embed_service.py        singleton CPU SentenceTransformer
    llm_service.py          singleton llama.cpp model, locking, retries
  screening/
    jd_parser.py            structured JD parsing and query vectors
    retrieval.py            shortlist and per-resume evidence search
    stage1_filter.py        conservative known-skill filter
    anonymize.py            evaluator-copy PII reduction
    evaluator.py            evaluator prompt and output schema
    verifier.py             quote and level verification
    ranking.py              scores, ties, recommendations
    report.py               result sections and short summary
    github_agent.py         resume project matching and profile scoring
    graph.py                graphs, queue, checkpoints, recovery
  integrations/
    github_client.py        public REST requests and conditional caching
db/init.sql                 extensions, tables, indexes
scripts/                    setup, acceptance, benchmark, calibration
tests/                      unit/API/orchestration tests
data/                       private runtime data; git-ignored
```

`AGENTS .md` is the original build specification. The implementation is authoritative when it differs from that document.

## Runtime pipeline

### 1. API startup

The lifespan in `app/api.py`:

1. Creates `DATA_DIR` and a rotating `data/service.log`.
2. Checks that both database vector columns match `EMBED_DIM`.
3. Loads the embedding model once.
4. Loads the llama.cpp model once and records GPU-offloaded layers.
5. Creates LangGraph checkpoint tables and resumes unfinished graph threads.

Startup fails on a vector-dimension mismatch. `/health` is healthy only when the database responds and both models are loaded.

### 2. Resume ingestion

Entry point: `app.ingest.pipeline.process_upload(content, file_name)`.

1. Reject empty files and files over `MAX_UPLOAD_BYTES` (20 MiB by default).
2. Hash content with SHA-256. A PostgreSQL advisory lock prevents concurrent duplicate inserts.
3. Parse DOCX paragraphs/tables or position-sorted PyMuPDF blocks. Pages with under 30 extracted characters use Tesseract when available. Other formats and protected PDFs fail.
4. Normalize Unicode, bullets, whitespace, repeated page edges, page numbers, line breaks, month names, and date separators. Case/punctuation remain so `C++`, `C#`, `.NET`, and `Node.js` survive.
5. Extract skills, education, non-overlapping employment duration, and validated GitHub profile/repository links. GitHub links can appear as visible text, PDF link annotations, or DOCX hyperlinks.
6. Split by sections; experience/projects split by role/project. Oversized or heading-free text uses tokenizer-aware splitting and overlap. A sentence above `CHUNK_MAX` fails rather than being silently truncated.
7. BGE batch-encodes normalized chunk vectors on CPU. Resume chunks never receive the query prefix.
8. Write the hash-named source under `DATA_DIR/uploads`, then insert the resume and chunks in one DB transaction.

Success returns `{id, status: "uploaded", chunks}`; duplicates return `{id, status: "duplicate"}`. Multi-file uploads isolate errors per file.

### 3. Queue and checkpoint

`POST /screenings` validates input, inserts a placeholder JD row, persists the initial LangGraph state through `PostgresSaver` with `thread_id = jd_id`, and submits the ID to one process-local runner. The runner only calls `graph.stream(..., stream_mode="updates")`; LangGraph owns routing, fan-out, retries, checkpoints, and resume state. The route returns HTTP 202 immediately.

The typed top-level state holds the request, weights, parsed JD, query vectors, shortlist, reducer-merged candidate results, ranking, report state, and status. `Send` maps the candidate subgraph over the shortlist. PostgreSQL checkpoints preserve completed sends, while `screening_results` remains the API source of truth and prevents finished candidates from being evaluated twice.

Run uvicorn with exactly one worker. Multiple workers would load separate models and maintain independent queues.

### 4. JD parsing

`screening/jd_parser.py` preprocesses the JD and makes one constrained LLM call for:

```text
title, must_have[], nice_to_have[], min_years, responsibilities[]
```

Post-processing moves degree/graduation clauses out of skill lists, preserves alternative skills as one `Any of: A | B | C` item, deduplicates requirements, and removes nice-to-haves when the source has no preference language.

The code separately embeds required skills, responsibilities, preferred skills, and minimum experience. Long parts are split to the embedding token budget. Query vectors use an in-process LRU, are computed once per cold JD, and are reused by shortlist/evidence retrieval; their normalized mean is stored on the JD row. A PostgreSQL JD cache avoids both parsing and embedding when normalized text, model ID, and prompt version match.

### 5. Retrieval

`screening/retrieval.py` uses pgvector cosine distance (`<=>`). For each JD vector it materializes a bounded nearest-chunk pool (allowing HNSW use), applies embed-model/optional resume scope/soft experience filters, and aggregates the best chunk or top-three mean per resume. The highest score across query vectors wins.

Unknown experience does not exclude a resume. The screening graph evaluates every eligible resume in the requested scope. It records similarity for tie-breaking. The bounded pgvector shortlist remains available to standalone retrieval callers.

### 6. Candidate evaluation

Candidates run sequentially:

1. Stage-1 filtering rejects only when more than half of must-haves are known dictionary skills missing from metadata and full text. Unknown requirements are not counted as misses.
2. Retrieval selects at most five unique best chunks across JD query vectors. Evidence for the whole shortlist is fetched in a batched query.
3. Anonymization replaces common names/contact lines, email, phone, address, profiles, demographic terms, and photo references in the evaluator copy only.
4. Evaluator receives a byte-stable system prefix containing the rubric and ordered JD requirements, followed by candidate evidence. llama.cpp's RAM state cache reuses the longest prompt prefix. JSON-schema generation returns one entry per skill plus fit/red flags; fixed positions restore exact names and flags.
5. Python replaces experience fit: 100 if no minimum; otherwise `min(100, years / minimum * 100)`, or 50 when experience is unknown.
6. Verifier keeps only quotes found after whitespace/case normalization. Bad evidence halves confidence and adds a red flag. Unsupported positive levels fall to 1 when the skill appears elsewhere, otherwise 0. Skills-list-only evidence caps level at 1. Last use at least five years ago reduces a level once. Only this verified value enters the evaluation cache.
7. The verified result is saved immediately. An exception marks only that candidate `error`.

Filtered candidates are saved as `filtered_out` and use no Evaluator call.

The GitHub worker processes the highest resume-based ranks first. It prioritizes direct resume links, reuses repository catalog metadata, and samples up to five plausible repositories. The top three are eligible for full language and workflow checks. Each request uses a shared PostgreSQL quota governor. A lease makes interrupted jobs claimable again, and duplicate usernames are processed serially so later candidates can reuse the endpoint cache. Ambiguous project matches still use the local model and verified quotes.

GitHub results distinguish queued, partial, stale, rate-limited, inaccessible, missing, and failed evidence. The score uses 40% project match and 20% each for activity, repository quality, and JD relevance. On a cache miss, the default catalog scans up to 300 repositories and reads up to 10 attributed commits per analyzed repository. Stale entries can be reused during a quota pause and are shown with their age.

Rubric:

```text
0 None: no evidence
1 Beginner: only listed in a skills list, or minimal/coursework exposure
2 Intermediate: used in real projects/jobs with moderate responsibility
3 Advanced: owned significant work, measurable impact, multi-year use
4 Expert: architect/lead level, large scale, deep specialization
```

### 7. Scoring, ranking, reporting

Verified levels map to `0: 0.00, 1: 0.25, 2: 0.55, 3: 0.85, 4: 1.00`, multiplied by confidence. Proficiency is the must-have average, or `75% must-have + 25% nice-to-have` when preferred skills exist.

```text
final = 0.55 * proficiency
      + 0.20 * experience fit
      + 0.15 * domain fit
      + 0.10 * education fit
```

Supply four base weights summing to 1, or five weights summing to 1 with a positive base total. GitHub defaults to 10% when an `ok` result has a score. Scored partial or stale evidence uses half that weight by default. The four base weights use their full normalized proportions when a GitHub score is unavailable. A missing must-have (level 0) caps final score at 60 after weighting. Re-ranking uses stored results.

Tie order: final score, advanced must-have count, similarity, years of experience, resume UUID. Defaults are `interview >= 75`, `maybe >= 55`, otherwise `not_a_fit`.

Ranked structured results are available while the graph is `notifying` and summaries populate. The Report agent writes exactly two factual sentences (at most 35 words), uses a verified-results cache, and stores a deterministic fallback on failure. Strengths, gaps, red flags, and evidence are always derived from structured stored fields.

### 8. Candidate email review

Python proposes candidates within the interview shortlist who have an `interview` recommendation and meet the threshold and every must-have. A complete GitHub score must meet `GITHUB_MIN_SCORE` (40 by default). Candidates with unavailable GitHub evidence remain eligible. `EMAIL_MAX_SELECTED` can further cap eligible ranks. Recruiters can opt to wait for GitHub before drafting email and can omit proposals or explicitly promote an evaluated candidate.

In **Results → Selected candidates**, review the drafts, adjust the checkboxes, and record approval. Use **Upload → Add or correct a contact email** for a missing address before approving. Approval resumes the PostgreSQL-backed LangGraph interrupt after a restart. One review records the approved set and skips the other drafts. With `EMAIL_DRY_RUN=true`, approved drafts are marked skipped with a dry-run audit entry. Test live delivery on a new screening after switching modes.

Decision records capture the rank and policy checks at drafting time. Changing ranking weights updates the results table while the email panel retains the original review ranks and drafts. Candidate email content is rendered from templates and includes only the greeting, role, company, decision, and configured instructions and recruiter signature.

## Persistence

PostgreSQL requires `vector` and `pgcrypto`.

| Table | Purpose | Main data |
|---|---|---|
| `resumes` | source documents | file path, clean text, GitHub links, metadata, private contact fields, embed model |
| `resume_chunks` | searchable evidence | section, label, text, tokens, `VECTOR(768)` |
| `job_descriptions` | raw/parsed JDs | requirements, minimum, responsibilities, mean vector |
| `screening_results` | one JD/candidate pair | status, evaluation, GitHub result, score, rank, summary |
| `jd_parse_cache` | reusable JD extraction | normalized-text hash, model/prompt version, parsed JSON/vectors |
| `embedding_cache` | reusable chunk vectors | text hash, embedding model, vector |
| `evaluation_cache` | verified candidate judgments | resume/content/requirements/chunks/model/prompt key |
| `report_cache` | reusable report text | verified-results hash, model/prompt version, summary |
| `github_profiles` | reusable public GitHub data | username, endpoint metadata and README heads, fetch times, ETags |
| `github_jobs` | durable GitHub queue | candidate, username, rank priority, retries, lease, coverage, request counts |
| `github_quota_state` | shared GitHub request governor | remaining quota, reset, cooldown, pacing by resource bucket |
| `notifications` | candidate email drafts and audit | masked API contact, private address and message, review decision, attempts, delivery state |
| `checkpoints*` | LangGraph recovery | graph state, writes, blobs, migrations |

Results are unique on `(jd_id, resume_id)` and notifications are unique on `(jd_id, resume_id, email_type)`. Deleting a resume cascades to its chunks, results, and notifications. The API also removes the source after confirming it is under `DATA_DIR/uploads`.

```text
data/uploads/       hash-named sources
data/service.log    rotating timings/model metadata
data/labeled/       optional calibration data
data/test_runs/     folder-run exports
data/latency/       optional benchmarks
```

Back up PostgreSQL, `uploads`, and `runs` together.

## Setup

Prerequisites: Python 3.12, Docker Compose, NVIDIA driver/CUDA runtime, and optionally Tesseract for scanned PDFs.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# If Windows lacks the runtime DLLs:
python -m pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
docker compose up -d
docker compose ps
python scripts/init_db.py
```

Linux uses `source .venv/bin/activate` and `cp .env.example .env`; install an appropriate llama-cpp-python build. Models download on first startup. `HF_TOKEN` is optional for public models.

For an existing installation, run `python scripts/migrate_add_github.py` to add the GitHub fields and backfill resume links. It reads the stored PDF/DOCX when available and falls back to stored text. API startup also applies the schema. Run a new screening to analyze GitHub links on previously completed candidates.

Run `python scripts/migrate_add_github_scaling.py` to apply the queue and governor schema explicitly. The unauthenticated core bucket supports small batches. With the default cap of 200, the effective reserve is 6 requests for a 60-request bucket and 200 for a 5,000-request bucket. `blocked_config` means the estimated work cannot fit even in a fresh quota window. This result has no retry time, and the candidate keeps their resume-based score.

To resolve a token-related block, set `GITHUB_TOKEN` in the local `.env`, restart the API and GitHub worker, and use **Retry GitHub**. Use a fine-grained personal access token with read-only access to public repositories or a classic token with no scopes. For an authenticated quota that is too small, adjust the repository budget or reserve cap before retrying. A temporary reserve deferral instead waits until the quota resets.

Run `python scripts/migrate_add_contact.py` to backfill private email addresses and names for existing uploads. The migration works from the stored PDF/DOCX or stored text and preserves contacts already saved. New screenings enter the email review workflow.

If 5432 is occupied, change both `POSTGRES_PORT` and the port in `DATABASE_URL`. `docker compose down` keeps data; `docker compose down -v` destroys the database volume. Never commit or print `.env`.

## Configuration

Only `app/config.py` reads environment values.

| Variable | Default | Meaning |
|---|---:|---|
| `EMBED_MODEL` / `EMBED_DIM` | BGE base / `768` | model and required DB vector size |
| `LLM_REPO` / `LLM_FILE` | Qwen 2.5 3B / Q4_K_M | GGUF source |
| `LLM_N_CTX` | `4096` | context window |
| `LLM_N_GPU_LAYERS` | `-1` | requested GPU offload |
| `LLM_MAX_TOKENS` | `600` | default output budget |
| `LLM_TIMEOUT_SECONDS` | `120` | sampling/caller deadline |
| `LLM_MAX_RETRIES` | `3` | generation attempts |
| `LLM_CONCURRENCY` | `1` | fixed; validation rejects other values |
| `SHORTLIST_SIZE` | `10` | default interview shortlist |
| `CHUNK_TARGET` / `CHUNK_MAX` | `256` / `400` | tokenizer token limits |
| `CHUNK_OVERLAP_PCT` | `0.125` | allowed 0.10-0.15 |
| `USE_QUERY_PREFIX` | `false` | BGE prefix on JD queries only |
| `AGG_MODE` | `top3_mean` | `best` or top-three mean |
| `EVALUATOR_SPLIT_CALLS` | `false` | separate skill/fit calls |
| `CACHE_ENABLED` | `true` | local derived-data caches |
| `CACHE_TTL_DAYS` | `7` | reserved for other external-data caches |
| `GITHUB_ENABLED` | `true` | evaluate public GitHub links |
| `GITHUB_TOKEN` | empty | optional GitHub REST API token |
| `GITHUB_API_BASE` | `https://api.github.com` | official GitHub REST origin |
| `GITHUB_CACHE_TTL_HOURS` / `GITHUB_STABLE_CACHE_TTL_HOURS` | `24` / `120` | freshness for volatile and stable endpoint data |
| `GITHUB_MAX_REPOS` | `8` | configured cap, with a five-repository tier for collection |
| `GITHUB_CATALOG_MAX_PAGES` / `GITHUB_COMMITS_PER_PAGE` | `3` / `10` | catalog page cap and author-filtered commit sample |
| `GITHUB_RESERVE_REMAINING` / `GITHUB_MAX_CONCURRENCY` | `200` / `4` | cap on the reserve of 10% of the bucket limit rounded up, and concurrent-request cap |
| `GITHUB_SHORTLIST_SIZE` | `100` | candidates considered for initial GitHub queue, ordered by resume rank |
| `GITHUB_PARTIAL_WEIGHT_FACTOR` | `0.5` | fraction of the GitHub weight used by partial or stale scores |
| `GITHUB_TIMEOUT_SECONDS` | `10` | HTTP timeout |
| `GITHUB_SCORE_WEIGHTS` | `[0.4,0.2,0.2,0.2]` | project match, activity, code quality, JD relevance |
| `RETRIEVAL_CHUNK_LIMIT` | `1000` | bounded nearest-chunk pool |
| `MAX_UPLOAD_BYTES` | `20971520` | per uploaded file |
| `INTERVIEW_THRESHOLD` / `MAYBE_THRESHOLD` | `75` / `55` | recommendations |
| `GITHUB_MIN_SCORE` / `EMAIL_MAX_SELECTED` | `40` / unlimited | GitHub eligibility threshold and optional maximum rank |
| `EMAIL_DRY_RUN` / `EMAIL_AUTO_SEND` / `SEND_REJECTION_EMAILS` | `true` / `false` / `false` | suppress delivery, bypass interview-email review, enable rejection drafts |
| `COMPANY_NAME` / `RECRUITER_NAME` / `RECRUITER_EMAIL` / `NEXT_STEP_TEXT` | see `.env.example` | template content and optional reply-to address |
| `SMTP_HOST` / `SMTP_PORT` / `EMAIL_FROM` | loopback / `1025` / example address | mail server and sender |
| `SMTP_USE_TLS` / `SMTP_STARTTLS` / `SMTP_USERNAME` / `SMTP_PASSWORD` | disabled / empty | TLS mode and optional SMTP authentication |
| `EMAIL_RATE_PER_MINUTE` / `EMAIL_MAX_RETRIES` | `20` / `3` | rate limit and additional attempts after transient SMTP failures |
| `DATA_DIR` | `data` | private runtime root |
| `API_URL` | `http://127.0.0.1:8000` | UI backend |

If 4 GB VRAM is insufficient, lower `LLM_N_CTX` (for example 3072) or `LLM_N_GPU_LAYERS`. Enable split calls when long skill lists exhaust output tokens.

## Run

```powershell
# Terminal 1: always one worker
uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1
# Terminal 2
python app.py
```

Start the GitHub worker in a separate terminal with `python -m app.workers.github_worker`. On a 4 GB GPU, set `LLM_N_GPU_LAYERS=0` in that terminal before startup so optional GitHub project matching uses CPU while the API owns the GPU.

In Streamlit, upload resumes and select which ones to screen under **Job description**. Open **Results** for the candidate ranking and GitHub evidence. Streamlit starts ranking-only runs and skips the email workflow.

### Decisions

- The requested REST tiers cost 20 calls for a cold five-repository sample with a one-page catalog, before optional workflows or uncatalogued links. The count includes the profile, catalog, three detail calls per repo, and three language calls. Preserve those evidence calls and size the reserve against the larger estimate.
- The existing per-username endpoint cache is shared across jobs. Concurrent claims serialize a username. Candidates with different linked projects can still require different endpoints.

Open `http://127.0.0.1:8501`; API docs are at `http://127.0.0.1:8000/docs`. Upload resumes, submit a JD, watch two-second result polling, then optionally re-rank stored results with new weights.

## API

| Method/path | Contract |
|---|---|
| `POST /resumes` | multipart `files`; PDF/DOCX independently processed |
| `GET /resumes` | metadata list; no full resume text |
| `PATCH /resumes/{id}/contact` | set `{"email":"candidate@example.org"}`; return a masked address |
| `DELETE /resumes/{id}` | DB cascade plus source removal |
| `POST /screenings` | queue JSON/multipart request; return `jd_id` (202) |
| `POST /screenings/rank-only` | same screening input and response, with email disabled for the run |
| `GET /screenings/{id}/status` | run state and candidate counts |
| `GET /screenings/{id}/results` | partial/final results and evidence |
| `POST /screenings/{id}/rerank` | score/rank stored evaluations |
| `GET /screenings/{id}/github` | status counts, queue depth, quota, estimated drain time, and network/cache totals |
| `POST /screenings/{id}/github/extend` | queue the next `{"count":25}` candidates in resume-based rank order |
| `POST /screenings/{id}/github/retry` | requeue failed or deferred `{"resume_ids":[...]}` through the same governor |
| `GET /screenings/{id}/notifications` | dry-run and review flags, masked addresses, previews, decisions, and delivery status |
| `POST /screenings/{id}/notifications/approve` | `{"resume_ids":[...]}` or `{"all_selected":true}`; use `"override":true` for an explicit promotion and `"rejected_resume_ids":[...]` for opt-in rejection drafts |
| `POST /screenings/{id}/notifications/reject` | skip the specified `resume_ids` during review; optional `email_type` defaults to `selected` |
| `POST /screenings/{id}/notifications/retry` | queue failed notifications for `resume_ids`; optional `email_type` defaults to `selected` |
| `GET /health` | DB/models and actual offloaded GPU layers |

JSON screening body:

```json
{
  "jd_text": "A sufficiently detailed job description...",
  "shortlist_size": 10,
  "resume_ids": ["optional-resume-uuid"],
  "weights": {
    "proficiency": 0.55,
    "experience": 0.20,
    "domain": 0.15,
    "education": 0.10
  }
}
```

`jd_text` is 10-30000 characters. Omit `resume_ids` to screen all eligible indexed resumes. Supply `wait_for_github: true` to wait for queued GitHub evidence before email review. Multipart also accepts these fields with `file` (PDF/DOCX/TXT), and expects JSON-encoded `resume_ids` and `weights`.

Candidate statuses are `pending`, `done`, `filtered_out`, and `error`. Graph states are `queued`, `running`, `evaluating`, `notifying`, `awaiting_approval`, `done`, and `error`. Status uses `interrupted` when no checkpoint exists.

Results and rerank responses include `github_status`, `github_score`, and structured `github_json` with project verdicts, verified evidence, profile signals, flags, and `score_explanation`. The breakdown shows what each factor found, the weights saved with new results, and why a score is missing when evidence is unavailable. The UI displays it inside each candidate’s results. Folder exports include it in JSON and in the `github_explanation` CSV column. Older results retain their recorded subscores and signals.

## LLM behavior

`services/llm_service.py` loads the model once, enables a 512 MiB llama.cpp RAM prefix cache, serializes native calls with a semaphore and lock, checks deadlines during sampling, validates constrained JSON with Pydantic, and retries generation/validation failures with backoff. It logs cache hit/miss and reused prefix-token counts, but never raw prompts or credentials.

A token-budgeted sliding window keeps system/developer instructions, original task/evidence, and newest user turn while dropping oldest complete continuation turns. Essential input that still cannot fit raises instead of being truncated.

A native CUDA hang cannot be killed safely from a Python thread. A caller may time out while the generation lock remains occupied until native code returns; restart the API if it never does.

## Tests and operational checks

To test the email workflow with a local SMTP catcher, start Mailpit with `docker compose --profile email-dev up -d mailpit` and open `http://127.0.0.1:8025`. Use synthetic resumes and leave `EMAIL_AUTO_SEND=false`. Set `EMAIL_DRY_RUN=false` only when ready to test delivery through the loopback Mailpit SMTP server. The optional `python scripts/check_email.py` command submits a synthetic message directly to that catcher. [Mailpit Docker image](https://hub.docker.com/r/axllent/mailpit/tags)

The automated notification tests use fake SMTP. With a test database that has the schema applied, set `RUN_POSTGRES_TESTS=1` to exercise the durable approval gate, masking, failure isolation, retries, and duplicate prevention.

```powershell
python -m pytest -q
python -c "from app.config import settings"
python scripts/init_db.py
python scripts/check_embeddings.py
python scripts/check_ingest.py
python scripts/check_llm.py
python scripts/check_retrieval.py
# with API running:
python scripts/check_flow.py
python scripts/check_ui.py
```

The tests cover normalization/chunking, dates, JD splitting/normalization, scoped retrieval, schema alignment, retries, LLM serialization/deadlines, evidence verification, scoring/ties, report fallback, API validation, graph routing, failure isolation, checkpoints, and vector reuse.

| Tool | Purpose |
|---|---|
| `benchmark_latency.py --output FILE` | ten-resume cold run plus identical zero-LLM warm rerun |
| `test_folders.py RESUMES JDS --output DIR` | scoped folder run and CSV/JSON export |
| `check_recovery.py prepare|verify|retry` | restart-recovery acceptance |
| `check_github.py` | public demo profile, direct project verification, and zero-request warm-cache acceptance |
| `test_github_batch.py` | run the supplied resumes and JD through GitHub evidence collection and save quota, cache, and outcome metrics |

GitHub batch reports and the progress API count `reserve_deferrals` and `config_blocks` separately from `primary_limit_hits` and `secondary_limit_hits`. The limit-hit counters count responses from GitHub. A local cooldown can therefore defer several candidates after one HTTP rejection.
| `create_calibration.py` | synthetic calibration fixtures |
| `eval_retrieval.py [labels] [--allow-synthetic] [--rechunk]` | recall@10/25 |
| `eval_evaluator.py JD_ID [labels] [--allow-synthetic]` | level agreement/MAE |
| `clear_cache.py` | clear derived caches without deleting sources/results |

```powershell
python scripts/test_folders.py "data/Aviators Resumes" data/jd --output data/test_runs/aviators
```

The folder manifest preserves IDs; reruns reuse finished screenings and refresh UTF-8 exports. `--upload-only` only indexes, and `--shortlist-size N` overrides configuration. Synthetic labels are smoke tests, not quality evidence. Use human-reviewed representative labels for calibration.

`data/service.log` records `Pipeline stage=... latency=... success=...`. Parent timings include children; do not sum both. The latency benchmark excludes upload transport, queue wait, and UI polling.

## Security and limitations

- The API has no authentication. Bind to loopback; add auth and TLS before remote/multi-user use.
- Raw files, clean text, PostgreSQL checkpoints, and exports contain private data. Protect both the database and `data/`.
- Regex anonymization is not certified de-identification and can miss unusual formats/languages.
- Skill metadata is finite and English-oriented. Unknown skills can still reach semantic retrieval/evaluation.
- PDF column ordering is heuristic; complex layouts may parse poorly.
- Tesseract is separate and OCR errors are not corrected automatically.
- The bounded approximate shortlist can underfill dense/scoped corpora; calibrate `RETRIEVAL_CHUNK_LIMIT`.
- One worker favors 4 GB GPU stability over throughput.
- Recommendations support human review; they are not autonomous hiring decisions.
- With live SMTP enabled, `EMAIL_AUTO_SEND=true` sends interview invitations without recruiter review. Optional rejection emails still require explicit approval.
- Follow applicable consent and data protection requirements for candidate communications. Keep the notification audit trail private. Use an authenticated sending domain with SPF and DKIM for production email.
- SMTP delivery with an unknown outcome stays in `sending`. Check the mail provider for confirmation. Only known failures are eligible for retry, which prevents duplicate delivery after a restart.
- GitHub activity reflects free time and opportunity as well as experience. GitHub is a supporting signal and has no automatic rejection rule.
- GitHub requests use the public REST API with validated user and repository paths. The service reads public metadata and README excerpts and never clones or runs candidate repositories. The token stays in request headers and is never logged. Conditional requests refresh expired cached data, and rate-limit responses pause live requests.
- Synthetic calibration does not establish accuracy, fairness, or production readiness.

## Change guide for coding agents

- New configuration: change `config.py`; update `.env.example` when user-settable and update this table.
- Schema change: change `db/init.sql`, all reads/writes, public models/UI/exports, and tests. `IF NOT EXISTS` does not migrate existing columns.
- Embedding/dimension change: update both `VECTOR(n)` columns, rebuild embeddings/index, and verify startup checks.
- Result field: update Pydantic, evaluator/verifier/ranking as relevant, SQL, API/UI/export, and tests.
- Prompt/schema change: preserve constrained validation, fixed-position skill alignment, retry privacy, and evidence verification.
- Graph change: preserve typed state/reducers, `Send` fan-out, immediate saves, candidate failure isolation, PostgreSQL checkpoint idempotence, and single-model concurrency.
- Scoring change: keep it in Python and update re-ranking/tie tests.
- Upload format: update parser, API/UI accepted types, fixtures, and tests together.

Run focused unit tests plus the smallest real acceptance script that crosses the changed boundary.
#   H i r e s m a r t V 2  
 