# AGENTS.md: Resume Screening Analyzer (Build Spec)

You are building a **Resume Screening Analyzer**. Recruiters upload resumes and a job description (JD). The system ranks candidates by proficiency for that JD and explains, with evidence, why each candidate matches or not.

Read this whole file before writing code. Follow the build order in section 14 and stop to verify each step's acceptance check before moving on.

---

## 1. Hard constraints

- **Fully open source.** No paid APIs.
- **Hardware target:** NVIDIA GTX 1650 (4 GB VRAM), 24 GB RAM, Intel i5. Design for this.
- **LLM:** `Qwen/Qwen2.5-3B-Instruct-GGUF` (Q4_K_M) via `llama-cpp-python` (CUDA build). Loaded **once** at app startup and reused for every run.
- **Embeddings:** `BAAI/bge-base-en-v1.5` (768 dims) via `sentence-transformers`, **on CPU** (keep VRAM free for the LLM). Loaded once at startup.
- **Database:** PostgreSQL + pgvector, run in Docker (section 4).
- **LLM concurrency = 1.** One local model handles one generation at a time. Use a lock plus a semaphore so requests queue instead of crashing the model.
- **Only 2 real agents:** Evaluator and Report. Everything else is plain functions or a single LLM call.
- **Scores are computed in code, never by the LLM.** The LLM only judges proficiency levels and writes short text.
- **Secrets never go in code or git.** Read from environment variables (section 3).

---

## 2. Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Schemas / validation | Pydantic v2 |
| LLM runtime | llama-cpp-python (GGUF, JSON-schema constrained output) |
| Model download/cache | huggingface-hub |
| Embeddings | sentence-transformers |
| Database | PostgreSQL 16 + pgvector, `psycopg[binary]`, `pgvector` |
| Parsing | PyMuPDF, python-docx, pytesseract (OCR fallback only) |
| Anonymizing | regex first, Presidio optional |
| Backend | FastAPI + uvicorn |
| UI | Streamlit |

Python 3.10+.

---

## 3. Environment variables

The user provides these values. Create `.env.example` (committed) and read `.env` (git-ignored).

`.env.example`:

```env
# --- Hugging Face ---
HF_TOKEN=your_huggingface_token_here

# --- Postgres (used by docker-compose AND the app) ---
POSTGRES_USER=resume_user
POSTGRES_PASSWORD=change_me
POSTGRES_DB=resume_db
POSTGRES_PORT=5432
DATABASE_URL=postgresql://resume_user:change_me@localhost:5432/resume_db

# --- Models ---
EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBED_DIM=768
EMBED_DEVICE=cpu
LLM_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF
LLM_FILE=*q4_k_m.gguf
LLM_N_CTX=4096
LLM_N_GPU_LAYERS=-1
LLM_MAX_TOKENS=600

# --- Pipeline ---
SHORTLIST_SIZE=10
LLM_CONCURRENCY=1
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=3
DATA_DIR=./data
```

Rules:
- Add `.env`, `data/`, `__pycache__/`, `.venv/` to `.gitignore`.
- Load config in one place (`app/config.py`, using `pydantic-settings` or `python-dotenv`). No other module reads `os.environ` directly.
- Export `HF_TOKEN` as `HUGGING_FACE_HUB_TOKEN` at startup so `huggingface-hub` picks it up.
- If `HF_TOKEN` is missing, log a warning and continue (the Qwen GGUF repo is public); do not crash.

---

## 4. Docker: database

Create `docker-compose.yml` in the repo root exactly like this:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: resume_pgvector
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/01_init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgdata:
```

### 4.1 Required files for the DB setup

```
resume-screening/
├── docker-compose.yml     # the YAML above
├── .env                   # user fills this (section 3)
└── db/
    └── init.sql           # schema from section 5, must be in db/ because compose mounts ./db/init.sql
```

Codex must create `docker-compose.yml` and `db/init.sql` exactly as specified in sections 4 and 5. The user provides `.env` with `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`, `DATABASE_URL` and `HF_TOKEN`. `DATABASE_URL` must match the three `POSTGRES_*` values.

### 4.2 Setup steps

1. Install Docker Desktop and make sure it is running. On Windows, WSL2 must be enabled in Docker Desktop settings.
2. Put `docker-compose.yml`, `.env` and `db/init.sql` in the project root as shown above.
3. Start the database:

```bash
docker compose up -d
```

4. Wait until the status is `healthy`:

```bash
docker compose ps
```

5. Verify the tables exist (replace the user and database with the `.env` values):

```bash
docker exec -it resume_pgvector psql -U resume_user -d resume_db -c "\dt"
docker exec -it resume_pgvector psql -U resume_user -d resume_db -c "\dx"
```

Expected: tables `resumes`, `resume_chunks`, `job_descriptions`, `screening_results`, and the extensions `vector` and `pgcrypto`.

### 4.3 Useful commands

| Task | Command |
|---|---|
| Stop the DB (keeps data) | `docker compose down` |
| Start again | `docker compose up -d` |
| View logs | `docker compose logs -f db` |
| Reset everything (deletes all data) | `docker compose down -v && docker compose up -d` |

### 4.4 Notes

- `db/init.sql` only runs on the **first** start (empty volume). If the schema changes later, either reset with `docker compose down -v` (deletes all data) or apply the change manually.
- Also provide `scripts/init_db.py` that applies `db/init.sql` idempotently (all statements use `IF NOT EXISTS`), for use on an existing database or on the user's own Postgres. That Postgres must have the pgvector extension installed.
- If port 5432 is already in use, set `POSTGRES_PORT=5433` in `.env` and update the port inside `DATABASE_URL` as well.
- Never commit `.env`. It contains the DB password and the HF token.
- Troubleshooting: if the container is not healthy, check `docker compose logs db`. If tables are missing, the volume was probably created before `init.sql` existed, so reset with `docker compose down -v`.

---

## 5. Database schema

Create `db/init.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

-- 1. Resumes: metadata + full clean text
CREATE TABLE IF NOT EXISTS resumes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name     TEXT NOT NULL,
    file_path     TEXT NOT NULL,              -- raw file stays on disk
    parse_method  TEXT,                       -- pymupdf | pdfplumber | docx | ocr
    clean_text    TEXT NOT NULL,              -- original cleaned text (not anonymized)
    skills        TEXT[] DEFAULT '{}',
    years_exp     FLOAT,
    education     TEXT,
    embed_model   TEXT NOT NULL,              -- which model produced the vectors
    uploaded_at   TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_resumes_years  ON resumes (years_exp);
CREATE INDEX IF NOT EXISTS idx_resumes_skills ON resumes USING GIN (skills);

-- 2. Resume chunks + embeddings (one row per section / role / project)
CREATE TABLE IF NOT EXISTS resume_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    section     TEXT NOT NULL,                -- skills | experience | project | education | summary | other
    label       TEXT,                         -- e.g. "Data Engineer at X"
    content     TEXT NOT NULL,                -- includes the context header
    token_count INT,
    embedding   VECTOR(768) NOT NULL          -- MUST equal EMBED_DIM
);
CREATE INDEX IF NOT EXISTS idx_chunks_resume ON resume_chunks (resume_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON resume_chunks USING hnsw (embedding vector_cosine_ops);

-- 3. Job descriptions
CREATE TABLE IF NOT EXISTS job_descriptions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title          TEXT,
    raw_text       TEXT NOT NULL,
    must_have      TEXT[] DEFAULT '{}',
    nice_to_have   TEXT[] DEFAULT '{}',
    min_years      FLOAT,
    responsibilities TEXT[] DEFAULT '{}',
    embedding      VECTOR(768),
    created_at     TIMESTAMP DEFAULT now()
);

-- 4. Screening results: one row per (JD, resume)
CREATE TABLE IF NOT EXISTS screening_results (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_id         UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    resume_id     UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | done | error | filtered_out
    error_message TEXT,
    similarity    FLOAT,
    final_score   FLOAT,
    rank          INT,
    skill_report  JSONB,                      -- per-skill level + evidence
    fit_scores    JSONB,                      -- experience / domain / education
    red_flags     TEXT[] DEFAULT '{}',
    explanation   TEXT,                       -- report text
    recommendation TEXT,                      -- interview | maybe | not_a_fit
    created_at    TIMESTAMP DEFAULT now(),
    UNIQUE (jd_id, resume_id)
);
CREATE INDEX IF NOT EXISTS idx_results_jd ON screening_results (jd_id, rank);
```

If `EMBED_DIM` is changed, the `VECTOR(n)` columns must change too. Add a startup check that compares `EMBED_DIM` with the actual column dimension and fails fast with a clear message on mismatch.

---

## 6. Project structure

```
resume-screening/
├── AGENTS.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── db/
│   └── init.sql
├── scripts/
│   ├── init_db.py
│   └── eval_retrieval.py          # recall@k on a labeled set (section 13)
├── data/                          # git-ignored: uploads, labeled test set
├── app/
│   ├── config.py
│   ├── db.py                      # connection pool, queries
│   ├── models.py                  # Pydantic schemas (section 9)
│   ├── services/
│   │   ├── llm_service.py         # load-once LLM + lock + generate_json
│   │   └── embed_service.py       # load-once embedder + batch encode
│   ├── ingest/
│   │   ├── parse.py
│   │   ├── preprocess.py
│   │   ├── chunk.py
│   │   ├── metadata.py
│   │   └── pipeline.py            # process_upload(file)
│   ├── screening/
│   │   ├── jd_parser.py
│   │   ├── retrieval.py
│   │   ├── stage1_filter.py
│   │   ├── anonymize.py
│   │   ├── evaluator.py           # Evaluator Agent
│   │   ├── verifier.py
│   │   ├── ranking.py
│   │   ├── report.py              # Report Agent
│   │   └── graph.py               # LangGraph wiring
│   ├── api.py                     # FastAPI
│   └── ui.py                      # Streamlit
└── tests/
```

---

## 7. Phase A: Upload pipeline (plain functions, NO agents)

`ingest/pipeline.py: process_upload(file)` runs: parse → preprocess → metadata → chunk → embed → store.

### 7.1 Parse (`parse.py`)
- PDF: PyMuPDF (`fitz`). Read **block by block sorted by position** to handle two-column layouts; do not read line by line across columns.
- DOCX: `python-docx` (paragraphs, headings, tables).
- OCR fallback: only if a page returns fewer than ~30 characters. Render the page to an image and run Tesseract. If Tesseract is not installed, log a clear warning and skip OCR rather than crash.
- Return text plus `{file_name, page_count, parse_method}`.

### 7.2 Preprocess (`preprocess.py`)
1. Unicode normalize (NFKC), replace odd bullets, ligatures, non-breaking spaces.
2. Remove repeated headers/footers and "Page X of Y".
3. Rejoin words and sentences broken by layout, but **keep bullet points on separate lines**.
4. Collapse extra whitespace and blank lines.
5. Detect section headings, case-insensitive, with variants: Summary/Profile/Objective, Skills/Technical Skills, Experience/Work History/Employment, Projects, Education, Certifications.
6. Standardize dates ("Jan 2021 – Present" → ISO-like) so experience years can be computed.
7. **Do not lowercase or strip punctuation.** Terms like `C++`, `.NET`, `C#`, `Node.js` must survive.

### 7.3 Metadata (`metadata.py`)
Extract `skills[]`, `years_exp`, `education` using rules (skills dictionary + regex, date-range arithmetic for years). Optionally one small LLM call as a fallback. These fill the filter columns in `resumes`.

### 7.4 Chunk (`chunk.py`): section-based, not blind fixed-size
- Split by section.
- Experience: **one chunk per job role** (title, company, dates, bullets).
- Projects: **one chunk per project**.
- Skills: one chunk. Education: one chunk.
- Target **150–300 tokens**, hard max **400**, minimum **~30** (merge tiny chunks into the neighbor).
- If a chunk exceeds 400 tokens, split at bullet boundaries (never mid-sentence) with **10–15% overlap**. Overlap is used **only** for oversized chunks and for the fallback.
- Fallback when no headings are detected: fixed-size chunks (~256 tokens) with ~15% overlap.
- Prepend a context header to each chunk, e.g. `Experience – Data Engineer at Acme (2021–2024):` so a bullet makes sense on its own.
- Make `CHUNK_TARGET`, `CHUNK_MAX`, `CHUNK_OVERLAP_PCT` constants in config so they can be tuned (section 13).
- Count tokens with the embedding model's tokenizer.

### 7.5 Embed (`embed_service.py`)
- Load `SentenceTransformer(EMBED_MODEL, device="cpu")` **once**, module-level, behind a lock.
- `encode(texts, batch_size=32, normalize_embeddings=True)`.
- **No prefix on resume chunks.**

### 7.6 Store
- Save the raw file under `DATA_DIR/uploads/`, only the path goes in the DB.
- Insert into `resumes`, then `resume_chunks` in one transaction. Store `embed_model` on the resume row.
- Duplicate protection: hash the file content; skip or report if the same hash already exists.

---

## 8. Phase B: Screening pipeline

Flow per JD run:

```
JD text
  → [1] JD Parser (one LLM call)           → save job_descriptions
  → [2] Retrieval (SQL + pgvector)         → top SHORTLIST_SIZE resumes
  → [3] Stage-1 filter (code)              → drop obvious must-have misses
  → [4] Per candidate, SEQUENTIAL (concurrency 1):
          Anonymize (code)
          → Evaluator Agent (LLM)
          → Verifier (code)
          → save result immediately
  → [5] Ranking (code)
  → [6] Report Agent (LLM, short)
  → save → UI
```

### 8.1 JD Parser (`jd_parser.py`)
One LLM call with JSON-schema output (`ParsedJD`): title, must_have[], nice_to_have[], min_years, responsibilities[]. Then embed the JD (see 8.2) and save to `job_descriptions`.

### 8.2 Retrieval (`retrieval.py`), no LLM
- Embed the JD. Because JDs can exceed the 512-token limit (silent truncation), embed these parts **separately**: must-have skills, responsibilities, experience requirement. Search with each and combine by taking the max (or mean) similarity per resume.
- Make the bge query prefix (`"Represent this sentence for searching relevant passages: "`) a config flag `USE_QUERY_PREFIX`, default **false**. It is evaluated in section 13. **Never** add it to resume chunks.
- Similarity: **cosine**, operator `<=>`. The query operator must match the HNSW opclass (`vector_cosine_ops`) or the index is not used.
- Aggregate chunks to resumes: **best chunk per resume**, or average of top 2–3 chunks (config `AGG_MODE`, default `top3_mean`).
- Apply metadata filter `years_exp >= min_years` **only as a soft filter**: if `years_exp` is NULL, do not exclude the resume.

Reference query shape:

```sql
SELECT r.id, MIN(c.embedding <=> %(q)s) AS distance
FROM resume_chunks c
JOIN resumes r ON r.id = c.resume_id
WHERE (r.years_exp IS NULL OR r.years_exp >= %(min_years)s)
GROUP BY r.id
ORDER BY distance
LIMIT %(n)s;
```

Similarity only **shortlists**. It cannot judge "5 years required, candidate has 1". Final ranking comes from the Evaluator + scoring.

### 8.3 Stage-1 filter (`stage1_filter.py`), code only
Compare the JD's must-have skills with `resumes.skills[]` (case-insensitive, with a small alias table: `js`↔`javascript`, `k8s`↔`kubernetes`, etc.). A candidate missing **most** must-haves is marked `filtered_out` (saved, but no LLM call). Keep this conservative: when in doubt, let the candidate through. Skills not in the `skills[]` list may still be in the text.

### 8.4 Anonymize (`anonymize.py`), code
Strip name, email, phone, address, URLs to personal profiles, gender/age/marital terms and photo references from the **evaluator copy** only. Keep the original in the DB.

### 8.5 Evaluator Agent (`evaluator.py`)
**Input to the prompt (keep it short for a 3B model):**
- the JD must-have and nice-to-have lists,
- the short rubric below,
- the **top 4–5 best matching chunks** for this JD from this resume (use `search within resume` by cosine),
- a one-line skills summary.

Do **not** send the entire resume.

**Rubric (use these exact words, one line each):**

```
0 None: no evidence
1 Beginner: only listed in a skills list, or minimal/coursework exposure
2 Intermediate: used in real projects/jobs with moderate responsibility
3 Advanced: owned significant work, measurable impact, multi-year use
4 Expert: architect/lead level, large scale, deep specialization
```

**Rules in the prompt:**
- Every level above 0 MUST include at least one exact quote from the provided text as evidence.
- A skill that appears only in a skills list is at most level 1.
- A skill last used 5+ years ago is downgraded one level.
- Treat equivalent technologies sensibly (e.g. PyTorch partly supports a TensorFlow requirement) and say so in the reasoning.
- If evidence is ambiguous, lower the confidence, do not inflate the level.
- Do not infer skills the text does not support.

**Output:** `EvaluatorOutput` (section 9) via JSON-schema constrained generation. Temperature 0.

**If quality on the 3B model is poor,** split into two smaller calls: (a) skill proficiency, (b) experience/domain/education fit. Make this a config switch `EVALUATOR_SPLIT_CALLS`.

**Version 1 uses NO tool calling.** Code gathers the chunks and the agent only judges. An optional tool `search_resume(resume_id, query)` may be added later behind a flag (max 3 calls), but must not be on by default.

### 8.6 Verifier (`verifier.py`), code
For every evidence quote, check it exists in the candidate's text (case-insensitive, whitespace-normalized, fuzzy ratio ≥ 0.9 allowed). If not found: halve that skill's confidence and add a red flag `"Unverified evidence for <skill>"`. If a level > 0 has **no** verified evidence at all, downgrade it to level 1 (or 0 if listed nowhere).

### 8.7 Save immediately
After each candidate, upsert into `screening_results` with `status='done'` (or `error`). Before evaluating, **skip any (jd_id, resume_id) already `done`**. A crash must never redo finished work.

### 8.8 Ranking (`ranking.py`), code only

```python
LEVEL_WEIGHT = {0: 0.0, 1: 0.25, 2: 0.55, 3: 0.85, 4: 1.0}
WEIGHTS = {"proficiency": 0.55, "experience": 0.20, "domain": 0.15, "education": 0.10}  # configurable

# skill score (0-100)
def avg(items):
    return sum(LEVEL_WEIGHT[s.level] * s.confidence for s in items) / len(items) if items else 0
prof = 100 * (0.75 * avg(must_have_skills) + 0.25 * avg(nice_to_have_skills))

final = (WEIGHTS["proficiency"] * prof
       + WEIGHTS["experience"]  * fit.experience     # each fit score is 0-100
       + WEIGHTS["domain"]      * fit.domain
       + WEIGHTS["education"]   * fit.education)

# must-have cap
if any(s.is_must_have and s.level == 0 for s in skills):
    final = min(final, 60)
```

- If there are no nice-to-have skills, use 100% must-have weight for the proficiency score.
- Tie-breaks: more must-haves at level ≥ 3, then higher `similarity`, then more years of experience.
- Optionally compare the top 3 head to head and add a "why A over B" line (config flag).
- Recommendation thresholds (config): `≥ 75` interview, `55–74` maybe, `< 55` not_a_fit.
- Weights must be adjustable from the UI and re-rankable **without** calling the LLM again (scores are stored per component).

### 8.9 Report Agent (`report.py`)
Mostly template-based so it is fast on a small model:
- Template fills **strengths** (skills with level ≥ 3 and their verified evidence), **gaps** (must-haves at level 0–1), **red flags**, and **recommendation** straight from the stored JSON.
- The LLM writes **only a 2–3 sentence summary** per candidate (max ~120 tokens), grounded in the stored data.
- Output per candidate: rank, score, recommendation, summary, strengths, gaps, red flags, evidence quotes.
- Also produce the overall ranked table.

---

## 9. Pydantic schemas (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ParsedJD(BaseModel):
    title: str
    must_have: List[str]
    nice_to_have: List[str]
    min_years: Optional[float] = None
    responsibilities: List[str]

class SkillProficiency(BaseModel):
    skill: str
    is_must_have: bool
    level: int = Field(ge=0, le=4)
    years_used: Optional[float] = None
    last_used_year: Optional[int] = None
    evidence: List[str] = Field(description="Exact quotes from the provided resume text")
    reasoning: str = Field(description="1-2 sentences")
    confidence: float = Field(ge=0, le=1)

class FitScores(BaseModel):
    experience: float = Field(ge=0, le=100)
    domain: float = Field(ge=0, le=100)
    education: float = Field(ge=0, le=100)
    notes: str

class EvaluatorOutput(BaseModel):
    skills: List[SkillProficiency]
    fit: FitScores
    red_flags: List[str]
```

Pass `Model.model_json_schema()` to `llama-cpp-python` via `response_format={"type": "json_object", "schema": ...}`. Always still validate with Pydantic after generation and retry on failure.

---

## 10. LLM service (`llm_service.py`)

Requirements:
- `get_llm()` loads `Llama.from_pretrained(repo_id=LLM_REPO, filename=LLM_FILE, n_ctx=4096, n_gpu_layers=-1, n_batch=256, verbose=False)` **once**, using double-checked locking. The GGUF downloads once to the Hugging Face cache and is reused after.
- `generate_json(messages, schema, max_tokens)` takes the global lock so only **one generation runs at a time**, uses `temperature=0`, and returns parsed JSON.
- Per-call timeout (`LLM_TIMEOUT_SECONDS`) and retry with exponential backoff + jitter (`LLM_MAX_RETRIES`, e.g. 1s, 2s, 4s). A candidate that still fails is marked `status='error'` and the batch **continues**.
- Call `get_llm()` and the embedder loader at **application startup** (FastAPI lifespan; `@st.cache_resource` if Streamlit loads them directly), so the first request is not slow and the model is never reloaded between runs.
- Log load time and tokens/second per call.
- `llama-cpp-python` must be the **CUDA** build. At startup, log how many layers were offloaded to the GPU; if it silently runs on CPU, log a clear warning.
- If out-of-memory occurs: lower `LLM_N_CTX` to 3072 or reduce `LLM_N_GPU_LAYERS`. An optional upgrade is `Qwen2.5-7B-Instruct` Q4_K_M with `n_gpu_layers` ≈ 16–20 (slower, better judgment), switchable only through env vars.

---

## 11. Bottleneck protection (must be implemented)

| Problem | Required fix |
|---|---|
| Too many simultaneous LLM requests | Semaphore (`LLM_CONCURRENCY=1`) + lock in `llm_service` |
| Timeouts / random failures | Per-call timeout, retry with backoff, mark candidate `error`, continue |
| Huge prompts | Shortlist of `SHORTLIST_SIZE`, only top 4–5 chunks per candidate |
| Wasted LLM calls | Stage-1 code filter first |
| Crash means redoing work | Save each result immediately, skip finished pairs |
| Frozen UI | Progress counter ("6 of 10 evaluated"), show partial results |
| VRAM pressure | Embeddings on CPU only |

Run screening as a **background task** (FastAPI `BackgroundTasks` or a simple worker thread), with a status endpoint the UI polls. Do not block the request.

---

## 12. API and UI

### FastAPI (`api.py`)
- `POST /resumes` upload one or many files, returns resume ids and per-file status.
- `GET /resumes` list resumes with metadata.
- `DELETE /resumes/{id}`
- `POST /screenings` body: JD text (or file) + optional `shortlist_size`, weights. Starts a background run, returns `jd_id`.
- `GET /screenings/{jd_id}/status` → `{total, done, errors, filtered_out, state}`.
- `GET /screenings/{jd_id}/results` → ranked results with evidence and explanation.
- `POST /screenings/{jd_id}/rerank` body: new weights → re-rank from stored scores, no LLM call.
- `GET /health` → DB connectivity, model loaded flags, GPU layers offloaded.

### Streamlit (`ui.py`)
- Upload page (multi-file), resume list.
- Screening page: paste/upload JD, run, **live progress bar**.
- Results page: ranked table (rank, anonymized id, score, recommendation), expandable per-candidate detail (skill levels with evidence, gaps, red flags, summary), weight sliders that call `rerank`.
- Show a visible note: **"This tool recommends. A human makes the final decision."**

---

## 13. Quality, evaluation, and bias

- **Bias reduction:** the evaluator never sees name, gender, age, photo, address. The UI shows candidates by id/anonymized label until the recruiter opens a detail view.
- **Evidence is mandatory** and verified in code.
- **Calibration set:** create `data/labeled/` with 10–20 resumes and 2–3 JDs with human-judged expected ranking/levels.
- `scripts/eval_retrieval.py`: compute **recall@10 / recall@25** on the labeled set. Use it to:
  1. choose chunk target size (try 128 / 256 / 384),
  2. compare `USE_QUERY_PREFIX` true vs false,
  3. compare `AGG_MODE` best-chunk vs top3-mean.
- A second script/report compares evaluator levels with the human labels (agreement rate, mean absolute error) so rubric and prompts can be tuned.
- Log every LLM call (prompt size, latency, validation failures) to a file for debugging.

---

## 14. Build order (follow strictly; verify each acceptance check)

1. **Scaffold + config.** Folder structure, `requirements.txt`, `.env.example`, `.gitignore`, `config.py`.
   *Check:* `python -c "from app.config import settings"` works with `.env` filled.
2. **Docker DB + schema.** `docker-compose.yml`, `db/init.sql`, `scripts/init_db.py`, `db.py`.
   *Check:* `docker compose up -d` becomes healthy; all 4 tables and the `vector` extension exist; dimension-check passes.
3. **Embed service.** Load-once bge model on CPU, batch encode.
   *Check:* encoding 32 texts returns shape `(32, 768)`, normalized, model loaded only once across repeated calls.
4. **Ingest pipeline.** parse → preprocess → metadata → chunk → embed → store.
   *Check:* a sample PDF and DOCX each produce rows in `resumes` and `resume_chunks` with sensible sections; two-column PDF is read in correct order; `C++`/`.NET` preserved.
5. **LLM service.** Load-once GGUF, JSON-schema generation, lock, retry, timeout.
   *Check:* two sequential calls show model loaded once; output validates against a test Pydantic schema; GPU offload logged.
6. **JD Parser.**
   *Check:* a sample JD yields must_have / nice_to_have / min_years and a saved row.
7. **Retrieval.** Multi-part JD embedding, cosine search, aggregation, soft filter.
   *Check:* returns top N resume ids with similarity; EXPLAIN shows the HNSW index is used.
8. **Stage-1 filter + anonymizer.**
   *Check:* unit tests on alias matching and PII stripping.
9. **Evaluator + Verifier.**
   *Check:* output validates; fabricated quote in a test case is caught and confidence halved.
10. **Ranking.**
   *Check:* unit tests for formula, must-have cap at 60, tie-breaks, rerank with new weights without LLM.
11. **Report agent.**
   *Check:* template sections filled from stored JSON; LLM summary ≤ 3 sentences.
12. **LangGraph wiring + background run.** Sequential per-candidate loop, save-as-you-go, skip-done, error isolation, progress counter.
   *Check:* kill the process mid-run, restart, finished candidates are not recomputed; one forced failure marks one candidate `error` and the rest finish.
13. **FastAPI + Streamlit.**
   *Check:* full flow from the UI: upload → screen → live progress → ranked results → re-weight.
14. **Evaluation scripts + README.** Setup steps, how to run, known limits, calibration results.

---

## 15. Setup commands (put these in README.md)

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
cp .env.example .env                                     # then fill HF_TOKEN and passwords

# 2. Database
docker compose up -d
docker compose ps

# 3. Python deps
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 4. Tesseract (optional, OCR fallback): install via the OS package manager

# 5. Run
uvicorn app.api:app --host 0.0.0.0 --port 8000
streamlit run app/ui.py
```

`requirements.txt` should include: `langgraph`, `pydantic`, `pydantic-settings`, `python-dotenv`, `sentence-transformers`, `huggingface-hub`, `psycopg[binary,pool]`, `pgvector`, `pymupdf`, `python-docx`, `pytesseract`, `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, `numpy`, `pytest`. (`llama-cpp-python` is installed separately because of the CUDA wheel.) Presidio is optional.

---

## 16. Rules for Codex

1. Follow the build order; do not skip ahead. Run the acceptance check after each step and fix failures before continuing.
2. Do not hardcode secrets, model names, paths, or sizes. Everything goes through `config.py` and `.env`.
3. Do not add extra agents, frameworks, or services beyond this spec. Keep the workflow light.
4. Never load a model per request. Models load once at startup.
5. Never let the LLM compute scores, years of experience, or rankings. Code does the math.
6. Never send the full resume to the Evaluator. Send top chunks only.
7. Write small, typed, testable functions. Add `pytest` tests for preprocess, chunk, stage-1 filter, anonymize, verifier, and ranking.
8. If something in this spec is ambiguous or impossible on the target hardware, choose the simplest option that works, note it in `README.md` under "Decisions", and continue.
9. Never delete user data or drop tables without an explicit flag.
10. Keep the UI note that the system recommends and a human decides.
