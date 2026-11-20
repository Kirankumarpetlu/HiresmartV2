# STEP 1 — LangGraph-native orchestration + caching + latency reduction

Read `README.md` first. Its "Non-negotiable design rules" and "Change guide for coding agents" still apply. Where this file conflicts with README, this file wins for the changes listed here; update README when done.

## 1. Goal

1. Move **all orchestration** into LangGraph. Today only part of it is: the queue (`ThreadPoolExecutor`), JSON checkpoints, recovery scan, retries and routing are hand-written.
2. Make each agent a proper LangGraph node (or subgraph) with typed state, so Steps 2 and 3 (GitHub agent, Email agent) plug in as nodes.
3. Add caching at every expensive boundary.
4. Cut end-to-end latency (resume upload and screening) without breaking any invariant.

## 2. Constraints that must stay true

- Fully local inference; no resume text to hosted APIs.
- One LLM generation at a time (`LLM_CONCURRENCY=1`). LangGraph parallelism is allowed ONLY for non-LLM work (embedding on CPU, DB, IO). All LLM calls still go through `llm_service` lock.
- Python computes scores/ranks. LLM judges proficiency + writes summaries only.
- Quote verification before scoring. Anonymized chunks only into Evaluator prompt.
- Every candidate saved immediately; one failure never stops the run.
- Re-rank makes zero LLM calls.
- Uvicorn stays at 1 worker.

## 3. Task A — Baseline first (do not skip)

1. Run `python scripts/benchmark_latency.py --output data/latency/before.json` with a fixed set of ~10 resumes and 1 JD. Keep the file.
2. From `data/service.log`, record per-stage latency: parse, OCR, chunk, embed, DB insert, JD parse LLM, retrieval, per-candidate evaluator LLM, report LLM.
3. Write the top 5 bottlenecks into `docs/latency_baseline.md`. Optimise in that order, not by guesswork.

Expected suspects (verify, do not assume): CPU BGE-base encoding on upload, OCR, one sequential 3B-model Evaluator call per candidate, JD-parse call repeated for the same JD, Report LLM call, no prompt-prefix reuse across candidates.

## 4. Task B — LangGraph-native orchestration

Add deps: `langgraph`, `langgraph-checkpoint-postgres`, `psycopg[binary,pool]` (check versions against Python 3.12; pin in `requirements.txt`).

### B1. One top-level graph, typed state

Create `app/screening/state.py` with `TypedDict`/Pydantic state:

```text
ScreeningState: jd_id, request, weights, jd_parsed, query_vectors, shortlist[], candidate_results[], ranked[], report, status
CandidateState: jd_id, resume_id, stage1_decision, chunks, anonymized_chunks, evaluation, verified, error
```

Use reducers (`Annotated[list, operator.add]`) for lists written by parallel/mapped nodes.

### B2. Replace hand-written pieces

| Today (bare metal) | Replace with |
|---|---|
| `data/runs/<jd_id>.json` checkpoints | LangGraph `PostgresSaver` checkpointer, `thread_id = jd_id` |
| Startup scan of queued/running checkpoints | On startup, list unfinished threads and `graph.invoke(None, config)` to resume |
| `ThreadPoolExecutor(max_workers=1)` | Keep ONE background worker only as the *runner* that calls `graph.stream()`; no custom scheduling logic. Alternative: a LangGraph-driven FIFO using a DB queue table. Pick one and document it |
| Manual per-candidate loop | `Send` API (map step) over the shortlist; the LLM lock still serializes evaluator calls |
| Hand-rolled retry/backoff in graph code | LangGraph `RetryPolicy` on nodes (LLM-level JSON validation retry stays in `llm_service`) |
| Candidate failure isolation try/except | Node-level error handling that writes `status='error'` and returns normally so the run continues |

Keep immediate per-candidate DB save (the `save` node) in addition to the checkpointer. The checkpointer is for resume-after-restart; `screening_results` stays the source of truth for the API.

### B3. Make every stage a node

Top-level graph:

```text
parse_jd -> retrieve -> fan_out_candidates -> [candidate subgraph x N] -> rank -> report -> END
```

Candidate subgraph:

```text
route -> (filtered_out -> save) | anonymize -> evaluator -> verifier -> save
```

Design the candidate subgraph with a clear extension point after `verifier` where Step 2 (`github_agent`) and Step 3 add nodes. Do not implement those here.

Agents must be thin wrappers: each node function calls the existing module (`jd_parser`, `evaluator`, `verifier`, `ranking`, `report`) and returns a state update. Do not rewrite the business logic.

### B4. Streaming progress

Use `graph.stream(..., stream_mode="updates")` so `/screenings/{id}/status` and `/results` reflect progress from graph events, not only DB polling. Keep the DB writes.

### B5. Tests

- Graph routing, `Send` fan-out count, reducer correctness.
- Kill/restart simulation: run graph, stop after N candidates, resume from the Postgres checkpoint, assert finished candidates are not re-evaluated.
- Failure isolation: one candidate raises, others finish.
- Update `scripts/check_recovery.py` to the new mechanism and keep it passing.

## 5. Task C — Caching (all local, no new infrastructure)

Create `app/cache.py` plus tables in `db/init.sql`. Every cache key MUST include `model id + prompt_version`, so changing a model or prompt invalidates automatically. Add `PROMPT_VERSION` constants in `evaluator.py`, `jd_parser.py`, `report.py`.

| Cache | Key | Value | Saves |
|---|---|---|---|
| JD parse cache (`jd_parse_cache`) | sha256(normalized JD text) + llm model + prompt_version | parsed JSON + query vectors | 1 LLM call + embeddings when the same JD is re-run |
| Chunk embedding cache (`embedding_cache`) | sha256(chunk text) + embed model | vector | re-embedding identical text (re-uploads, overlapping sections) |
| Query vector LRU (in-process) | hash(query text) + embed model | vector | repeated JD parts |
| Evaluation cache (`evaluation_cache`) | resume content hash + JD requirements hash + selected chunk ids hash + model + prompt_version | verified evaluation JSON | full Evaluator call when the same resume/JD pair is rerun |
| Report cache | hash(verified results) | 2-sentence summary | Report call |
| GitHub cache | (Step 2) | | |

Rules:
- Caches store derived data only; never log raw prompts.
- Cache hits are still verified by `verifier` rules only if the stored result was verified (store verified output; no need to re-verify).
- Add `CACHE_ENABLED=true` and `CACHE_TTL_DAYS` (only for GitHub-type external data) to `config.py` and `.env.example`.
- Deleting a resume must cascade to its cache rows.
- Add a `scripts/clear_cache.py` utility.

## 6. Task D — Latency reductions (apply in measured order)

### D1. LLM (usually the biggest cost)
1. **Prompt prefix KV reuse.** Restructure the Evaluator prompt as `[static system + rubric + JD requirements] + [candidate-specific evidence]`. Keep the static block byte-identical across candidates in a run, then enable llama.cpp prompt caching (`LlamaCache`/state reuse via llama-cpp-python) so the shared prefix is not re-evaluated per candidate. Verify with timing logs that prompt-eval time drops after the first candidate.
2. Tune llama.cpp: `n_batch`/`n_ubatch`, `flash_attn=True` if supported, `n_threads` = physical cores, `offload_kqv=True`. Re-check VRAM headroom on 4 GB.
3. Shrink the prompt: cap evidence at fewer chunks when similarity is high, drop low-value chunks, de-duplicate overlapping chunk text, compact skill-metadata formatting.
4. Shrink the output: shortest valid JSON keys, short evidence quotes (cap word count), lower `LLM_MAX_TOKENS` for the evaluator to what the schema needs. Fixed-position schema stays.
5. **Remove the Report LLM call from the critical path**: either run it after results are already returned and update asynchronously, or fold the two-sentence summary into the Evaluator output. The deterministic fallback already exists, so the run can finish without waiting.
6. Skip the Evaluator when the evaluation cache hits or Stage-1 filters the candidate.
7. Evaluate in **similarity order** and stream each result as it finishes so the UI shows the top candidates first.

### D2. Ingestion
1. Parse/OCR multiple uploaded files in a `ProcessPoolExecutor` (CPU bound), bounded to cores - 1. Keep the per-file error isolation.
2. Batch-encode all chunks of a file together (already) and also across files; tune batch size for the i5.
3. Check embedding cache before encoding.
4. Insert chunks with `executemany`/`COPY` in one transaction.
5. Optional (behind a flag, measure quality before enabling): ONNX Runtime or int8-quantized BGE on CPU. Changing to `bge-small` (384-d) requires the full embedding-dimension change process in the README change guide; do NOT do this silently.
6. Return from `POST /resumes` as soon as the file is durably stored if ingestion exceeds a threshold, and expose status via `GET /resumes`. Only do this if the baseline shows upload blocking is the user-visible problem.

### D3. Retrieval / DB
1. Confirm HNSW index exists and is used (`EXPLAIN ANALYZE`); set `hnsw.ef_search` explicitly.
2. Run the per-JD-vector retrieval queries concurrently (pure DB, no LLM).
3. Fetch evidence chunks for all shortlisted resumes in one query instead of per candidate.
4. Size the pooled connections (`db.py`) to match concurrency.

### D4. Startup
Warm up both models with a tiny dummy call at startup so the first request is not slow.

## 7. Acceptance criteria

- `python -m pytest -q` passes.
- `python scripts/check_recovery.py prepare|verify|retry` passes on the new LangGraph checkpointer.
- Same JD re-run on the same resumes performs **0** JD-parse LLM calls and **0** Evaluator calls (log proof).
- Second and later candidates in a run show lower prompt-eval time than the first (prefix cache proof).
- `data/latency/after.json` vs `before.json` shows improvement; summarize in `docs/latency_report.md` with numbers. Target: at least 40% lower end-to-end screening time for a warm re-run and a clear drop for a cold run. Report honestly if a target is missed.
- Scores for a fixed fixture set are identical (or within rounding) before and after. Caching and prompt restructuring must not change ranking. Add a regression test.
- No invariant from README is violated.

## 8. Out of scope

GitHub agent (Step 2), email agent (Step 3), auth, UI redesign.

## 9. Deliverables

Code, tests, updated `README.md` (architecture, config table, persistence table, change guide), `.env.example`, `docs/latency_baseline.md`, `docs/latency_report.md`.
