# Step 1 latency report

## Implemented reductions

- JD parse cache keys normalized JD text, LLM model ID, and `jd-v2`; hits skip both the parse LLM call and JD embeddings.
- Evaluation cache keys resume/content, normalized requirements, selected chunk IDs, model ID, and `evaluator-v2`; only verified output is stored, so hits skip the Evaluator and verifier without changing scores.
- Report cache keys verified report facts, model ID, and `report-v2`.
- Identical chunk text reuses PostgreSQL embedding vectors; repeated JD parts reuse an in-process query-vector LRU.
- Evaluator messages now place static rubric and JD requirements in a byte-stable system prefix. llama.cpp uses a 512 MiB RAM state cache and logs `prompt_cache hit prefix_tokens=...`.
- llama.cpp now uses `n_batch=512`, `n_ubatch=128`, physical-core-oriented generation threads, all logical cores for batch prompt evaluation, flash attention, and K/Q/V offload.
- Per-JD retrieval queries run concurrently, set `hnsw.ef_search=100`, and evidence chunks for the shortlist are fetched in one batched operation.
- LangGraph maps candidates with `Send`. The runner processes them in similarity order at concurrency one. Retrieval still parallelizes independent database queries.
- The rank node publishes `status=done` before the Report node runs, so ranked results are available from graph events while summaries continue to populate; warm reports skip generation entirely.

## Correctness checks

- The Python suite passed 46 tests with `RUN_POSTGRES_TESTS=1` and Docker pgvector on port 5433.
- The live checkpoint test interrupted dispatch, committed one candidate, and resumed through the production runner. The committed candidate was skipped. The test uses stub inference to isolate recovery behavior.
- With real models, `check_recovery.py prepare` captured one committed result before the test API process was terminated. After automatic recovery on restart, `check_recovery.py verify` confirmed all three candidates completed with zero errors and the committed candidate was skipped. The artifact is `data/latency/recovery_step1/recovery_acceptance.json`.
- The live database test also ran `check_recovery.py retry` after injecting a synthetic candidate failure. Only the failed candidate was evaluated again.
- `EXPLAIN ANALYZE` selected `idx_chunks_embedding` on the small fixture with sequential scans disabled. Retrieval sets `hnsw.ef_search` transaction-locally before executing the query.
- Tests prove fan-out/reducer behavior, route behavior, candidate failure isolation, finished-candidate resume skips, all four cache boundaries, and score equality after evaluation serialization/cache restoration.
- Re-ranking still calls only deterministic stored-result scoring.
- Immediate candidate DB writes and quote verification remain before ranking.

## Live benchmark

The November 19 run used Docker PostgreSQL with pgvector, BGE-base on CPU, and Qwen with all 37 layers offloaded to the GTX 1650. Both screenings used the same JD and ten synthetic resumes. The cold screening started with empty JD and evaluation caches for this request. An earlier upload attempt had already populated some chunk embeddings.

| Measurement | Cold screening | Identical warm rerun |
|---|---:|---:|
| Screening through output read | 150.368 s | 0.497 s |
| JD parse, including embeddings | 4.726 s | 0.006 s |
| Report stage | 22.135 s | 0.035 s |
| LLM generation calls | 21 | 0 |

The warm rerun took 99.7% less screening time and returned the same candidate order and scores. Ten uploads plus the cold screening took 151.934 s, excluding model startup. The successful run had zero validation failures. Measurements are in `data/latency/after.json` and `data/latency/after.log`. The older after artifacts are preserved as `after_pre_step1.json` and `after_pre_step1.log`.

Native llama.cpp counters recorded 546 prompt tokens evaluated in 2,723.97 ms for the first candidate. The second reused 510 prefix tokens and evaluated 39 prompt tokens in 211.76 ms. Later candidates took 211–426 ms for prompt evaluation. These counters measure prompt evaluation separately from generation.

The retained `before.json` measured one resume and 46.059 s of screening. The ten-resume cold run has a different fixture size, so a matched pre-Step 1 cold speedup remains unmeasured.

## Deferred measured non-bottlenecks

- Multiprocess parsing/OCR: parse was 0.0097 s and OCR was not exercised.
- ONNX/int8 or a smaller embedding model: requires separate quality and dimension migration work.
- Asynchronous upload acknowledgement: baseline ingestion was 2.5722 s and did not justify changing API semantics.
