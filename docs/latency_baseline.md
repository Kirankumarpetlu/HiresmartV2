# Latency baseline

## Source and scope

The retained real-model benchmark at `data/latency/before.json` was recorded on 2026-11-17 with PostgreSQL, BGE-base on CPU, Qwen2.5-3B Q4_K_M, and all 37 model layers offloaded to the GTX 1650. It measured local function calls from one fresh synthetic DOCX through ranked output and excluded upload transport, queue wait, and UI polling.

That retained benchmark contains one resume, not the ten-resume default now enforced by `scripts/benchmark_latency.py`. A second retained service-log run contains 25 candidates and confirms that per-candidate model generation dominates at shortlist scale. PostgreSQL/Docker was unavailable when Step 1 began, so the mandated ten-resume command could not be rerun without inventing data; the original artifact was preserved.

## Recorded stages

| Stage | Seconds | Notes |
|---|---:|---|
| Resume parse | 0.0097 | DOCX; no OCR |
| OCR | not invoked | No scanned page in the fixture |
| Chunk | 0.2017 | Includes tokenizer use |
| Chunk embedding | 2.2424 | CPU BGE-base |
| DB insert | 0.0889 | Resume and chunks transaction |
| JD parse | 8.4009 | Includes one constrained LLM call and 0.3179 s JD embedding |
| Retrieval | 0.1598 | pgvector shortlist |
| Evaluator | 27.3802 | One candidate LLM call |
| Candidate save | 0.3229 | Verified JSON persistence |
| Report | 9.6244 | One summary LLM call |
| Screening total | 46.0588 | Excludes startup and ingestion |
| Ingestion to output | 48.6441 | User-path total |

Model startup was measured separately: embedder 18.3703 s and LLM 9.5528 s.

The retained 25-candidate service-log run recorded 736.7292 s in candidate evaluation, 103.9503 s in reports, and 868.7733 s total screening time. It is supporting scale evidence, not substituted into `before.json`.

## Top five measured bottlenecks

1. Evaluator LLM: 27.3802 s per candidate in the benchmark; 736.7292 s across the retained 25-candidate run.
2. Report LLM: 9.6244 s for one summary; 103.9503 s across the retained multi-candidate run.
3. JD parse LLM: 8.4009 s for every cold repeated JD.
4. CPU embedding: 2.2424 s for the small synthetic resume.
5. Candidate DB save/chunking: 0.3229 s and 0.2017 s respectively; all other measured request stages were lower.

This order drove the implementation: verified evaluation/JD/report caches and llama.cpp prefix reuse first, embedding/query caches second, then batched evidence reads and concurrent JD-vector retrieval. Process-pool parsing and a new embedding runtime were deliberately skipped because parse time was 0.0097 s and changing them would not address a measured bottleneck.
