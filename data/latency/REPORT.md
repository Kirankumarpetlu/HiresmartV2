# Latency measurement, 2026-11-17

One synthetic DOCX resume, one JD, one scoped candidate, real PostgreSQL and local Qwen2.5-3B Q4_K_M on GTX 1650. All 37 model layers offloaded to CUDA. Both runs finished without validation retries.

Ingestion to ranked output and summary: **48.644 s before, 37.804 s after**, an observed reduction of **10.840 s (22.3%)**. These are single-run observations, not averages, percentiles, or a guaranteed speedup. Model startup is excluded.

| Stage | Before (s) | After (s) |
|---|---:|---:|
| Duplicate check / database lock | 0.019293 | 0.003288 |
| DOCX parsing | 0.009738 | 0.012865 |
| Preprocessing | 0.003075 | 0.004467 |
| Metadata extraction | 0.003606 | 0.009109 |
| Chunking / token counting | 0.201651 | 0.456485 |
| Resume embedding | 2.242432 | 0.398410 |
| Raw file write | 0.001160 | 0.000876 |
| Resume / chunk storage and commit | 0.088869 | 0.158082 |
| **Ingestion total** | **2.572241** | **1.047107** |
| Screening creation | 0.006411 | 0.008562 |
| JD parsing, embedding and save | 8.400872 | 8.111324 |
| Candidate retrieval and pending-result saves | 0.159826 | 0.027573 |
| Stage-1 filter | 0.000043 | 0.000029 |
| Candidate chunk retrieval | 0.011724 | 0.006697 |
| Anonymization | 0.001284 | 0.001322 |
| Evaluator, including validation/retries | 27.380235 | 24.559979 |
| Evidence verification | 0.000375 | 0.000314 |
| Evaluation save | 0.322926 | 0.006227 |
| Ranking and persistence | 0.005886 | 0.005075 |
| Report generation and save | 9.631075 | 3.958107 |
| **Screening total** | **46.058764** | **36.743350** |
| Output database read and section formatting | 0.005769 | 0.004113 |

Totals include nested stages and small amounts of orchestration/logging overhead. Do not add parent totals to their child stages. JD embeddings were computed three times before (0.530062 s combined) and once after (0.319690 s, included in JD parsing).

## Changes

- Reuse the JD query-vector matrix in LangGraph state across parsing, retrieval, and candidate evaluation, including resumed runs.
- Insert resume chunks with `executemany` inside the existing transaction. The single-sample storage measurement increased; no measured storage speedup is claimed.
- Send each report evidence quote once, compact its JSON, and request two sentences with at most 35 words. Observed summary generation fell from 9.624 s to 3.948 s; summary length fell from 55 to 22 words.
- Ask the evaluator to omit optional year fields when they are unknown. Stored defaults, evidence verification, score arithmetic, and strict validation remain in place.
- Add stage timing logs and a repeatable benchmark: `python scripts/benchmark_latency.py --output data/latency/run.json`.

## Limits and quality checks

Model startup was 27.956 s before and 38.147 s after, including database setup and model repository metadata checks; these timings vary with network/cache state and are paid once per process. The benchmark starts after models load. Network upload, backend queue waiting, response serialization/transfer, PDF/OCR, and Streamlit's two-second polling interval were not benchmarked.

This comparison checks execution and latency, not equivalent model judgments or hiring accuracy. Stored scores differed (60.125 before, 72.8438 after). The JD parser introduced an unrelated JavaScript/Java/C++ alternative requirement in both runs, despite its absence from the synthetic JD. This existing model-grounding issue requires human review and separate calibration; do not use these synthetic ranks as a quality validation. Timing variations in unchanged stages also show that one sample cannot attribute the entire end-to-end reduction to code changes.

Raw timings: `before.json`, `after.json`; inference and stage logs: `before.log`, `after.log`. Both synthetic uploads and screenings were retained; no user data was deleted.
