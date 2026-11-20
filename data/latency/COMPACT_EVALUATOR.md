# Resume evaluation latency

The completed Aviators run (`d0ea9896-24cd-4688-a7e3-5c270aabe1ce`) processed 25 resumes in 868.77 seconds (14 minutes 29 seconds).

| Work | Total | Per resume |
|---|---:|---:|
| JD parsing/embedding (once) | 27.95 s | — |
| Evaluator | 736.30 s | 29.45 s |
| Report summaries | 103.76 s | 4.15 s |

The run had no validation retries. Generation, rather than ingestion or graph orchestration, dominated elapsed time.

## Change

The constrained evaluator output no longer generates full requirement names and must-have flags for every resume. Each fixed array position has the exact requirement name as a validated default, and code restores the must-have flag from the JD. Wrong explicit names, missing array entries, invalid confidence, and invalid proficiency remain rejected. Stored evidence, verification, scoring, API fields, and the local model are retained.

The final prompt numbers each requirement explicitly to make the fixed-position mapping clear. Retry feedback refers to the supplied schema instead of asking the model to emit names that the schema omits. All 41 tests pass. The local backend was restarted and its health check confirms PostgreSQL, embeddings, the loaded LLM, and 37 GPU-offloaded layers.

## Paired exploratory check

Three already-uploaded resumes were evaluated with both formats using the same loaded model, JD, and anonymized chunks. Test order alternated. No existing screening results were updated.

| Resume | Previous format | Compact format |
|---|---:|---:|
| Jayanth_Resume_mondee.pdf | 88.62 s | 67.88 s |
| T Mohan Lakshmi Ganesh_Resume.pdf | 71.75 s | 48.34 s |
| henilsinh_raj.pdf | 53.40 s | 60.89 s |
| **Mean** | **71.26 s** | **59.04 s** |

This trial observed 17.1% lower average evaluation time; one resume was slower. Absolute timings were substantially slower than the earlier folder run, so they cannot be used to promise a new production average. The final explicit numbering was added after this trial and checked by unit tests, but was not separately timed.

Model scores and evidence flags changed between formats; this is not a quality-equivalence validation. Human-labeled calibration is still required. The entire 25-resume run has not been repeated with the final code. Start a new screening (or use a fresh folder-test output directory) to exercise the new evaluator; existing manifests reuse completed runs.

Raw results: `compact_resume_comparison.json`. Existing full results remain under `data/test_runs/aviators_nov17/`.
