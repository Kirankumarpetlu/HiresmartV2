# AgentHS GitHub API rate limiting

## Review request

I’m building AgentHS, a resume screening application. The GitHub agent looks at public GitHub evidence linked from a resume and returns a status plus an optional score. In a test with 25 resumes and one software engineering job description, only one candidate received a GitHub score. GitHub reported zero unauthenticated REST requests remaining during the run.

Please recommend the smallest reliable fix for screening batches of roughly 10 to 25 resumes. Cover both the immediate configuration change and any changes you would make to the GitHub client. Prioritize preserving useful repository evidence, handling unavailable evidence consistently, and explaining rate-limit outcomes clearly to a recruiter.

This handoff contains aggregate results and code behavior. Use the details below as the starting point.

## Application context

- FastAPI exposes the screening API and Streamlit provides the UI.
- LangGraph runs the local proficiency evaluator and GitHub evidence collection in parallel for each shortlisted candidate. Both branches join at project matching before verification, persistence, ranking, and reporting.
- Python computes the GitHub score from project match, activity, repository quality, and job relevance. The local LLM can help match ambiguous project claims. GitHub rate limiting blocked the collection of public evidence upstream of that matching.
- PostgreSQL stores public endpoint responses and their timestamps and ETags in `github_profiles`. The client uses this cache across test runs when caching is enabled.
- The ranking code uses the GitHub weight only for an `ok` result with a score. When GitHub evidence is unavailable, the resume-based criteria determine the score. Interview selection and email are downstream stages.

## Observed test results

The test used a new-graduate software engineering JD and 25 supplied resumes. The test explicitly set the shortlist size to 25 to exercise every indexed resume. The application’s usual configured shortlist size was 10.

| Observation | Result |
|---|---|
| Resumes indexed and shortlisted | 25 |
| GitHub authentication | No personal access token configured |
| GitHub result with a score | 1 candidate scored 59/100 |
| `rate_limited` | 23 candidates received no GitHub score |
| `not_provided` | 1 resume yielded no detected GitHub profile |
| GitHub REST core quota during the run | `GET /rate_limit` reported `remaining: 0` |
| LangGraph cutoff checked | 10 candidates in a scoped candidate graph that stopped after GitHub matching |
| Full GitHub-only coverage | All 25 tested using the actual filter, GitHub collection, and matching functions |
| Email activity | Zero notification rows and zero emails |

The observed quota exhaustion explains why uncached GitHub evidence became unavailable during the run. The saved results contain per-candidate statuses and scores. The exact endpoint, request count per candidate, and response headers for the first failed request are unknown. The 23 statuses may reflect one rejection followed by local cooldowns for later candidates. The test ended before final candidate ranking and interview selection.

## How one resume turns into multiple API requests

1. `app/screening/github_agent.py::collect` reads the detected GitHub username and repository links. If there is no username, it returns `not_provided` before creating the HTTP client.
2. `app/integrations/github_client.py::profile` fetches the user profile and a paginated public repository catalog. The code requests up to 100 repositories per page and caps the scan at 10 pages.
3. The agent prioritizes repository links from the resume and sorts other repositories using their match to resume projects, relevance to the JD, fork status, and recent pushes. The repository detail budget is `min(GITHUB_MAX_REPOS, max(5, number_of_direct_links, 3 * number_of_projects))`. The default `GITHUB_MAX_REPOS` is 30. A candidate with fewer available repositories can use less than that budget.
4. For each selected repository, `details` fetches languages, README, up to 30 commits filtered to the candidate’s username, and the root directory. The presence of a root `.github` directory adds a workflow lookup. A directly linked repository absent from the catalog can require another metadata lookup.
5. Once collection finishes, the workflow matches public repositories to resume project claims and computes the score.

**Illustrative cold-cache cost.** For a profile with one catalog page and five repositories to inspect, the basic cost is `1 profile + 1 catalog + 5 × 4 detail requests = 22 requests`. Workflow checks, additional catalog pages, and metadata for directly linked repositories can increase it. Fresh cache hits reduce the network cost. This is an example from the call structure. Actual request counts vary by candidate and cache state.

GitHub documents a primary REST limit of 60 requests per hour per originating IP for unauthenticated public calls and typically 5,000 requests per hour for a personal access token. Secondary limits can still apply to authenticated traffic. Uploading or screening one resume at a time draws from the same quota when the originating IP and authentication stay the same. [GitHub REST API rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

## Current caching and cooldown behavior

| Area | Current behavior |
|---|---|
| Authentication | `app/config.py` reads `GITHUB_TOKEN`. `GithubClient` sends a bearer header when the setting is populated. |
| Fresh cache | A cached endpoint response younger than `GITHUB_CACHE_TTL_HOURS` is returned before the client checks the cooldown. The default TTL is 24 hours. |
| Expired cache | When an ETag exists, the client makes a conditional request and reuses the cached value on HTTP 304. |
| Partial collection | Successfully fetched responses are written to PostgreSQL when the client context exits, including when a later call raises. The current candidate receives `rate_limited` if collection raises `RateLimited`. |
| Error classification | The client treats every HTTP 403 or 429 as a rate limit. This makes a permission-related 403 indistinguishable from a quota rejection in the stored candidate status. |
| Shared cooldown | The module-level `_blocked_until` timestamp applies to GitHub client instances in the same process. It is set to the latest of the current time plus 60 seconds, a valid `X-RateLimit-Reset` timestamp, and the current time plus a valid numeric `Retry-After` duration. |
| Requests during cooldown | A request that needs the network raises `RateLimited` locally while the cooldown is active. Fresh cache hits can still be served. |
| Candidate status | `app/screening/graph.py::github_agent_node` catches `RateLimited` and returns a `GithubResult` with status `rate_limited` and no GitHub score. |

## Code map

| File | Relevant code |
|---|---|
| `app/config.py`, `.env.example` | `GITHUB_TOKEN`, `GITHUB_MAX_REPOS`, `GITHUB_CACHE_TTL_HOURS`, and cache settings |
| `app/integrations/github_client.py` | `GithubClient.get`, `request`, `profile`, `repository`, `details`, and `__exit__` |
| `app/screening/github_agent.py` | `collect`, `match_projects`, and `score_profile` |
| `app/screening/graph.py` | `github_agent_node`, `github_match_node`, and the candidate graph |
| `app/models.py` | `GithubResult` and its status and optional score |
| `app/screening/ranking.py` | `rank_candidates` and `score`, which gate the GitHub contribution on a usable result |
| `db/init.sql` | `github_profiles` and stored screening GitHub fields |
| `tests/test_github.py`, `scripts/check_github.py` | Existing HTTP, cache, rate-limit, and matching tests, plus a public API and warm-cache check |

## Questions for ChatGPT or Claude

1. **Immediate test setup.** Is a GitHub personal access token in local `.env` enough for routine 10–25-resume batches with the current implementation? What minimum access is appropriate for querying public repositories?
2. **Request budget.** Which calls should the agent keep, defer, or make optional to retain the strongest evidence? Compare a smaller repository cap, prioritizing direct links, reusing catalog metadata, and fetching expensive detail only for plausible project matches. Estimate cold-cache and warm-cache costs separately.
3. **Status handling.** How should the client distinguish primary quota exhaustion, secondary limits, and permission-related 403s? Which response headers or response fields should it inspect, and what should it do during cooldown?
4. **Partial evidence.** If the quota runs out midway through a candidate, should the app keep the current unavailable score, use sufficiently complete cached evidence, or queue the candidate for later? Explain how the choice affects comparability when some candidates have a GitHub score and others do not.
5. **Order of implementation.** Separate changes needed for the next authenticated test from changes needed for repeatable batches. Recommend a small set of tests and the metrics that would tell us whether the changes work.

## Verification for a proposed fix

- Run a single explicitly scoped resume with a known public GitHub link. Capture cache hits, network calls, quota headers, and the final GitHub status.
- Repeat the same input while its responses are fresh. Verify how many network calls remain.
- Run a small authenticated batch and then the same 25-resume GitHub-only test. Compare usable scores, missing profiles, quota-related statuses, and request counts.
- Mock primary quota exhaustion, a secondary limit with `Retry-After`, and a permission 403. Verify that each case is reported correctly and that the client respects GitHub’s retry guidance.
- Record sanitized diagnostics such as HTTP status, endpoint category, cache hits and misses, and available rate-limit headers. Keep tokens, candidate identities, and resume contents out of the shared diagnostics.
- Preserve the ability to continue screening with unavailable GitHub evidence. Keep this verification run scoped through GitHub matching, with the notification stages excluded.

An authenticated run has yet to be measured. The first comparison should use the same JD and the same scoped resumes so the effect of authentication and cache behavior can be seen clearly.
