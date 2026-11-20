# AgentHS: GitHub Evidence at Scale (Implementation Spec for Codex)

## 0. How to use this document

Implement the phases in order. Each phase must be independently shippable, keep the existing test suite green, and leave screening usable when GitHub evidence is unavailable. Before changing code, read the files listed in the code map and confirm the current behavior. Where this spec and the code disagree on a detail (names, signatures), follow the code and note the deviation in the PR description.

Do not add features beyond this spec. Do not log tokens, candidate names, or resume contents anywhere.

## 1. Problem summary

- In a test of 25 resumes with no `GITHUB_TOKEN`, only 1 candidate received a GitHub score; 23 were `rate_limited` and 1 was `not_provided`. GitHub reported `remaining: 0` for unauthenticated REST (limit: 60 requests/hour per IP).
- One cold-cache candidate costs roughly 22 requests (1 profile + 1 catalog + 5 repos x 4 detail calls), and up to about 120 at the default `GITHUB_MAX_REPOS=30`.
- The current client treats every HTTP 403/429 as a rate limit, and the cooldown (`_blocked_until`) is module-level, so it only works inside one process.
- **Target:** screen batches of 10 to 25 reliably now, and 1,000+ resumes in the future, without losing evidence or giving ambiguous results to recruiters.

## 2. Design principles

1. GitHub collection is asynchronous and quota-aware. It must never block the screening request.
2. Resume-based ranking is always available immediately. GitHub scores arrive later and re-rank.
3. Every missing GitHub score has an explicit, recruiter-readable reason.
4. Spend quota only where it can change the outcome (shortlist only, plausible repos only).
5. Respect GitHub's rate-limit rules. Never rotate multiple tokens or accounts to bypass limits.

## 3. Code map (read first)

| File | Relevance |
|---|---|
| `app/config.py`, `.env.example` | `GITHUB_TOKEN`, `GITHUB_MAX_REPOS`, `GITHUB_CACHE_TTL_HOURS` |
| `app/integrations/github_client.py` | `GithubClient.get`, `request`, `profile`, `repository`, `details`, `__exit__`, module-level `_blocked_until` |
| `app/screening/github_agent.py` | `collect`, `match_projects`, `score_profile` |
| `app/screening/graph.py` | `github_agent_node`, `github_match_node`, candidate graph |
| `app/models.py` | `GithubResult` (status, optional score) |
| `app/screening/ranking.py` | `rank_candidates`, `score` (GitHub weight only for `ok` + score) |
| `db/init.sql` | `github_profiles` cache table |
| `tests/test_github.py`, `scripts/check_github.py` | existing tests and live check script |

## 4. Phase 1: Immediate fixes (next authenticated test)

### 4.1 Token and config
- Document `GITHUB_TOKEN` in `.env.example`. Recommended: fine-grained PAT with **Public repositories (read-only)** and no other permissions (a classic PAT with no scopes also works). Never `repo` scope.
- Change `GITHUB_MAX_REPOS` default from 30 to **8**.
- Add settings: `GITHUB_CATALOG_MAX_PAGES` (default 3), `GITHUB_COMMITS_PER_PAGE` (default 10), `GITHUB_RESERVE_REMAINING` (default 200), `GITHUB_MAX_CONCURRENCY` (default 4).

### 4.2 Quota diagnostics
- Add `GithubClient.rate_limit()` calling `GET /rate_limit` (does not consume quota). Return `core`, `graphql`, `search` buckets with `limit`, `remaining`, `reset`.
- Call it at the start and end of every batch and log `remaining`/`reset`. Log a clear warning at startup when no token is configured.
- Per request, log sanitized diagnostics only: HTTP status, endpoint category (`profile`, `catalog`, `languages`, `readme`, `commits`, `root`, `workflow`), cache hit/miss/304, and the headers `x-ratelimit-remaining`, `x-ratelimit-reset`, `x-ratelimit-resource`, `retry-after` when present.

### 4.3 Error classification
Replace "every 403/429 is a rate limit" with a classifier returning one of:

| Class | Detection | Action |
|---|---|---|
| `primary_limit` | 403/429 and `x-ratelimit-remaining == 0` | Cooldown until `x-ratelimit-reset` for that resource bucket |
| `secondary_limit` | 403/429 with `retry-after`, or body message mentions "secondary rate limit" / "abuse" | Cooldown for `retry-after` seconds; if absent, 60s then exponential backoff |
| `forbidden` | 403 with `remaining > 0`, no `retry-after`, no secondary message | **No cooldown.** Mark endpoint forbidden, continue with other repos |
| `not_found` | 404 | Record as missing evidence, not an error |

- Raise distinct exceptions: `PrimaryRateLimited`, `SecondaryRateLimited`, `GithubForbidden`, `GithubNotFound`. Keep `RateLimited` as a base class for backward compatibility.
- Remove the unconditional "now + 60s" cooldown floor. Cooldown = the classified wait only.
- Cooldown is tracked **per resource bucket** (`core`, `graphql`, `search`).
- Cache lookup order is unchanged: fresh cache is served before the cooldown check.

### 4.4 Reduce cost per candidate
- Reuse catalog metadata (language, topics, description, fork, `pushed_at`, size, stars) for ranking repositories. Do **not** call per-repo metadata when the repo is in the catalog; only fetch it for directly linked repos missing from the catalog.
- Two tiers in `github_agent.collect`:
  - **Tier 1 (cheap):** profile + catalog (capped by `GITHUB_CATALOG_MAX_PAGES`). Score and rank repos from catalog metadata.
  - **Tier 2 (selective):** top K repos (K = `min(GITHUB_MAX_REPOS, 5)` by default; direct resume links first, then best project/JD matches). Fetch README, author-filtered commits (`per_page=GITHUB_COMMITS_PER_PAGE`), root directory. Fetch the full language breakdown only for the top 3 repos; use the catalog `language` field otherwise.
  - Workflow lookup (`.github`) only for the top 3 repos and only if the root listing shows `.github`.
- Cold-cache target: **about 17 or fewer requests per candidate** at K=5.

### 4.5 Acceptance criteria (Phase 1)
- A single candidate with a known public GitHub link, with a token, yields a score; diagnostics show requests used and `remaining`.
- A repeat run within TTL makes **0** network calls for that candidate.
- Mocked primary exhaustion, secondary limit, and permission 403 are classified correctly; the permission 403 sets no cooldown.
- The 25-resume GitHub-only test with a token yields a usable score for every candidate that has a GitHub link (24/25 expected for the sample set).

## 5. Phase 2: Two-phase screening (shortlist-only GitHub)

- Phase A (all resumes): parse, index, run the local proficiency evaluator, compute resume-based score and rank. No GitHub calls.
- Phase B (top N only): collect GitHub evidence for the shortlist. `GITHUB_SHORTLIST_SIZE` (default 100; separate from the interview shortlist size).
- Add an API path to extend GitHub collection to the next N candidates on demand.
- `ranking.py`: unchanged rule that the GitHub weight applies only to a usable result. Do not penalize candidates lacking a GitHub score.
- Deduplicate by GitHub username: cache and job logic must key on username, not candidate id.
- Cache TTLs: `GITHUB_CACHE_TTL_HOURS` stays for volatile data (commits, catalog). Add `GITHUB_STABLE_CACHE_TTL_HOURS` (default 120) for README, languages, root tree. Keep ETag revalidation; a 304 does not consume primary quota.
- **Stale-if-error:** if cache is expired and the refresh fails with a limit error, use the stale value and mark evidence `stale=true`.

### Acceptance criteria (Phase 2)
- A 1,000-resume run makes zero GitHub calls for candidates outside the GitHub shortlist.
- The same username in two candidates results in one set of network calls.

## 6. Phase 3: Job queue and shared rate governor

### 6.1 `github_jobs` table (add to `db/init.sql` plus a migration)
Columns: `id`, `candidate_id`, `username`, `run_id`, `status` (`pending`, `running`, `done`, `rate_limited`, `partial`, `forbidden`, `not_found`, `failed`), `priority` (higher resume score first), `attempts`, `max_attempts` (default 5), `retry_after` (timestamptz), `last_error_category`, `evidence_coverage` (`full`, `partial`, `none`), `created_at`, `updated_at`.
Index on `(status, retry_after, priority DESC)`.

### 6.2 Worker
- A background worker process (separate entry point, for example `python -m app.workers.github_worker`) claims jobs with `SELECT ... FOR UPDATE SKIP LOCKED` where `status = 'pending' AND (retry_after IS NULL OR retry_after <= now())`, ordered by `priority DESC`.
- Jobs are idempotent and resumable. A crash leaves the job claimable again (use a lease timeout on `running`).
- On a limit error: set `status='pending'`, `retry_after` from the classifier, increment `attempts` only when the failure is not a pure quota wait. Quota waits must not burn attempts.
- Exponential backoff with jitter for transient 5xx. After `max_attempts`, set `failed` with a category.
- Partial evidence rule (see 7).

### 6.3 Shared rate governor
- Persist quota state per bucket in Postgres (table `github_quota_state`: `bucket`, `limit`, `remaining`, `reset_at`, `blocked_until`, `updated_at`), updated from every response's headers. Use Redis instead only if it already exists in the stack.
- Interface (suggested):
  - `acquire(bucket, cost) -> Grant | Wait(seconds)`
  - `record(bucket, headers)`
  - `block(bucket, until)`
- Rules: refuse to start a candidate if `remaining - estimated_cost < GITHUB_RESERVE_REMAINING`; cap in-flight requests at `GITHUB_MAX_CONCURRENCY`; pace requests across the hour rather than bursting; on exhaustion pause all workers until reset.
- Replace module-level `_blocked_until` with the governor. All `GithubClient` instances and processes share it.
- Remove the LangGraph fan-out of GitHub collection. The graph (or API) enqueues jobs and proceeds with resume-based results; the worker fills GitHub results in later.

### Acceptance criteria (Phase 3)
- Killing a worker mid-batch loses no jobs and double-counts none.
- With mocked GitHub, a 1,000-job run never exceeds the configured budget and secondary-limit hits are 0.
- Primary exhaustion pauses all workers; they resume automatically after the reset time.

## 7. Partial evidence and statuses

Extend `GithubResult.status` (update `app/models.py` and any DB/schema fields) with: `ok`, `partial`, `stale`, `pending`, `rate_limited`, `forbidden`, `not_found`, `not_provided`, `failed`.

- **Minimum evidence set** for a score: profile + catalog, plus README or metadata for all directly linked repos or at least N=3 matched repos. If met: score and set `evidence_coverage='partial'` (status `partial`). If not met: no score, status `rate_limited` (or `pending` when queued), keep the retry time.
- Always store `evidence_coverage`, `stale`, `repos_inspected`, `repos_total`, and `retry_after` with the result.
- Ranking: resume score stays primary. GitHub contributes only for `ok` and `partial` results with a score (apply a smaller weight or flag for `partial`/`stale`; make this configurable). Missing GitHub never reduces a candidate's score.

### Recruiter-facing messages (use exactly these patterns)
- `pending`: "GitHub evidence queued."
- `rate_limited`: "GitHub evidence temporarily unavailable; retry after HH:MM."
- `partial`: "Based on N of M repositories."
- `stale`: "Based on cached data from <date>."
- `forbidden`: "Some repositories couldn't be accessed."
- `not_provided`: "No GitHub profile found on resume."
- `not_found`: "GitHub profile not found."

## 8. Phase 4: Progressive results and UI

- API: endpoint returning GitHub progress for a run: counts by status, queue depth, quota `remaining`/`reset`, estimated time to drain.
- Streamlit: show progress ("GitHub evidence: 62 of 100 collected, quota resets 14:10"), per-candidate status badges with the messages above, automatic re-rank as scores arrive, and a "mixed evidence" flag while jobs are incomplete. Add a "Retry GitHub" action for failed or rate-limited candidates.
- Interview selection and email stages must not block on pending GitHub jobs unless the user opts in.

## 9. Phase 5 (optional): GraphQL batching

- Add a GraphQL path (separate 5,000-point hourly budget, tracked as the `graphql` bucket) fetching, per candidate in one query: profile, top repos with primary language, topics, README text, root tree entries, and author-filtered commit history.
- Use REST only for what GraphQL cannot do cleanly. Keep the REST path as a fallback behind a setting (`GITHUB_USE_GRAPHQL`).
- Target cost: about 4 requests or fewer per candidate.

## 10. Tests

Add or extend `tests/test_github.py` and add `tests/test_github_governor.py`, `tests/test_github_jobs.py`:

1. Primary exhaustion (403 + `remaining: 0` + reset) sets cooldown until reset for that bucket only.
2. Secondary limit with `retry-after` waits the stated time; without it, 60s plus backoff.
3. Permission 403 (remaining > 0, no `retry-after`) sets **no** cooldown and marks the endpoint forbidden.
4. 404 is recorded as `not_found` without cooldown.
5. 304 conditional hit reuses cache and counts no primary quota.
6. Mid-candidate failure still saves earlier cache writes.
7. Stale-if-error returns stale data flagged `stale`.
8. Tiered collection issues at most the budgeted number of requests per candidate (assert counts with a mocked transport).
9. Username deduplication: two candidates, one username, one set of calls.
10. Job queue: claim with SKIP LOCKED is exclusive; crash recovery via lease timeout; quota waits do not consume attempts.
11. Governor: refuses start below reserve; enforces concurrency cap; shared across clients.
12. Ranking: missing/unavailable GitHub never lowers a candidate's score versus resume-only; partial/stale weighting follows config.
13. Load test (mocked): 1,000 candidates, assert budget never exceeded and queue drains.

## 11. Metrics to record (per run)

- Usable-score rate among candidates with a GitHub link (target: 100% eventually; 24/25 for the sample set).
- Requests per scored candidate, cold and warm.
- Quota consumed per batch (`remaining` start vs end) per bucket.
- Cache hit ratio on rerun (target near 100% within TTL).
- Counts by status (`ok`, `partial`, `stale`, `rate_limited`, `forbidden`, `not_provided`, `failed`).
- Secondary-limit hits (target 0).
- Queue depth and time to drain.

## 12. Verification run order

1. Single scoped resume with a known public GitHub link and a token: record cache hits, network calls, quota headers, final status.
2. Same input again within TTL: expect 0 network calls.
3. Authenticated 10-resume batch, then the same 25-resume GitHub-only test (same JD and resumes as the baseline of 1 scored out of 25). Exclude notification/email stages.
4. Mocked failure modes (primary, secondary, permission 403).
5. Mocked 1,000-candidate load test.

## 13. Out of scope

- Rotating multiple tokens or accounts to raise limits (violates GitHub's terms).
- Changes to the local LLM project matching logic beyond consuming the new evidence structure.
- Email and interview-selection changes, other than not blocking on pending GitHub jobs.
