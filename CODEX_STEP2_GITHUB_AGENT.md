# STEP 2 — GitHub Profile Agent

Prerequisite: Step 1 is complete (LangGraph-native orchestration, caching layer, typed state). Read `README.md` and `CODEX_STEP1_LANGGRAPH_AND_LATENCY.md` first. All README invariants apply.

## 1. Goal

Add a **GitHub agent** as a LangGraph node in the candidate subgraph. For each shortlisted candidate who has a GitHub profile in their resume, it must:

1. Fetch the public GitHub profile and repositories.
2. Check whether the **projects claimed in the resume** actually exist and match repositories.
3. Rate the overall GitHub profile.
4. Return a structured, verified result that feeds the final score and Step 3 (email decision).

## 2. Design rules (follow strictly)

- **Same split as the rest of the project:** Python collects facts and computes scores; the local LLM only judges fuzzy semantic matches and writes short reasons. No score comes from the LLM alone.
- No resume text or candidate-identifying data goes to a hosted model. The only external call is the public GitHub REST API, and it sends only a username/repo path.
- Missing GitHub is **not** a penalty. It yields `github_status = "not_provided"` and a neutral/absent score that is excluded from weighting (re-normalize weights). Many good candidates, e.g. in enterprise roles, have no public GitHub.
- Private repos and inaccessible profiles are reported as "unverifiable", not "fake".
- Failure of this agent must never fail the candidate. Return `github_status = "error"` and continue.
- Anonymization: do not put the username or real name into the LLM prompt. Use repo names and anonymized project descriptions only (repo names are needed for matching; strip owner names).

## 3. Task A — Extract the GitHub URL

In `app/ingest/metadata.py` (and `parse.py` for hyperlink annotations):

- Extract `github.com/<user>` from resume text AND from PDF link annotations / DOCX hyperlinks (PyMuPDF `page.get_links()`, python-docx relationships). Many resumes show only "GitHub" as link text.
- Validate: host must be exactly `github.com`; username matches `^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$`. Reject everything else (SSRF protection). Also accept `github.com/<user>/<repo>` links and record those repos as direct project links.
- Store `github_username` and `github_repo_links[]` in resume metadata. Schema change: update `db/init.sql`, models, reads/writes, tests (per README change guide; `IF NOT EXISTS` does not migrate, so provide a migration script `scripts/migrate_add_github.py`).
- Extract **resume project list** (name + short description + tech) from experience/projects chunks for matching. Reuse existing chunk/section data; do this in Python, no LLM.

## 4. Task B — GitHub client (`app/integrations/github_client.py`)

- Use `httpx` with timeouts, a response size cap, and a descriptive User-Agent.
- Config in `config.py` + `.env.example`: `GITHUB_TOKEN` (optional), `GITHUB_API_BASE=https://api.github.com`, `GITHUB_CACHE_TTL_HOURS=24`, `GITHUB_MAX_REPOS=30`, `GITHUB_TIMEOUT_SECONDS=10`, `GITHUB_ENABLED=true`. Never log the token.
- Endpoints: `/users/{u}`, `/users/{u}/repos?per_page=100&sort=pushed`, per selected repo: `/repos/{u}/{r}/languages`, README (`/readme`), recent commits (`/commits?author={u}&per_page=30`), and the repo tree root for presence of `tests/`, CI config (`.github/workflows`), `Dockerfile`, license.
- Handle: 404 (user not found), 403/429 rate limit (read `X-RateLimit-Reset`, mark `rate_limited`, do not hammer), network errors.
- **Cache** with conditional requests (`ETag`/`If-None-Match`; 304 responses do not consume rate limit when using a token) and a DB table `github_profiles(username, fetched_at, payload JSONB, etag)`. Respect TTL. Key includes username only.
- Only analyze the top N relevant repos (see matching), not all repos, to save API calls.
- Do NOT clone repos or execute any code from them. Metadata and README text only.

## 5. Task C — Deterministic signals (Python only)

Compute, with no LLM:

- Account age, public repo count, original (non-fork) repo count, followers (low weight; do not reward popularity heavily).
- Languages distribution vs JD must-have skills (overlap).
- Activity: commits in last 12 months, last push recency, contribution spread (flag a single giant commit burst or one-day uploads).
- Repo quality signals: README present and non-trivial, tests present, CI present, license, description, repo size > trivial, multiple commits by the user.
- Fork-only profile flag; empty-repo flag; repos created in a single day flag (possible bulk upload).
- Authorship check: the candidate's commits exist in the repo (not just a fork or copy).

## 6. Task D — Project matching (hybrid)

For each resume project:

1. **Candidate repos:** exact/fuzzy name match (`rapidfuzz`), direct links from the resume, and embedding similarity between the resume project chunk and repo `description + README head` using the existing CPU embedding service and cache. Keep top 3 repos per project.
2. **Python decides the clear cases:** exact link → `verified`; no repo above the similarity floor → `not_found`.
3. **LLM only for ambiguous cases** (one batched call per candidate, constrained JSON, schema validated by Pydantic, fixed output positions like the Evaluator). Input: anonymized project claim + repo description/README excerpt/languages. Output per project: `match: verified | partial | unverified`, `reason` (max 20 words), `evidence_quote` from the README or description.
4. **Verify the quote in code** exactly like `verifier.py` (whitespace/case-normalized substring check). If the quote is not found, downgrade the match one level.
5. Never conclude "fake"; the wording is `unverified (no public repo found)`. Humans decide.

## 7. Task E — Scoring (Python)

Output model in `app/models.py`:

```text
GithubResult:
  status: ok | not_provided | not_found | rate_limited | error
  github_score: int 0-100 | null
  subscores: project_match, activity, code_quality, jd_relevance
  project_matches: [{resume_project, repo, match, reason, evidence_quote, verified}]
  strengths[], red_flags[], summary (<= 35 words, deterministic template or LLM fallback)
```

Default weighting for `github_score`:

```text
0.40 project_match + 0.20 activity + 0.20 code_quality + 0.20 jd_relevance
```

- `project_match` score = share of resume projects verified (partial counts 0.5).
- `jd_relevance` = overlap between repo languages/topics and JD must-have skills.
- Put weights in config, validate they sum to 1, unit test.
- Integrate into final ranking as an **optional fifth weight**: `github` default `0.10`; the other four weights re-normalize proportionally when github is present. When `status != ok`, drop it and use the original four weights. Extend the `weights` request model and `POST /screenings/{id}/rerank` (re-rank must remain LLM-free and use the stored GitHub result). Keep the "missing must-have caps final at 60" rule.
- Tie-break order stays the same.

## 8. Task F — LangGraph integration

- Add node `github_agent` to the candidate subgraph. It does **not** depend on the Evaluator, so run it in **parallel** with `anonymize -> evaluator` (IO-bound vs LLM-bound) and join before `verifier/save`. The LLM lock still serializes the optional LLM matching call, so run that call after the Evaluator finishes if the model is busy.
- Retry policy for network calls only; no retry on 404.
- State additions: `github: GithubResult | None`.
- Persist in `screening_results`: `github_json JSONB`, `github_score INT`. Update the save node, API result models, UI (show profile rating, matched/unverified project table, red flags), and CSV/JSON export in `scripts/test_folders.py`.
- Cache GitHub results per username (Task B) and the project-matching LLM output via the Step 1 cache (key includes resume hash + repo pushed_at + prompt_version).

## 9. Tests

- URL extraction: plain text, hyperlink-only, with/without trailing slash, invalid hosts (`github.com.evil.com`), SSRF attempts, username edge cases.
- GitHub client with mocked HTTP (`respx`/`httpx.MockTransport`): 200, 404, 403 rate limit, 304 ETag, timeouts. No real network in unit tests.
- Deterministic signals on fixture payloads (fork-only, empty repos, bulk upload, healthy profile).
- Project matching: exact link, fuzzy name, ambiguous → LLM path with fake LLM, quote verification downgrade.
- Scoring and weight re-normalization with/without GitHub; re-rank uses stored result and makes no LLM or network call.
- Graph test: GitHub failure does not fail the candidate; parallel branch joins correctly.
- Add `scripts/check_github.py` (real network, optional, uses a public demo account) as the acceptance script.

## 10. Acceptance criteria

- Candidate with a valid GitHub link gets `github_score`, per-project verdicts, and red flags in API results and UI.
- Candidate without GitHub is not penalized and shows `not_provided`.
- Same username within TTL causes zero GitHub API calls (cache proof in logs).
- Evaluator latency is unchanged by this step (parallel branch).
- All previous tests still pass; ranking for candidates without GitHub is identical to Step 1.

## 11. Security and fairness notes (document in README)

- GitHub activity correlates with free time and background, so it is a **supporting signal**, never a gate. Do not auto-reject on GitHub results.
- Only public data is read. Do not scrape the HTML site; use the official API.
- Store the minimum needed from GitHub; honor the TTL; do not log tokens.

## 12. Out of scope

Email sending (Step 3), cloning or running candidate code, LinkedIn or other platforms.
