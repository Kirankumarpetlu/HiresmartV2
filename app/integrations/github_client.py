import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from time import sleep

import httpx
from psycopg.types.json import Jsonb

from app.config import settings
from app.db import execute, fetch
from app.ingest.metadata import GITHUB_REPO, GITHUB_USER
from app.integrations import github_governor as governor

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_blocked_until = {}
_secondary_failures = {}


class RateLimited(Exception):
    category = "quota"
    def __init__(self, bucket: str = "core", retry_after: datetime | None = None):
        self.bucket = bucket
        self.retry_after = retry_after
        super().__init__(bucket)


class PrimaryRateLimited(RateLimited):
    category = "primary_limit"


class QuotaReserved(PrimaryRateLimited):
    category = "reserve"


class SecondaryRateLimited(RateLimited):
    category = "secondary_limit"


class GithubForbidden(Exception):
    pass


class GithubNotFound(Exception):
    pass


def endpoint_category(path: str) -> str:
    if path == "/rate_limit":
        return "quota"
    if path.startswith("/users/"):
        return "catalog" if "/repos?" in path else "profile"
    suffix = path.split("?", 1)[0].rsplit("/", 1)[-1]
    return {"languages": "languages", "readme": "readme", "commits": "commits",
            "contents": "root", "workflows": "workflow"}.get(suffix, "repository")


def int_header(headers, key):
    value = headers.get(key, "")
    return int(value) if value.isdecimal() and len(value) <= 12 else None


def quota_log(stage: str) -> None:
    try:
        with GithubClient("octocat") as client:
            for bucket, quota in client.rate_limit().items():
                log.info("GitHub quota stage=%s bucket=%s limit=%s remaining=%s reset=%s", stage, bucket,
                         quota.get("limit"), quota.get("remaining"), quota.get("reset"))
    except Exception as exc:
        log.warning("GitHub quota stage=%s unavailable type=%s", stage, type(exc).__name__)


def warn_if_unauthenticated() -> None:
    if settings.github_enabled and not settings.github_token.get_secret_value():
        log.warning("GitHub is enabled without a token. The unauthenticated core limit is 60 requests/hour.")


def now() -> datetime:
    return datetime.now(timezone.utc)


def fields(item: dict, names: str) -> dict:
    return {name: item.get(name) for name in names.split()}


def repo_metadata(item: dict) -> dict:
    return fields(item, "name description topics language fork private size created_at pushed_at license stargazers_count") | {
        "license": bool(item.get("license")),
        "description": (item.get("description") or "")[:500],
    }


def readme_head(item: dict) -> str:
    if item.get("encoding") != "base64":
        return ""
    encoded = "".join(item.get("content", "").split())
    return base64.b64decode(encoded[:8192]).decode("utf-8", errors="replace")[:4000]


class GithubClient:
    def __init__(self, username: str, transport: httpx.BaseTransport | None = None):
        if not GITHUB_USER.fullmatch(username):
            raise ValueError("Invalid GitHub username")
        self.username = username.lower()
        self.dirty = {}
        self.network_calls = 0
        self.cache_hits = 0
        self.stale_at = None
        self.stale_limit = None
        self.primary_limit_hits = 0
        self.secondary_limits = 0
        self.shared = transport is None
        self.budget_checked = set()
        headers = {"User-Agent": "ResumeScreeningAnalyzer/1.0", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        token = settings.github_token.get_secret_value()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=settings.github_api_base, headers=headers,
                                 timeout=settings.github_timeout_seconds, follow_redirects=False,
                                 trust_env=False, transport=transport)

    def __enter__(self):
        try:
            rows = fetch("SELECT payload FROM github_profiles WHERE username=%s", (self.username,)) if settings.cache_enabled else []
        except Exception:
            self.http.close()
            raise
        self.cached = rows[0]["payload"] if rows else {}
        return self

    def __exit__(self, *exc):
        self.http.close()
        user = self.dirty.get(f"/users/{self.username}", {})
        if self.dirty and settings.cache_enabled:
            execute("""INSERT INTO github_profiles (username,fetched_at,payload,etag)
                VALUES (%s,COALESCE(%s::timestamptz,now()),%s,%s)
                ON CONFLICT (username) DO UPDATE SET
                    fetched_at=COALESCE(%s::timestamptz,github_profiles.fetched_at),
                    payload=github_profiles.payload || EXCLUDED.payload,
                    etag=CASE WHEN %s::timestamptz IS NULL THEN github_profiles.etag ELSE EXCLUDED.etag END""",
                (self.username, user.get("at"), Jsonb(self.dirty), user.get("etag"), user.get("at"), user.get("at")))
        log.info("GitHub cache hits=%d network_calls=%d", self.cache_hits, self.network_calls)

    def get(self, path: str, compact, *, empty_statuses=(404,), bucket="core"):
        entry = self.dirty.get(path) or self.cached.get(path)
        category = endpoint_category(path)
        ttl = (settings.github_stable_cache_ttl_hours if category in {"readme", "languages", "root", "workflow"}
               else settings.github_cache_ttl_hours)
        if entry and now() - datetime.fromisoformat(entry["at"]) < timedelta(hours=ttl):
            self.cache_hits += 1
            log.info("GitHub endpoint=%s cache=hit", category)
            return entry["value"]
        headers = {"If-None-Match": entry["etag"]} if entry and entry.get("etag") else {}
        for attempt in range(2):
            try:
                value, etag = self.request(path, headers, entry, compact, empty_statuses, bucket)
                break
            except GithubNotFound:
                if 404 not in empty_statuses:
                    raise
                value, etag = None, None
                break
            except RateLimited as exc:
                if entry is None:
                    raise
                self.cache_hits += 1
                self.stale_limit = exc
                at = datetime.fromisoformat(entry["at"])
                self.stale_at = min(self.stale_at, at) if self.stale_at else at
                log.info("GitHub endpoint=%s cache=stale", category)
                return entry["value"]
            except httpx.TransportError:
                if attempt:
                    raise
        self.dirty[path] = {"at": now().isoformat(), "etag": etag, "value": value}
        return value

    def rate_limit(self) -> dict:
        if self.shared:
            state = governor.quota_state()
            if any(row["blocked_category"] == "secondary_limit" and row["blocked_until"] and row["blocked_until"] > now()
                   for row in state.values()):
                return {key: {"limit": row["limit"], "remaining": row["remaining"],
                              "reset": int(row["reset_at"].timestamp()) if row["reset_at"] else None}
                        for key, row in state.items()}
            with governor.request_slot():
                return self._rate_limit()
        return self._rate_limit()

    def _rate_limit(self) -> dict:
        with self.http.stream("GET", "/rate_limit") as response:
            if self.shared and response.status_code in (403, 429):
                self.classify(response, "core")
            response.raise_for_status()
            resources = self.read_json(response).get("resources", {})
        buckets = {bucket: {key: value for key in ("limit", "remaining", "reset")
                         if isinstance(value := resources.get(bucket, {}).get(key), int)}
                for bucket in ("core", "graphql", "search") if bucket in resources}
        if self.shared:
            for bucket, counts in buckets.items():
                governor.record(bucket, counts.get("limit"), counts.get("remaining"), counts.get("reset"))
        return buckets

    @staticmethod
    def read_json(response):
        body = bytearray()
        for block in response.iter_bytes(chunk_size=65536):
            if len(body) + len(block) > MAX_RESPONSE_BYTES:
                raise ValueError("GitHub response exceeds the size limit")
            body.extend(block)
        return json.loads(body)

    def request(self, path, headers, entry, compact, empty_statuses, bucket="core"):
        if self.shared:
            try:
                with governor.request_slot():
                    if bucket not in self.budget_checked:
                        estimate = 1 + settings.github_catalog_max_pages + 5 * min(5, settings.github_max_repos) + 3
                        governor.acquire(bucket, estimate, admission=True)
                        self.budget_checked.add(bucket)
                    while delay := governor.acquire(bucket):
                        sleep(delay)
                    return self._request(path, headers, entry, compact, empty_statuses, bucket)
            except governor.QuotaWait as exc:
                error = {"secondary_limit": SecondaryRateLimited, "reserve": QuotaReserved}.get(exc.category, PrimaryRateLimited)
                raise error(exc.bucket, exc.until) from None
        blocked = _blocked_until.get(bucket)
        if blocked and blocked.retry_after > now():
            raise type(blocked)(bucket, blocked.retry_after)
        return self._request(path, headers, entry, compact, empty_statuses, bucket)

    def classify(self, response, bucket):
        reset = int_header(response.headers, "x-ratelimit-reset")
        delay = int_header(response.headers, "retry-after")
        delay = min(delay, 604800) if delay is not None else None
        try:
            message = str(self.read_json(response).get("message", "")).lower()
        except (ValueError, AttributeError):
            message = ""
        if int_header(response.headers, "x-ratelimit-remaining") == 0:
            self.primary_limit_hits += 1
            retry = datetime.fromtimestamp(reset, timezone.utc) if reset and reset < 253402300800 else now() + timedelta(seconds=60)
            retry = max(retry, now() + timedelta(seconds=delay or 0))
            error_type = PrimaryRateLimited
        elif delay is not None or response.status_code == 429 or "secondary rate limit" in message or "abuse" in message:
            attempt = _secondary_failures.get(bucket, 0)
            retry = now() + timedelta(seconds=delay if delay is not None else 60 * 2 ** attempt)
            if not self.shared:
                _secondary_failures[bucket] = min(attempt + 1, 6)
            error_type = SecondaryRateLimited
            self.secondary_limits += 1
        else:
            raise GithubForbidden
        if self.shared:
            retry = governor.block(bucket, error_type.category, retry,
                                   fallback=error_type is SecondaryRateLimited and delay is None)
        error = error_type(bucket, retry)
        if not self.shared:
            _blocked_until[bucket] = error
        raise error

    def _request(self, path, headers, entry, compact, empty_statuses, bucket):
        self.network_calls += 1
        with self.http.stream("GET", path, headers=headers) as response:
            etag = response.headers.get("etag")
            counts = {key: int_header(response.headers, key) for key in
                      ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after")}
            resource = response.headers.get("x-ratelimit-resource")
            bucket = resource if resource in {"core", "graphql", "search"} else bucket
            log.info("GitHub endpoint=%s HTTP=%s cache=%s resource=%s headers=%s", endpoint_category(path),
                      response.status_code, "304" if response.status_code == 304 else "miss", bucket, counts)
            if self.shared:
                governor.record(bucket, counts["x-ratelimit-limit"], counts["x-ratelimit-remaining"],
                                counts["x-ratelimit-reset"],
                                refund=response.status_code == 304 and "Authorization" in self.http.headers)
            if response.status_code == 304 and entry:
                self.cache_hits += 1
                return entry["value"], etag or entry.get("etag")
            if response.status_code in {403, 429}:
                self.classify(response, bucket)
            if response.status_code == 404:
                raise GithubNotFound
            if response.status_code in empty_statuses:
                return None, None
            response.raise_for_status()
            if self.shared:
                governor.clear_secondary(bucket)
            else:
                _secondary_failures.pop(bucket, None)
            return compact(self.read_json(response)), etag

    def profile(self) -> tuple[dict | None, list[dict]]:
        path = f"/users/{self.username}"
        user = self.get(path, lambda item: fields(item, "created_at followers public_repos"))
        if user is None:
            return None, []
        repos = {}
        pages = min(settings.github_catalog_max_pages, max(1, (user["public_repos"] + 99) // 100))
        for page in range(1, pages + 1):
            batch = self.get(path + f"/repos?per_page=100&sort=pushed&type=owner&page={page}",
                             lambda items: [repo_metadata(item) for item in items if item.get("private") is False])
            for repo in batch or []:
                repos[repo["name"].lower()] = repo
            if not batch or len(batch) < 100:
                break
        return user, list(repos.values())

    def repository(self, owner: str, name: str) -> dict | None:
        path = self.repo_path(owner, name)
        return self.get(path, lambda item: repo_metadata(item) if item.get("private") is False else None)

    @staticmethod
    def repo_path(owner: str, name: str) -> str:
        if not GITHUB_USER.fullmatch(owner) or not GITHUB_REPO.fullmatch(name):
            raise ValueError("Invalid GitHub repository path")
        return f"/repos/{owner.lower()}/{name.lower()}"

    def details(self, owner: str, name: str, *, deep: bool = True, language: str | None = None) -> dict:
        path = self.repo_path(owner, name)
        languages = (self.get(path + "/languages", dict) or {}) if deep else ({language: 1} if language else {})
        readme = self.get(path + "/readme", readme_head) or ""
        commits = self.get(path + f"/commits?author={self.username}&per_page={settings.github_commits_per_page}",
                           lambda items: [item["commit"]["author"]["date"] for item in items
                                          if ((item.get("author") or {}).get("login") or "").lower() == self.username],
                           empty_statuses=(404, 409)) or []
        root = self.get(path + "/contents", lambda items: [
            item["name"].lower() for item in items if item.get("type") in {"dir", "file"}
        ]) or []
        ci = deep and ".github" in root and bool(self.get(path + "/contents/.github/workflows", lambda items: any(
            item.get("type") == "file" and item["name"].lower().endswith((".yml", ".yaml")) for item in items)))
        return {"languages": languages, "readme": readme, "commits": commits, "ci": ci,
                "tests": bool({"tests", "test", "__tests__"}.intersection(root)),
                "dockerfile": "dockerfile" in root,
                "license_file": any(name.startswith(("license", "copying")) for name in root)}
