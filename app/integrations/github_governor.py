from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from time import sleep

from app.config import settings
from app.db import fetch, get_pool

_slot_connection = ContextVar("github_slot_connection", default=None)


class QuotaWait(Exception):
    def __init__(self, bucket: str, until: datetime, category: str):
        self.bucket, self.until, self.category = bucket, until, category


class ConfigBlocked(Exception):
    def __init__(self, category: str):
        self.category = category


def effective_reserve(limit: int | None) -> int:
    return min(settings.github_reserve_remaining, ((limit or 0) + 9) // 10)


def now() -> datetime:
    return datetime.now(timezone.utc)


def quota_state() -> dict:
    return {row["bucket"]: row for row in fetch("SELECT * FROM github_quota_state")}


@contextmanager
def transaction():
    conn = _slot_connection.get()
    if conn is not None:
        with conn.transaction():
            yield conn
    else:
        with get_pool().connection() as conn:
            yield conn


def acquire(bucket: str, cost: int = 1, *, admission: bool = False) -> float:
    """Reserve a request, check a candidate budget, or return the pacing delay."""
    with transaction() as conn:
        conn.execute("INSERT INTO github_quota_state (bucket) VALUES (%s) ON CONFLICT DO NOTHING", (bucket,))
        row = conn.execute("SELECT * FROM github_quota_state WHERE bucket=%s FOR UPDATE", (bucket,)).fetchone()
        current = now()
        limit = row["limit"]
        reserve = effective_reserve(limit)
        if limit is not None and cost > limit - reserve:
            raise ConfigBlocked("reserve_exceeds_limit" if settings.github_token.get_secret_value() else "no_token")
        if row["blocked_until"] and row["blocked_until"] > current:
            raise QuotaWait(bucket, row["blocked_until"], row["blocked_category"] or "quota")
        remaining, reset = row["remaining"], row["reset_at"]
        expired = reset is not None and reset <= current
        if expired:
            remaining, reset = None, None
            conn.execute("UPDATE github_quota_state SET remaining=NULL,reset_at=NULL,next_request_at=NULL WHERE bucket=%s",
                         (bucket,))
        if remaining is not None and remaining - cost < reserve:
            category = "primary_limit" if remaining == 0 else "reserve"
            raise QuotaWait(bucket, reset or current + timedelta(minutes=1), category)
        if admission:
            return 0
        next_at = None if expired else row["next_request_at"]
        if next_at and next_at > current:
            return (next_at - current).total_seconds()
        usable = max(1, (remaining or 0) - reserve)
        interval = max(0.25, (reset - current).total_seconds() / usable) if reset else 0.25
        conn.execute("""UPDATE github_quota_state SET remaining=%s,next_request_at=%s,updated_at=now()
            WHERE bucket=%s""", (remaining - cost if remaining is not None else None,
                                  current + timedelta(seconds=min(interval, 60)), bucket))
        return 0


def record(bucket: str, limit: int | None, remaining: int | None, reset: int | None, *, refund: bool = False) -> None:
    reset_at = datetime.fromtimestamp(reset, timezone.utc) if reset and reset < 253402300800 else None
    with transaction() as conn:
        conn.execute("""INSERT INTO github_quota_state (bucket,"limit",remaining,reset_at)
            VALUES (%s,%s,%s,%s) ON CONFLICT (bucket) DO UPDATE SET
            "limit"=COALESCE(EXCLUDED."limit",github_quota_state."limit"),
            remaining=CASE WHEN EXCLUDED.reset_at > github_quota_state.reset_at
                OR (EXCLUDED."limit" IS NOT NULL AND EXCLUDED."limit" IS DISTINCT FROM github_quota_state."limit")
                THEN EXCLUDED.remaining
                ELSE LEAST(EXCLUDED.remaining, github_quota_state.remaining + %s) END,
            reset_at=CASE WHEN EXCLUDED."limit" IS NOT NULL AND EXCLUDED."limit" IS DISTINCT FROM github_quota_state."limit"
                THEN EXCLUDED.reset_at ELSE GREATEST(EXCLUDED.reset_at,github_quota_state.reset_at) END,
            blocked_until=CASE WHEN EXCLUDED."limit" IS NOT NULL AND EXCLUDED."limit" IS DISTINCT FROM github_quota_state."limit"
                THEN NULL ELSE github_quota_state.blocked_until END,
            next_request_at=CASE WHEN EXCLUDED."limit" IS NOT NULL AND EXCLUDED."limit" IS DISTINCT FROM github_quota_state."limit"
                THEN NULL ELSE github_quota_state.next_request_at END,updated_at=now()
            WHERE EXCLUDED.reset_at IS NULL OR github_quota_state.reset_at IS NULL
                OR EXCLUDED.reset_at >= github_quota_state.reset_at
                OR (EXCLUDED."limit" IS NOT NULL AND EXCLUDED."limit" IS DISTINCT FROM github_quota_state."limit")""",
            (bucket, limit, remaining, reset_at, int(refund)))


def block(bucket: str, category: str, until: datetime | None, *, fallback: bool = False) -> datetime:
    with transaction() as conn:
        conn.execute("INSERT INTO github_quota_state (bucket) VALUES (%s) ON CONFLICT DO NOTHING", (bucket,))
        row = conn.execute("SELECT * FROM github_quota_state WHERE bucket=%s FOR UPDATE", (bucket,)).fetchone()
        if fallback:
            until = now() + timedelta(seconds=60 * 2 ** min(row["secondary_failures"], 6))
        if row["blocked_until"] and row["blocked_until"] > until:
            return row["blocked_until"]
        conn.execute("""UPDATE github_quota_state SET blocked_until=%s,blocked_category=%s,
            secondary_failures=CASE WHEN %s='secondary_limit' THEN LEAST(secondary_failures+1,6) ELSE 0 END,
            updated_at=now() WHERE bucket=%s""", (until, category, category, bucket))
        return until


def clear_secondary(bucket: str) -> None:
    with transaction() as conn:
        conn.execute("""UPDATE github_quota_state SET secondary_failures=0 WHERE bucket=%s
            AND secondary_failures>0 AND (blocked_until IS NULL OR blocked_until<=now())""", (bucket,))


@contextmanager
def request_slot():
    # Session locks release on process death, including a worker killed during HTTP.
    with get_pool().connection() as conn:
        while True:
            for slot in range(settings.github_max_concurrency):
                available = conn.execute("SELECT pg_try_advisory_lock(813271, %s) AS acquired", (slot,)).fetchone()
                if available["acquired"]:
                    break
            else:
                conn.commit()
                sleep(0.05)
                continue
            break
        conn.commit()
        token = _slot_connection.set(conn)
        try:
            yield
        finally:
            _slot_connection.reset(token)
            conn.execute("SELECT pg_advisory_unlock(813271, %s)", (slot,))
