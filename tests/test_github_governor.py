import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from app.db import execute
from app.integrations import github_governor as governor

pytestmark = pytest.mark.skipif(os.environ.get("RUN_POSTGRES_TESTS") != "1", reason="requires PostgreSQL")


def test_quota_reserve_refund_and_reset_are_shared(monkeypatch):
    current = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr(governor, "now", lambda: current)
    monkeypatch.setattr(governor.settings, "github_reserve_remaining", 200)
    reset = current + timedelta(hours=1)
    governor.record("test-core", 60, 57, int(reset.timestamp()))
    try:
        with governor.request_slot():
            with pytest.raises(governor.QuotaWait):
                governor.acquire("test-core", 52, admission=True)
            assert governor.acquire("test-core", 51, admission=True) == 0
            assert governor.acquire("test-core") == 0
            assert governor.acquire("test-core") > 0
            governor.record("test-core", 60, 57, int(reset.timestamp()), refund=True)
        assert governor.quota_state()["test-core"]["remaining"] == 57
        governor.block("test-core", "primary_limit", reset)
        with pytest.raises(governor.QuotaWait) as blocked:
            governor.acquire("test-core")
        assert blocked.value.until == reset
        assert governor.acquire("test-search") == 0
        current = reset + timedelta(seconds=1)
        assert governor.acquire("test-core") == 0
    finally:
        execute("DELETE FROM github_quota_state WHERE bucket IN ('test-core','test-search')")


def test_request_slot_caps_concurrency(monkeypatch):
    monkeypatch.setattr(governor.settings, "github_max_concurrency", 1)
    entered = Event()

    def second():
        with governor.request_slot():
            entered.set()

    with ThreadPoolExecutor(1) as pool:
        with governor.request_slot():
            task = pool.submit(second)
            assert not entered.wait(.15)
        assert entered.wait(3)
        task.result()
