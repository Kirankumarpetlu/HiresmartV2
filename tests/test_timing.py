import logging

import pytest

from app import timing


@pytest.mark.parametrize("fails", [False, True])
def test_stage_records_duration_without_swallowing_errors(monkeypatch, caplog, fails):
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(timing, "perf_counter", lambda: next(ticks))

    @timing.timed_stage("test.stage")
    def work():
        if fails:
            raise ValueError("private input")
        return "done"

    with caplog.at_level(logging.INFO, logger="app.timing"):
        if fails:
            with pytest.raises(ValueError, match="private input"):
                work()
        else:
            assert work() == "done"
    record = caplog.records[-1]
    assert record.stage == "test.stage" and record.duration_s == 0.25
    assert record.succeeded is not fails
    assert "private input" not in caplog.text
