import logging
from contextlib import contextmanager
from time import perf_counter

log = logging.getLogger(__name__)


@contextmanager
def timed_stage(name: str):
    start = perf_counter()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        elapsed = perf_counter() - start
        log.info("Pipeline stage=%s latency=%.6fs success=%s", name, elapsed, succeeded,
            extra={"stage": name, "duration_s": elapsed, "succeeded": succeeded})
