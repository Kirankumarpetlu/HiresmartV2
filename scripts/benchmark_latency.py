"""Measure a fixed ten-resume ingestion-to-results run; retains its data."""
import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document

from app.api import screening_results
from app.config import settings
from app.db import fetch
from app.ingest.pipeline import process_upload
from app.models import ScreeningRequest
from app.screening.graph import graph, read_run, run_screening, setup_checkpointer, thread_config
from app.services import embed_service, llm_service
from app.timing import timed_stage


class Measurements(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stages = []
        self.validation_failures = 0

    def emit(self, record):
        if hasattr(record, "stage"):
            self.stages.append({"stage": record.stage, "seconds": record.duration_s, "succeeded": record.succeeded})
        if "LLM call/validation failed" in record.getMessage():
            self.validation_failures += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements = Measurements()
    logging.basicConfig(level=logging.INFO, handlers=[
        measurements, logging.FileHandler(args.output.with_suffix(".log"), encoding="utf-8"), logging.StreamHandler()])
    with timed_stage("startup.database"):
        fetch("SELECT 1")
        setup_checkpointer()
    with timed_stage("startup.embedder"):
        embed_service.get_embedder()
    with timed_stage("startup.llm"):
        llm_service.get_llm()
    started = perf_counter()
    uploaded = []
    for index in range(10):
        document = Document()
        document.core_properties.identifier = str(uuid4())
        for line in [f"Synthetic Candidate {index}", "Skills", "Python SQL PostgreSQL",
                "Experience", f"Software Engineer 202{index % 2} - 2024",
                f"- Built Python APIs and SQL reporting systems for production service {index}.",
                "- Designed PostgreSQL data pipelines and worked with a team of four engineers.",
                "Education", "BSc Computer Science"]:
            document.add_paragraph(line)
        content = io.BytesIO()
        document.save(content)
        uploaded.append(process_upload(content.getvalue(), f"latency_synthetic_{index}.docx"))
    request = ScreeningRequest(jd_text="Backend engineer. Must have Python and SQL. At least 2 years of professional experience. Build Python APIs and SQL reporting systems. PostgreSQL is preferred.",
        shortlist_size=10, resume_ids=[row["id"] for row in uploaded])
    screening_started = perf_counter()
    with timed_stage("screening.create"):
        jd_id = str(fetch("INSERT INTO job_descriptions (raw_text) VALUES (%s) RETURNING id", (request.jd_text,))[0]["id"])
        graph.update_state(thread_config(jd_id), {"jd_id": jd_id, "request": request.model_dump(mode="json"),
            "weights": request.weights.model_dump(), "candidate_results": [], "ranked": [], "report": {},
            "status": "queued", "error": None}, as_node="__start__")
    run_screening(jd_id)
    rows = screening_results(UUID(jd_id))
    finished = perf_counter()
    run = read_run(jd_id)
    calls_before_warm = llm_service.generation_calls
    warm_started = perf_counter()
    warm_jd_id = str(fetch("INSERT INTO job_descriptions (raw_text) VALUES (%s) RETURNING id", (request.jd_text,))[0]["id"])
    graph.update_state(thread_config(warm_jd_id), {"jd_id": warm_jd_id, "request": request.model_dump(mode="json"),
        "weights": request.weights.model_dump(), "candidate_results": [], "ranked": [], "report": {},
        "status": "queued", "error": None}, as_node="__start__")
    run_screening(warm_jd_id)
    warm_rows = screening_results(UUID(warm_jd_id))
    warm_elapsed = perf_counter() - warm_started
    warm_calls = llm_service.generation_calls - calls_before_warm
    result = {"measured_at": datetime.now(timezone.utc).isoformat(), "synthetic": True,
        "scope": "Local function calls with real models/database; excludes network upload, queue wait and UI polling",
        "jd_id": jd_id, "resume_ids": [row["id"] for row in uploaded], "state": run["status"],
        "ingestion_to_output_seconds": finished - started, "cold_screening_seconds": finished - screening_started,
        "gpu_layers_offloaded": llm_service.gpu_layers_offloaded,
        "warm_rerun": {"jd_id": warm_jd_id, "seconds": warm_elapsed, "llm_calls": warm_calls},
        "validation_failures": measurements.validation_failures, "stages": measurements.stages}
    assert all(row["status"] == "uploaded" for row in uploaded), "Benchmark must measure fresh ingestion"
    assert run["status"] == "done" and len(rows) == 10 and all(row["status"] == "done" for row in rows), "Screening failed; inspect the benchmark log"
    assert rows[0]["final_score"] is not None and rows[0]["summary"], "Ranked output and report must exist"
    assert len(warm_rows) == 10 and warm_calls == 0, "Warm rerun must use JD, evaluation, and report caches"
    assert [(row["resume_id"], row["final_score"]) for row in warm_rows] == [
        (row["resume_id"], row["final_score"]) for row in rows], "Warm cache changed scores or ranking"
    assert llm_service.gpu_layers_offloaded > 0, "Benchmark requires CUDA offload"
    assert all(stage["succeeded"] for stage in measurements.stages), "A measured stage failed"
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
