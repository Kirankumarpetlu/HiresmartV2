"""Acceptance test against the running backend. Uses only synthetic fixtures."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from app.config import settings


def request(method: str, path: str, **kwargs):
    response = requests.request(method, settings.api_url + path, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    health = request("GET", "/health")
    assert health["database"] and health["llm_loaded"] and health["embeddings_loaded"]
    assert health["gpu_layers_offloaded"] > 0
    for name in ("sample.pdf", "sample.docx"):
        path = settings.data_dir / "samples" / name
        result = request("POST", "/resumes", files=[("files", (name, path.read_bytes()))])
        assert result[0]["status"] in ("uploaded", "duplicate")
    jd = "Backend engineer. Must have Python and SQL. At least 2 years of professional experience. Build Python APIs and SQL reporting systems. PostgreSQL is preferred."
    started = request("POST", "/screenings", json={"jd_text": jd, "shortlist_size": 2})
    key = started["jd_id"]
    print("Screening", key, flush=True)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        progress = request("GET", f"/screenings/{key}/status")
        print(json.dumps(progress), flush=True)
        if progress["state"] in ("done", "awaiting_approval", "error"):
            break
        time.sleep(3)
    else:
        raise TimeoutError("End-to-end screening did not finish within 15 minutes")
    assert progress["state"] in ("done", "awaiting_approval"), progress
    assert progress["done"] >= 1 and progress["errors"] == 0, progress
    rows = request("GET", f"/screenings/{key}/results")
    assert any(row["final_score"] is not None and row["skill_report"] for row in rows)
    logs = settings.data_dir / "service.log"
    calls_before = logs.read_text(encoding="utf-8").count("LLM call prompt_tokens=")
    reranked = request("POST", f"/screenings/{key}/rerank", json={"proficiency": 0.7, "experience": 0.1, "domain": 0.1, "education": 0.1})
    assert reranked
    assert logs.read_text(encoding="utf-8").count("LLM call prompt_tokens=") == calls_before
    (settings.data_dir / "flow_acceptance.json").write_text(json.dumps({"jd_id": key, "status": progress, "rerank_without_llm": True}, indent=2))
    print("Full API flow passed; reranking made no LLM calls.", flush=True)
