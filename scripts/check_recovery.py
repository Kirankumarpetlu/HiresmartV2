"""Prepare/verify a real interrupted screening around an external backend restart."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from app.config import settings


def get(path: str):
    response = requests.get(settings.api_url + path, timeout=10)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "retry"])
    args = parser.parse_args()
    path = settings.data_dir / "recovery_acceptance.json"
    log_path = settings.data_dir / "service.log"
    if args.action == "retry":
        from app.db import execute
        from app.screening.graph import graph, setup_checkpointer, thread_config

        key = json.loads(path.read_text())["jd_id"]
        setup_checkpointer()
        execute("UPDATE screening_results SET status='pending',error_message=NULL WHERE jd_id=%s AND status='error'", (key,))
        graph.update_state(thread_config(key), {"status": "evaluating", "error": None}, as_node="fan_out_candidates")
        print("Unfinished/failed candidates queued for the next backend startup.")
        sys.exit(0)
    if args.action == "prepare":
        response = requests.post(settings.api_url + "/screenings", json={"jd_text": "Backend engineer. Required Python and SQL. Minimum 2 years. Build reliable production APIs and data reporting. PostgreSQL preferred.", "shortlist_size": 3}, timeout=10)
        response.raise_for_status()
        key = response.json()["jd_id"]
        for _ in range(600):
            progress = get(f"/screenings/{key}/status")
            if progress["done"] >= 1 and progress["state"] in ("running", "evaluating"):
                rows = get(f"/screenings/{key}/results")
                finished = {row["resume_id"]: row["skill_report"] for row in rows if row["status"] == "done"}
                path.write_text(json.dumps({"jd_id": key, "finished": finished, "log_offset": log_path.stat().st_size}, indent=2))
                print("Checkpoint captured; stop the backend now.", key, flush=True)
                break
            if progress["state"] in ("done", "awaiting_approval", "error"):
                raise AssertionError("Run ended before an interruption checkpoint could be captured")
            time.sleep(1)
        else:
            raise TimeoutError("No completed candidate checkpoint appeared")
    else:
        checkpoint = json.loads(path.read_text())
        key = checkpoint["jd_id"]
        for _ in range(600):
            progress = get(f"/screenings/{key}/status")
            if progress["state"] in ("done", "awaiting_approval", "error"):
                break
            time.sleep(1)
        assert progress["state"] in ("done", "awaiting_approval") and progress["errors"] == 0, progress
        rows = {row["resume_id"]: row for row in get(f"/screenings/{key}/results")}
        with log_path.open("rb") as log:
            log.seek(checkpoint["log_offset"])
            after = log.read().decode("utf-8")
        for resume_id, skills in checkpoint["finished"].items():
            assert rows[resume_id]["skill_report"] == skills
            assert f"Candidate saved jd={key} resume={resume_id}" not in after
        checkpoint.update(passed=True, status=progress)
        path.write_text(json.dumps(checkpoint, indent=2))
        print("Restart recovery passed; completed candidates were not evaluated again.")
