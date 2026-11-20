import json
from pathlib import Path

from app.db import execute
from app.screening.graph import read_run, write_run

manifest = json.loads(Path("data/test_runs/aviators/manifest.json").read_text(encoding="utf-8"))
record = next(iter(manifest["screenings"].values()))
key = record["jd_id"]
execute("UPDATE screening_results SET status='pending',error_message=NULL WHERE jd_id=%s AND status='error'", (key,))
run = read_run(key)
run.update(state="queued", error_message=None)
write_run(key, run)
print("Unfinished/failed test candidates queued; valid completed evaluations preserved.")
