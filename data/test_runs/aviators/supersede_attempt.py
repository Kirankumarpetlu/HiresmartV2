import json
from pathlib import Path

from app.screening.graph import read_run, write_run

path = Path("data/test_runs/aviators/manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
for name, record in list(manifest["screenings"].items()):
    run = read_run(record["jd_id"])
    run.update(state="error", error_message="Test superseded after correcting requirement extraction or evaluator output alignment.")
    write_run(record["jd_id"], run)
    manifest.setdefault("superseded_attempts", []).append(dict(record, file_name=name, reason=run["error_message"]))
    del manifest["screenings"][name]
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Original test attempt retained; corrected test will start a new screening.")
