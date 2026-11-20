"""Upload a resume folder, screen only those candidates, and export local test results."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from app.config import settings


def request(method: str, path: str, **kwargs):
    response = requests.request(method, settings.api_url.rstrip("/") + path, timeout=180, **kwargs)
    response.raise_for_status()
    return response.json()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def export_csv(path: Path, rows: list[dict]) -> None:
    columns = ["rank", "file_name", "resume_id", "status", "final_score", "recommendation", "similarity", "years_exp", "summary", "error_message"]
    columns += ["github_status", "github_score", "github_explanation", "github_json"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            github = row.get("github_json")
            if github is None:
                writer.writerow(row)
                continue
            writer.writerow(dict(row, github_explanation=github.get("score_explanation") or github.get("summary", ""),
                                 github_json=json.dumps(github, ensure_ascii=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resumes", type=Path)
    parser.add_argument("jds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, default=settings.shortlist_size)
    parser.add_argument("--upload-only", action="store_true")
    args = parser.parse_args()
    assert args.resumes.is_dir() and args.jds.is_dir(), "Both source directories must exist"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"uploads": [], "screenings": {}}
    request("GET", "/health")
    files = sorted(path for path in args.resumes.iterdir() if path.suffix.lower() in (".pdf", ".docx"))
    assert files, "No PDF/DOCX resumes found"
    uploaded = []
    for index, file in enumerate(files, 1):
        try:
            result = request("POST", "/resumes", files=[("files", (file.name, file.read_bytes()))])[0]
        except requests.RequestException as exc:
            result = {"file_name": file.name, "status": "error", "error": type(exc).__name__}
        uploaded.append(result)
        manifest["uploads"] = uploaded
        save(manifest_path, manifest)
        print(f"Upload {index}/{len(files)}: {file.name}: {result['status']}", flush=True)
    ids = list(dict.fromkeys(item["id"] for item in uploaded if item["status"] in ("uploaded", "duplicate")))
    assert ids, "No resumes could be ingested"
    if args.upload_only:
        print(f"Uploaded/indexed {len(ids)} of {len(files)} resumes", flush=True)
        return
    jd_files = sorted(path for path in args.jds.iterdir() if path.suffix.lower() in (".pdf", ".docx", ".txt"))
    assert jd_files, "No PDF/DOCX/TXT job descriptions found"
    for file in jd_files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        record = manifest["screenings"].get(file.name)
        if not record or record.get("sha256") != digest:
            started = request("POST", "/screenings", data={"shortlist_size": args.shortlist_size, "resume_ids": json.dumps(ids)}, files={"file": (file.name, file.read_bytes())})
            record = {"jd_id": started["jd_id"], "sha256": digest}
            manifest["screenings"][file.name] = record
            save(manifest_path, manifest)
        key = record["jd_id"]
        print(f"Screening {file.name}: {key}", flush=True)
        previous = None
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            progress = request("GET", f"/screenings/{key}/status")
            if progress != previous:
                print(json.dumps(progress), flush=True)
                previous = progress
            if progress["total"]:
                rows = request("GET", f"/screenings/{key}/results")
                assert all(row["resume_id"] in ids for row in rows), "Test included a resume outside the selected folder"
                save(args.output / f"{key}_results.json", rows)
                export_csv(args.output / f"{key}_results.csv", rows)
            if progress["state"] in ("done", "awaiting_approval", "error"):
                break
            time.sleep(3)
        else:
            raise TimeoutError("Screening exceeded the two-hour test deadline; server progress remains saved")
        record.update(status=progress)
        save(manifest_path, manifest)
        assert progress["state"] in ("done", "awaiting_approval"), progress
        assert progress["done"] > 0, "No candidates successfully evaluated; inspect the saved results"
        print(f"Finished: {progress['done']} evaluated, {progress['errors']} errors, {progress['filtered_out']} filtered", flush=True)


if __name__ == "__main__":
    main()
