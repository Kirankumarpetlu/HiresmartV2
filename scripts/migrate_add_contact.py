"""Apply notification schema and backfill private contact metadata for existing resumes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import execute, fetch, init_database
from app.ingest.metadata import contact_metadata
from app.ingest.parse import parse


def main() -> None:
    init_database()
    rows = fetch("SELECT id,file_path,clean_text FROM resumes WHERE contact_email IS NULL OR candidate_name IS NULL")
    root = (settings.data_dir / "uploads").resolve()
    for row in rows:
        path = Path(row["file_path"]).resolve()
        raw, links = None, []
        if path.is_relative_to(root):
            try:
                with path.open("rb") as source:
                    content = source.read(settings.max_upload_bytes + 1)
                if len(content) <= settings.max_upload_bytes:
                    raw, parsed = parse(content, path.name)
                    links = parsed["hyperlinks"]
            except (OSError, RuntimeError, ValueError):
                pass
        meta = contact_metadata(row["clean_text"], links, raw_text=raw)
        execute("""UPDATE resumes SET contact_email=COALESCE(contact_email,%s),
            candidate_name=COALESCE(candidate_name,%s) WHERE id=%s""",
            (meta["contact_email"], meta["candidate_name"], row["id"]))
    print(f"Contact schema applied. Checked {len(rows)} existing resumes.")


if __name__ == "__main__":
    main()
