"""Add GitHub columns and populate links for previously uploaded resumes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import execute, fetch, init_database
from app.ingest.pipeline import stored_github_metadata


def main() -> None:
    init_database()
    rows = fetch("""SELECT id,file_path,clean_text,github_username,github_repo_links
        FROM resumes WHERE github_repo_links IS NULL""")
    for row in rows:
        links = stored_github_metadata(row)
        execute("""UPDATE resumes SET github_username=%s,github_repo_links=%s
            WHERE id=%s AND github_repo_links IS NULL""",
            (links["github_username"], links["github_repo_links"], row["id"]))
    print(f"GitHub schema applied. Checked {len(rows)} existing resumes.")


if __name__ == "__main__":
    main()
