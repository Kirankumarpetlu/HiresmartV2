import hashlib
from pathlib import Path

from app.config import settings
from app.cache import cached_embeddings
from app.db import get_pool
from app.ingest.chunk import chunk
from app.ingest.metadata import github_metadata, metadata
from app.ingest.parse import parse
from app.ingest.preprocess import preprocess
from app.services.embed_service import encode
from app.timing import timed_stage


def stored_github_metadata(resume: dict) -> dict:
    if resume.get("github_repo_links") is not None:
        return {"github_username": resume.get("github_username"), "github_repo_links": resume["github_repo_links"]}
    text = resume.get("clean_text", "")
    if resume.get("file_path"):
        path = Path(resume["file_path"]).resolve()
        if path.is_relative_to((settings.data_dir / "uploads").resolve()):
            try:
                with path.open("rb") as source:
                    raw = source.read(settings.max_upload_bytes + 1)
                if len(raw) <= settings.max_upload_bytes:
                    parsed_text, parsed = parse(raw, path.name)
                    return github_metadata(parsed_text, parsed["hyperlinks"])
            except (OSError, RuntimeError, ValueError):
                pass
    meta = github_metadata(text)
    meta["github_username"] = resume.get("github_username") or meta["github_username"]
    return meta


@timed_stage("ingest.total")
def process_upload(content: bytes, file_name: str) -> dict:
    if not content or len(content) > settings.max_upload_bytes:
        raise ValueError("File is empty or exceeds the upload size limit")
    digest = hashlib.sha256(content).hexdigest()
    suffix = Path(file_name).suffix.lower()
    path = (settings.data_dir / "uploads" / (digest + suffix)).resolve()
    with get_pool().connection() as conn:
        # Cross-process lock makes duplicate detection and file writes atomic for this hash.
        with timed_stage("ingest.duplicate_check"):
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (int(digest[:15], 16),))
            existing = conn.execute("SELECT id FROM resumes WHERE file_path = %s", (str(path),)).fetchone()
        if existing:
            return {"id": str(existing["id"]), "status": "duplicate"}
        with timed_stage("ingest.parse"):
            raw, parsed = parse(content, file_name)
        with timed_stage("ingest.preprocess"):
            text = preprocess(raw)
        if not text.strip():
            raise ValueError("No readable text found; scanned documents may need Tesseract")
        with timed_stage("ingest.metadata"):
            meta = metadata(text, parsed["hyperlinks"], link_text=raw)
        with timed_stage("ingest.chunk"):
            chunks = chunk(text)
        with timed_stage("ingest.embed"):
            vectors = cached_embeddings([item["content"] for item in chunks], encode)
        with timed_stage("ingest.write_file"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        # Keep the source if a database commit fails; an orphan is safer than lost data.
        with timed_stage("ingest.store"):
            row = conn.execute("""INSERT INTO resumes
                (file_name, file_path, parse_method, clean_text, skills, years_exp, education, embed_model,
                 github_username, github_repo_links, contact_email, candidate_name)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (Path(file_name).name, str(path), parsed["parse_method"], text, meta["skills"], meta["years_exp"],
                 meta["education"], settings.embed_model, meta["github_username"], meta["github_repo_links"],
                 meta["contact_email"], meta["candidate_name"])).fetchone()
            with conn.cursor() as cursor:
                cursor.executemany("""INSERT INTO resume_chunks (resume_id, section, label, content, token_count, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s)""", [(row["id"], item["section"], item["label"], item["content"], item["token_count"], vector)
                    for item, vector in zip(chunks, vectors)])
            conn.commit()
    return {"id": str(row["id"]), "status": "uploaded", "chunks": len(chunks)}
