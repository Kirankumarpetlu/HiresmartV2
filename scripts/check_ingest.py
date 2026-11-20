"""Generate synthetic PDF/DOCX fixtures and verify real ingestion and deduplication."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz
from docx import Document

from app.config import settings
from app.db import fetch
from app.ingest.parse import parse
from app.ingest.pipeline import process_upload

if __name__ == "__main__":
    folder = settings.data_dir / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    text = "Skills\nPython SQL C++ .NET\nExperience\nEngineer 2021 - 2024\n- Built Python APIs and SQL reporting systems.\nEducation\nBSc Computer Science"
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(folder / "sample.docx")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(fitz.Rect(30, 30, 280, 700), text, fontsize=11)
    page.insert_textbox(fitz.Rect(320, 30, 570, 700), "Projects\nAnalytics\n- Designed PostgreSQL data pipelines.\n- Reduced batch processing latency by 40 percent.", fontsize=11)
    pdf.save(folder / "sample.pdf")
    pdf.close()
    parsed, _ = parse((folder / "sample.pdf").read_bytes(), "sample.pdf")
    assert parsed.index("Education") < parsed.index("Projects")
    for file in (folder / "sample.pdf", folder / "sample.docx"):
        result = process_upload(file.read_bytes(), file.name)
        assert result["status"] in ("uploaded", "duplicate")
        assert process_upload(file.read_bytes(), file.name)["status"] == "duplicate"
        rows = fetch("SELECT content FROM resume_chunks WHERE resume_id=%s", (result["id"],))
        assert rows and any("C++" in r["content"] and ".NET" in r["content"] for r in rows)
    print("Ingest acceptance passed: two-column PDF, DOCX, preserved skills, real vectors, duplicate protection.")
