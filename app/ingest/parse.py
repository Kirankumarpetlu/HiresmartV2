import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse(content: bytes, file_name: str) -> tuple[str, dict]:
    links = []
    suffix = Path(file_name).suffix.lower()
    if suffix == ".docx":
        from docx import Document
        from docx.table import Table

        document = Document(io.BytesIO(content))
        for part in document.part.package.parts:
            links.extend(rel.target_ref for rel in part.rels.values()
                         if rel.is_external and rel.reltype.endswith("/hyperlink"))
        parts = []
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            else:
                parts.append(block.text)
        return "\n".join(parts), {"file_name": file_name, "page_count": 1, "parse_method": "docx", "hyperlinks": links}
    if suffix != ".pdf":
        raise ValueError("Only PDF and DOCX files are supported")

    import fitz

    pages, used_ocr = [], False
    with fitz.open(stream=content, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        for page in document:
            links.extend(link["uri"] for link in page.get_links()
                         if link["kind"] == fitz.LINK_URI and link.get("uri"))
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            midpoint = page.rect.width / 2
            left = [b for b in blocks if b[2] <= midpoint + 10]
            right = [b for b in blocks if b[0] >= midpoint - 10]
            # A full-width header belongs before both columns, not between their rows.
            if left and right:
                spanning = [b for b in blocks if b not in left and b not in right]
                blocks = sorted(spanning, key=lambda b: b[1]) + sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
            else:
                blocks.sort(key=lambda b: (b[1], b[0]))
            text = "\n".join(b[4].strip() for b in blocks)
            if len(text.strip()) < 30:
                try:
                    import pytesseract
                    from PIL import Image

                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    text = pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
                    used_ocr = True
                except (pytesseract.TesseractNotFoundError, OSError):
                    log.warning("Tesseract is unavailable; skipping OCR for page %d", page.number + 1)
            pages.append(text)
        return "\f".join(pages), {"file_name": file_name, "page_count": len(pages),
                                 "parse_method": "ocr" if used_ocr else "pymupdf", "hyperlinks": links}
