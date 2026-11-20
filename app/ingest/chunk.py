import re
from collections.abc import Callable

from app.config import settings
from app.ingest.metadata import DATE_RANGE
from app.ingest.preprocess import heading
from app.services.embed_service import token_count


def chunk(text: str, count: Callable[[str], int] = token_count) -> list[dict]:
    sections, current, lines = [], "other", []
    for line in text.splitlines():
        found = heading(line)
        if found:
            if lines:
                sections.append((current, lines))
            current, lines = found, []
        else:
            lines.append(line)
    if lines:
        sections.append((current, lines))
    fallback = all(section == "other" for section, _ in sections)
    result = []
    for section, lines in sections:
        groups, group = [], []
        for line in lines:
            if section in ("experience", "project") and group and (not line.strip() or (DATE_RANGE.search(line) and any(DATE_RANGE.search(item) for item in group))):
                groups.append(group)
                group = []
            if line.strip():
                group.append(line)
        if group:
            groups.append(group)
        for group in groups:
            label = group[0] if section in ("experience", "project") else section.title()
            header = f"{section.title()} - {label}:\n"
            units = [part for line in group for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if part]
            oversized = count(header + "\n".join(group)) > settings.chunk_max
            limit = settings.chunk_target if fallback or oversized else settings.chunk_max
            pieces, pending = [], []
            for unit in units:
                if count(header + unit) > settings.chunk_max:
                    raise ValueError("A single sentence exceeds CHUNK_MAX; shorten it or increase the configured limit")
                if pending and count(header + "\n".join(pending + [unit])) > limit:
                    pieces.append(pending)
                    overlap = []
                    for previous in reversed(pending):
                        if count("\n".join([previous] + overlap)) > int(limit * settings.chunk_overlap_pct):
                            break
                        overlap.insert(0, previous)
                    while overlap and count(header + "\n".join(overlap + [unit])) > settings.chunk_max:
                        overlap.pop(0)
                    pending = overlap
                pending.append(unit)
            if pending:
                pieces.append(pending)
            for piece in pieces:
                content = header + "\n".join(piece)
                if result and count(content) < 30 and result[-1]["section"] == section and count(result[-1]["content"] + "\n" + content) <= settings.chunk_max:
                    result[-1]["content"] += "\n" + content
                    result[-1]["token_count"] = count(result[-1]["content"])
                else:
                    result.append({"section": section, "label": label, "content": content, "token_count": count(content)})
    return result
