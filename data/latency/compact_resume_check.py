"""Paired evaluator timing on three saved resumes; does not update screening results."""
import importlib.util
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db import fetch
from app.screening import evaluator
from app.screening.anonymize import anonymize
from app.screening.jd_parser import jd_vectors, load_jd
from app.screening.ranking import score
from app.screening.retrieval import best_chunks
from app.screening.verifier import verify
from app.services import llm_service

folder = Path(__file__).parent
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler(folder / "compact_resume_check.log", encoding="utf-8"), logging.StreamHandler()])
spec = importlib.util.spec_from_file_location("evaluator_before_compact", folder / "evaluator_before_compact.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
jd_id = "d0ea9896-24cd-4688-a7e3-5c270aabe1ce"
jd = load_jd(jd_id)
vectors = jd_vectors(jd)
llm_service.get_llm()
assert llm_service.gpu_layers_offloaded > 0
saved = fetch("""SELECT r.*, s.rank FROM resumes r JOIN screening_results s ON r.id=s.resume_id
    WHERE s.jd_id=%s AND s.rank IN (1,13,25) ORDER BY s.rank""", (jd_id,))
assert len(saved) == 3
results = []
for index, resume in enumerate(saved):
    chunks = [dict(item, content=anonymize(item["content"], resume["clean_text"]))
        for item in best_chunks(str(resume["id"]), vectors)]
    row = {"resume_id": str(resume["id"]), "file_name": resume["file_name"]}
    # Alternate order to reduce the benefit of shared-prefix caching for either version.
    versions = [("before", reference), ("after", evaluator)] if index % 2 == 0 else [("after", evaluator), ("before", reference)]
    for label, module in versions:
        start = perf_counter()
        output = module.evaluate(jd, chunks, resume["skills"] or [], resume["years_exp"])
        elapsed = perf_counter() - start
        checked = verify(output, "\n\n".join(chunk["content"] for chunk in chunks), chunks)
        assert [skill.skill for skill in checked.skills] == list(dict.fromkeys(jd.must_have + jd.nice_to_have))
        assert all(0 <= skill.confidence <= 1 and 0 <= skill.level <= 4 for skill in checked.skills)
        row[label] = {"seconds": elapsed, "verified_score": score(checked), "red_flags": len(checked.red_flags)}
        print(json.dumps({"rank": resume["rank"], "version": label, "seconds": round(elapsed, 2)}), flush=True)
    results.append(row)
    (folder / "compact_resume_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
before = sum(row["before"]["seconds"] for row in results)
after = sum(row["after"]["seconds"] for row in results)
print(json.dumps({"resumes": len(results), "before_average_s": before / len(results),
    "after_average_s": after / len(results), "reduction_percent": (1 - after / before) * 100}), flush=True)
