"""Verify the Streamlit UI renders and can re-rank a completed real screening."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from app.config import settings

if __name__ == "__main__":
    result = json.loads((settings.data_dir / "flow_acceptance.json").read_text())
    test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py")).run(timeout=30)
    assert not test.exception
    assert any("A human makes the final decision" in info.value for info in test.info)
    test.text_input[0].set_value(result["jd_id"]).run(timeout=30)
    assert not test.exception and not test.error
    assert len(test.dataframe) >= 2
    rerank = next(button for button in test.button if button.label == "Re-rank stored evaluations")
    rerank.click().run(timeout=30)
    assert not test.exception and not test.error
    assert any("Ranking updated" in success.value for success in test.success)
    print("Streamlit acceptance passed: rendering, live results, details, and re-weight action.")
