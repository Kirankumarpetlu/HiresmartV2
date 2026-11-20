import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from app.services import llm_service


class Answer(BaseModel):
    answer: str


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = llm_service.get_llm()
    for _ in range(2):
        result = llm_service.generate_json([{"role": "user", "content": "Return JSON with answer set to ready."}], Answer, 40)
        assert result.answer == "ready"
        assert llm_service.get_llm() is model
    assert llm_service.gpu_layers_offloaded > 0, "CUDA offload acceptance failed"
    print("LLM acceptance passed: schema validation, load-once, two calls, CUDA offload.")
