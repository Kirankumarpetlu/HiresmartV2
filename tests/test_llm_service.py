import sys
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from app.services import llm_service
from app.models import EvaluatorOutput


class Answer(BaseModel):
    answer: str


def test_validation_retries_and_reuses_model(monkeypatch):
    calls = []

    class FakeModel:
        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"wrong": 1}' if len(calls) == 1 else '{"answer": "ready"}'}, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}

    fake = FakeModel()
    monkeypatch.setattr(llm_service, "_model", fake)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(LogitsProcessorList=list))
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    result = llm_service.generate_json([{"role": "user", "content": "ready"}], Answer, 40)
    assert result.answer == "ready"
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0
    assert calls[0]["response_format"]["schema"] == Answer.model_json_schema()
    assert llm_service.get_llm() is fake


def test_concurrent_callers_never_overlap_generation(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()

    class FakeModel:
        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return {"choices": [{"message": {"content": '{"answer": "ready"}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm_service, "_model", FakeModel())
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(LogitsProcessorList=list))
    with ThreadPoolExecutor(max_workers=2) as callers:
        tasks = [callers.submit(llm_service.generate_json, [{"role": "user", "content": "ready"}], Answer, 40) for _ in range(2)]
        assert all(task.result().answer == "ready" for task in tasks)
    assert peak == 1


def test_native_sampling_callback_stops_at_deadline(monkeypatch):
    class FakeModel:
        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            monkeypatch.setattr(llm_service.time, "monotonic", lambda: 1000)
            return kwargs["logits_processor"][0]([], [])

    monkeypatch.setattr(llm_service, "_model", FakeModel())
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(LogitsProcessorList=list))
    monkeypatch.setattr(llm_service.time, "monotonic", lambda: 0)
    with pytest.raises(TimeoutError):
        llm_service._generate([{"role": "user", "content": "ready"}], Answer, 40)


def test_percentage_confidence_is_rejected_and_retry_gets_private_field_feedback(monkeypatch):
    attempts = []

    class FakeModel:
        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            attempts.append(kwargs["messages"])
            payload = {"skills": [{"skill": "Python", "is_must_have": True, "level": 2,
                "evidence": ["PRIVATE_EVIDENCE"], "reasoning": "Test", "confidence": 80 if len(attempts) == 1 else 0.8}],
                "fit": {"experience": 100, "domain": 50, "education": 100, "notes": "Test"}, "red_flags": []}
            return {"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm_service, "_model", FakeModel())
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(LogitsProcessorList=list))
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    result = llm_service.generate_json([{"role": "user", "content": "Evaluate"}], EvaluatorOutput, 600)
    assert result.skills[0].confidence == 0.8
    feedback = attempts[1][-1]["content"]
    assert "skills.0.confidence" in feedback and "0-1" in feedback
    assert "PRIVATE_EVIDENCE" not in feedback


def test_sliding_window_drops_oldest_whole_turns_and_keeps_instructions(monkeypatch):
    monkeypatch.setattr(llm_service.settings, "llm_n_ctx", 512)
    model = SimpleNamespace(tokenize=lambda text, **kwargs: text.split())
    messages = [{"role": role, "content": f"message{index} " * size}
        for index, (role, size) in enumerate([
            ("system", 10), ("user", 20), ("assistant", 60), ("user", 40), ("assistant", 60),
            ("developer", 5), ("user", 10), ("assistant", 10), ("user", 10), ("assistant", 5)])]
    original = [dict(message) for message in messages]
    window, tokens = llm_service._window_messages(messages, model, 40)
    assert window == [messages[index] for index in [0, 1, 5, 6, 7, 8, 9]]
    assert tokens + 40 + 256 <= 512
    assert messages == original


@pytest.mark.parametrize("task_tokens", [190, 191])
def test_sliding_window_budget_boundary_preserves_original_task(monkeypatch, task_tokens):
    monkeypatch.setattr(llm_service.settings, "llm_n_ctx", 512)
    model = SimpleNamespace(tokenize=lambda text, **kwargs: text.split())
    messages = [{"role": "system", "content": "instructions " * 10},
        {"role": "user", "content": "evidence " * task_tokens}]
    if task_tokens == 190:
        window, tokens = llm_service._window_messages(messages, model, 40)
        assert window == messages
        assert tokens + 40 + 256 == 512
    else:
        with pytest.raises(ValueError, match="original task"):
            llm_service._window_messages(messages, model, 40)


def test_sliding_window_applies_to_validation_retry(monkeypatch):
    calls = []

    class FakeModel:
        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            calls.append(kwargs["messages"])
            payload = '{"wrong": 1}' if len(calls) == 1 else '{"answer": "ready"}'
            return {"choices": [{"message": {"content": payload}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm_service, "_model", FakeModel())
    monkeypatch.setattr(llm_service.settings, "llm_n_ctx", 512)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(LogitsProcessorList=list))
    monkeypatch.setattr(llm_service.time, "sleep", lambda seconds: None)
    messages = [{"role": "user", "content": "original_evidence " * 10},
        {"role": "user", "content": "old_continuation " * 140},
        {"role": "user", "content": "latest_request " * 10}]
    assert llm_service.generate_json(messages, Answer, 40).answer == "ready"
    assert calls[0] == messages
    assert calls[1][:2] == [messages[0], messages[2]]
    assert "schema validation" in calls[1][-1]["content"]
    assert len(messages) == 3
