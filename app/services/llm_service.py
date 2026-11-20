import json
import logging
import random
import re
import time
import sys
import ctypes
import importlib.util
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import Lock, Semaphore

from pydantic import BaseModel, ValidationError

from app.config import settings

log = logging.getLogger(__name__)
_model = None
_load_lock = Lock()
_generation_lock = Lock()
_semaphore = Semaphore(settings.llm_concurrency)
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-llm")
_callback = None
_native_library = None
gpu_layers_offloaded = 0
prompt_cache_hits = 0
generation_calls = 0


def get_llm():
    global _model, _callback, _native_library
    if _model is None:
        with _load_lock:
            if _model is None:
                if sys.platform == "win32":
                    spec = importlib.util.find_spec("llama_cpp")
                    if spec is None:
                        raise RuntimeError("Install the CUDA llama-cpp-python wheel before starting the backend")
                    # Preload with Windows' safe DLL search flags; the binding uses legacy flags.
                    _native_library = ctypes.CDLL(str(Path(spec.origin).parent / "lib" / "llama.dll"))
                import llama_cpp

                class LoggedRAMCache(llama_cpp.LlamaRAMCache):
                    def __getitem__(self, key):
                        global prompt_cache_hits
                        try:
                            state = super().__getitem__(key)
                        except KeyError:
                            log.info("LLM prompt_cache miss")
                            raise
                        prompt_cache_hits += 1
                        prefix = llama_cpp.Llama.longest_token_prefix(state.input_ids.tolist(), key)
                        log.info("LLM prompt_cache hit prefix_tokens=%d", prefix)
                        return state

                @llama_cpp.llama_log_callback
                def capture(level, message, user_data):
                    global gpu_layers_offloaded
                    text = message.decode("utf-8", errors="replace")
                    match = re.search(r"offloaded (\d+)/(\d+) layers", text)
                    if match:
                        gpu_layers_offloaded = int(match[1])

                _callback = capture
                llama_cpp.llama_log_set(capture, None)
                start = time.monotonic()
                model = llama_cpp.Llama.from_pretrained(
                    repo_id=settings.llm_repo, filename=settings.llm_file,
                    n_ctx=settings.llm_n_ctx, n_gpu_layers=settings.llm_n_gpu_layers,
                    n_batch=512, n_ubatch=128, n_threads=max(1, (os.cpu_count() or 4) // 2),
                    n_threads_batch=max(1, os.cpu_count() or 4), flash_attn=True,
                    offload_kqv=True, verbose=False,
                )
                model.set_cache(LoggedRAMCache(capacity_bytes=512 * 1024 * 1024))
                _model = model
                log.info("LLM loaded in %.2fs; GPU layers offloaded=%d", time.monotonic() - start, gpu_layers_offloaded)
                if not gpu_layers_offloaded:
                    log.warning("No GPU offload detected: install the CUDA build and check CUDA runtime libraries")
    return _model


def _window_messages(messages: list[dict], model, max_tokens: int) -> tuple[list[dict], int]:
    # shortcut: formatting uses a 256-token reserve plus 8 per message; use exact template counts for other chat formats.
    costs = [len(model.tokenize(message["content"].encode(), add_bos=False)) + 8 for message in messages]
    budget = settings.llm_n_ctx - max_tokens - 256
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    first_user = users[0] if users else -1
    last_user = users[-1] if users else len(messages) - 1
    turns = []
    for index, message in enumerate(messages):
        if message["role"] in ("system", "developer") or index == first_user or index >= last_user:
            continue
        if message["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(index)
    total = sum(costs)
    removed = set()
    for turn in turns:
        if total <= budget:
            break
        removed.update(turn)
        total -= sum(costs[index] for index in turn)
    if total > budget:
        raise ValueError("Required instructions, original task, and latest context exceed the model context budget; reduce chunks or split the evaluation")
    if removed:
        log.info("LLM sliding window removed_messages=%d", len(removed))
    return [message for index, message in enumerate(messages) if index not in removed], total


def _generate(messages: list[dict], schema: type[BaseModel], max_tokens: int):
    global generation_calls
    with _semaphore, _generation_lock:
        generation_calls += 1
        model = get_llm()
        import llama_cpp
        messages, prompt_tokens = _window_messages(messages, model, max_tokens)
        context = getattr(getattr(model, "_ctx", None), "ctx", None)
        if context is not None:
            llama_cpp.llama_perf_context_reset(context)
        start = time.monotonic()
        deadline = start + settings.llm_timeout_seconds
        def check_deadline(tokens, logits):
            if time.monotonic() > deadline:
                raise TimeoutError("LLM generation deadline exceeded")
            return logits
        output = model.create_chat_completion(
            messages=messages, temperature=0, max_tokens=max_tokens,
            response_format={"type": "json_object", "schema": schema.model_json_schema()},
            logits_processor=llama_cpp.LogitsProcessorList([check_deadline]),
        )
        elapsed = time.monotonic() - start
        usage = output.get("usage", {})
        log.info("LLM call prompt_tokens=%d latency=%.2fs tokens_per_second=%.2f", prompt_tokens, elapsed, usage.get("completion_tokens", 0) / max(elapsed, 0.001))
        if context is not None:
            performance = llama_cpp.llama_perf_context(context)
            log.info("LLM prompt_eval schema=%s tokens=%d milliseconds=%.2f",
                schema.__name__, performance.n_p_eval, performance.t_p_eval_ms)
        if time.monotonic() > deadline:
            raise TimeoutError("LLM generation deadline exceeded")
        if output["choices"][0].get("finish_reason") == "length":
            raise ValueError("LLM output was truncated; enable EVALUATOR_SPLIT_CALLS or increase LLM_MAX_TOKENS")
        return schema.model_validate(json.loads(output["choices"][0]["message"]["content"]))


def generate_json(messages: list[dict], schema: type[BaseModel], max_tokens: int | None = None):
    for attempt in range(settings.llm_max_retries):
        future = _worker.submit(_generate, messages, schema, max_tokens or settings.llm_max_tokens)
        try:
            return future.result(timeout=settings.llm_timeout_seconds)
        except FutureTimeout:
            future.cancel()
            log.warning("LLM call timed out attempt=%d", attempt + 1)
            error = TimeoutError("LLM timed out; an active native call retains its lock until it stops")
        except Exception as exc:
            # Log types only: validation exception text may include private resume content.
            log.warning("LLM call/validation failed attempt=%d type=%s", attempt + 1, type(exc).__name__)
            if isinstance(exc, ValidationError):
                failures = [(".".join(map(str, error["loc"])), error["type"]) for error in exc.errors(include_input=False, include_url=False)]
                log.warning("LLM validation fields=%s", failures)
                messages = messages + [{"role": "user", "content": f"The output failed schema validation at {failures}. Return the complete corrected JSON. Keep skill entries in the requested order and follow the supplied schema. Confidence is 0-1, proficiency level 0-4, and fit values 0-100."}]
            error = exc
        if attempt + 1 < settings.llm_max_retries:
            time.sleep(2 ** attempt + random.random() * 0.25)
    raise error
