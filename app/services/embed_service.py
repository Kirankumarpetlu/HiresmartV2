import logging
import time
from collections import OrderedDict
from threading import Lock

import numpy as np

from app.config import settings

_model = None
_lock = Lock()
_encode_lock = Lock()
_query_lock = Lock()
_query_cache = OrderedDict()
log = logging.getLogger(__name__)


def get_embedder():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                start = time.monotonic()
                model = SentenceTransformer(settings.embed_model, device=settings.embed_device)
                if model.get_sentence_embedding_dimension() != settings.embed_dim:
                    raise RuntimeError("Embedding model dimensions do not match EMBED_DIM")
                _model = model
                log.info("CPU embedder loaded in %.2fs", time.monotonic() - start)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, settings.embed_dim), dtype=np.float32)
    with _encode_lock:
        return get_embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def encode_queries(texts: list[str]) -> np.ndarray:
    keys = [(settings.embed_model, text) for text in texts]
    with _query_lock:
        missing = list(dict.fromkeys(text for key, text in zip(keys, texts) if key not in _query_cache))
        if missing:
            for text, vector in zip(missing, encode(missing)):
                _query_cache[(settings.embed_model, text)] = tuple(float(value) for value in vector)
            while len(_query_cache) > 256:
                _query_cache.popitem(last=False)
        for key in keys:
            _query_cache.move_to_end(key)
        return np.asarray([_query_cache[key] for key in keys], dtype=np.float32)


def token_count(text: str) -> int:
    return len(get_embedder().tokenizer.encode(text, add_special_tokens=False))
