"""Real-model acceptance check; downloads once into the Hugging Face cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.services.embed_service import encode, get_embedder

if __name__ == "__main__":
    first = get_embedder()
    vectors = encode(["Python engineer with SQL experience"] * 32)
    assert vectors.shape == (32, 768)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert get_embedder() is first
    encode(["Second call"])
    assert get_embedder() is first
    print("Embedding acceptance passed: (32, 768), normalized, same model reused.")
