import pytest
from pydantic import ValidationError

from app.config import Settings


def test_hardware_constraints_and_chunk_validation():
    config = Settings(_env_file=None)
    assert config.embed_device == "cpu"
    assert config.llm_concurrency == 1
    assert config.embed_dim == 768
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embed_device="cuda")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_target=500, chunk_max=400)
