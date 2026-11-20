from types import SimpleNamespace

import numpy as np

from app.models import ParsedJD
from app.screening import jd_parser


def test_long_jd_is_embedded_without_truncation(monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split(), decode=lambda tokens, **kwargs: " ".join(tokens))
    embedder = SimpleNamespace(tokenizer=tokenizer, max_seq_length=32)
    monkeypatch.setattr(jd_parser, "get_embedder", lambda: embedder)
    monkeypatch.setattr(jd_parser.settings, "use_query_prefix", False)
    words = [f"responsibility{i}" for i in range(100)]
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[" ".join(words)])
    parts = jd_parser.query_parts(jd)
    assert all(len(part.split()) <= 30 for part in parts)
    assert all(word in " ".join(parts) for word in words)


def test_degree_and_alternative_language_requirements_are_not_independent_skills():
    jd = ParsedJD(title="New Grad Engineer", must_have=["Recent college graduates with a degree starting in 2026", "Proficiency in at least one of Java, Python, C++", "Collaboration"],
        nice_to_have=["Experience with VisaNet"], responsibilities=["Maintain VisaNet"])
    result = jd_parser.normalize_requirements(jd, "Degree required. At least one of Java, Python, C++. You might maintain VisaNet.")
    assert result.must_have == ["Any of: Proficiency in at least one of Java, Python, C++", "Collaboration"]
    assert result.nice_to_have == []
    assert "Recent college graduates with a degree starting in 2026" in result.responsibilities
    assert jd.nice_to_have == ["Experience with VisaNet"]
    preferred = jd_parser.normalize_requirements(jd, "VisaNet experience is preferred")
    assert preferred.nice_to_have == ["Experience with VisaNet"]


def test_jd_parser_returns_the_same_vectors_it_uses_for_storage(monkeypatch):
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    embedded = []
    stored = []
    monkeypatch.setattr(jd_parser, "generate_json", lambda *args: jd)
    monkeypatch.setattr(jd_parser, "jd_vectors", lambda parsed: embedded.append(parsed) or vectors)
    monkeypatch.setattr(jd_parser, "execute", lambda sql, params: stored.append(params))
    monkeypatch.setattr(jd_parser, "get_jd", lambda *args: None)
    monkeypatch.setattr(jd_parser, "put_jd", lambda *args: None)
    key, parsed, returned = jd_parser.parse_jd("Python engineer required", "test")
    assert key == "test" and parsed == jd and returned is vectors
    assert len(embedded) == 1
    np.testing.assert_allclose(stored[0][-2], [2 ** -0.5, 2 ** -0.5])
    np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])
