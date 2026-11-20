import operator
from typing import Annotated, TypedDict

from app.models import GithubResult
from app.screening.github_agent import GithubEvidence


class ScreeningState(TypedDict, total=False):
    jd_id: str
    request: dict
    weights: dict
    jd_parsed: dict
    query_vectors: list[list[float]]
    shortlist: list[dict]
    candidate_results: Annotated[list[dict], operator.add]
    ranked: list[dict]
    report: dict
    email_selection: list[dict]
    status: str
    error: str | None


class EvaluationUpdate(TypedDict, total=False):
    chunks: list[dict]
    anonymized_chunks: list[dict]
    evaluation: dict
    verified: dict
    cache_hit: bool


class CandidateState(EvaluationUpdate, total=False):
    jd_id: str
    resume_id: str
    resume: dict
    jd_parsed: dict
    query_vectors: list[list[float]]
    stage1_decision: str
    error: str | None
    github_evidence: GithubEvidence | None
    github: GithubResult | None
    candidate_results: Annotated[list[dict], operator.add]
