from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ParsedJD(BaseModel):
    title: str
    must_have: list[str]
    nice_to_have: list[str]
    min_years: float | None = Field(default=None, ge=0)
    responsibilities: list[str]


class SkillProficiency(BaseModel):
    skill: str
    is_must_have: bool
    level: int = Field(ge=0, le=4)
    years_used: float | None = Field(default=None, ge=0)
    last_used_year: int | None = None
    evidence: list[str] = Field(description="Exact quotes from the provided resume text")
    reasoning: str = Field(description="1-2 sentences")
    confidence: float = Field(ge=0, le=1)


class FitScores(BaseModel):
    experience: float = Field(ge=0, le=100)
    domain: float = Field(ge=0, le=100)
    education: float = Field(ge=0, le=100)
    notes: str


class EvaluatorOutput(BaseModel):
    skills: list[SkillProficiency]
    fit: FitScores
    red_flags: list[str]


class Weights(BaseModel):
    proficiency: float = Field(default=0.55, ge=0, le=1)
    experience: float = Field(default=0.20, ge=0, le=1)
    domain: float = Field(default=0.15, ge=0, le=1)
    education: float = Field(default=0.10, ge=0, le=1)
    github: float = Field(default=0.10, ge=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self):
        core_total = sum(self.model_dump(exclude={"github"}).values())
        base_weights = abs(core_total - 1) <= 1e-6
        five_weights = "github" in self.model_fields_set and abs(core_total + self.github - 1) <= 1e-6
        if core_total <= 0 or not (base_weights or five_weights):
            raise ValueError("The four base weights, or all five weights, must sum to 1")
        return self


GithubStatus = Literal["ok", "partial", "stale", "pending", "not_queued", "rate_limited", "blocked_config", "forbidden", "not_found",
                       "not_provided", "failed", "error"]


class GithubProjectMatch(BaseModel):
    resume_project: str
    repo: str | None = None
    match: Literal["verified", "partial", "unverified", "not_found"]
    reason: str
    evidence_quote: str | None = None
    verified: bool = False
    authored: bool | None = None


class GithubSubscores(BaseModel):
    project_match: float = Field(ge=0, le=100)
    activity: float = Field(ge=0, le=100)
    code_quality: float = Field(ge=0, le=100)
    jd_relevance: float = Field(ge=0, le=100)


class GithubSignals(BaseModel):
    account_age_days: int | None
    public_repos: int
    original_repos: int
    followers: int
    languages: dict[str, int]
    sampled_repos: int
    authored_repos: int
    recent_commits: int
    active_days: int
    days_since_push: int | None


class GithubResult(BaseModel):
    status: GithubStatus
    github_score: int | None = Field(default=None, ge=0, le=100)
    subscores: GithubSubscores | None = None
    signals: GithubSignals | None = None
    project_matches: list[GithubProjectMatch] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    summary: str = ""
    evidence_coverage: Literal["full", "partial", "none"] = "none"
    stale: bool = False
    repos_inspected: int = Field(default=0, ge=0)
    repos_total: int = Field(default=0, ge=0)
    retry_after: datetime | None = None
    error_category: str | None = None
    network_calls: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    secondary_limits: int = Field(default=0, ge=0)
    primary_limit_hits: int = Field(default=0, ge=0)
    score_weights: tuple[float, float, float, float] | None = None
    score_evidence: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def score_explanation(self) -> str:
        if self.github_score is None:
            return f"No GitHub score. {self.summary or self.status.replace('_', ' ').capitalize() + '.'}"
        lines = [f"GitHub score {self.github_score}/100.", self.summary]
        if self.stale and self.status != "stale":
            lines.append("Some evidence came from cache beyond its normal freshness window.")
        if self.subscores is None:
            return "\n".join(filter(None, lines))
        labels = ("Project match", "Activity", "Repository quality", "Job relevance")
        if self.score_weights is None:
            lines.append("This result retains the subscores. Weights are available for results saved with a score breakdown.")
        for index, (key, value) in enumerate(self.subscores.model_dump().items()):
            part = f"{labels[index]} {value:g}/100"
            if self.score_weights is not None:
                weight = self.score_weights[index]
                points = round(value * weight, 4)
                part += f" at {100 * weight:g}% contributes {points:g} points"
            evidence = self.score_evidence.get(key)
            if not evidence and key == "activity" and self.signals:
                signals = self.signals
                evidence = (f"{signals.recent_commits} attributed commits over {signals.active_days} active days "
                            "in the past year.")
                if signals.days_since_push is not None:
                    evidence += f" Latest catalog push was {signals.days_since_push} days before scoring."
            if not evidence:
                evidence = {
                    "project_match": "Matches link resume project claims to sampled public repositories.",
                    "code_quality": "The aggregate checks cover README, tests, CI, license, description, size, attributed commits, and Dockerfile.",
                    "jd_relevance": "The stored result contains the aggregate match to required job skills.",
                }.get(key, "")
            lines.append(f"{part}. {evidence}".rstrip())
        return "\n".join(filter(None, lines))


class ScreeningResult(BaseModel):
    id: UUID
    jd_id: UUID
    resume_id: UUID
    file_name: str
    status: Literal["pending", "done", "error", "filtered_out"]
    error_message: str | None
    similarity: float | None
    years_exp: float | None
    final_score: float | None
    rank: int | None
    skill_report: list[SkillProficiency] | None
    fit_scores: FitScores | None
    red_flags: list[str]
    explanation: str | None
    recommendation: str | None
    created_at: datetime | None
    strengths: list[dict]
    gaps: list[str]
    evidence: list[dict]
    summary: str
    github_json: GithubResult | None = None
    github_score: int | None = None
    github_status: GithubStatus | None = None
    github_retryable: bool = False


class Summary(BaseModel):
    summary: str = Field(max_length=1000)


class ScreeningRequest(BaseModel):
    jd_text: str = Field(min_length=10, max_length=30000)
    shortlist_size: int | None = Field(default=None, ge=1, le=100)
    resume_ids: list[UUID] | None = Field(default=None, min_length=1)
    weights: Weights = Field(default_factory=Weights)
    wait_for_github: bool = False
    send_notifications: bool = True


class GithubExtendRequest(BaseModel):
    count: int = Field(default=100, ge=1, le=1000)


class GithubRetryRequest(BaseModel):
    resume_ids: list[UUID] = Field(min_length=1, max_length=1000)


class RunStatus(BaseModel):
    total: int = 0
    done: int = 0
    errors: int = 0
    filtered_out: int = 0
    state: Literal["queued", "running", "evaluating", "notifying", "awaiting_github", "awaiting_approval", "done", "error", "interrupted"] = "queued"
    error_message: str | None = None


class ResumeContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str


class NotificationCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_ids: list[UUID] = Field(min_length=1, max_length=100)
    email_type: Literal["selected", "rejected"] = "selected"


class NotificationApproval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_ids: list[UUID] | None = Field(default=None, max_length=100)
    rejected_resume_ids: list[UUID] = Field(default_factory=list, max_length=100)
    all_selected: bool = False
    override: bool = False

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.all_selected == (self.resume_ids is not None):
            raise ValueError("Provide either resume_ids or all_selected")
        return self
