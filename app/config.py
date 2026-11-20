import logging
import os
import sys
import sysconfig
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False, hide_input_in_errors=True)

    hf_token: SecretStr = SecretStr("")
    postgres_user: str = "resume_user"
    postgres_password: SecretStr = SecretStr("change_me")
    postgres_db: str = "resume_db"
    postgres_port: int = 5432
    database_url: SecretStr = SecretStr("postgresql://resume_user:change_me@localhost:5432/resume_db")
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = Field(default=768, gt=0)
    embed_device: Literal["cpu"] = "cpu"
    llm_repo: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    llm_file: str = "*q4_k_m.gguf"
    llm_n_ctx: int = Field(default=4096, ge=512)
    llm_n_gpu_layers: int = -1
    llm_max_tokens: int = Field(default=600, gt=0)
    shortlist_size: int = Field(default=10, ge=1, le=100)
    llm_concurrency: int = Field(default=1, ge=1, le=1)
    llm_timeout_seconds: float = Field(default=120, gt=0)
    llm_max_retries: int = Field(default=3, ge=1)
    data_dir: Path = Path("data")
    chunk_target: int = Field(default=256, ge=30)
    chunk_max: int = Field(default=400, ge=30)
    chunk_overlap_pct: float = Field(default=0.125, ge=0.10, le=0.15)
    use_query_prefix: bool = False
    agg_mode: Literal["best", "top3_mean"] = "top3_mean"
    evaluator_split_calls: bool = False
    cache_enabled: bool = True
    cache_ttl_days: int = Field(default=7, ge=1)
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    interview_threshold: float = 75
    maybe_threshold: float = 55
    api_url: str = "http://127.0.0.1:8000"
    max_upload_bytes: int = 20 * 1024 * 1024
    retrieval_chunk_limit: int = 1000
    github_enabled: bool = True
    github_token: SecretStr = SecretStr("")
    github_api_base: str = "https://api.github.com"
    github_cache_ttl_hours: float = Field(default=24, gt=0)
    github_max_repos: int = Field(default=8, ge=1, le=100)
    github_catalog_max_pages: int = Field(default=3, ge=1, le=10)
    github_commits_per_page: int = Field(default=10, ge=1, le=100)
    github_reserve_remaining: int = Field(default=200, ge=0)
    github_max_concurrency: int = Field(default=4, ge=1)
    github_shortlist_size: int = Field(default=100, ge=1)
    github_stable_cache_ttl_hours: float = Field(default=120, gt=0)
    github_partial_weight_factor: float = Field(default=0.5, ge=0, le=1)
    github_timeout_seconds: float = Field(default=10, gt=0)
    github_score_weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    github_min_score: int = Field(default=40, ge=0, le=100)
    email_max_selected: int | None = Field(default=None, ge=1)
    send_rejection_emails: bool = False
    email_auto_send: bool = False
    email_dry_run: bool = True
    company_name: str = "Our hiring team"
    recruiter_name: str = "Recruiting Team"
    recruiter_email: str = ""
    next_step_text: str = "Please reply with your availability."
    smtp_host: str = "127.0.0.1"
    smtp_port: int = Field(default=1025, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_use_tls: bool = False
    smtp_starttls: bool = False
    email_from: str = "recruiting@example.org"
    email_rate_per_minute: int = Field(default=20, ge=1)
    email_max_retries: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.smtp_use_tls and self.smtp_starttls:
            raise ValueError("Choose SMTP_USE_TLS or SMTP_STARTTLS")
        if self.smtp_username and not (self.smtp_use_tls or self.smtp_starttls):
            raise ValueError("SMTP authentication requires TLS")
        if self.chunk_target > self.chunk_max:
            raise ValueError("CHUNK_TARGET must not exceed CHUNK_MAX")
        if self.github_api_base.rstrip("/") != "https://api.github.com":
            raise ValueError("GITHUB_API_BASE must use the public GitHub API")
        if not all(0 <= weight <= 1 for weight in self.github_score_weights) or abs(sum(self.github_score_weights) - 1) > 1e-6:
            raise ValueError("GITHUB_SCORE_WEIGHTS must contain four weights summing to 1")
        return self


settings = Settings()
# Windows CUDA wheels need NVIDIA's pip-installed DLL directories on the loader path.
_dll_directories = []
if sys.platform == "win32":
    _site_packages = Path(sysconfig.get_paths()["purelib"])
    for directory in [*(_site_packages / "nvidia").glob("*/bin"), _site_packages / "llama_cpp" / "lib"]:
        if not directory.is_dir():
            continue
        _dll_directories.append(os.add_dll_directory(str(directory)))
if settings.hf_token.get_secret_value():
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token.get_secret_value()
    os.environ["HF_TOKEN"] = settings.hf_token.get_secret_value()
else:
    logging.getLogger(__name__).warning("HF_TOKEN is missing; public model downloads will continue anonymously")
