"""Check the public GitHub API and prove a repeated lookup uses the database cache."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import init_database
from app.integrations.github_client import GithubClient
from app.models import ParsedJD
from app.screening.github_agent import GithubEvidence, Project, match_projects, score_profile


def main() -> None:
    assert settings.cache_enabled, "Enable CACHE_ENABLED to run the warm-cache check"
    logging.basicConfig(level=logging.INFO)
    init_database()
    demo = "https://github.com/octocat/spoon-knife"
    for attempt in range(2):
        with GithubClient("octocat") as client:
            profile, catalog = client.profile()
            assert profile is not None, "Public demo profile lookup failed"
            repo = client.repository("octocat", "spoon-knife")
            assert repo is not None, "Public demo repository lookup failed"
            details = client.details("octocat", "spoon-knife")
        evidence = GithubEvidence("octocat", profile, catalog, {demo: repo | details},
                                  [Project("Spoon-Knife", "", [], [demo])])
        jd = ParsedJD(title="Demo", must_have=["HTML"], nice_to_have=[], responsibilities=[])
        result = score_profile(evidence, match_projects(evidence, {}), jd)
        assert result.project_matches[0].verified and result.github_score is not None
        print(f"Pass {attempt + 1}: GitHub score {result.github_score}, API requests {client.network_calls}")
        if attempt:
            assert client.network_calls == 0, "Expected no GitHub API requests on cache hit"


if __name__ == "__main__":
    main()
