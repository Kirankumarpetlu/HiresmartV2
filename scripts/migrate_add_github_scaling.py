"""Apply the idempotent GitHub job, quota, configuration-block, and metrics schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_database

if __name__ == "__main__":
    init_database()
    print("GitHub scaling schema and quota metrics applied.")
