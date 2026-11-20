"""Apply the idempotent schema without dropping existing data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_database
from app.screening.graph import setup_checkpointer

if __name__ == "__main__":
    init_database()
    setup_checkpointer()
    print("Schema applied; vector dimensions verified.")
