"""Clear all local derived-data caches without deleting resumes or screening results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import clear

if __name__ == "__main__":
    clear()
    print("Derived-data caches cleared.")
