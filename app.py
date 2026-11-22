import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    frontend = Path(__file__).resolve().parent / "app" / "ui.py"
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(frontend),
        "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true",
    ]))
