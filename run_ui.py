from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_backend() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT),
        env=env,
    )


def run_frontend() -> subprocess.Popen | None:
    frontend_dir = ROOT / "frontend"
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found. Backend started on http://127.0.0.1:8000. Install Node.js to run the React UI.")
        return None
    return subprocess.Popen([npm, "run", "dev"], cwd=str(frontend_dir))


if __name__ == "__main__":
    backend = run_backend()
    frontend = run_frontend()
    print("Backend: http://127.0.0.1:8000")
    if frontend is not None:
        print("Frontend: http://127.0.0.1:3000")
    try:
        backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        if frontend is not None:
            frontend.terminate()
