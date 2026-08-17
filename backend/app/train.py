from pathlib import Path
from fastapi import Body
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "demo.py"
TRAINING_PROCESS = None


def start_training(epochs: int = 3):
    global TRAINING_PROCESS

    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        return False, "training_already_running"

    TRAINING_PROCESS = subprocess.Popen(
        [sys.executable, str(DEMO_PATH), "--epochs", str(epochs), "--ui", "none"],
        cwd=str(PROJECT_ROOT),
    )
    return True, "training_started"
