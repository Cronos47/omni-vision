from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = Path(os.getenv("OMNIVISION_STATE_PATH", PROJECT_ROOT / "states/omnivision_state.json")).resolve()
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
API_PREFIX = "/api"
DEMO_PATH = PROJECT_ROOT / "demo.py"

