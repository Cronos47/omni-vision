# OmniVision UI Refactor

## What changed
- Your callback, state persistence, and training workflow remain intact.
- FastAPI now exposes the live training state at `/api/state` and a lighter `/api/summary`.
- A modern React + Tailwind + Framer Motion frontend replaces Streamlit as the primary UI.
- Streamlit remains in the project as a fallback while you transition.

## Run the new stack

### Python dependencies
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Frontend dependencies
```bash
cd frontend
npm install
```

### Start UI + API
```bash
python run_ui.py
```

Then open:
- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`

## Training workflow
Run your trainer exactly as before. The callback still writes `omnivision_state.json`.
The backend reads that file and the frontend polls the API every 2 seconds.

## Fallback
You can still use the existing `dashboard.py` if needed during migration.
