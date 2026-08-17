from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import API_PREFIX, FRONTEND_DIST, DEMO_PATH, STATE_PATH
from .state_reader import summarize_state
from state import load_state
from .train import start_training  
from fastapi import Body


app = FastAPI(title="OmniVision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"status": "OmniVision Backend Running"}


@app.get("/api/state")
def get_state():
    return load_state(str(STATE_PATH))

@app.get(f"{API_PREFIX}/summary")
def get_summary() -> dict:
    return summarize_state(load_state())

@app.get("/api/logs")
def get_logs():
    state = load_state(str(STATE_PATH))
    return state.get("logs", [])

@app.post("/train")
def train(payload: dict = Body(default={})):
    epochs = int(payload.get("epochs", 5))
    ok, status = start_training(epochs=epochs)
    return {"status": status, "epochs": epochs}

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/app")
    def serve_frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")