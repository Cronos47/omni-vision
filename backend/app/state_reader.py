from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import STATE_PATH


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": {"message": "no_state"}}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = state.get("meta", {})
    history = state.get("history", {})
    metrics = history.get("metrics", {})
    gradients = state.get("gradients", {})
    diffusion = state.get("diffusion", {})

    latest_epoch = str(meta.get("current_epoch") or 0)
    loss_series = metrics.get("loss", []) or []
    acc_series = metrics.get("accuracy", []) or metrics.get("acc", []) or []

    latest_loss = _safe_float(loss_series[-1]) if loss_series else None
    latest_acc = _safe_float(acc_series[-1]) if acc_series else None

    diffusion_epochs = sorted(diffusion.keys(), key=lambda x: int(x)) if diffusion else []
    latest_diff = diffusion.get(diffusion_epochs[-1], {}) if diffusion_epochs else {}
    energies = latest_diff.get("energies", []) or []

    gradient_layers = 0
    if latest_epoch in gradients and isinstance(gradients[latest_epoch], dict):
        gradient_layers = len(gradients[latest_epoch])

    return {
        "meta": meta,
        "latest": {
            "loss": latest_loss,
            "accuracy": latest_acc,
            "gradient_layers": gradient_layers,
            "diffusion_energy_start": _safe_float(energies[0]) if energies else None,
            "diffusion_energy_end": _safe_float(energies[-1]) if energies else None,
        },
        "available": {
            "metrics": list(metrics.keys()),
            "gradient_epochs": list(gradients.keys()),
            "diffusion_epochs": diffusion_epochs,
        },
    }
