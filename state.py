import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict

import numpy as np
from filelock import FileLock


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def init_state(
    run_name: str = "omnivision_run",
    total_epochs: int = 0,
    arch_type: str = "generic",
) -> Dict[str, Any]:
    return {
        "meta": {
            "run_name": run_name,
            "arch_type": arch_type,
            "total_epochs": total_epochs,
            "current_epoch": 0,
            "last_updated": None,
        },
        "history": {
            "metrics": {},
            "loss_delta": [],
        },
        "weights": {},
        "gradients": {},
        "activations": {},
        "labels": {},
        "layer_blocks": {},
        "attention": {},
        "routing": {},
        "diffusion": {},
        "sequence": {},
        "vision": {},
        "custom": {},
        "logs": [],
        "status": {
            "training": False,
            "done": False,
            "error": None,
        },
    }


def touch_state(state: Dict[str, Any]) -> None:
    state["meta"]["last_updated"] = datetime.utcnow().isoformat()


def _lock_path(path: str) -> str:
    return f"{path}.lock"


def save_state(state: Dict[str, Any], path: str, retries: int = 20, delay: float = 0.1) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    touch_state(state)
    payload = _to_serializable(state)
    lock = FileLock(_lock_path(path), timeout=10)

    last_error = None

    for _ in range(retries):
        tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            with lock:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp_path, path)
            return

        except PermissionError as e:
            last_error = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            time.sleep(delay)

        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    raise last_error if last_error else PermissionError(f"Could not write state file: {path}")


def load_state(path: str, retries: int = 12, delay: float = 0.05) -> Dict[str, Any]:
    if not os.path.exists(path):
        return init_state()

    lock = FileLock(_lock_path(path), timeout=10)
    last_error = None

    for _ in range(retries):
        try:
            with lock:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, PermissionError) as e:
            last_error = e
            time.sleep(delay)

    if last_error:
        raise last_error

    return init_state()


def trim_epoch_history(state: Dict[str, Any], max_epochs: int = 20) -> None:
    for key in [
        "weights",
        "gradients",
        "activations",
        "labels",
        "attention",
        "routing",
        "diffusion",
        "sequence",
        "vision",
        "custom",
    ]:
        epoch_map = state.get(key, {})
        if not isinstance(epoch_map, dict):
            continue

        epoch_keys = sorted(
            epoch_map.keys(),
            key=lambda x: int(x) if str(x).isdigit() else x
        )

        if len(epoch_keys) <= max_epochs:
            continue

        keys_to_remove = epoch_keys[:-max_epochs]
        for k in keys_to_remove:
            epoch_map.pop(k, None)