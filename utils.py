from typing import Any, Dict, List, Tuple

import numpy as np


def ema_smooth(values: List[float], smoothing: int) -> List[float]:
    if not values:
        return []

    if smoothing <= 0:
        return values

    alpha = 1.0 / (1.0 + smoothing)
    smoothed = [float(values[0])]

    for v in values[1:]:
        smoothed.append(alpha * float(v) + (1.0 - alpha) * smoothed[-1])

    return smoothed


def compute_tensor_stats(tensor: Any) -> Dict[str, float]:
    arr = np.asarray(tensor)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "l2": 0.0}

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "l2": float(np.linalg.norm(arr.ravel(), ord=2)),
    }


def compute_gradient_stats(grad: Any) -> Dict[str, float]:
    if grad is None:
        return {"mean_abs": 0.0, "l2": 0.0}

    arr = np.asarray(grad)
    if arr.size == 0:
        return {"mean_abs": 0.0, "l2": 0.0}

    return {
        "mean_abs": float(np.mean(np.abs(arr))),
        "l2": float(np.linalg.norm(arr.ravel(), ord=2)),
    }


def detect_architecture(model: Any) -> str:
    layer_class_names = [layer.__class__.__name__ for layer in model.layers]
    lowered_layer_names = [layer.name.lower() for layer in model.layers]
    model_name = getattr(model, "name", "").lower()

    custom_tags = []
    for layer in model.layers:
        tags = getattr(layer, "omnivision_tags", None)
        if tags is None:
            continue
        if isinstance(tags, (list, tuple, set)):
            custom_tags.extend([str(t).lower() for t in tags])
        else:
            custom_tags.append(str(tags).lower())

    tag_space = lowered_layer_names + custom_tags + [model_name]

    has_conv1d = any(name == "Conv1D" for name in layer_class_names)
    has_conv2d = any(name == "Conv2D" for name in layer_class_names)
    has_conv3d = any(name == "Conv3D" for name in layer_class_names)
    has_rnn = any(name in ["LSTM", "GRU", "Bidirectional"] for name in layer_class_names)
    has_attention = any(name == "MultiHeadAttention" for name in layer_class_names)

    if any("moe" in t or "expert" in t or "router" in t for t in tag_space):
        return "moe"

    if any("diffusion" in t or "unet" in t or "denoise" in t for t in tag_space):
        return "diffusion"

    if has_attention and (has_conv1d or has_conv2d or has_conv3d or has_rnn):
        return "hybrid"

    if has_attention or any("transformer" in t or "attention" in t for t in tag_space):
        return "transformer"

    if has_rnn and (has_conv1d or has_conv2d):
        return "hybrid"

    if has_rnn or any("lstm" in t or "gru" in t or "bilstm" in t for t in tag_space):
        return "rnn"

    if has_conv2d or has_conv3d or any("vision" in t or "resnet" in t or "efficientnet" in t for t in tag_space):
        return "cnn"

    if has_conv1d:
        return "sequence_cnn"

    return "generic"


def build_layer_blocks(model: Any, block_size: int = 5) -> Dict[str, str]:
    if len(model.layers) <= 20:
        return {}

    mapping = {}
    for idx, layer in enumerate(model.layers):
        block_id = (idx // block_size) + 1
        mapping[layer.name] = f"block_{block_id}"
    return mapping


def sample_activation_array(
    arr: Any,
    threshold: int = 512,
    top_percent: float = 0.10,
) -> Dict[str, Any]:
    np_arr = np.asarray(arr)
    flat = np_arr.reshape(-1)

    if flat.size <= threshold:
        return {
            "shape": list(np_arr.shape),
            "sampled": False,
            "indices": [],
            "values": flat.tolist(),
        }

    k = max(1, int(flat.size * top_percent))
    abs_flat = np.abs(flat)
    top_idx = np.argsort(abs_flat)[-k:]

    return {
        "shape": list(np_arr.shape),
        "sampled": True,
        "indices": top_idx.tolist(),
        "values": flat[top_idx].tolist(),
    }


def get_final_non_output_layer(model: Any) -> Any:
    if len(model.layers) < 2:
        return model.layers[-1]
    return model.layers[-2]


def rank_top_neurons(activation: Any, top_k: int = 20) -> Tuple[List[int], List[float]]:
    arr = np.asarray(activation)

    if arr.size == 0:
        return [], []

    if arr.ndim == 1:
        scores = np.abs(arr)
    else:
        last_dim = arr.shape[-1]
        reshaped = arr.reshape(-1, last_dim)
        scores = np.mean(np.abs(reshaped), axis=0)

    top_idx = np.argsort(scores)[-top_k:][::-1]
    top_vals = scores[top_idx]

    return top_idx.tolist(), top_vals.astype(float).tolist()


def find_layers_by_class_name(model: Any, class_name: str) -> List[Any]:
    return [layer for layer in model.layers if layer.__class__.__name__ == class_name]


def get_layer_tags(layer: Any) -> List[str]:
    tags = []
    tags.append(layer.__class__.__name__.lower())
    tags.append(getattr(layer, "name", "").lower())

    custom_tags = getattr(layer, "omnivision_tags", None)
    if custom_tags is not None:
        if isinstance(custom_tags, (list, tuple, set)):
            tags.extend([str(t).lower() for t in custom_tags])
        else:
            tags.append(str(custom_tags).lower())

    return [t for t in tags if t]


def layer_matches_any_tag(layer: Any, candidates: List[str]) -> bool:
    layer_tags = get_layer_tags(layer)
    candidates = [c.lower() for c in candidates]

    for c in candidates:
        for t in layer_tags:
            if c in t:
                return True
    return False


def extract_custom_payload(layer: Any) -> Dict[str, Any]:
    if hasattr(layer, "get_omnivision_payload") and callable(layer.get_omnivision_payload):
        try:
            payload = layer.get_omnivision_payload()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}

    payload = getattr(layer, "omnivision_payload", None)
    if isinstance(payload, dict):
        return payload

    return {}