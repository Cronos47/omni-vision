from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from state import init_state, save_state, trim_epoch_history
from utils import (
    build_layer_blocks,
    compute_gradient_stats,
    compute_tensor_stats,
    detect_architecture,
    extract_custom_payload,
    get_final_non_output_layer,
    layer_matches_any_tag,
    sample_activation_array,
)


class OmniVisionCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        held_out_batch: Any,
        labels: Optional[Any] = None,
        state_path: str = "omnivision_state.json",
        total_epochs: int = 0,
        max_stored_epochs: int = 20,
        activation_sample_threshold: int = 512,
        activation_top_percent: float = 0.10,
    ) -> None:
        super().__init__()
        self.held_out_batch = held_out_batch
        self.labels = labels
        self.state_path = state_path
        self.total_epochs = total_epochs
        self.max_stored_epochs = max_stored_epochs
        self.activation_sample_threshold = activation_sample_threshold
        self.activation_top_percent = activation_top_percent

        self.state: Dict[str, Any] = init_state(total_epochs=total_epochs)
        self.activation_model: Optional[tf.keras.Model] = None
        self.final_non_output_layer_name: Optional[str] = None
        self.state.setdefault("logs", [])

    def _log(self, msg):
        import datetime
        self.state["logs"].append({
            "time": datetime.datetime.utcnow().isoformat(),
            "msg": msg
        })

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        arch_type = detect_architecture(self.model)
        self.state["meta"]["arch_type"] = arch_type
        self.state["meta"]["run_name"] = getattr(self.model, "name", "omnivision_run")
        self.state["meta"]["total_epochs"] = self.total_epochs
        self.state["status"]["training"] = True
        self.state["status"]["done"] = False
        self.state["status"]["error"] = None

        self.state["meta"]["current_epoch"] = 0
        self.state["history"]["metrics"] = {}
        self.state["history"]["loss_delta"] = []
        self.state["weights"] = {}
        self.state["gradients"] = {}
        self.state["activations"] = {}
        self.state["labels"] = {}
        self.state["attention"] = {}
        self.state["routing"] = {}
        self.state["diffusion"] = {}
        self.state["sequence"] = {}
        self.state["vision"] = {}
        self.state["custom"] = {}
        self.state["logs"] = []

        self.state["layer_blocks"] = build_layer_blocks(self.model, block_size=5)

        # diffusion models do not need the normal activation model path
        if arch_type != "diffusion":
            self._build_activation_model()
        else:
            self.activation_model = None
            self.final_non_output_layer_name = None

        save_state(self.state, self.state_path)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        epoch_idx = int(epoch) + 1
        epoch_key = str(epoch_idx)

        self._log("Training started")
        self._log(f"Epoch {epoch+1} started")
        self._log(f"Epoch {epoch+1} completed")
        self._log("Training finished")

        self.state["meta"]["current_epoch"] = epoch_idx

        self._capture_metrics(logs)
        self._capture_weight_stats(epoch_key)
        self._capture_gradient_stats(epoch_key)

        if self.state["meta"].get("arch_type") != "diffusion":
            self._capture_activations(epoch_key)

        self._capture_custom_payloads(epoch_key)
        self._capture_diffusion(epoch_key)

        if self.labels is not None:
            self.state["labels"][epoch_key] = np.asarray(self.labels).tolist()

        trim_epoch_history(self.state, max_epochs=self.max_stored_epochs)
        save_state(self.state, self.state_path)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.state["status"]["training"] = False
        self.state["status"]["done"] = True
        save_state(self.state, self.state_path)

    def _capture_diffusion(self, epoch_key: str) -> None:
        if not hasattr(self.model, "run_diffusion_probe"):
            return

        try:
            result = self.model.run_diffusion_probe()
        except Exception as exc:
            self.state["status"]["error"] = f"Diffusion probe failed: {str(exc)}"
            return

        if not isinstance(result, dict):
            return

        self.state["diffusion"][epoch_key] = result

    def _capture_metrics(self, logs: Dict[str, Any]) -> None:
        metrics_history = self.state["history"]["metrics"]

        for metric_name, metric_value in logs.items():
            if metric_value is None:
                continue

            if metric_name not in metrics_history:
                metrics_history[metric_name] = []

            metrics_history[metric_name].append(float(metric_value))

        loss_series = metrics_history.get("loss", [])
        if len(loss_series) >= 2:
            delta = float(loss_series[-1] - loss_series[-2])
        else:
            delta = 0.0

        self.state["history"]["loss_delta"].append(delta)

    def _capture_weight_stats(self, epoch_key: str) -> None:
        self.state["weights"][epoch_key] = {}

        for variable in self.model.trainable_variables:
            layer_name = self._extract_layer_name_from_variable(variable.name)
            tensor_stats = compute_tensor_stats(variable.numpy())

            if layer_name not in self.state["weights"][epoch_key]:
                self.state["weights"][epoch_key][layer_name] = {}

            self.state["weights"][epoch_key][layer_name][variable.name] = tensor_stats

    def _capture_gradient_stats(self, epoch_key: str) -> None:
        self.state["gradients"][epoch_key] = {}

        last_gradients = getattr(self.model, "_omnivision_last_gradients", None)
        if not last_gradients:
            return

        for full_name, grad_array in last_gradients.items():
            if "::" in full_name:
                layer_name, variable_name = full_name.split("::", 1)
            else:
                layer_name = self._extract_layer_name_from_variable(full_name)
                variable_name = full_name

            grad_stats = compute_gradient_stats(grad_array)

            if layer_name not in self.state["gradients"][epoch_key]:
                self.state["gradients"][epoch_key][layer_name] = {}

            self.state["gradients"][epoch_key][layer_name][variable_name] = grad_stats

    def _capture_activations(self, epoch_key: str) -> None:
        if self.activation_model is None:
            return

        try:
            activation_outputs = self.activation_model(self.held_out_batch, training=False)
        except Exception as exc:
            self.state["status"]["error"] = f"Activation capture failed: {str(exc)}"
            return

        if not isinstance(activation_outputs, (list, tuple)):
            activation_outputs = [activation_outputs]

        self.state["activations"][epoch_key] = {}

        model_layers_with_outputs = []
        for layer in self.model.layers:
            try:
                _ = layer.output
                model_layers_with_outputs.append(layer)
            except Exception:
                continue

        for layer, activation in zip(model_layers_with_outputs, activation_outputs):
            try:
                act_np = activation.numpy()
            except Exception:
                act_np = np.asarray(activation)

            should_store_full = False

            if layer.name == self.final_non_output_layer_name:
                should_store_full = True

            if layer_matches_any_tag(layer, ["conv"]) and act_np.ndim == 4:
                should_store_full = True

            if layer_matches_any_tag(layer, ["attention", "mha", "transformer"]) and act_np.ndim >= 3:
                should_store_full = True

            if layer_matches_any_tag(layer, ["lstm", "gru", "bilstm", "rnn"]) and act_np.ndim >= 2:
                should_store_full = True

            if should_store_full:
                self.state["activations"][epoch_key][layer.name] = {
                    "shape": list(act_np.shape),
                    "sampled": False,
                    "indices": [],
                    "values": act_np.tolist(),
                }
                continue

            sampled = sample_activation_array(
                act_np,
                threshold=self.activation_sample_threshold,
                top_percent=self.activation_top_percent,
            )
            self.state["activations"][epoch_key][layer.name] = sampled

    def _capture_custom_payloads(self, epoch_key: str) -> None:
        self.state["custom"][epoch_key] = {}

        attention_bucket = {}
        routing_bucket = {}
        sequence_bucket = {}
        diffusion_bucket = {}
        vision_bucket = {}

        for layer in self.model.layers:
            payload = extract_custom_payload(layer)
            if not payload:
                continue

            layer_name = layer.name
            self.state["custom"][epoch_key][layer_name] = payload

            payload_type = str(payload.get("type", "")).lower()

            if payload_type == "attention":
                attention_bucket[layer_name] = payload
            elif payload_type == "routing":
                routing_bucket[layer_name] = payload
            elif payload_type == "sequence":
                sequence_bucket[layer_name] = payload
            elif payload_type == "diffusion":
                diffusion_bucket[layer_name] = payload
            elif payload_type == "vision":
                vision_bucket[layer_name] = payload

        if attention_bucket:
            self.state["attention"][epoch_key] = attention_bucket
        if routing_bucket:
            self.state["routing"][epoch_key] = routing_bucket
        if sequence_bucket:
            self.state["sequence"][epoch_key] = sequence_bucket
        if diffusion_bucket:
            self.state["diffusion"][epoch_key] = diffusion_bucket
        if vision_bucket:
            self.state["vision"][epoch_key] = vision_bucket

    def _build_activation_model(self) -> None:
        try:
            layer_outputs = []

            for layer in self.model.layers:
                try:
                    _ = layer.output
                    layer_outputs.append(layer.output)
                except Exception:
                    continue

            if not layer_outputs:
                self.activation_model = None
                self.final_non_output_layer_name = None
                return

            self.activation_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=layer_outputs,
                name=f"{self.model.name}_activations",
            )

            final_non_output_layer = get_final_non_output_layer(self.model)
            self.final_non_output_layer_name = final_non_output_layer.name

        except Exception as exc:
            self.activation_model = None
            self.final_non_output_layer_name = None
            self.state["status"]["error"] = f"Activation model build failed: {str(exc)}"

    @staticmethod
    def _extract_layer_name_from_variable(variable_name: str) -> str:
        if "/" in variable_name:
            return variable_name.split("/")[0]
        if ":" in variable_name:
            return variable_name.split(":")[0]
        return variable_name