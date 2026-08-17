import argparse
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf

from callback import OmniVisionCallback


class OmniVisionModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omnivision_last_gradients = {}

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        grad_store = {}

        for var, grad in zip(trainable_vars, gradients):
            if grad is None:
                continue

            layer_name = "unknown_layer"
            for layer in self.layers:
                for layer_var in layer.trainable_variables:
                    if layer_var is var:
                        layer_name = layer.name
                        break
                if layer_name != "unknown_layer":
                    break

            if isinstance(grad, tf.IndexedSlices):
                grad_array = tf.convert_to_tensor(grad).numpy()
            else:
                grad_array = grad.numpy()

            grad_store[f"{layer_name}::{var.name}"] = grad_array

        self._omnivision_last_gradients = grad_store

        results = {metric.name: metric.result() for metric in self.metrics}
        results["loss"] = loss
        return results


def build_demo_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv_1")(inputs)
    x = tf.keras.layers.MaxPooling2D(name="pool_1")(x)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_2")(x)
    x = tf.keras.layers.MaxPooling2D(name="pool_2")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)

    model = OmniVisionModel(inputs=inputs, outputs=outputs, name="omnivision_cnn_demo")
    return model


def load_data():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 10)

    return x_train, y_train_cat, x_val, y_val, y_val_cat


def launch_streamlit():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except Exception as exc:
        print(f"Could not launch dashboard: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", choices=["react", "streamlit", "none"], default="none")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    epochs = int(args.epochs)

    x_train, y_train, x_val, y_val_raw, y_val = load_data()

    model = build_demo_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )

    cb = OmniVisionCallback(
        held_out_batch=x_val[:32],
        labels=y_val_raw[:32],
        state_path="states/omnivision_state.json",
        total_epochs=epochs,
    )

    if args.ui == "streamlit":
        launch_streamlit()

    model.fit(
        x_train[:4000],
        y_train[:4000],
        validation_data=(x_val[:800], y_val[:800]),
        epochs=epochs,
        batch_size=32,
        callbacks=[cb],
        verbose=1,
    )


if __name__ == "__main__":
    main()