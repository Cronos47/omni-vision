import numpy as np
import tensorflow as tf
from callback import OmniVisionCallback

x = np.random.rand(32, 28, 28, 1).astype("float32")
y = tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(32,)), 10)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

cb = OmniVisionCallback(
    held_out_batch=x[:8],
    labels=np.argmax(y[:8], axis=1),
    total_epochs=2,
)

model.fit(x, y, epochs=2, callbacks=[cb], verbose=1)