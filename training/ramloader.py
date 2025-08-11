# %%
import gc
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# %%
gc.enable()

# %%
MAX_SEQ_LEN = 130
STROKE_FEATURES = 3
IMG_HEIGHT, IMG_WIDTH = 28, 28
IMG_CHANNELS = 1
NUM_CLASSES = 345
SAMPLES_PER_CLASS = 30_000
DATA_DIR_STROKES = "dataset/combined_strokes"
DATA_DIR_IMAGES = "dataset/images"
LABEL_MAP_PATH = "dataset/label_map.json"
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 20

with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAP = json.load(f)

# %%
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# %%
def preprocess_stroke(stroke, max_len=MAX_SEQ_LEN):
    """
    Improved stroke preprocessing with consistent normalization
    Centers to (0,0) and scales to [-100, 100] range
    """
    stroke = stroke.astype(np.float32)

    # Convert to absolute coordinates
    stroke[:, 0] = np.cumsum(stroke[:, 0])
    stroke[:, 1] = np.cumsum(stroke[:, 1])

    # Center to (0, 0)
    stroke[:, 0] -= stroke[:, 0].mean()
    stroke[:, 1] -= stroke[:, 1].mean()

    # Scale to [-100, 100] range
    if len(stroke) > 0:
        # Find the maximum absolute coordinate value
        max_coord = max(
            np.abs(stroke[:, 0]).max() if len(stroke) > 0 else 1,
            np.abs(stroke[:, 1]).max() if len(stroke) > 0 else 1,
        )

        # Avoid division by zero
        if max_coord > 0:
            # Scale to [-100, 100] range
            scale_factor = 100.0 / max_coord
            stroke[:, 0] *= scale_factor
            stroke[:, 1] *= scale_factor

    # Truncate or pad as before
    if len(stroke) > max_len:
        return stroke[:max_len]

    pad = np.zeros((max_len - len(stroke), STROKE_FEATURES), dtype=np.float32)
    return np.vstack([stroke, pad])


# %%
def build_hybrid_model(num_classes):
    inp_str = Input(shape=(MAX_SEQ_LEN, STROKE_FEATURES), name="stroke_input")
    x = Bidirectional(LSTM(128, return_sequences=True))(inp_str)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(128, activation="relu")(x)

    inp_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image_input")
    y = Conv2D(32, 3, activation="relu", padding="same")(inp_img)
    y = MaxPooling2D()(y)
    y = Conv2D(64, 3, activation="relu", padding="same")(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Conv2D(128, 3, activation="relu", padding="same")(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Conv2D(256, 3, activation="relu", padding="same")(y)
    y = GlobalAveragePooling2D()(y)
    y = Dense(128, activation="relu")(y)

    merged = Concatenate()([x, y])
    merged = Dropout(0.5)(merged)
    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(0.3)(merged)
    out = Dense(num_classes, activation="softmax", dtype="float32")(merged)

    return Model(inputs=[inp_str, inp_img], outputs=out, name="hybrid_model")


# %%
model = build_hybrid_model(num_classes=NUM_CLASSES)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

model.summary()


# %%
def load_hybrid_data(N=SAMPLES_PER_CLASS):
    X_img_list, X_str_list, y_list = [], [], []
    for cls, idx in tqdm(LABEL_MAP.items()):
        img_arr = np.load(
            os.path.join(DATA_DIR_IMAGES, f"{cls}.npy"),
            allow_pickle=True,
            encoding="latin1",
        )[:N]
        img_arr = (
            img_arr.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype("float16")
            / 255.0
        )
        X_img_list.append(img_arr)
        del img_arr
        gc.collect()

        data = np.load(
            os.path.join(DATA_DIR_STROKES, f"{cls}.npz"),
            allow_pickle=True,
            encoding="latin1",
        )
        strokes = data["strokes"][:N]
        proc = np.stack([preprocess_stroke(s) for s in strokes], axis=0).astype(
            "float16"
        )
        X_str_list.append(proc)
        del data, strokes, proc
        gc.collect()

        y_list.append(np.full((N,), idx, dtype=np.int32))

    X_img = np.concatenate(X_img_list, axis=0)
    X_str = np.concatenate(X_str_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    X_img, X_str, y = shuffle(X_img, X_str, y, random_state=42)
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)

    return (X_str, X_img), y_cat


def tf_dataset(X_img, X_str, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(((X_str, X_img), y))
    ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# %%
(X_str, X_img), y = load_hybrid_data(N=SAMPLES_PER_CLASS)

total = X_img.shape[0]
split = int((1 - VALIDATION_SPLIT) * total)

X_str_train, X_str_val = X_str[:split], X_str[split:]
X_img_train, X_img_val = X_img[:split], X_img[split:]
y_train, y_val = y[:split], y[split:]
del X_str, X_img, y
gc.collect()

train_ds = tf_dataset(X_img_train, X_str_train, y_train, batch_size=BATCH_SIZE)
val_ds = tf_dataset(X_img_val, X_str_val, y_val, batch_size=BATCH_SIZE)

# %%
model_name = f"best_model_{NUM_CLASSES}_classes_{SAMPLES_PER_CLASS}_examples.keras"
callbacks = [
    ModelCheckpoint(model_name, monitor="val_accuracy", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# %%
val_acc = history.history["val_accuracy"][-1]
train_acc = history.history["accuracy"][-1]
val_loss = history.history["val_loss"][-1]

val_acc_str = f"{val_acc:.4f}"
train_acc_str = f"{train_acc:.4f}"
val_loss_str = f"{val_loss:.4f}"

new_model_name = (
    f"best_model_{NUM_CLASSES}c_{SAMPLES_PER_CLASS}x_"
    f"valacc_{val_acc_str}_trainacc_{train_acc_str}_valloss_{val_loss_str}.keras"
)

os.rename(model_name, new_model_name)
print(f"Renamed model file to: {new_model_name}")

with open("training_summary.txt", "w") as f:
    f.write(f"NUM_CLASSES: {NUM_CLASSES}\n")
    f.write(f"SAMPLES_PER_CLASS: {SAMPLES_PER_CLASS}\n")
    f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Final Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Final Validation Loss: {val_loss:.4f}\n")
    f.write(f"Model file: {new_model_name}\n")

# %%
