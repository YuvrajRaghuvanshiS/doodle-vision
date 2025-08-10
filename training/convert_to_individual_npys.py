# Convert each example of image and each example of stroke into an individual npy file for efficient loading.


import json
import os

import numpy as np
from tqdm import tqdm

# Data props (stroke)
MAX_STROKES_LEN = 130
STROKES_FEATURES = 3


def preprocess_image(flat_img):
    """
    Input shape: (784,)
    Output shape: (28, 28, 1), normalized to [0, 1]
    """
    img = flat_img.reshape(28, 28).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)  # shape: (28, 28, 1)


def preprocess_strokes(strokes, max_len=MAX_STROKES_LEN):
    """
    Improved stroke preprocessing with consistent normalization
    Centers to (0,0) and scales to [-100, 100] range
    """
    strokes = strokes.astype(np.float32)

    # Convert to absolute coordinates
    strokes[:, 0] = np.cumsum(strokes[:, 0])
    strokes[:, 1] = np.cumsum(strokes[:, 1])

    # Center to (0, 0)
    strokes[:, 0] -= strokes[:, 0].mean()
    strokes[:, 1] -= strokes[:, 1].mean()

    # Scale to [-100, 100] range
    if len(strokes) > 0:
        # Find the maximum absolute coordinate value
        max_coord = max(
            np.abs(strokes[:, 0]).max() if len(strokes) > 0 else 1,
            np.abs(strokes[:, 1]).max() if len(strokes) > 0 else 1,
        )

        # Avoid division by zero
        if max_coord > 0:
            # Scale to [-100, 100] range
            scale_factor = 100.0 / max_coord
            strokes[:, 0] *= scale_factor
            strokes[:, 1] *= scale_factor

    # Truncate or pad as before
    if len(strokes) > max_len:
        return strokes[:max_len]

    pad = np.zeros((max_len - len(strokes), STROKES_FEATURES), dtype=np.float32)

    return np.vstack([strokes, pad])


def preprocess_and_save(
    DATA_DIR_IMAGES,
    DATA_DIR_STROKES,
    OUTPUT_DIR,
    LABEL_MAP,
    max_samples_per_class=100_000,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def ensure_dirs(base_dir, label):
        for subdir in ["images", "strokes", "labels"]:
            os.makedirs(os.path.join(base_dir, subdir, label), exist_ok=True)

    global_idx = 0

    for label_name, _ in tqdm(LABEL_MAP.items(), desc="Processing classes"):
        # Paths for class
        img_path = os.path.join(DATA_DIR_IMAGES, f"{label_name}.npy")
        stroke_path = os.path.join(DATA_DIR_STROKES, f"{label_name}.npz")

        # Load data
        images = np.load(img_path, allow_pickle=True, encoding="latin1", mmap_mode="r")
        strokes = np.load(stroke_path, allow_pickle=True, encoding="latin1")["strokes"]

        N = min(len(images), len(strokes), max_samples_per_class)

        ensure_dirs(OUTPUT_DIR, label_name)

        for i in range(N):
            idx = global_idx + i
            np.save(
                os.path.join(OUTPUT_DIR, "images", label_name, f"{idx:06d}.npy"),
                preprocess_image(images[i]),
            )
            np.save(
                os.path.join(OUTPUT_DIR, "strokes", label_name, f"{idx:06d}.npy"),
                preprocess_strokes(strokes[i]),
            )

        global_idx += N


if __name__ == "__main__":
    with open("dataset/label_map.json", "r") as f:
        LABEL_MAP = json.load(f)

    preprocess_and_save(
        "dataset/images",
        "dataset/combined_strokes",
        "dataset/processed",
        LABEL_MAP,
        max_samples_per_class=100_000,
    )
