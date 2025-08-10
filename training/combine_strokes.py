import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Files
DATA_DIR_STROKES = "dataset/strokes"
COMBINED_DIR_STROKES = "dataset/combined_strokes"  # Strokes are given in splits (test, train, val) so we must combine all to fully utilize the data


def process_and_save_class(fname):
    """
    Process a single class: combine 'train', 'val', 'test' strokes and save to disk.
    """
    cls_name = os.path.splitext(fname)[0]
    path_in = os.path.join(DATA_DIR_STROKES, fname)

    try:
        data = np.load(path_in, allow_pickle=True, encoding="latin1")

        splits = []
        for split in ("train", "val", "test"):
            if split in data:
                splits.append(data[split])

        combined = np.concatenate(splits, axis=0)
        out_path = os.path.join(COMBINED_DIR_STROKES, f"{cls_name}.npz")
        np.savez_compressed(out_path, strokes=combined)

        del data, splits, combined
        gc.collect()

    except Exception as e:
        return f"Error processing {cls_name}: {e}"


def combine_and_save_strokes_parallel():
    """
    Combine and save all class strokes using multiprocessing.
    """
    os.makedirs(COMBINED_DIR_STROKES, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR_STROKES) if f.endswith(".npz")]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_and_save_class, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    combine_and_save_strokes_parallel()
