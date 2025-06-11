def load_hybrid_data(N=SAMPLES_PER_CLASS):
    img_files_full = sorted(
        f for f in os.listdir(DATA_DIR_IMAGES) if f.endswith(".npy")
    )
    stroke_files_full = sorted(
        f for f in os.listdir(DATA_DIR_STROKES) if f.endswith(".npz")
    )

    img_names = {os.path.splitext(f)[0] for f in img_files_full}
    stroke_names = {os.path.splitext(f)[0] for f in stroke_files_full}
    common = sorted(img_names & stroke_names)[:NUM_CLASSES]

    X_img_list, X_str_list, y_list = [], [], []
    for idx, cls in enumerate(common):
        img_arr = np.load(
            os.path.join(DATA_DIR_IMAGES, f"{cls}.npy"),
            allow_pickle=True,
            encoding="latin1",
        )[:N]
        img_arr = (
            img_arr.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype("float32")
            / 255.0
        )
        X_img_list.append(img_arr)

        data = np.load(
            os.path.join(DATA_DIR_STROKES, f"{cls}.npz"),
            allow_pickle=True,
            encoding="latin1",
        )
        strokes = data["train"][:N]
        proc = np.stack([preprocess_stroke(s) for s in strokes], axis=0)
        X_str_list.append(proc)

        y_list.append(
            np.full((N,), idx, dtype=np.int32)
        )  #  putting labels same for each class

    X_img = np.concatenate(X_img_list, axis=0)
    X_str = np.concatenate(X_str_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    X_img, X_str, y = shuffle(
        X_img, X_str, y, random_state=42
    )  #  mix everything randomly

    true_num_classes = len(common)
    y_cat = to_categorical(y, num_classes=true_num_classes)
    return (X_str, X_img), y_cat
