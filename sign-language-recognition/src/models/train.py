import os
import json
import numpy as np
from pathlib import Path
import shutil
from tensorflow import keras
from sklearn.model_selection import train_test_split

import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import config
Config = config.Config


# ==========================================================
# AUTO-CLEAR OLD MODEL FILES BEFORE TRAINING
# ==========================================================
def clear_old_models():
    models_dir = Path(Config.MODELS_PATH)
    checkpoints_dir = models_dir / "checkpoints"

    print("\n=====================================")
    print(" Clearing old model artifacts...")
    print("=====================================")

    # Delete best_model.h5
    best_model = models_dir / "best_model.h5"
    if best_model.exists():
        best_model.unlink()
        print(f"Deleted: {best_model}")

    # Delete any other .h5 files
    for file in models_dir.glob("*.h5"):
        try:
            file.unlink()
            print(f"Deleted: {file}")
        except:
            pass

    # Delete checkpoints folder
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        print(f"Deleted folder: {checkpoints_dir}")

    print("✔ Model directory cleaned.\n")


# Call it immediately when training starts
clear_old_models()
# ==========================================================



# ==========================================================
# LOAD PROCESSED DATA
# ==========================================================
def load_dataset():
    processed_dir = Path(Config.PROCESSED_DATA_PATH)
    index_file = processed_dir / "index.json"

    if not index_file.exists():
        print("ERROR: No processed keypoints found.")
        print("Run extract_keypoints.py first.")
        exit()

    with open(index_file, "r") as f:
        entries = json.load(f)

    X = []
    y = []

    for item in entries:
        keypoints_path = item["keypoints_file"]
        keypoints = np.load(keypoints_path)

        # Pad or trim sequences to fixed length
        seq_len = Config.SEQUENCE_LENGTH
        if len(keypoints) >= seq_len:
            keypoints = keypoints[:seq_len]
        else:
            pad = np.zeros((seq_len - len(keypoints), 126))
            keypoints = np.vstack([keypoints, pad])

        X.append(keypoints)
        y.append(item["label"])

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples.")
    return X, y



# ==========================================================
# BUILD MODEL
# ==========================================================
def build_model(num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(Config.SEQUENCE_LENGTH, 126)),
        keras.layers.Masking(mask_value=0.0),

        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(64),

        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# ==========================================================
# TRAIN MODEL
# ==========================================================
def train():
    X, y = load_dataset()

    num_classes = len(set(y))
    models_dir = Path(Config.MODELS_PATH)
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(num_classes)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=models_dir / "best_model.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[checkpoint_callback],
        shuffle=True
    )

    print("\nTraining complete!")
    print("Best model saved to:", models_dir / "best_model.h5")



if __name__ == "__main__":
    train()
