"""
train_cnn.py - Eye-state classifier for a drowsy-driver warning system.
Optimized training pipeline (TensorFlow / Keras 2.9+ or Keras 3).

What changed vs. the original train_cnn.py
------------------------------------------
* One consistent `tf.keras` import (no mixing standalone `keras` with `tensorflow`,
  which is a common source of silent version bugs).
* tf.data input pipeline (image_dataset_from_directory + cache + prefetch) instead
  of the deprecated, single-threaded ImageDataGenerator  ->  noticeably faster.
* Augmentation via Keras preprocessing layers that run on-device (GPU), including
  brightness/contrast jitter that matters a lot for changing in-car lighting.
* Pixel normalization (Rescaling) is baked INTO the model. See the IMPORTANT note
  at the bottom about your real-time inference code.
* Lighter, slightly deeper net with GlobalAveragePooling -> fewer parameters,
  less overfitting, and fast per-frame inference.
* Class weighting so an imbalanced open/closed dataset does not bias the model.
* Drowsiness-relevant metrics: precision, recall, AUC (accuracy alone hides the
  thing you care about -> catching CLOSED eyes).
* ModelCheckpoint + a fixed random seed for reproducibility.
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------------------------------------------------
# 0. Reproducibility & optional performance switches
# ----------------------------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Set True on a modern NVIDIA GPU for ~1.5-2x faster training (no accuracy loss).
USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Let the GPU allocate memory as needed instead of grabbing all of it up front.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
TRAIN_DIR = 'C:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\data\\train'  
VAL_DIR = 'C:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\data\\val'

IMG_SIZE   = 48
BATCH_SIZE = 64
EPOCHS     = 50            # EarlyStopping will usually stop well before this
MODEL_OUT  = "eye_state_classifier.keras"
LABELS_OUT = "class_indices.json"

AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------------------------------------------------
# 2. Data pipeline (tf.data - faster than ImageDataGenerator)
# ----------------------------------------------------------------------
print("Loading datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",          # 0/1 targets for sigmoid + binary_crossentropy
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Capture the label mapping BEFORE applying .map()/.prefetch() (it is lost after).
class_names = train_ds.class_names                 # e.g. ['Closed', 'Open']
class_indices = {name: i for i, name in enumerate(class_names)}
print("Class mapping (name -> index):", class_indices)

# --- Augmentation: training set only, executes on-device ---

#train_cnn.py
data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),                  
        layers.RandomRotation(0.05),          # ~ +/- 18 degrees
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.20),          # độ sáng/độ tương phản thay đổi 
        layers.RandomBrightness(0.20, value_range=(0, 255)),
    ],
    name="augmentation",
)

# cache() stores the raw decoded images; shuffle + augment run fresh every epoch.
# If you hit RAM limits on a huge dataset, change .cache() to .cache("train_cache").
train_ds = (
    train_ds
    .cache()
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ----------------------------------------------------------------------
# 3. Class weights (handles an imbalanced open/closed split)
# ----------------------------------------------------------------------
def count_images(directory, names):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.pgm")
    counts = []
    for name in names:
        n = 0
        for ext in exts:
            n += len(glob.glob(os.path.join(directory, name, ext)))
        counts.append(n)
    return counts

counts = count_images(TRAIN_DIR, class_names)
total = sum(counts)
class_weight = {
    i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0
}
print("Train images per class:", dict(zip(class_names, counts)))
print("Class weights:", class_weight)

# ----------------------------------------------------------------------
# 4. Model (normalization baked in; GAP head for fast inference)
# ----------------------------------------------------------------------
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)               # normalize INSIDE the model

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    # dtype="float32" keeps the output stable when mixed precision is on.
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return models.Model(inputs, outputs, name="eye_state_cnn")

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
model.summary()

# ----------------------------------------------------------------------
# 5. Train
# ----------------------------------------------------------------------
cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1
    ),
]

print("\nTRAINING...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
)

# ----------------------------------------------------------------------
# 6. Save + report
# ----------------------------------------------------------------------
model.save(MODEL_OUT)
with open(LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)

print(f"\nSaved model -> {MODEL_OUT}")
print(f"Saved label map -> {LABELS_OUT}")
print("Class indices:", class_indices)

print("\nFinal validation metrics:")
results = model.evaluate(val_ds, verbose=0)
for name, value in zip(model.metrics_names, results):
    print(f"  {name}: {value:.4f}")