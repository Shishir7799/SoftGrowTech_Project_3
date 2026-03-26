import cv2
import numpy as np
import pandas as pd
import joblib
import os

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
MODEL_FILE   = "rf_malaria_model"
TOP_CONTOURS = 5
IMG_SIZE     = (64, 64)

# ──────────────────────────────────────────────
#  CHANGE THIS TO YOUR IMAGE PATH ⬇️
# ──────────────────────────────────────────────
TEST_IMAGE = r"cell_images\Parasitized\C33P1thinF_IMG_20150619_114756a_cell_179.png"


# ──────────────────────────────────────────────
#  FEATURE EXTRACTION (same as step2)
# ──────────────────────────────────────────────
def extract_features(img_path):
    im = cv2.imread(img_path)
    if im is None:
        return None

    im        = cv2.resize(im, IMG_SIZE)
    im        = cv2.GaussianBlur(im, (5, 5), 2)
    im_gray   = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(im_gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = sorted(contours, key=cv2.contourArea, reverse=True)

    features = []

    for i in range(TOP_CONTOURS):
        try:
            features.append(round(cv2.contourArea(contours[i]), 2))
        except IndexError:
            features.append(0)

    try:
        features.append(round(cv2.arcLength(contours[0], True), 2))
    except IndexError:
        features.append(0)

    try:
        area      = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        circ      = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        features.append(round(circ, 4))
    except IndexError:
        features.append(0)

    features.append(round(float(np.mean(im_gray)), 4))
    features.append(round(float(np.std(im_gray)),  4))

    return features


# ──────────────────────────────────────────────
#  PREDICT
# ──────────────────────────────────────────────
print("=" * 55)
print("  STEP 4: Predict Single Image")
print("=" * 55)

# Check model exists
if not os.path.exists(MODEL_FILE):
    print("❌ Model not found!")
    print("   Please run step3_train.py first.")
    exit()

# Check image exists
if not os.path.exists(TEST_IMAGE):
    print(f"❌ Image not found: {TEST_IMAGE}")
    print("   Update TEST_IMAGE path at top of this file.")
    exit()

# Load model
model    = joblib.load(MODEL_FILE)
print(f"\n✅ Model loaded: {MODEL_FILE}")
print(f"📷 Image      : {TEST_IMAGE}")

# Extract features
features = extract_features(TEST_IMAGE)
if features is None:
    print("❌ Could not read image.")
    exit()

# Predict
feature_names = (
    [f"area_{i}" for i in range(TOP_CONTOURS)]
    + ["perimeter_0", "circularity_0", "mean_intensity", "std_intensity"]
)

df       = pd.DataFrame([features], columns=feature_names)
pred     = model.predict(df)[0]
proba    = model.predict_proba(df)[0]
classes  = model.classes_

print(f"\n{'=' * 55}")
print(f"  🔬 RESULT: {pred}")
print(f"{'=' * 55}")
for cls, p in zip(classes, proba):
    bar = "█" * int(p * 30)
    print(f"  {cls:15s} : {p*100:6.2f}%  {bar}")

if pred == "Parasitized":
    print("\n  ⚠️  Malaria parasite DETECTED!")
else:
    print("\n  ✅ No malaria parasite detected.")
print("=" * 55)
