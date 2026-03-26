import cv2
import os
import numpy as np
import csv
import glob

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
LABELS       = ["Parasitized", "Uninfected"]
IMAGE_DIR    = "cell_images"
OUTPUT_CSV   = "dataset.csv"
TOP_CONTOURS = 5
IMG_SIZE     = (64, 64)

# ──────────────────────────────────────────────
#  FEATURE EXTRACTION FUNCTION
# ──────────────────────────────────────────────
def extract_features(img_path):
    """
    Reads one image and returns 9 features:
    [area_0..4, perimeter_0, circularity_0, mean_intensity, std_intensity]
    Returns None if image cannot be read.
    """
    im = cv2.imread(img_path)
    if im is None:
        return None

    # Resize for consistency
    im = cv2.resize(im, IMG_SIZE)

    # Gaussian Blur — removes noise
    im = cv2.GaussianBlur(im, (5, 5), 2)

    # Grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Binary Threshold
    _, thresh = cv2.threshold(im_gray, 127, 255, 0)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area — largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    features = []

    # Feature 1 to 5: Top-5 contour areas
    for i in range(TOP_CONTOURS):
        try:
            features.append(round(cv2.contourArea(contours[i]), 2))
        except IndexError:
            features.append(0)

    # Feature 6: Perimeter of largest contour
    try:
        features.append(round(cv2.arcLength(contours[0], True), 2))
    except IndexError:
        features.append(0)

    # Feature 7: Circularity (1.0 = perfect circle)
    try:
        area      = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        circ      = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        features.append(round(circ, 4))
    except IndexError:
        features.append(0)

    # Feature 8 & 9: Mean and Std of pixel intensities
    features.append(round(float(np.mean(im_gray)), 4))
    features.append(round(float(np.std(im_gray)),  4))

    return features


# ──────────────────────────────────────────────
#  MAIN — LOOP OVER BOTH LABELS
# ──────────────────────────────────────────────
print("=" * 55)
print("  STEP 2: Feature Extraction → dataset.csv")
print("=" * 55)

# CSV Header
header = (
    ["Label"]
    + [f"area_{i}" for i in range(TOP_CONTOURS)]
    + ["perimeter_0", "circularity_0", "mean_intensity", "std_intensity"]
)

total_saved   = 0
total_skipped = 0

with open(OUTPUT_CSV, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

    for label in LABELS:
        dir_path = os.path.join(IMAGE_DIR, label, "*.png")
        img_list = glob.glob(dir_path)

        if not img_list:
            print(f"\n❌ No images found for: {label}")
            print(f"   Checked path: {dir_path}")
            print("   Make sure cell_images folder exists in the same directory!")
            continue

        print(f"\n📂 Processing: {label} ({len(img_list)} images)")

        saved   = 0
        skipped = 0

        for idx, img_path in enumerate(img_list):
            features = extract_features(img_path)

            if features is None:
                skipped += 1
                continue

            writer.writerow([label] + features)
            saved += 1

            # Progress update every 1000 images
            if (idx + 1) % 1000 == 0:
                print(f"   {idx + 1}/{len(img_list)} done...")

        print(f"   ✅ Saved  : {saved}")
        print(f"   ⚠️  Skipped: {skipped}")

        total_saved   += saved
        total_skipped += skipped

print(f"\n{'=' * 55}")
print(f"  ✅ DONE! dataset.csv created")
print(f"  Total saved  : {total_saved}")
print(f"  Total skipped: {total_skipped}")
print(f"{'=' * 55}")
print("\n▶️  Now run: python step3_train.py")
