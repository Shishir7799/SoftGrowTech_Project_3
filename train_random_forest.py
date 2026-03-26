import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import os

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
DATASET_CSV  = "dataset.csv"
MODEL_FILE   = "rf_malaria_model"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH    = 10

# ══════════════════════════════════════════════
print("=" * 55)
print("  STEP 3: Train Random Forest Model")
print("=" * 55)

# ──────────────────────────────────────────────
#  1. LOAD DATASET
# ──────────────────────────────────────────────
print("\n📂 Loading dataset...")

if not os.path.exists(DATASET_CSV):
    print("❌ dataset.csv not found!")
    print("   Please run step2_extract.py first.")
    exit()

dataframe = pd.read_csv(DATASET_CSV)

print(f"   Total samples  : {len(dataframe)}")
print(f"   Columns        : {dataframe.columns.tolist()}")
print(f"\n   Class Balance:")
print(dataframe["Label"].value_counts().to_string())

# ──────────────────────────────────────────────
#  2. PREPARE FEATURES & LABELS
# ──────────────────────────────────────────────
print("\n⚙️  Preparing features...")

x = dataframe.drop(["Label"], axis=1).fillna(0)
y = dataframe["Label"]

print(f"   Feature shape  : {x.shape}")

# ──────────────────────────────────────────────
#  3. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n✂️  Splitting data (80% train / 20% test)...")

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y
)

print(f"   Training samples : {len(x_train)}")
print(f"   Testing  samples : {len(x_test)}")

# ──────────────────────────────────────────────
#  4. TRAIN MODEL
# ──────────────────────────────────────────────
print(f"\n🌲 Training Random Forest...")
print(f"   Trees (n_estimators) : {N_ESTIMATORS}")
print(f"   Max Depth            : {MAX_DEPTH}")

model = RandomForestClassifier(
    n_estimators = N_ESTIMATORS,
    max_depth    = MAX_DEPTH,
    random_state = RANDOM_STATE,
    n_jobs       = -1
)

model.fit(x_train, y_train)

# Save model
joblib.dump(model, MODEL_FILE)
print(f"\n   ✅ Model saved as: {MODEL_FILE}")

# ──────────────────────────────────────────────
#  5. EVALUATE
# ──────────────────────────────────────────────
print("\n📊 Evaluating Model...")

predictions = model.predict(x_test)
accuracy    = model.score(x_test, y_test)

print("\n--- Classification Report ---")
print(metrics.classification_report(y_test, predictions))

print(f"--- Overall Accuracy ---")
print(f"   {accuracy:.4f}  ({accuracy * 100:.2f}%)")

print("\n--- Confusion Matrix ---")
cm = metrics.confusion_matrix(y_test, predictions)
print(f"   Labels: {list(model.classes_)}")
print(f"   {cm}")

# ──────────────────────────────────────────────
#  6. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
print("\n--- Feature Importance ---")
importance_df = pd.DataFrame({
    "Feature"    : x.columns,
    "Importance" : model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))

print(f"\n{'=' * 55}")
print(f"  ✅ ALL DONE! Model accuracy: {accuracy * 100:.2f}%")
print(f"{'=' * 55}")
print("\n▶️  Now run: python step4_predict.py")
