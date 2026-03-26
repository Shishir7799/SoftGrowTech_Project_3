import kagglehub
import os
import shutil

# ──────────────────────────────────────────────
#  STEP 1 — DOWNLOAD MALARIA DATASET
# ──────────────────────────────────────────────

print("Downloading Malaria Cell Images Dataset...")
print("This may take a few minutes (~340MB)...\n")

path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
print(f"✅ Downloaded to: {path}")

# ──────────────────────────────────────────────
#  COPY cell_images TO YOUR PROJECT FOLDER
# ──────────────────────────────────────────────

# Your project folder (where this script is running from)
project_dir  = os.path.dirname(os.path.abspath(__file__))
destination  = os.path.join(project_dir, "cell_images")

# Find cell_images inside downloaded path
src = os.path.join(path, "cell_images")

if not os.path.exists(src):
    # Sometimes files are directly in path
    print(f"\n📁 Files found at: {os.listdir(path)}")
    print("Please check the path above and set src manually.")
else:
    if os.path.exists(destination):
        print(f"\n⚠️  'cell_images' folder already exists at:\n   {destination}")
        print("Skipping copy. Delete it first if you want to re-copy.")
    else:
        print(f"\nCopying to project folder...")
        shutil.copytree(src, destination)
        print(f"✅ Copied to: {destination}")

# ──────────────────────────────────────────────
#  VERIFY
# ──────────────────────────────────────────────
p_path = os.path.join(project_dir, "cell_images", "Parasitized")
u_path = os.path.join(project_dir, "cell_images", "Uninfected")

p_count = len([f for f in os.listdir(p_path) if f.endswith(".png")]) if os.path.exists(p_path) else 0
u_count = len([f for f in os.listdir(u_path) if f.endswith(".png")]) if os.path.exists(u_path) else 0

print(f"\n📊 Dataset Summary:")
print(f"   Parasitized images : {p_count}")
print(f"   Uninfected  images : {u_count}")
print(f"   Total              : {p_count + u_count}")

if p_count > 0 and u_count > 0:
    print("\n✅ Dataset ready! Now run: python step2_extract.py")
else:
    print("\n❌ Something went wrong. Check the cell_images folder manually.")
