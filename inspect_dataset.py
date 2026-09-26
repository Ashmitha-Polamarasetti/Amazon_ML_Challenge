import os
import glob
import pandas as pd
import platform
import psutil

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"

# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)

print("Python:", platform.python_version())
print("Operating System:", platform.system(), platform.release())

try:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM: {ram_gb:.2f} GB")
except Exception:
    print("RAM information unavailable")

print()

# ============================================================
# CHECK DATASET DIRECTORY
# ============================================================

print("=" * 70)
print("DATASET DIRECTORY")
print("=" * 70)

print("Dataset path:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset path does not exist!")
    print("Check the path carefully.")
    raise SystemExit

print("\nDataset path exists: YES")
print()

# ============================================================
# FIND TSV FILES
# ============================================================

files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "train", "*.tsv"))
    + glob.glob(os.path.join(DATASET_PATH, "test", "*.tsv"))
)

print("=" * 70)
print("TSV FILES")
print("=" * 70)

for file in files:
    size_mb = os.path.getsize(file) / (1024 ** 2)

    print()
    print("File:", os.path.basename(file))
    print("Path:", file)
    print(f"Size: {size_mb:.2f} MB")

print()

# ============================================================
# INSPECT EACH FILE
# ============================================================

for file in files:

    print("\n")
    print("=" * 70)
    print("FILE:", os.path.basename(file))
    print("=" * 70)

    try:
        # Only read the first 5 rows.
        # This is safe even for very large files.
        df = pd.read_csv(
            file,
            sep="\t",
            nrows=5
        )

        print("\nColumns:")
        print(list(df.columns))

        print("\nFirst 5 rows:")
        print(df.to_string(index=False))

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values in first 5 rows:")
        print(df.isnull().sum())

        # Show unique countries in the sample
        if "country" in df.columns:
            print("\nCountries appearing in first 5 rows:")
            print(df["country"].tolist())

    except Exception as e:
        print("\nERROR while reading file:")
        print(e)

# ============================================================
# EXPECTED STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("EXPECTED DATASET STRUCTURE")
print("=" * 70)

print("""
dataset/
|
+-- train/
|   +-- train_source1.tsv
|   +-- train_source2.tsv
|   +-- train_source3.tsv
|   +-- train_ground_truth.tsv
|
+-- test/
    +-- test_source1.tsv
    +-- test_source2.tsv
    +-- test_source3.tsv
""")

print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)