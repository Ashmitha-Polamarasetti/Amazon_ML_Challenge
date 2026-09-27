from pathlib import Path
import zipfile

ROOT = Path(r"D:\AmazonMLChallenge")

FINAL = ROOT / "final_submission"
OUTPUT = ROOT / "output"
ZIP_PATH = ROOT / "Mission_Impossible_submission.zip"

files = []

# Documentation
files.append((
    FINAL / "Documentation_template.md",
    "Documentation_template.md"
))

# Output files - use existing files directly
files.append((
    OUTPUT / "matching_results.tsv",
    "output/matching_results.tsv"
))

files.append((
    OUTPUT / "candidate_pairs.tsv",
    "output/candidate_pairs.tsv"
))

# Code files
code_root = FINAL / "code" / "business_entity_resolution"

for path in code_root.rglob("*"):
    if path.is_file():
        relative = path.relative_to(FINAL)
        files.append((path, relative.as_posix()))

print("=" * 70)
print("AMAZON ML CHALLENGE 2026 - FINAL PACKAGING")
print("=" * 70)

# Verify everything before starting
for source, archive_name in files:
    if not source.exists():
        raise FileNotFoundError(f"Missing: {source}")

print(f"Files to package: {len(files)}")
print(f"ZIP: {ZIP_PATH}")
print()

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1,
    allowZip64=True
) as zf:

    for i, (source, archive_name) in enumerate(files, 1):
        size_mb = source.stat().st_size / (1024 * 1024)

        print(
            f"[{i}/{len(files)}] Adding {archive_name} "
            f"({size_mb:.1f} MB)"
        )

        zf.write(source, archive_name)

print()
print("=" * 70)
print("ZIP CREATED SUCCESSFULLY")
print("=" * 70)
print(ZIP_PATH)
print(f"ZIP size: {ZIP_PATH.stat().st_size / (1024**3):.2f} GB")