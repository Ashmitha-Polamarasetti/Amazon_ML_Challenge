import os
import csv
from collections import Counter

DATASET_PATH = r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"


def analyze_source_file(path):
    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(path))
    print("=" * 70)

    row_count = 0
    countries = Counter()
    missing_name = 0
    missing_address = 0
    missing_country = 0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        print("Columns:", reader.fieldnames)

        for row in reader:
            row_count += 1

            country = (row.get("country") or "").strip()

            if country:
                countries[country] += 1
            else:
                missing_country += 1

            if not (row.get("business_name") or "").strip():
                missing_name += 1

            if not (row.get("business_address") or "").strip():
                missing_address += 1

    print("Rows:", f"{row_count:,}")
    print("Missing business names:", f"{missing_name:,}")
    print("Missing addresses:", f"{missing_address:,}")
    print("Missing countries:", f"{missing_country:,}")

    print("\nCountries:")
    for country, count in countries.most_common():
        print(f"  {country}: {count:,}")


def analyze_ground_truth(path):
    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)

    row_count = 0
    singleton_count = 0
    total_matches = 0

    source2_matches = 0
    source3_matches = 0

    matches_per_s1 = Counter()

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        print("Columns:", reader.fieldnames)

        for row in reader:
            row_count += 1

            matched = (row.get("matched_entity_ids") or "").strip()

            if not matched:
                singleton_count += 1
                matches_per_s1[0] += 1
                continue

            ids = [x.strip() for x in matched.split(",") if x.strip()]

            total_matches += len(ids)
            matches_per_s1[len(ids)] += 1

            for entity_id in ids:
                if entity_id.startswith("S2-"):
                    source2_matches += 1
                elif entity_id.startswith("S3-"):
                    source3_matches += 1

    print("Source 1 ground-truth rows:", f"{row_count:,}")
    print("Singletons / no-match S1:", f"{singleton_count:,}")
    print("Total listed matches:", f"{total_matches:,}")
    print("S2 matches:", f"{source2_matches:,}")
    print("S3 matches:", f"{source3_matches:,}")

    print("\nDistribution of number of matches per S1:")
    for number, count in sorted(matches_per_s1.items()):
        print(f"  {number} matches: {count:,}")


# ============================================================
# TRAINING DATA
# ============================================================

analyze_source_file(
    os.path.join(DATASET_PATH, "train", "train_source1.tsv")
)

analyze_source_file(
    os.path.join(DATASET_PATH, "train", "train_source2.tsv")
)

analyze_source_file(
    os.path.join(DATASET_PATH, "train", "train_source3.tsv")
)

analyze_ground_truth(
    os.path.join(DATASET_PATH, "train", "train_ground_truth.tsv")
)


# ============================================================
# TEST DATA
# ============================================================

analyze_source_file(
    os.path.join(DATASET_PATH, "test", "test_source1.tsv")
)

analyze_source_file(
    os.path.join(DATASET_PATH, "test", "test_source2.tsv")
)

analyze_source_file(
    os.path.join(DATASET_PATH, "test", "test_source3.tsv")
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)