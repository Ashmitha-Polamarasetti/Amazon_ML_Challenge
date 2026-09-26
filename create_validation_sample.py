import csv
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"
)

TRAIN = DATASET / "train"

S1_FILE = TRAIN / "train_source1.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

OUTPUT_DIR = Path(
    r"D:\AmazonMLChallenge\validation_sample"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_SIZE = 100_000
RANDOM_SEED = 42


# ============================================================
# STEP 1: SAMPLE S1 IDS
# ============================================================

def sample_s1_ids():

    print("=" * 70)
    print("STEP 1: Sampling S1 records")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    reservoir = []

    total = 0

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            total += 1

            if len(reservoir) < SAMPLE_SIZE:

                reservoir.append(
                    row["entity_id"]
                )

            else:

                j = random.randint(
                    1,
                    total
                )

                if j <= SAMPLE_SIZE:

                    reservoir[j - 1] = (
                        row["entity_id"]
                    )

            if total % 500_000 == 0:

                print(
                    f"Scanned {total:,} S1 rows..."
                )

    sampled_ids = set(reservoir)

    print()
    print(
        f"Total S1 scanned: {total:,}"
    )

    print(
        f"Sampled S1 IDs: "
        f"{len(sampled_ids):,}"
    )

    return sampled_ids


# ============================================================
# STEP 2: WRITE SAMPLED S1 FILE
# ============================================================

def write_sample_s1(sampled_ids):

    print("\n" + "=" * 70)
    print("STEP 2: Creating validation_source1.tsv")
    print("=" * 70)

    output_file = (
        OUTPUT_DIR /
        "validation_source1.tsv"
    )

    written = 0

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as infile, open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        reader = csv.DictReader(
            infile,
            delimiter="\t"
        )

        writer = csv.DictWriter(
            outfile,
            fieldnames=reader.fieldnames,
            delimiter="\t"
        )

        writer.writeheader()

        for row in reader:

            if row["entity_id"] in sampled_ids:

                writer.writerow(row)

                written += 1

    print(
        f"Written: {written:,} S1 records"
    )

    print(
        f"Output: {output_file}"
    )


# ============================================================
# STEP 3: WRITE MATCHING GROUND TRUTH
# ============================================================

def write_sample_ground_truth(
    sampled_ids
):

    print("\n" + "=" * 70)
    print("STEP 3: Creating validation_ground_truth.tsv")
    print("=" * 70)

    output_file = (
        OUTPUT_DIR /
        "validation_ground_truth.tsv"
    )

    written = 0
    singleton_count = 0
    total_matches = 0

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as infile, open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        reader = csv.DictReader(
            infile,
            delimiter="\t"
        )

        writer = csv.DictWriter(
            outfile,
            fieldnames=reader.fieldnames,
            delimiter="\t"
        )

        writer.writeheader()

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if s1_id not in sampled_ids:
                continue

            writer.writerow(row)

            written += 1

            matches = (
                row["matched_entity_ids"]
                or ""
            ).strip()

            if not matches:

                singleton_count += 1

            else:

                total_matches += len(
                    [
                        x
                        for x in matches.split(",")
                        if x.strip()
                    ]
                )

    print(
        f"Ground-truth rows: "
        f"{written:,}"
    )

    print(
        f"Singleton S1: "
        f"{singleton_count:,}"
    )

    print(
        f"True matches represented: "
        f"{total_matches:,}"
    )

    print(
        f"Output: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("VALIDATION SAMPLE CREATOR")
    print("=" * 70)

    print(
        f"\nSample size: {SAMPLE_SIZE:,}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    sampled_ids = sample_s1_ids()

    write_sample_s1(
        sampled_ids
    )

    write_sample_ground_truth(
        sampled_ids
    )

    print("\n" + "=" * 70)
    print("VALIDATION SAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()