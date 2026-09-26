import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


# ============================================================
# DATASET PATH
# ============================================================

DATASET = Path(
    r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"
)

TRAIN = DATASET / "train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Basic normalization of business names.

    Example:
    "ABC Pvt. Ltd." -> "abc pvt ltd"
    """

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BUILD EXACT-NAME INDEX
# ============================================================

def build_exact_name_index(file_path):

    print("\n" + "=" * 70)
    print(f"Building index: {file_path.name}")
    print("=" * 70)

    index = defaultdict(list)

    count = 0

    with open(
        file_path,
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

            entity_id = row["entity_id"]

            business_name = normalize(
                row["business_name"]
            )

            country = row["country"]

            # Only index records that have a name
            if business_name:

                key = (
                    country,
                    business_name
                )

                index[key].append(entity_id)

            count += 1

            if count % 1_000_000 == 0:

                print(
                    f"Processed {count:,} rows..."
                )

    print(
        f"Finished {file_path.name}"
    )

    print(
        f"Total rows: {count:,}"
    )

    print(
        f"Unique name blocks: {len(index):,}"
    )

    return index


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\n" + "=" * 70)
    print("Loading ground truth")
    print("=" * 70)

    truth = {}

    count = 0

    with open(
        GT_FILE,
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

            s1_id = row["source1_entity_id"]

            matches_text = (
                row["matched_entity_ids"]
                or ""
            ).strip()

            if matches_text:

                matches = {
                    x.strip()
                    for x in matches_text.split(",")
                    if x.strip()
                }

            else:

                matches = set()

            truth[s1_id] = matches

            count += 1

            if count % 500_000 == 0:

                print(
                    f"Loaded {count:,} "
                    f"ground-truth rows..."
                )

    print(
        f"Ground-truth rows loaded: "
        f"{len(truth):,}"
    )

    return truth


# ============================================================
# EVALUATE CANDIDATE GENERATION
# ============================================================

def evaluate_candidates(
    s2_index,
    s3_index,
    truth
):

    print("\n" + "=" * 70)
    print("Evaluating candidate generation")
    print("=" * 70)

    total_s1 = 0

    s1_with_truth = 0

    s1_with_candidates = 0

    total_true_matches = 0

    recovered_true_matches = 0

    total_candidates = 0


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

            s1_id = row["entity_id"]

            business_name = normalize(
                row["business_name"]
            )

            country = row["country"]

            key = (
                country,
                business_name
            )

            candidates = set()

            # Get exact-name candidates
            if business_name:

                candidates.update(
                    s2_index.get(
                        key,
                        []
                    )
                )

                candidates.update(
                    s3_index.get(
                        key,
                        []
                    )
                )

            # Get actual true matches
            actual = truth.get(
                s1_id,
                set()
            )

            # Statistics
            total_s1 += 1

            total_true_matches += len(actual)

            total_candidates += len(candidates)

            recovered_true_matches += len(
                actual.intersection(candidates)
            )

            if actual:
                s1_with_truth += 1

            if candidates:
                s1_with_candidates += 1


            # Progress display
            if total_s1 % 500_000 == 0:

                current_recall = (
                    recovered_true_matches
                    / total_true_matches
                    if total_true_matches
                    else 0
                )

                avg_candidates = (
                    total_candidates
                    / total_s1
                )

                print(
                    f"Processed {total_s1:,} S1 | "
                    f"Recall = "
                    f"{current_recall:.4%} | "
                    f"Avg candidates = "
                    f"{avg_candidates:.2f}"
                )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    candidate_recall = (
        recovered_true_matches
        / total_true_matches
        if total_true_matches
        else 0
    )

    avg_candidates = (
        total_candidates
        / total_s1
        if total_s1
        else 0
    )


    print("\n" + "=" * 70)

    print("BASELINE RESULTS")

    print("=" * 70)


    print(
        f"S1 records evaluated: "
        f"{total_s1:,}"
    )

    print(
        f"S1 records with true matches: "
        f"{s1_with_truth:,}"
    )

    print(
        f"S1 records receiving candidates: "
        f"{s1_with_candidates:,}"
    )

    print(
        f"Total true matches: "
        f"{total_true_matches:,}"
    )

    print(
        f"Recovered true matches: "
        f"{recovered_true_matches:,}"
    )

    print(
        f"Candidate Recall: "
        f"{candidate_recall:.4%}"
    )

    print(
        f"Total generated candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates per S1: "
        f"{avg_candidates:.2f}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("CANDIDATE GENERATION BASELINE")
    print("=" * 70)

    print(
        "\nStrategy: "
        "Exact normalized business name "
        "+ same country"
    )


    # STEP 1
    # Build S2 index

    s2_index = build_exact_name_index(
        S2_FILE
    )


    # STEP 2
    # Build S3 index

    s3_index = build_exact_name_index(
        S3_FILE
    )


    # STEP 3
    # Load known training matches

    truth = load_ground_truth()


    # STEP 4
    # Evaluate candidate recall

    evaluate_candidates(
        s2_index,
        s3_index,
        truth
    )


if __name__ == "__main__":
    main()