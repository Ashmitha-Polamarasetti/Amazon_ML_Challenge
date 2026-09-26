import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
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

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # punctuation -> spaces
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # collapse repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ADDRESS TOKEN FUNCTION
# ============================================================

def useful_address_tokens(address):
    """
    Extract useful tokens from an address.

    Very short tokens are ignored to avoid
    creating enormous candidate blocks.
    """

    address = normalize(address)

    if not address:
        return []

    tokens = address.split()

    result = []

    for token in tokens:

        # Keep numbers such as 12, 45, 1234
        if token.isdigit():
            result.append(token)

        # Keep reasonably informative words
        elif len(token) >= 5:
            result.append(token)

    # Remove duplicate tokens
    return list(set(result))


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(file_path):

    print("\n" + "=" * 70)
    print(f"Building index: {file_path.name}")
    print("=" * 70)

    name_index = defaultdict(list)
    address_index = defaultdict(list)

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

            country = row["country"]

            name = normalize(
                row["business_name"]
            )

            address = row["business_address"] or ""

            # ----------------------------------------
            # Exact normalized name index
            # ----------------------------------------

            if name:

                name_key = (
                    country,
                    name
                )

                name_index[name_key].append(
                    entity_id
                )

            # ----------------------------------------
            # Address token index
            # ----------------------------------------

            tokens = useful_address_tokens(
                address
            )

            for token in tokens:

                address_key = (
                    country,
                    token
                )

                address_index[address_key].append(
                    entity_id
                )

            count += 1

            if count % 1_000_000 == 0:

                print(
                    f"Processed {count:,} rows..."
                )

    print(
        f"Finished {file_path.name}"
    )

    print(
        f"Rows: {count:,}"
    )

    print(
        f"Unique name blocks: "
        f"{len(name_index):,}"
    )

    print(
        f"Unique address-token blocks: "
        f"{len(address_index):,}"
    )

    return name_index, address_index


# ============================================================
# GROUND TRUTH
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

            s1_id = row[
                "source1_entity_id"
            ]

            matches_text = (
                row["matched_entity_ids"]
                or ""
            ).strip()

            if matches_text:

                matches = {
                    x.strip()
                    for x
                    in matches_text.split(",")
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
        f"Ground-truth rows: "
        f"{len(truth):,}"
    )

    return truth


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    s2_name,
    s2_address,
    s3_name,
    s3_address,
    truth
):

    print("\n" + "=" * 70)
    print("Evaluating Baseline V2")
    print("=" * 70)

    total_s1 = 0

    total_true = 0

    recovered = 0

    total_candidates = 0

    s1_with_candidates = 0

    s1_with_truth = 0


    # Candidate cap prevents giant blocks from
    # consuming huge amounts of RAM/time.

    MAX_BLOCK_SIZE = 200


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

            country = row["country"]

            name = normalize(
                row["business_name"]
            )

            address = (
                row["business_address"]
                or ""
            )

            candidates = set()


            # =================================================
            # STRATEGY 1
            # EXACT NORMALIZED NAME
            # =================================================

            if name:

                key = (
                    country,
                    name
                )

                candidates.update(
                    s2_name.get(
                        key,
                        []
                    )
                )

                candidates.update(
                    s3_name.get(
                        key,
                        []
                    )
                )


            # =================================================
            # STRATEGY 2
            # ADDRESS TOKEN BLOCKING
            # =================================================

            tokens = useful_address_tokens(
                address
            )

            for token in tokens:

                key = (
                    country,
                    token
                )

                block2 = s2_address.get(
                    key,
                    []
                )

                block3 = s3_address.get(
                    key,
                    []
                )

                # Ignore extremely common tokens.
                #
                # Example:
                # "road", "street", etc.
                #
                # Large blocks would create too many
                # meaningless candidates.

                if len(block2) <= MAX_BLOCK_SIZE:

                    candidates.update(
                        block2
                    )

                if len(block3) <= MAX_BLOCK_SIZE:

                    candidates.update(
                        block3
                    )


            # =================================================
            # GROUND TRUTH
            # =================================================

            actual = truth.get(
                s1_id,
                set()
            )

            total_s1 += 1

            total_true += len(actual)

            total_candidates += len(
                candidates
            )

            recovered += len(
                actual.intersection(
                    candidates
                )
            )

            if actual:
                s1_with_truth += 1

            if candidates:
                s1_with_candidates += 1


            # =================================================
            # PROGRESS
            # =================================================

            if total_s1 % 500_000 == 0:

                recall = (
                    recovered /
                    total_true
                    if total_true
                    else 0
                )

                avg_candidates = (
                    total_candidates /
                    total_s1
                )

                print(
                    f"Processed "
                    f"{total_s1:,} S1 | "
                    f"Recall = "
                    f"{recall:.4%} | "
                    f"Avg candidates = "
                    f"{avg_candidates:.2f}"
                )


    # =========================================================
    # FINAL RESULTS
    # =========================================================

    recall = (
        recovered /
        total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates /
        total_s1
        if total_s1
        else 0
    )


    print("\n" + "=" * 70)

    print("BASELINE V2 RESULTS")

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
        f"S1 receiving candidates: "
        f"{s1_with_candidates:,}"
    )

    print(
        f"Total true matches: "
        f"{total_true:,}"
    )

    print(
        f"Recovered true matches: "
        f"{recovered:,}"
    )

    print(
        f"Candidate Recall: "
        f"{recall:.4%}"
    )

    print(
        f"Total candidates: "
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

    print(
        "AMAZON ML CHALLENGE 2026"
    )

    print(
        "CANDIDATE BASELINE V2"
    )

    print("=" * 70)

    print(
        "\nStrategies:"
    )

    print(
        "1. Exact normalized business name"
    )

    print(
        "2. Informative address-token blocking"
    )

    print(
        "3. Same-country blocking"
    )


    # S2

    s2_name, s2_address = build_index(
        S2_FILE
    )


    # S3

    s3_name, s3_address = build_index(
        S3_FILE
    )


    # Ground truth

    truth = load_ground_truth()


    # Evaluate

    evaluate(
        s2_name,
        s2_address,
        s3_name,
        s3_address,
        truth
    )


if __name__ == "__main__":
    main()