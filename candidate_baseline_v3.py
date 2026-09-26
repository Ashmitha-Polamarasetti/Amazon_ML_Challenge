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

S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

VALIDATION_DIR = Path(
    r"D:\AmazonMLChallenge\validation_sample"
)

S1_FILE = VALIDATION_DIR / "validation_source1.tsv"
GT_FILE = VALIDATION_DIR / "validation_ground_truth.tsv"


# ============================================================
# SETTINGS
# ============================================================

MAX_ADDRESS_BLOCK = 200
MAX_SIGNATURE_BLOCK = 100


# Common legal/business suffixes.
# We remove these ONLY for the additional signature.
# The original exact-name strategy remains unchanged.

LEGAL_WORDS = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "pvt",
    "private",
    "llp",
    "lp",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NAME SIGNATURE
# ============================================================

def name_signature(name):

    name = normalize(name)

    if not name:
        return ""

    tokens = []

    for token in name.split():

        if token in LEGAL_WORDS:
            continue

        # Ignore one-character noise
        if len(token) <= 1:
            continue

        tokens.append(token)

    # Remove duplicates and sort.
    #
    # Example:
    #
    # "Lakshmi Medical Center"
    # "Medical Center Lakshmi"
    #
    # both become:
    #
    # center lakshmi medical

    tokens = sorted(set(tokens))

    return " ".join(tokens)


# ============================================================
# ADDRESS TOKENS
# ============================================================

def useful_address_tokens(address):

    address = normalize(address)

    if not address:
        return []

    result = set()

    for token in address.split():

        if token.isdigit():

            result.add(token)

        elif len(token) >= 5:

            result.add(token)

    return result


# ============================================================
# BUILD SOURCE INDEXES
# ============================================================

def build_index(file_path):

    print("\n" + "=" * 70)
    print(f"Building index: {file_path.name}")
    print("=" * 70)

    name_index = defaultdict(list)

    signature_index = defaultdict(list)

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

            raw_name = (
                row["business_name"]
                or ""
            )

            raw_address = (
                row["business_address"]
                or ""
            )

            # -----------------------------------------
            # Strategy 1 index:
            # exact normalized name
            # -----------------------------------------

            norm_name = normalize(
                raw_name
            )

            if norm_name:

                name_index[
                    (
                        country,
                        norm_name
                    )
                ].append(entity_id)

            # -----------------------------------------
            # Strategy 2 index:
            # business-name signature
            # -----------------------------------------

            signature = name_signature(
                raw_name
            )

            if signature:

                signature_index[
                    (
                        country,
                        signature
                    )
                ].append(entity_id)

            # -----------------------------------------
            # Strategy 3 index:
            # address tokens
            # -----------------------------------------

            for token in useful_address_tokens(
                raw_address
            ):

                address_index[
                    (
                        country,
                        token
                    )
                ].append(entity_id)

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
        f"Unique exact-name blocks: "
        f"{len(name_index):,}"
    )

    print(
        f"Unique name-signature blocks: "
        f"{len(signature_index):,}"
    )

    print(
        f"Unique address blocks: "
        f"{len(address_index):,}"
    )

    return (
        name_index,
        signature_index,
        address_index
    )


# ============================================================
# LOAD VALIDATION GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\n" + "=" * 70)
    print("Loading validation ground truth")
    print("=" * 70)

    truth = {}

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

            text = (
                row["matched_entity_ids"]
                or ""
            ).strip()

            if text:

                truth[s1_id] = {
                    x.strip()
                    for x in text.split(",")
                    if x.strip()
                }

            else:

                truth[s1_id] = set()

    print(
        f"Ground-truth rows loaded: "
        f"{len(truth):,}"
    )

    return truth


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    s2_name,
    s2_signature,
    s2_address,
    s3_name,
    s3_signature,
    s3_address,
    truth
):

    print("\n" + "=" * 70)
    print("Evaluating Candidate Baseline V3")
    print("=" * 70)

    total_s1 = 0

    total_true = 0

    recovered = 0

    total_candidates = 0

    s1_with_candidates = 0

    s1_with_truth = 0


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

            raw_name = (
                row["business_name"]
                or ""
            )

            raw_address = (
                row["business_address"]
                or ""
            )

            candidates = set()


            # =================================================
            # 1. EXACT NORMALIZED NAME
            # =================================================

            norm_name = normalize(
                raw_name
            )

            if norm_name:

                key = (
                    country,
                    norm_name
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
            # 2. NAME SIGNATURE
            # =================================================

            signature = name_signature(
                raw_name
            )

            if signature:

                key = (
                    country,
                    signature
                )

                block2 = s2_signature.get(
                    key,
                    []
                )

                block3 = s3_signature.get(
                    key,
                    []
                )

                # Avoid extremely common names.

                if (
                    len(block2)
                    <= MAX_SIGNATURE_BLOCK
                ):

                    candidates.update(
                        block2
                    )

                if (
                    len(block3)
                    <= MAX_SIGNATURE_BLOCK
                ):

                    candidates.update(
                        block3
                    )


            # =================================================
            # 3. ADDRESS TOKEN BLOCKING
            # =================================================

            for token in useful_address_tokens(
                raw_address
            ):

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

                if (
                    len(block2)
                    <= MAX_ADDRESS_BLOCK
                ):

                    candidates.update(
                        block2
                    )

                if (
                    len(block3)
                    <= MAX_ADDRESS_BLOCK
                ):

                    candidates.update(
                        block3
                    )


            # =================================================
            # EVALUATE
            # =================================================

            actual = truth.get(
                s1_id,
                set()
            )

            total_s1 += 1

            total_true += len(actual)

            recovered += len(
                actual.intersection(
                    candidates
                )
            )

            total_candidates += len(
                candidates
            )

            if actual:
                s1_with_truth += 1

            if candidates:
                s1_with_candidates += 1


            if total_s1 % 10_000 == 0:

                recall = (
                    recovered / total_true
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
    # RESULTS
    # =========================================================

    recall = (
        recovered / total_true
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
    print("BASELINE V3 RESULTS")
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
    print("AMAZON ML CHALLENGE 2026")
    print("CANDIDATE BASELINE V3")
    print("=" * 70)

    print("\nValidation S1: 100,000")

    print("\nStrategies:")
    print("1. Exact normalized business name")
    print("2. Legal-suffix-free sorted name signature")
    print("3. Informative address tokens")
    print("4. Same-country blocking")

    print(
        f"\nMax signature block: "
        f"{MAX_SIGNATURE_BLOCK}"
    )

    print(
        f"Max address block: "
        f"{MAX_ADDRESS_BLOCK}"
    )


    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    (
        s2_name,
        s2_signature,
        s2_address
    ) = build_index(
        S2_FILE
    )


    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    (
        s3_name,
        s3_signature,
        s3_address
    ) = build_index(
        S3_FILE
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    truth = load_ground_truth()


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate(
        s2_name,
        s2_signature,
        s2_address,
        s3_name,
        s3_signature,
        s3_address,
        truth
    )


if __name__ == "__main__":
    main()