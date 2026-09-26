import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio


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

# V3 settings
MAX_ADDRESS_BLOCK = 200
MAX_SIGNATURE_BLOCK = 100

# V4 settings
MAX_FUZZY_BLOCK = 300
FUZZY_THRESHOLD = 85
PREFIX_LENGTH = 3


# ============================================================
# COMMON LEGAL / BUSINESS WORDS
# ============================================================

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
    """
    Basic text normalization.

    Example:
        "ABC Pvt. Ltd."
    becomes:
        "abc pvt ltd"
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    # Convert punctuation to spaces
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Collapse repeated spaces
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
    """
    Removes common legal words and sorts the remaining
    business-name tokens.

    Example:

        "Lakshmi Medical Center Pvt Ltd"

    becomes:

        "center lakshmi medical"
    """

    name = normalize(name)

    if not name:
        return ""

    tokens = []

    for token in name.split():

        # Remove legal/business suffixes
        if token in LEGAL_WORDS:
            continue

        # Ignore one-character noise
        if len(token) <= 1:
            continue

        tokens.append(token)

    # Remove duplicate tokens
    # and ignore word ordering
    tokens = sorted(set(tokens))

    return " ".join(tokens)


# ============================================================
# FUZZY BLOCK KEY
# ============================================================

def fuzzy_block_key(name, country):
    """
    Creates a small blocking key before fuzzy matching.

    We do NOT fuzzy compare an S1 record against millions
    of S2/S3 records.

    Instead:
        name
          ->
        remove legal words
          ->
        sort useful tokens
          ->
        first token
          ->
        first 3 characters
          ->
        same-country fuzzy block
    """

    name = normalize(name)

    if not name:
        return None

    tokens = []

    for token in name.split():

        if token in LEGAL_WORDS:
            continue

        if len(token) < 3:
            continue

        tokens.append(token)

    if not tokens:
        return None

    # Sorting makes the block more resistant
    # to word-order changes.
    tokens.sort()

    first_token = tokens[0]

    prefix = first_token[:PREFIX_LENGTH]

    if not prefix:
        return None

    return (
        country,
        prefix
    )


# ============================================================
# ADDRESS TOKENS
# ============================================================

def useful_address_tokens(address):
    """
    Extract reasonably informative address tokens.

    Numbers are retained.

    Words shorter than 5 characters are ignored because
    very short words can create huge candidate blocks.
    """

    address = normalize(address)

    if not address:
        return set()

    result = set()

    for token in address.split():

        if token.isdigit():

            result.add(token)

        elif len(token) >= 5:

            result.add(token)

    return result


# ============================================================
# BUILD S2 / S3 INDEX
# ============================================================

def build_index(file_path):

    print("\n" + "=" * 70)
    print(f"Building index: {file_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # V1 index
    # --------------------------------------------------------

    name_index = defaultdict(list)

    # --------------------------------------------------------
    # V3 index
    # --------------------------------------------------------

    signature_index = defaultdict(list)

    # --------------------------------------------------------
    # V2 index
    # --------------------------------------------------------

    address_index = defaultdict(list)

    # --------------------------------------------------------
    # NEW V4 index
    #
    # Each entry stores:
    #
    # (entity_id, normalized_business_name)
    #
    # because we need the name for RapidFuzz scoring later.
    # --------------------------------------------------------

    fuzzy_index = defaultdict(list)

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

            country = (
                row["country"]
                or ""
            )

            raw_name = (
                row["business_name"]
                or ""
            )

            raw_address = (
                row["business_address"]
                or ""
            )

            # =================================================
            # INDEX 1:
            # EXACT NORMALIZED NAME
            # =================================================

            norm_name = normalize(
                raw_name
            )

            if norm_name:

                name_key = (
                    country,
                    norm_name
                )

                name_index[
                    name_key
                ].append(
                    entity_id
                )


            # =================================================
            # INDEX 2:
            # NAME SIGNATURE
            # =================================================

            signature = name_signature(
                raw_name
            )

            if signature:

                signature_key = (
                    country,
                    signature
                )

                signature_index[
                    signature_key
                ].append(
                    entity_id
                )


            # =================================================
            # INDEX 3:
            # ADDRESS TOKENS
            # =================================================

            address_tokens = useful_address_tokens(
                raw_address
            )

            for token in address_tokens:

                address_key = (
                    country,
                    token
                )

                address_index[
                    address_key
                ].append(
                    entity_id
                )


            # =================================================
            # INDEX 4:
            # FUZZY NAME BLOCK
            # =================================================

            fuzzy_key = fuzzy_block_key(
                raw_name,
                country
            )

            if fuzzy_key:

                fuzzy_index[
                    fuzzy_key
                ].append(
                    (
                        entity_id,
                        norm_name
                    )
                )


            # =================================================
            # PROGRESS
            # =================================================

            count += 1

            if count % 1_000_000 == 0:

                print(
                    f"Processed "
                    f"{count:,} rows..."
                )


    # =========================================================
    # INDEX SUMMARY
    # =========================================================

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

    print(
        f"Unique fuzzy blocks: "
        f"{len(fuzzy_index):,}"
    )

    return (
        name_index,
        signature_index,
        address_index,
        fuzzy_index
    )


# ============================================================
# LOAD VALIDATION GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\n" + "=" * 70)
    print("Loading validation ground truth")
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


    print(
        f"Ground-truth rows loaded: "
        f"{count:,}"
    )

    return truth


# ============================================================
# EVALUATE V4
# ============================================================

def evaluate(
    s2_name,
    s2_signature,
    s2_address,
    s2_fuzzy,
    s3_name,
    s3_signature,
    s3_address,
    s3_fuzzy,
    truth
):

    print("\n" + "=" * 70)
    print("Evaluating Candidate Baseline V4")
    print("=" * 70)

    total_s1 = 0

    total_true = 0

    recovered = 0

    total_candidates = 0

    s1_with_candidates = 0

    s1_with_truth = 0


    # --------------------------------------------------------
    # Extra diagnostics
    # --------------------------------------------------------

    fuzzy_blocks_used = 0

    fuzzy_blocks_skipped = 0

    fuzzy_candidates_added = 0


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

            country = (
                row["country"]
                or ""
            )

            raw_name = (
                row["business_name"]
                or ""
            )

            raw_address = (
                row["business_address"]
                or ""
            )

            norm_name = normalize(
                raw_name
            )

            candidates = set()


            # =================================================
            # STRATEGY 1:
            # EXACT NORMALIZED BUSINESS NAME
            # =================================================

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
            # STRATEGY 2:
            # LEGAL-SUFFIX-FREE NAME SIGNATURE
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

                # Ignore huge ambiguous blocks.

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
            # STRATEGY 3:
            # PREFIX-BLOCKED FUZZY NAME MATCHING
            # =================================================

            fuzzy_key = fuzzy_block_key(
                raw_name,
                country
            )

            if fuzzy_key and norm_name:

                block2 = s2_fuzzy.get(
                    fuzzy_key,
                    []
                )

                block3 = s3_fuzzy.get(
                    fuzzy_key,
                    []
                )

                fuzzy_records = []


                # ---------------------------------------------
                # S2 fuzzy block
                # ---------------------------------------------

                if (
                    len(block2)
                    <= MAX_FUZZY_BLOCK
                ):

                    fuzzy_records.extend(
                        block2
                    )

                    if block2:
                        fuzzy_blocks_used += 1

                else:

                    fuzzy_blocks_skipped += 1


                # ---------------------------------------------
                # S3 fuzzy block
                # ---------------------------------------------

                if (
                    len(block3)
                    <= MAX_FUZZY_BLOCK
                ):

                    fuzzy_records.extend(
                        block3
                    )

                    if block3:
                        fuzzy_blocks_used += 1

                else:

                    fuzzy_blocks_skipped += 1


                # ---------------------------------------------
                # Fuzzy scoring
                # ---------------------------------------------

                for (
                    entity_id,
                    candidate_name
                ) in fuzzy_records:

                    score = token_set_ratio(
                        norm_name,
                        candidate_name
                    )

                    if (
                        score
                        >= FUZZY_THRESHOLD
                    ):

                        old_size = len(
                            candidates
                        )

                        candidates.add(
                            entity_id
                        )

                        if (
                            len(candidates)
                            > old_size
                        ):

                            fuzzy_candidates_added += 1


            # =================================================
            # STRATEGY 4:
            # ADDRESS TOKEN BLOCKING
            # =================================================

            address_tokens = useful_address_tokens(
                raw_address
            )

            for token in address_tokens:

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


                # Ignore very large address blocks.

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
            # GROUND TRUTH
            # =================================================

            actual = truth.get(
                s1_id,
                set()
            )

            total_s1 += 1

            total_true += len(
                actual
            )

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

            if (
                total_s1
                % 10_000
                == 0
            ):

                recall = (
                    recovered
                    / total_true
                    if total_true
                    else 0
                )

                avg_candidates = (
                    total_candidates
                    / total_s1
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
        recovered
        / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates
        / total_s1
        if total_s1
        else 0
    )


    print("\n" + "=" * 70)

    print(
        "BASELINE V4 RESULTS"
    )

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


    print("\n" + "-" * 70)

    print(
        "FUZZY RETRIEVAL DIAGNOSTICS"
    )

    print("-" * 70)

    print(
        f"Fuzzy blocks used: "
        f"{fuzzy_blocks_used:,}"
    )

    print(
        f"Fuzzy blocks skipped "
        f"(>{MAX_FUZZY_BLOCK}): "
        f"{fuzzy_blocks_skipped:,}"
    )

    print(
        f"New fuzzy candidates added: "
        f"{fuzzy_candidates_added:,}"
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
        "CANDIDATE BASELINE V4"
    )

    print("=" * 70)


    print(
        "\nValidation S1: 100,000"
    )


    print(
        "\nStrategies:"
    )

    print(
        "1. Exact normalized business name"
    )

    print(
        "2. Legal-suffix-free sorted name signature"
    )

    print(
        "3. Prefix-blocked fuzzy name matching"
    )

    print(
        "4. Informative address tokens"
    )

    print(
        "5. Same-country blocking"
    )


    print(
        f"\nMax signature block: "
        f"{MAX_SIGNATURE_BLOCK}"
    )

    print(
        f"Max address block: "
        f"{MAX_ADDRESS_BLOCK}"
    )

    print(
        f"Fuzzy prefix length: "
        f"{PREFIX_LENGTH}"
    )

    print(
        f"Max fuzzy block: "
        f"{MAX_FUZZY_BLOCK}"
    )

    print(
        f"Fuzzy threshold: "
        f"{FUZZY_THRESHOLD}"
    )


    # ========================================================
    # BUILD S2 INDEX
    # ========================================================

    (
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy
    ) = build_index(
        S2_FILE
    )


    # ========================================================
    # BUILD S3 INDEX
    # ========================================================

    (
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy
    ) = build_index(
        S3_FILE
    )


    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    truth = load_ground_truth()


    # ========================================================
    # EVALUATE
    # ========================================================

    evaluate(
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
        truth
    )


if __name__ == "__main__":
    main()