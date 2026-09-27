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
# BUILD LABELED MATCHING PAIRS FROM V4 CANDIDATES
# ============================================================

OUTPUT_FILE = Path(
    r"D:\AmazonMLChallenge\matching_training_pairs.tsv"
)

NEGATIVES_PER_POSITIVE = 5
NO_RECOVERED_POSITIVE_NEGATIVES = 10


def get_split(s1_id):
    """Deterministic 70/30 split by Source-1 entity ID."""
    import hashlib

    value = int(
        hashlib.md5(s1_id.encode("utf-8")).hexdigest()[:8],
        16,
    )

    return "train" if value % 10 < 7 else "holdout"


def generate_v4_candidates(
    raw_name,
    raw_address,
    country,
    s2_name,
    s2_signature,
    s2_address,
    s2_fuzzy,
    s3_name,
    s3_signature,
    s3_address,
    s3_fuzzy,
):
    """Generate candidates using the same four retrieval strategies as V4."""

    norm_name = normalize(raw_name)
    candidates = set()

    # --------------------------------------------------------
    # STRATEGY 1: EXACT NORMALIZED BUSINESS NAME
    # --------------------------------------------------------
    if norm_name:
        key = (country, norm_name)
        candidates.update(s2_name.get(key, []))
        candidates.update(s3_name.get(key, []))

    # --------------------------------------------------------
    # STRATEGY 2: LEGAL-SUFFIX-FREE NAME SIGNATURE
    # --------------------------------------------------------
    signature = name_signature(raw_name)

    if signature:
        key = (country, signature)
        block2 = s2_signature.get(key, [])
        block3 = s3_signature.get(key, [])

        if len(block2) <= MAX_SIGNATURE_BLOCK:
            candidates.update(block2)

        if len(block3) <= MAX_SIGNATURE_BLOCK:
            candidates.update(block3)

    # --------------------------------------------------------
    # STRATEGY 3: PREFIX-BLOCKED FUZZY NAME MATCHING
    # --------------------------------------------------------
    fuzzy_key = fuzzy_block_key(raw_name, country)

    if fuzzy_key and norm_name:
        block2 = s2_fuzzy.get(fuzzy_key, [])
        block3 = s3_fuzzy.get(fuzzy_key, [])
        fuzzy_records = []

        if len(block2) <= MAX_FUZZY_BLOCK:
            fuzzy_records.extend(block2)

        if len(block3) <= MAX_FUZZY_BLOCK:
            fuzzy_records.extend(block3)

        for entity_id, candidate_name in fuzzy_records:
            score = token_set_ratio(norm_name, candidate_name)

            if score >= FUZZY_THRESHOLD:
                candidates.add(entity_id)

    # --------------------------------------------------------
    # STRATEGY 4: INFORMATIVE ADDRESS TOKENS
    # --------------------------------------------------------
    address_tokens = useful_address_tokens(raw_address)

    for token in address_tokens:
        key = (country, token)
        block2 = s2_address.get(key, [])
        block3 = s3_address.get(key, [])

        if len(block2) <= MAX_ADDRESS_BLOCK:
            candidates.update(block2)

        if len(block3) <= MAX_ADDRESS_BLOCK:
            candidates.update(block3)

    return candidates


def build_training_pairs(
    s2_name,
    s2_signature,
    s2_address,
    s2_fuzzy,
    s3_name,
    s3_signature,
    s3_address,
    s3_fuzzy,
    truth,
):
    """
    Build a compact labeled pair file from V4 candidates.

    All recovered positives are retained. Negatives are sampled so that the
    next feature/model stage does not need to process all ~13M candidates.
    """
    import random

    rng = random.Random(42)

    print("\n" + "=" * 70)
    print("BUILDING MATCHING TRAINING PAIRS")
    print("=" * 70)

    total_s1 = 0
    train_s1 = 0
    holdout_s1 = 0
    s1_with_candidates = 0
    s1_with_truth = 0

    total_true_matches = 0
    recovered_true_matches = 0
    total_candidates = 0
    positive_rows = 0
    negative_rows = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "label",
                "split",
            ]
        )

        with open(
            S1_FILE,
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                s1_id = row["entity_id"]
                country = row["country"] or ""
                raw_name = row["business_name"] or ""
                raw_address = row["business_address"] or ""

                candidates = generate_v4_candidates(
                    raw_name,
                    raw_address,
                    country,
                    s2_name,
                    s2_signature,
                    s2_address,
                    s2_fuzzy,
                    s3_name,
                    s3_signature,
                    s3_address,
                    s3_fuzzy,
                )

                # Defensive: Source-1 IDs should not be candidate IDs, but
                # discard it if an ID collision ever occurs.
                candidates.discard(s1_id)

                actual = truth.get(s1_id, set())
                recovered_positives = candidates.intersection(actual)
                negatives = candidates.difference(actual)
                split = get_split(s1_id)

                total_s1 += 1
                total_true_matches += len(actual)
                recovered_true_matches += len(recovered_positives)
                total_candidates += len(candidates)

                if actual:
                    s1_with_truth += 1

                if candidates:
                    s1_with_candidates += 1

                if split == "train":
                    train_s1 += 1
                else:
                    holdout_s1 += 1

                # Keep every positive pair recovered by V4.
                for candidate_id in sorted(recovered_positives):
                    writer.writerow([s1_id, candidate_id, 1, split])
                    positive_rows += 1

                # Sample negatives. Sorting first makes the sample reproducible
                # across Python runs rather than depending on set iteration order.
                negative_list = sorted(negatives)

                if recovered_positives:
                    number_to_sample = min(
                        len(negative_list),
                        max(
                            5,
                            len(recovered_positives)
                            * NEGATIVES_PER_POSITIVE,
                        ),
                    )
                else:
                    number_to_sample = min(
                        len(negative_list),
                        NO_RECOVERED_POSITIVE_NEGATIVES,
                    )

                if number_to_sample > 0:
                    sampled_negatives = rng.sample(
                        negative_list,
                        number_to_sample,
                    )

                    for candidate_id in sampled_negatives:
                        writer.writerow([s1_id, candidate_id, 0, split])
                        negative_rows += 1

                if total_s1 % 10_000 == 0:
                    recall = (
                        recovered_true_matches / total_true_matches
                        if total_true_matches
                        else 0
                    )
                    avg_candidates = total_candidates / total_s1

                    print(
                        f"Processed {total_s1:,} S1 | "
                        f"Candidate recall = {recall:.4%} | "
                        f"Avg candidates = {avg_candidates:.2f} | "
                        f"Positives = {positive_rows:,} | "
                        f"Negatives = {negative_rows:,}"
                    )

    candidate_recall = (
        recovered_true_matches / total_true_matches
        if total_true_matches
        else 0
    )
    avg_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0
    )

    print("\n" + "=" * 70)
    print("TRAINING PAIRS COMPLETE")
    print("=" * 70)
    print(f"S1 processed: {total_s1:,}")
    print(f"S1 with true matches: {s1_with_truth:,}")
    print(f"S1 receiving candidates: {s1_with_candidates:,}")
    print(f"Train S1: {train_s1:,}")
    print(f"Holdout S1: {holdout_s1:,}")
    print(f"Total true matches: {total_true_matches:,}")
    print(f"Recovered positive pairs: {recovered_true_matches:,}")
    print(f"Candidate recall: {candidate_recall:.4%}")
    print(f"Original V4 candidates considered: {total_candidates:,}")
    print(f"Average candidates per S1: {avg_candidates:.2f}")
    print(f"Positive rows written: {positive_rows:,}")
    print(f"Sampled negative rows written: {negative_rows:,}")
    print(f"Total labeled rows: {positive_rows + negative_rows:,}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUILD MATCHING TRAINING DATA FROM V4")
    print("=" * 70)

    print(f"\nValidation S1: {S1_FILE}")
    print(f"Ground truth: {GT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nCandidate strategies:")
    print("1. Exact normalized business name")
    print("2. Legal-suffix-free sorted name signature")
    print("3. Prefix-blocked fuzzy name matching")
    print("4. Informative address tokens")
    print("5. Same-country blocking")

    print(f"\nMax signature block: {MAX_SIGNATURE_BLOCK}")
    print(f"Max address block: {MAX_ADDRESS_BLOCK}")
    print(f"Fuzzy prefix length: {PREFIX_LENGTH}")
    print(f"Max fuzzy block: {MAX_FUZZY_BLOCK}")
    print(f"Fuzzy threshold: {FUZZY_THRESHOLD}")
    print(f"Negatives per recovered positive: {NEGATIVES_PER_POSITIVE}")
    print(
        "Negatives when no positive is recovered: "
        f"{NO_RECOVERED_POSITIVE_NEGATIVES}"
    )

    # Build Source-2 indexes.
    (
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
    ) = build_index(S2_FILE)

    # Build Source-3 indexes.
    (
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
    ) = build_index(S3_FILE)

    # Load validation labels.
    truth = load_ground_truth()

    # Generate V4 candidates and write compact labeled pairs.
    build_training_pairs(
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
        truth,
    )


if __name__ == "__main__":
    main()
