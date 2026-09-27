import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio,
)
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"D:\AmazonMLChallenge")

DATASET = Path(
    r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"
)

TRAIN = DATASET / "train"

S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

VALIDATION_DIR = PROJECT / "validation_sample"

S1_FILE = VALIDATION_DIR / "validation_source1.tsv"
GT_FILE = VALIDATION_DIR / "validation_ground_truth.tsv"

FEATURE_FILE = PROJECT / "matching_features.tsv"
MODEL_FILE = PROJECT / "matching_model.joblib"


# ============================================================
# V4 SETTINGS
# ============================================================

MAX_ADDRESS_BLOCK = 200
MAX_SIGNATURE_BLOCK = 100

MAX_FUZZY_BLOCK = 300
FUZZY_THRESHOLD = 85
PREFIX_LENGTH = 3


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
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "name_exact",
    "name_signature_exact",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_length_ratio",

    "address_exact",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "address_length_ratio",

    "address_number_overlap",
    "address_number_count",
    "address_number_exact",
    "address_missing",

    "same_country",

    "candidate_is_s2",
    "candidate_is_s3",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
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


def name_signature(name):

    name = normalize(name)

    if not name:
        return ""

    tokens = []

    for token in name.split():

        if token in LEGAL_WORDS:
            continue

        if len(token) <= 1:
            continue

        tokens.append(token)

    return " ".join(
        sorted(set(tokens))
    )


def fuzzy_block_key(
    name,
    country
):

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

    tokens.sort()

    prefix = (
        tokens[0][:PREFIX_LENGTH]
    )

    if not prefix:
        return None

    return (
        country,
        prefix
    )


def useful_address_tokens(
    address
):

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
# FEATURE HELPERS
# ============================================================

def text_tokens(text):

    text = normalize(text)

    if not text:
        return set()

    return set(
        text.split()
    )


def address_numbers(text):

    text = normalize(text)

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


def safe_length_ratio(
    a,
    b
):

    a = normalize(a)
    b = normalize(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return (
        min(len(a), len(b))
        /
        max(len(a), len(b))
    )


def jaccard(
    set1,
    set2
):

    if not set1 and not set2:
        return 0.0

    union = set1 | set2

    if not union:
        return 0.0

    return (
        len(set1 & set2)
        /
        len(union)
    )


# ============================================================
# DETERMINE HOLDOUT S1 IDs
# ============================================================

def load_holdout_ids():

    print()
    print("=" * 70)
    print("STEP 1: LOADING HOLDOUT S1 IDS")
    print("=" * 70)

    holdout_ids = set()

    with open(
        FEATURE_FILE,
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

            if (
                row["split"]
                == "holdout"
            ):

                holdout_ids.add(
                    row[
                        "source1_entity_id"
                    ]
                )

    print(
        "Holdout S1 IDs:",
        f"{len(holdout_ids):,}"
    )

    return holdout_ids


# ============================================================
# LOAD HOLDOUT S1
# ============================================================

def load_holdout_s1(
    holdout_ids
):

    print()
    print("=" * 70)
    print("STEP 2: LOADING HOLDOUT S1 RECORDS")
    print("=" * 70)

    records = {}

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

            entity_id = (
                row["entity_id"]
            )

            if (
                entity_id
                not in holdout_ids
            ):
                continue

            records[entity_id] = {
                "business_name":
                    row["business_name"] or "",

                "business_address":
                    row["business_address"] or "",

                "country":
                    row["country"] or "",
            }

    print(
        "Holdout records loaded:",
        f"{len(records):,}"
    )

    if (
        len(records)
        != len(holdout_ids)
    ):

        raise RuntimeError(
            "Some holdout S1 records "
            "could not be loaded."
        )

    return records


# ============================================================
# LOAD ORIGINAL GROUND TRUTH
# ============================================================

def load_ground_truth(
    holdout_ids
):

    print()
    print("=" * 70)
    print("STEP 3: LOADING ORIGINAL GROUND TRUTH")
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

            if (
                s1_id
                not in holdout_ids
            ):
                continue

            text = (
                row[
                    "matched_entity_ids"
                ]
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
        "Ground-truth S1 loaded:",
        f"{len(truth):,}"
    )

    if (
        len(truth)
        != len(holdout_ids)
    ):

        raise RuntimeError(
            "Holdout ground truth incomplete."
        )

    return truth


# ============================================================
# COLLECT HOLDOUT QUERY KEYS
#
# Important memory optimization:
# only build S2/S3 blocks that can actually be queried by
# our 30K holdout S1 records.
# ============================================================

def collect_query_keys(
    s1_records
):

    print()
    print("=" * 70)
    print("STEP 4: COLLECTING HOLDOUT QUERY KEYS")
    print("=" * 70)

    name_keys = set()
    signature_keys = set()
    address_keys = set()
    fuzzy_keys = set()

    for record in (
        s1_records.values()
    ):

        country = (
            record["country"]
            or ""
        )

        raw_name = (
            record["business_name"]
            or ""
        )

        raw_address = (
            record[
                "business_address"
            ]
            or ""
        )

        norm_name = normalize(
            raw_name
        )

        if norm_name:

            name_keys.add(
                (
                    country,
                    norm_name
                )
            )

        signature = (
            name_signature(
                raw_name
            )
        )

        if signature:

            signature_keys.add(
                (
                    country,
                    signature
                )
            )

        fuzzy_key = (
            fuzzy_block_key(
                raw_name,
                country
            )
        )

        if fuzzy_key:
            fuzzy_keys.add(
                fuzzy_key
            )

        for token in (
            useful_address_tokens(
                raw_address
            )
        ):

            address_keys.add(
                (
                    country,
                    token
                )
            )

    print(
        "Exact-name query keys:",
        f"{len(name_keys):,}"
    )

    print(
        "Signature query keys:",
        f"{len(signature_keys):,}"
    )

    print(
        "Address query keys:",
        f"{len(address_keys):,}"
    )

    print(
        "Fuzzy query keys:",
        f"{len(fuzzy_keys):,}"
    )

    return (
        name_keys,
        signature_keys,
        address_keys,
        fuzzy_keys
    )


# ============================================================
# BUILD FILTERED INDEX
# ============================================================

def build_filtered_index(
    file_path,
    source_label,
    name_keys,
    signature_keys,
    address_keys,
    fuzzy_keys
):

    print()
    print("=" * 70)
    print(
        f"STEP 5: INDEXING {file_path.name}"
    )
    print("=" * 70)

    name_index = defaultdict(list)
    signature_index = defaultdict(list)
    address_index = defaultdict(list)
    fuzzy_index = defaultdict(list)

    # Candidate records needed later for features.
    candidate_records = {}

    count = 0
    retained_records = 0

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

            count += 1

            entity_id = row[
                "entity_id"
            ]

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

            relevant = False

            # --------------------------------------------
            # Exact name
            # --------------------------------------------

            if norm_name:

                key = (
                    country,
                    norm_name
                )

                if key in name_keys:

                    name_index[
                        key
                    ].append(
                        entity_id
                    )

                    relevant = True

            # --------------------------------------------
            # Signature
            # --------------------------------------------

            signature = (
                name_signature(
                    raw_name
                )
            )

            if signature:

                key = (
                    country,
                    signature
                )

                if (
                    key
                    in signature_keys
                ):

                    signature_index[
                        key
                    ].append(
                        entity_id
                    )

                    relevant = True

            # --------------------------------------------
            # Fuzzy block
            # --------------------------------------------

            fuzzy_key = (
                fuzzy_block_key(
                    raw_name,
                    country
                )
            )

            if (
                fuzzy_key
                and fuzzy_key
                in fuzzy_keys
            ):

                fuzzy_index[
                    fuzzy_key
                ].append(
                    (
                        entity_id,
                        norm_name
                    )
                )

                relevant = True

            # --------------------------------------------
            # Address
            # --------------------------------------------

            for token in (
                useful_address_tokens(
                    raw_address
                )
            ):

                key = (
                    country,
                    token
                )

                if (
                    key
                    in address_keys
                ):

                    address_index[
                        key
                    ].append(
                        entity_id
                    )

                    relevant = True

            # --------------------------------------------
            # Store feature data only if potentially useful
            # --------------------------------------------

            if relevant:

                candidate_records[
                    entity_id
                ] = {
                    "business_name":
                        raw_name,

                    "business_address":
                        raw_address,

                    "country":
                        country,

                    "source":
                        source_label,
                }

                retained_records += 1

            if (
                count
                % 1_000_000
                == 0
            ):

                print(
                    f"Processed "
                    f"{count:,} rows | "
                    f"retained = "
                    f"{retained_records:,}"
                )

    print(
        f"Finished {file_path.name}"
    )

    print(
        "Rows:",
        f"{count:,}"
    )

    print(
        "Potential candidate records retained:",
        f"{len(candidate_records):,}"
    )

    return (
        name_index,
        signature_index,
        address_index,
        fuzzy_index,
        candidate_records
    )


# ============================================================
# GENERATE V4 CANDIDATES
# ============================================================

def generate_candidates(
    s1,
    indexes
):

    (
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
    ) = indexes

    country = (
        s1["country"]
        or ""
    )

    raw_name = (
        s1["business_name"]
        or ""
    )

    raw_address = (
        s1["business_address"]
        or ""
    )

    norm_name = normalize(
        raw_name
    )

    candidates = set()

    # Exact name

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

    # Signature

    signature = (
        name_signature(
            raw_name
        )
    )

    if signature:

        key = (
            country,
            signature
        )

        block2 = (
            s2_signature.get(
                key,
                []
            )
        )

        block3 = (
            s3_signature.get(
                key,
                []
            )
        )

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

    # Fuzzy

    fuzzy_key = (
        fuzzy_block_key(
            raw_name,
            country
        )
    )

    if (
        fuzzy_key
        and norm_name
    ):

        block2 = (
            s2_fuzzy.get(
                fuzzy_key,
                []
            )
        )

        block3 = (
            s3_fuzzy.get(
                fuzzy_key,
                []
            )
        )

        fuzzy_records = []

        if (
            len(block2)
            <= MAX_FUZZY_BLOCK
        ):

            fuzzy_records.extend(
                block2
            )

        if (
            len(block3)
            <= MAX_FUZZY_BLOCK
        ):

            fuzzy_records.extend(
                block3
            )

        for (
            entity_id,
            candidate_name
        ) in fuzzy_records:

            score = (
                token_set_ratio(
                    norm_name,
                    candidate_name
                )
            )

            if (
                score
                >= FUZZY_THRESHOLD
            ):

                candidates.add(
                    entity_id
                )

    # Address

    for token in (
        useful_address_tokens(
            raw_address
        )
    ):

        key = (
            country,
            token
        )

        block2 = (
            s2_address.get(
                key,
                []
            )
        )

        block3 = (
            s3_address.get(
                key,
                []
            )
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

    return candidates


# ============================================================
# CALCULATE MODEL FEATURES
# ============================================================

def calculate_features(
    s1,
    candidate
):

    name1 = (
        s1["business_name"]
        or ""
    )

    name2 = (
        candidate["business_name"]
        or ""
    )

    address1 = (
        s1["business_address"]
        or ""
    )

    address2 = (
        candidate["business_address"]
        or ""
    )

    norm_name1 = normalize(
        name1
    )

    norm_name2 = normalize(
        name2
    )

    norm_address1 = normalize(
        address1
    )

    norm_address2 = normalize(
        address2
    )

    name_exact = int(
        bool(norm_name1)
        and norm_name1
        == norm_name2
    )

    sig1 = name_signature(
        name1
    )

    sig2 = name_signature(
        name2
    )

    signature_exact = int(
        bool(sig1)
        and sig1 == sig2
    )

    name_r = ratio(
        norm_name1,
        norm_name2
    )

    name_sort = (
        token_sort_ratio(
            norm_name1,
            norm_name2
        )
    )

    name_set = (
        token_set_ratio(
            norm_name1,
            norm_name2
        )
    )

    name_len = (
        safe_length_ratio(
            norm_name1,
            norm_name2
        )
    )

    address_missing = int(
        not norm_address1
        or not norm_address2
    )

    address_exact = int(
        bool(norm_address1)
        and norm_address1
        == norm_address2
    )

    if (
        norm_address1
        and norm_address2
    ):

        address_r = ratio(
            norm_address1,
            norm_address2
        )

        address_sort = (
            token_sort_ratio(
                norm_address1,
                norm_address2
            )
        )

        address_set = (
            token_set_ratio(
                norm_address1,
                norm_address2
            )
        )

    else:

        address_r = 0.0
        address_sort = 0.0
        address_set = 0.0

    tokens1 = text_tokens(
        norm_address1
    )

    tokens2 = text_tokens(
        norm_address2
    )

    address_j = jaccard(
        tokens1,
        tokens2
    )

    address_len = (
        safe_length_ratio(
            norm_address1,
            norm_address2
        )
    )

    numbers1 = (
        address_numbers(
            norm_address1
        )
    )

    numbers2 = (
        address_numbers(
            norm_address2
        )
    )

    common_numbers = (
        numbers1 & numbers2
    )

    number_overlap = int(
        bool(common_numbers)
    )

    number_count = len(
        common_numbers
    )

    number_exact = int(
        bool(numbers1)
        and bool(numbers2)
        and numbers1 == numbers2
    )

    same_country = int(
        s1["country"]
        == candidate["country"]
    )

    source = (
        candidate["source"]
    )

    return [
        name_exact,
        signature_exact,
        name_r,
        name_sort,
        name_set,
        name_len,

        address_exact,
        address_r,
        address_sort,
        address_set,
        address_j,
        address_len,

        number_overlap,
        number_count,
        number_exact,
        address_missing,

        same_country,

        int(source == "S2"),
        int(source == "S3"),
    ]


# ============================================================
# MACRO F0.5
# ============================================================

def f05_for_sets(
    truth,
    predicted
):

    if (
        not truth
        and not predicted
    ):
        return 1.0

    if (
        not truth
        and predicted
    ):
        return 0.0

    tp = len(
        truth & predicted
    )

    fp = len(
        predicted - truth
    )

    fn = len(
        truth - predicted
    )

    if tp == 0:
        return 0.0

    precision = (
        tp / (tp + fp)
    )

    recall = (
        tp / (tp + fn)
    )

    beta2 = 0.25

    return (
        (1 + beta2)
        * precision
        * recall
        /
        (
            beta2 * precision
            + recall
        )
    )


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate(
    s1_records,
    truth,
    indexes,
    candidate_records,
    model
):

    print()
    print("=" * 70)
    print("STEP 6: FULL HOLDOUT PIPELINE EVALUATION")
    print("=" * 70)

    thresholds = [
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.72,
        0.74,
        0.76,
        0.78,
        0.80,
        0.82,
        0.85,
        0.88,
        0.90,
        0.92,
        0.95,
        0.97,
    ]

    score_totals = {
        t: 0.0
        for t in thresholds
    }

    predicted_counts = {
        t: 0
        for t in thresholds
    }

    total_s1 = 0
    total_candidates = 0

    total_truth = 0
    candidate_recovered = 0

    missing_candidate_records = 0

    for (
        s1_id,
        s1
    ) in s1_records.items():

        candidates = (
            generate_candidates(
                s1,
                indexes
            )
        )

        actual = truth.get(
            s1_id,
            set()
        )

        total_truth += len(
            actual
        )

        candidate_recovered += len(
            actual & candidates
        )

        total_candidates += len(
            candidates
        )

        candidate_ids = []
        feature_rows = []

        for candidate_id in candidates:

            candidate = (
                candidate_records.get(
                    candidate_id
                )
            )

            if candidate is None:

                missing_candidate_records += 1
                continue

            candidate_ids.append(
                candidate_id
            )

            feature_rows.append(
                calculate_features(
                    s1,
                    candidate
                )
            )

        # ----------------------------------------------------
        # Score all candidates for this S1 in one batch
        # ----------------------------------------------------

        probabilities = []

        if feature_rows:

            X = np.asarray(
                feature_rows,
                dtype=np.float32
            )

            probabilities = (
                model.predict_proba(
                    X
                )[:, 1]
            )

        # ----------------------------------------------------
        # Test all thresholds without re-running model
        # ----------------------------------------------------

        for threshold in thresholds:

            predicted = {
                candidate_id
                for (
                    candidate_id,
                    probability
                )
                in zip(
                    candidate_ids,
                    probabilities
                )
                if probability
                >= threshold
            }

            predicted_counts[
                threshold
            ] += len(
                predicted
            )

            score_totals[
                threshold
            ] += f05_for_sets(
                actual,
                predicted
            )

        total_s1 += 1

        if (
            total_s1
            % 2_500
            == 0
        ):

            recall = (
                candidate_recovered
                / total_truth
                if total_truth
                else 0
            )

            avg_candidates = (
                total_candidates
                / total_s1
            )

            print(
                f"Processed "
                f"{total_s1:,} S1 | "
                f"candidate recall = "
                f"{recall:.4%} | "
                f"avg candidates = "
                f"{avg_candidates:.2f}"
            )

    print()
    print("-" * 70)
    print("CANDIDATE GENERATION")
    print("-" * 70)

    candidate_recall = (
        candidate_recovered
        / total_truth
        if total_truth
        else 0
    )

    print(
        "Holdout S1:",
        f"{total_s1:,}"
    )

    print(
        "Total true matches:",
        f"{total_truth:,}"
    )

    print(
        "True matches retrieved:",
        f"{candidate_recovered:,}"
    )

    print(
        "Candidate recall:",
        f"{candidate_recall:.4%}"
    )

    print(
        "Average candidates/S1:",
        f"{total_candidates / total_s1:.2f}"
    )

    print(
        "Missing candidate records:",
        f"{missing_candidate_records:,}"
    )

    print()
    print("-" * 70)
    print("FULL-PIPELINE MACRO F0.5")
    print("-" * 70)

    best_threshold = None
    best_score = -1.0

    for threshold in thresholds:

        macro_score = (
            score_totals[
                threshold
            ]
            / total_s1
        )

        print(
            f"Threshold "
            f"{threshold:.2f} | "
            f"Macro F0.5 = "
            f"{macro_score:.6f} | "
            f"Predicted matches = "
            f"{predicted_counts[threshold]:,}"
        )

        if (
            macro_score
            > best_score
        ):

            best_score = (
                macro_score
            )

            best_threshold = (
                threshold
            )

    print()
    print("=" * 70)
    print("FULL HOLDOUT VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "Best threshold:",
        f"{best_threshold:.2f}"
    )

    print(
        "Best full-pipeline "
        "macro F0.5:",
        f"{best_score:.6f}"
    )

    print(
        "Candidate recall:",
        f"{candidate_recall:.4%}"
    )

    print(
        "Missing candidate records:",
        f"{missing_candidate_records:,}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("FULL HOLDOUT PIPELINE EVALUATION")
    print("=" * 70)

    holdout_ids = (
        load_holdout_ids()
    )

    s1_records = (
        load_holdout_s1(
            holdout_ids
        )
    )

    truth = (
        load_ground_truth(
            holdout_ids
        )
    )

    (
        name_keys,
        signature_keys,
        address_keys,
        fuzzy_keys
    ) = collect_query_keys(
        s1_records
    )

    (
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
        s2_records
    ) = build_filtered_index(
        S2_FILE,
        "S2",
        name_keys,
        signature_keys,
        address_keys,
        fuzzy_keys
    )

    (
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
        s3_records
    ) = build_filtered_index(
        S3_FILE,
        "S3",
        name_keys,
        signature_keys,
        address_keys,
        fuzzy_keys
    )

    candidate_records = {}

    candidate_records.update(
        s2_records
    )

    candidate_records.update(
        s3_records
    )

    indexes = (
        s2_name,
        s2_signature,
        s2_address,
        s2_fuzzy,
        s3_name,
        s3_signature,
        s3_address,
        s3_fuzzy,
    )

    print()
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved[
        "model"
    ]

    saved_features = saved[
        "features"
    ]

    if (
        saved_features
        != FEATURE_COLUMNS
    ):

        raise RuntimeError(
            "Model feature order does "
            "not match evaluation features."
        )

    print(
        "Model loaded successfully."
    )

    evaluate(
        s1_records,
        truth,
        indexes,
        candidate_records,
        model
    )


if __name__ == "__main__":
    main()