import csv
import re
import unicodedata
from pathlib import Path

from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio


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

VALIDATION_S1 = (
    PROJECT
    / "validation_sample"
    / "validation_source1.tsv"
)

PAIRS_FILE = PROJECT / "matching_training_pairs.tsv"

OUTPUT_FILE = PROJECT / "matching_features.tsv"


# ============================================================
# SETTINGS
# ============================================================

PROGRESS_EVERY = 200_000

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
    Same basic normalization used by V4.
    """

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
    """
    Same legal-word-free sorted signature idea used by V4.
    """

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

    tokens = sorted(set(tokens))

    return " ".join(tokens)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def text_tokens(text):

    text = normalize(text)

    if not text:
        return set()

    return set(text.split())


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


def safe_length_ratio(a, b):
    """
    Ratio between shorter and longer normalized string lengths.
    Range: 0..1
    """

    a = normalize(a)
    b = normalize(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    short = min(
        len(a),
        len(b)
    )

    long = max(
        len(a),
        len(b)
    )

    if long == 0:
        return 0.0

    return short / long


def jaccard(set1, set2):

    if not set1 and not set2:
        return 0.0

    union = set1 | set2

    if not union:
        return 0.0

    return len(
        set1 & set2
    ) / len(union)


# ============================================================
# STEP 1
# LOAD PAIRS AND COLLECT REQUIRED IDs
# ============================================================

def collect_required_ids():

    print()
    print("=" * 70)
    print("STEP 1: COLLECTING REQUIRED ENTITY IDs")
    print("=" * 70)

    required_s1 = set()
    required_candidates = set()

    pair_count = 0

    with open(
        PAIRS_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        print(
            "Pair columns:",
            reader.fieldnames
        )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            candidate_id = row[
                "candidate_entity_id"
            ]

            required_s1.add(
                s1_id
            )

            required_candidates.add(
                candidate_id
            )

            pair_count += 1

            if (
                pair_count
                % PROGRESS_EVERY
                == 0
            ):

                print(
                    f"Read "
                    f"{pair_count:,} pairs..."
                )

    print()
    print(
        "Total labeled pairs:",
        f"{pair_count:,}"
    )

    print(
        "Required S1 IDs:",
        f"{len(required_s1):,}"
    )

    print(
        "Unique candidate IDs:",
        f"{len(required_candidates):,}"
    )

    return (
        required_s1,
        required_candidates,
        pair_count
    )


# ============================================================
# STEP 2
# LOAD VALIDATION S1 RECORDS
# ============================================================

def load_s1_records(required_s1):

    print()
    print("=" * 70)
    print("STEP 2: LOADING VALIDATION S1 RECORDS")
    print("=" * 70)

    records = {}

    with open(
        VALIDATION_S1,
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

            entity_id = row[
                "entity_id"
            ]

            if entity_id not in required_s1:
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
        "S1 records loaded:",
        f"{len(records):,}"
    )

    missing = (
        len(required_s1)
        - len(records)
    )

    print(
        "Missing required S1 records:",
        f"{missing:,}"
    )

    if missing != 0:

        raise RuntimeError(
            "Some required S1 records "
            "could not be loaded."
        )

    return records


# ============================================================
# STEP 3
# LOAD ONLY REQUIRED S2/S3 RECORDS
# ============================================================

def scan_candidate_source(
    file_path,
    source_label,
    required_candidates,
    records
):

    print()
    print("-" * 70)
    print(
        f"Scanning {file_path.name}"
    )
    print("-" * 70)

    scanned = 0
    found = 0

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

            scanned += 1

            entity_id = row[
                "entity_id"
            ]

            if (
                entity_id
                not in required_candidates
            ):
                if (
                    scanned
                    % 1_000_000
                    == 0
                ):

                    print(
                        f"Scanned "
                        f"{scanned:,} rows | "
                        f"Required records found = "
                        f"{found:,}"
                    )

                continue

            records[entity_id] = {
                "business_name":
                    row["business_name"] or "",

                "business_address":
                    row["business_address"] or "",

                "country":
                    row["country"] or "",

                "source":
                    source_label,
            }

            found += 1

            if (
                scanned
                % 1_000_000
                == 0
            ):

                print(
                    f"Scanned "
                    f"{scanned:,} rows | "
                    f"Required records found = "
                    f"{found:,}"
                )

    print(
        f"Finished {file_path.name}"
    )

    print(
        "Rows scanned:",
        f"{scanned:,}"
    )

    print(
        "Required records found:",
        f"{found:,}"
    )


def load_candidate_records(
    required_candidates
):

    print()
    print("=" * 70)
    print(
        "STEP 3: LOADING REQUIRED "
        "S2/S3 RECORDS"
    )
    print("=" * 70)

    records = {}

    scan_candidate_source(
        S2_FILE,
        "S2",
        required_candidates,
        records
    )

    scan_candidate_source(
        S3_FILE,
        "S3",
        required_candidates,
        records
    )

    print()
    print(
        "Unique candidate records loaded:",
        f"{len(records):,}"
    )

    missing = (
        required_candidates
        - set(records.keys())
    )

    print(
        "Missing candidate IDs:",
        f"{len(missing):,}"
    )

    if missing:

        print()
        print(
            "WARNING: Some candidate IDs "
            "were not found."
        )

        print(
            "First 10 missing IDs:"
        )

        for entity_id in list(
            missing
        )[:10]:

            print(entity_id)

    return records


# ============================================================
# FEATURE EXTRACTION
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

    country1 = (
        s1["country"]
        or ""
    )

    country2 = (
        candidate["country"]
        or ""
    )

    # --------------------------------------------------------
    # NORMALIZED VALUES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_exact = int(
        bool(norm_name1)
        and norm_name1 == norm_name2
    )

    signature1 = name_signature(
        name1
    )

    signature2 = name_signature(
        name2
    )

    name_signature_exact = int(
        bool(signature1)
        and signature1 == signature2
    )

    name_ratio = ratio(
        norm_name1,
        norm_name2
    )

    name_token_sort = (
        token_sort_ratio(
            norm_name1,
            norm_name2
        )
    )

    name_token_set = (
        token_set_ratio(
            norm_name1,
            norm_name2
        )
    )

    name_length_ratio = (
        safe_length_ratio(
            norm_name1,
            norm_name2
        )
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

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

        address_ratio = ratio(
            norm_address1,
            norm_address2
        )

        address_token_sort = (
            token_sort_ratio(
                norm_address1,
                norm_address2
            )
        )

        address_token_set = (
            token_set_ratio(
                norm_address1,
                norm_address2
            )
        )

    else:

        address_ratio = 0.0
        address_token_sort = 0.0
        address_token_set = 0.0

    tokens1 = text_tokens(
        norm_address1
    )

    tokens2 = text_tokens(
        norm_address2
    )

    address_jaccard = jaccard(
        tokens1,
        tokens2
    )

    address_length_ratio = (
        safe_length_ratio(
            norm_address1,
            norm_address2
        )
    )

    # --------------------------------------------------------
    # ADDRESS NUMBER FEATURES
    # --------------------------------------------------------

    numbers1 = address_numbers(
        norm_address1
    )

    numbers2 = address_numbers(
        norm_address2
    )

    common_numbers = (
        numbers1 & numbers2
    )

    address_number_overlap = int(
        bool(common_numbers)
    )

    address_number_count = len(
        common_numbers
    )

    address_number_exact = int(
        bool(numbers1)
        and bool(numbers2)
        and numbers1 == numbers2
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    same_country = int(
        country1 == country2
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source = candidate.get(
        "source",
        ""
    )

    candidate_is_s2 = int(
        source == "S2"
    )

    candidate_is_s3 = int(
        source == "S3"
    )

    return [
        name_exact,
        name_signature_exact,
        round(name_ratio, 4),
        round(name_token_sort, 4),
        round(name_token_set, 4),
        round(name_length_ratio, 6),

        address_exact,
        round(address_ratio, 4),
        round(address_token_sort, 4),
        round(address_token_set, 4),
        round(address_jaccard, 6),
        round(address_length_ratio, 6),

        address_number_overlap,
        address_number_count,
        address_number_exact,
        address_missing,

        same_country,

        candidate_is_s2,
        candidate_is_s3,
    ]


# ============================================================
# STEP 4
# GENERATE FEATURE FILE
# ============================================================

def generate_features(
    s1_records,
    candidate_records
):

    print()
    print("=" * 70)
    print(
        "STEP 4: GENERATING MATCHING FEATURES"
    )
    print("=" * 70)

    feature_names = [
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

    processed = 0
    written = 0
    skipped = 0

    positive_count = 0
    negative_count = 0

    train_count = 0
    holdout_count = 0

    with open(
        PAIRS_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as output_file:

        reader = csv.DictReader(
            input_file,
            delimiter="\t"
        )

        writer = csv.writer(
            output_file,
            delimiter="\t"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_id",
                *feature_names,
                "label",
                "split",
            ]
        )

        for row in reader:

            processed += 1

            s1_id = row[
                "source1_entity_id"
            ]

            candidate_id = row[
                "candidate_entity_id"
            ]

            label = int(
                row["label"]
            )

            split = row[
                "split"
            ]

            s1 = s1_records.get(
                s1_id
            )

            candidate = (
                candidate_records.get(
                    candidate_id
                )
            )

            if (
                s1 is None
                or candidate is None
            ):

                skipped += 1
                continue

            features = calculate_features(
                s1,
                candidate
            )

            writer.writerow(
                [
                    s1_id,
                    candidate_id,
                    *features,
                    label,
                    split,
                ]
            )

            written += 1

            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

            if split == "train":
                train_count += 1

            elif split == "holdout":
                holdout_count += 1

            if (
                processed
                % PROGRESS_EVERY
                == 0
            ):

                print(
                    f"Processed "
                    f"{processed:,} pairs | "
                    f"Written = "
                    f"{written:,} | "
                    f"Skipped = "
                    f"{skipped:,}"
                )

    print()
    print("=" * 70)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Pairs processed:",
        f"{processed:,}"
    )

    print(
        "Feature rows written:",
        f"{written:,}"
    )

    print(
        "Rows skipped:",
        f"{skipped:,}"
    )

    print(
        "Positive rows:",
        f"{positive_count:,}"
    )

    print(
        "Negative rows:",
        f"{negative_count:,}"
    )

    print(
        "Train rows:",
        f"{train_count:,}"
    )

    print(
        "Holdout rows:",
        f"{holdout_count:,}"
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print("=" * 70)

    if skipped != 0:

        print()
        print(
            "WARNING: Rows were skipped."
        )

        print(
            "Do NOT train the model until "
            "we investigate why."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUILD MATCHING FEATURES")
    print("=" * 70)

    print()
    print("Input pairs:")
    print(PAIRS_FILE)

    print()
    print("Output features:")
    print(OUTPUT_FILE)

    (
        required_s1,
        required_candidates,
        pair_count
    ) = collect_required_ids()

    s1_records = load_s1_records(
        required_s1
    )

    candidate_records = (
        load_candidate_records(
            required_candidates
        )
    )

    generate_features(
        s1_records,
        candidate_records
    )


if __name__ == "__main__":
    main()