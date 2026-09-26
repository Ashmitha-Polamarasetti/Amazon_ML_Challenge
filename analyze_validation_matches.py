import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio


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
# NORMALIZATION
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


def normalize(text):

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

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


def name_signature(text):

    text = normalize(text)

    if not text:
        return ""

    tokens = []

    for token in text.split():

        if token in LEGAL_WORDS:
            continue

        if len(token) <= 1:
            continue

        tokens.append(token)

    return " ".join(
        sorted(set(tokens))
    )


def address_tokens(text):

    text = normalize(text)

    if not text:
        return set()

    result = set()

    for token in text.split():

        if token.isdigit():
            result.add(token)

        elif len(token) >= 5:
            result.add(token)

    return result


# ============================================================
# LOAD VALIDATION S1
# ============================================================

def load_s1():

    print("=" * 70)
    print("STEP 1: Loading validation S1")
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

            records[row["entity_id"]] = {
                "country": row["country"],
                "name": row["business_name"] or "",
                "address": row["business_address"] or ""
            }

    print(
        f"Validation S1 loaded: {len(records):,}"
    )

    return records


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\n" + "=" * 70)
    print("STEP 2: Loading validation ground truth")
    print("=" * 70)

    truth = {}

    required_source_ids = set()

    total_matches = 0

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

            text = (
                row["matched_entity_ids"]
                or ""
            ).strip()

            if text:

                matches = [
                    x.strip()
                    for x in text.split(",")
                    if x.strip()
                ]

            else:

                matches = []

            truth[s1_id] = matches

            for entity_id in matches:

                required_source_ids.add(
                    entity_id
                )

            total_matches += len(matches)

    print(
        f"Ground-truth rows: {len(truth):,}"
    )

    print(
        f"Total true matches: {total_matches:,}"
    )

    print(
        f"Unique source IDs needed: "
        f"{len(required_source_ids):,}"
    )

    return truth, required_source_ids


# ============================================================
# LOAD ONLY REQUIRED S2/S3 RECORDS
# ============================================================

def load_required_records(
    file_path,
    required_ids
):

    print("\n" + "=" * 70)
    print(f"STEP 3: Scanning {file_path.name}")
    print("=" * 70)

    records = {}

    scanned = 0

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

            entity_id = row["entity_id"]

            if entity_id in required_ids:

                records[entity_id] = {
                    "country": row["country"],
                    "name": row["business_name"] or "",
                    "address": row["business_address"] or ""
                }

            if scanned % 1_000_000 == 0:

                print(
                    f"Scanned {scanned:,} | "
                    f"Found {len(records):,}"
                )

    print(
        f"Finished scanning: {scanned:,}"
    )

    print(
        f"Relevant records found: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# ANALYSIS
# ============================================================

def analyze(
    s1_records,
    truth,
    source_records
):

    print("\n" + "=" * 70)
    print("STEP 4: Analysing true-match similarity")
    print("=" * 70)

    categories = Counter()

    name_ratio_bins = Counter()
    token_sort_bins = Counter()
    token_set_bins = Counter()

    address_overlap_bins = Counter()

    total = 0
    missing_source = 0

    examples = {
        "exact_name": [],
        "signature": [],
        "fuzzy_high": [],
        "fuzzy_medium": [],
        "fuzzy_low": [],
    }

    for s1_id, matches in truth.items():

        s1 = s1_records.get(s1_id)

        if not s1:
            continue

        s1_name = normalize(
            s1["name"]
        )

        s1_signature = name_signature(
            s1["name"]
        )

        s1_address = address_tokens(
            s1["address"]
        )

        for matched_id in matches:

            source = source_records.get(
                matched_id
            )

            if source is None:

                missing_source += 1
                continue

            total += 1

            source_name = normalize(
                source["name"]
            )

            source_signature = name_signature(
                source["name"]
            )

            source_address = address_tokens(
                source["address"]
            )

            # -----------------------------------------
            # Name similarities
            # -----------------------------------------

            r = ratio(
                s1_name,
                source_name
            )

            tsort = token_sort_ratio(
                s1_name,
                source_name
            )

            tset = token_set_ratio(
                s1_name,
                source_name
            )

            # -----------------------------------------
            # Categorize name relationship
            # -----------------------------------------

            if (
                s1_name
                and s1_name == source_name
            ):

                categories[
                    "exact_normalized_name"
                ] += 1

                if len(
                    examples["exact_name"]
                ) < 5:

                    examples[
                        "exact_name"
                    ].append(
                        (
                            s1["name"],
                            source["name"]
                        )
                    )

            elif (
                s1_signature
                and
                s1_signature
                == source_signature
            ):

                categories[
                    "same_name_signature"
                ] += 1

                if len(
                    examples["signature"]
                ) < 5:

                    examples[
                        "signature"
                    ].append(
                        (
                            s1["name"],
                            source["name"]
                        )
                    )

            elif tset >= 90:

                categories[
                    "fuzzy_name_90_plus"
                ] += 1

                if len(
                    examples["fuzzy_high"]
                ) < 5:

                    examples[
                        "fuzzy_high"
                    ].append(
                        (
                            s1["name"],
                            source["name"]
                        )
                    )

            elif tset >= 70:

                categories[
                    "fuzzy_name_70_89"
                ] += 1

                if len(
                    examples["fuzzy_medium"]
                ) < 5:

                    examples[
                        "fuzzy_medium"
                    ].append(
                        (
                            s1["name"],
                            source["name"]
                        )
                    )

            else:

                categories[
                    "fuzzy_name_below_70"
                ] += 1

                if len(
                    examples["fuzzy_low"]
                ) < 5:

                    examples[
                        "fuzzy_low"
                    ].append(
                        (
                            s1["name"],
                            source["name"]
                        )
                    )

            # -----------------------------------------
            # Similarity bins
            # -----------------------------------------

            name_ratio_bins[
                score_bin(r)
            ] += 1

            token_sort_bins[
                score_bin(tsort)
            ] += 1

            token_set_bins[
                score_bin(tset)
            ] += 1

            # -----------------------------------------
            # Address token overlap
            # -----------------------------------------

            if (
                s1_address
                and source_address
            ):

                intersection = len(
                    s1_address
                    & source_address
                )

                union = len(
                    s1_address
                    | source_address
                )

                overlap = (
                    intersection / union
                    if union
                    else 0
                )

                if overlap >= 0.75:

                    address_overlap_bins[
                        "75-100%"
                    ] += 1

                elif overlap >= 0.50:

                    address_overlap_bins[
                        "50-74%"
                    ] += 1

                elif overlap >= 0.25:

                    address_overlap_bins[
                        "25-49%"
                    ] += 1

                elif overlap > 0:

                    address_overlap_bins[
                        "1-24%"
                    ] += 1

                else:

                    address_overlap_bins[
                        "0%"
                    ] += 1

            else:

                address_overlap_bins[
                    "missing/empty"
                ] += 1


            if total % 100_000 == 0:

                print(
                    f"Analysed "
                    f"{total:,} true pairs..."
                )

    print_results(
        total,
        missing_source,
        categories,
        name_ratio_bins,
        token_sort_bins,
        token_set_bins,
        address_overlap_bins,
        examples
    )


# ============================================================
# SCORE BINS
# ============================================================

def score_bin(score):

    if score >= 90:
        return "90-100"

    if score >= 80:
        return "80-89"

    if score >= 70:
        return "70-79"

    if score >= 60:
        return "60-69"

    if score >= 50:
        return "50-59"

    return "0-49"


# ============================================================
# PRINT RESULTS
# ============================================================

def print_counter(title, counter, total):

    print("\n" + title)
    print("-" * 70)

    order = [
        "exact_normalized_name",
        "same_name_signature",
        "fuzzy_name_90_plus",
        "fuzzy_name_70_89",
        "fuzzy_name_below_70",

        "90-100",
        "80-89",
        "70-79",
        "60-69",
        "50-59",
        "0-49",

        "75-100%",
        "50-74%",
        "25-49%",
        "1-24%",
        "0%",
        "missing/empty",
    ]

    printed = set()

    for key in order:

        if key in counter:

            value = counter[key]

            percent = (
                value / total * 100
                if total
                else 0
            )

            print(
                f"{key:30s} "
                f"{value:10,} "
                f"({percent:6.2f}%)"
            )

            printed.add(key)

    for key, value in counter.items():

        if key not in printed:

            percent = (
                value / total * 100
                if total
                else 0
            )

            print(
                f"{key:30s} "
                f"{value:10,} "
                f"({percent:6.2f}%)"
            )


def print_results(
    total,
    missing_source,
    categories,
    name_ratio_bins,
    token_sort_bins,
    token_set_bins,
    address_overlap_bins,
    examples
):

    print("\n" + "=" * 70)
    print("TRUE-MATCH ANALYSIS RESULTS")
    print("=" * 70)

    print(
        f"True pairs analysed: "
        f"{total:,}"
    )

    print(
        f"Missing source records: "
        f"{missing_source:,}"
    )

    print_counter(
        "NAME RELATIONSHIP",
        categories,
        total
    )

    print_counter(
        "NORMAL STRING RATIO",
        name_ratio_bins,
        total
    )

    print_counter(
        "TOKEN SORT RATIO",
        token_sort_bins,
        total
    )

    print_counter(
        "TOKEN SET RATIO",
        token_set_bins,
        total
    )

    print_counter(
        "ADDRESS TOKEN OVERLAP",
        address_overlap_bins,
        total
    )

    print("\n" + "=" * 70)
    print("EXAMPLE NAME PAIRS")
    print("=" * 70)

    for category, pairs in examples.items():

        print(
            f"\n[{category}]"
        )

        for left, right in pairs:

            print(
                f"S1 : {left}"
            )

            print(
                f"SX : {right}"
            )

            print("-" * 50)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("VALIDATION TRUE-MATCH ANALYSIS")
    print("=" * 70)

    s1_records = load_s1()

    (
        truth,
        required_ids
    ) = load_ground_truth()

    source_records = {}

    s2_records = load_required_records(
        S2_FILE,
        required_ids
    )

    source_records.update(
        s2_records
    )

    del s2_records

    s3_records = load_required_records(
        S3_FILE,
        required_ids
    )

    source_records.update(
        s3_records
    )

    del s3_records

    print("\n" + "=" * 70)

    print(
        f"Total relevant S2/S3 records loaded: "
        f"{len(source_records):,}"
    )

    print("=" * 70)

    analyze(
        s1_records,
        truth,
        source_records
    )


if __name__ == "__main__":
    main()