import csv
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET = r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"

SOURCE2 = rf"{DATASET}\train\train_source2.tsv"
SOURCE3 = rf"{DATASET}\train\train_source3.tsv"

VALIDATION_S1 = (
    r"D:\AmazonMLChallenge\validation_sample\validation_source1.tsv"
)

VALIDATION_GT = (
    r"D:\AmazonMLChallenge\validation_sample\validation_ground_truth.tsv"
)


# ============================================================
# MEMBER 2 SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

MIN_TOKEN_LENGTH = 4

NAME_MIN_FREQ = 1
NAME_MAX_FREQ = 100

ADDRESS_MIN_FREQ = 1
ADDRESS_MAX_FREQ = 100

NUMBER_MIN_FREQ = 2
NUMBER_MAX_FREQ = 500


# ============================================================
# MEMBER 2 NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"'s\b", "", text)

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_dataframe(df):

    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_text)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_text)
    )

    return df


# ============================================================
# HELPERS
# ============================================================

def load_chunks(path):

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    )


def get_tokens(text):

    if not text:
        return []

    return [
        token
        for token in text.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def get_address_numbers(address):

    if not address:
        return set()

    return set(re.findall(r"\d+", address))


# ============================================================
# LOAD VALIDATION GROUND TRUTH
# ============================================================

def load_ground_truth():

    print()
    print("=" * 70)
    print("LOADING VALIDATION GROUND TRUTH")
    print("=" * 70)

    ground_truth = {}

    with open(
        VALIDATION_GT,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        print("Ground-truth columns:", reader.fieldnames)

        for row in reader:

            s1_id = row["source1_entity_id"]

            matches = row["matched_entity_ids"].strip()

            if matches:
                ground_truth[s1_id] = set(
                    x.strip()
                    for x in matches.split(",")
                    if x.strip()
                )
            else:
                ground_truth[s1_id] = set()

    print(
        "Ground-truth S1 records:",
        f"{len(ground_truth):,}"
    )

    return ground_truth


# ============================================================
# PASS 1
# COUNT MEMBER 2 FEATURES
# ============================================================

def count_features():

    print()
    print("=" * 70)
    print("PASS 1: COUNTING MEMBER 2 FEATURES")
    print("=" * 70)

    name_counts = Counter()
    address_counts = Counter()
    number_counts = Counter()

    for source_name, path in [
        ("SOURCE2", SOURCE2),
        ("SOURCE3", SOURCE3)
    ]:

        print()
        print("Counting:", source_name)

        rows_processed = 0

        for chunk_number, df in enumerate(
            load_chunks(path),
            start=1
        ):

            df = normalize_dataframe(df)

            for name in df["business_name_normalized"]:

                for token in set(get_tokens(name)):
                    name_counts[token] += 1

            for address in df["business_address_normalized"]:

                for token in set(get_tokens(address)):
                    address_counts[token] += 1

                for number in get_address_numbers(address):
                    number_counts[number] += 1

            rows_processed += len(df)

            if chunk_number % 10 == 0:

                print(
                    f"{source_name}: "
                    f"{rows_processed:,} rows processed"
                )

        print(
            f"{source_name} finished: "
            f"{rows_processed:,} rows"
        )

    print()
    print("Unique name tokens:", f"{len(name_counts):,}")
    print(
        "Unique address tokens:",
        f"{len(address_counts):,}"
    )
    print(
        "Unique address numbers:",
        f"{len(number_counts):,}"
    )

    return (
        name_counts,
        address_counts,
        number_counts
    )


# ============================================================
# PASS 2
# BUILD MEMBER 2 INDEXES
# ============================================================

def build_indexes(
    name_counts,
    address_counts,
    number_counts
):

    print()
    print("=" * 70)
    print("PASS 2: BUILDING MEMBER 2 INDEXES")
    print("=" * 70)

    eligible_name_tokens = {
        token
        for token, count in name_counts.items()
        if NAME_MIN_FREQ <= count <= NAME_MAX_FREQ
    }

    eligible_address_tokens = {
        token
        for token, count in address_counts.items()
        if ADDRESS_MIN_FREQ <= count <= ADDRESS_MAX_FREQ
    }

    eligible_numbers = {
        number
        for number, count in number_counts.items()
        if NUMBER_MIN_FREQ <= count <= NUMBER_MAX_FREQ
    }

    print(
        "Eligible name tokens:",
        f"{len(eligible_name_tokens):,}"
    )

    print(
        "Eligible address tokens:",
        f"{len(eligible_address_tokens):,}"
    )

    print(
        "Eligible address numbers:",
        f"{len(eligible_numbers):,}"
    )

    exact_index = defaultdict(set)
    name_index = defaultdict(set)
    address_index = defaultdict(set)
    number_index = defaultdict(set)

    for source_name, path in [
        ("SOURCE2", SOURCE2),
        ("SOURCE3", SOURCE3)
    ]:

        print()
        print("Indexing:", source_name)

        rows_processed = 0

        for chunk_number, df in enumerate(
            load_chunks(path),
            start=1
        ):

            df = normalize_dataframe(df)

            for row in df.itertuples(index=False):

                entity_id = row.entity_id

                country = (
                    row.country.strip().lower()
                )

                name = (
                    row.business_name_normalized
                )

                address = (
                    row.business_address_normalized
                )

                if not country:
                    continue

                # ----------------------------------------
                # Exact normalized name
                # ----------------------------------------

                if name:

                    exact_index[
                        (country, name)
                    ].add(entity_id)

                # ----------------------------------------
                # Rare name tokens
                # ----------------------------------------

                for token in set(get_tokens(name)):

                    if token in eligible_name_tokens:

                        name_index[
                            (country, token)
                        ].add(entity_id)

                # ----------------------------------------
                # Rare address tokens
                # ----------------------------------------

                for token in set(
                    get_tokens(address)
                ):

                    if token in eligible_address_tokens:

                        address_index[
                            (country, token)
                        ].add(entity_id)

                # ----------------------------------------
                # Address numbers
                # ----------------------------------------

                for number in get_address_numbers(
                    address
                ):

                    if number in eligible_numbers:

                        number_index[
                            (country, number)
                        ].add(entity_id)

            rows_processed += len(df)

            if chunk_number % 10 == 0:

                print(
                    f"{source_name}: "
                    f"{rows_processed:,} rows indexed"
                )

        print(
            f"{source_name} finished: "
            f"{rows_processed:,} rows"
        )

    print()
    print(
        "Exact-name blocks:",
        f"{len(exact_index):,}"
    )

    print(
        "Rare-name blocks:",
        f"{len(name_index):,}"
    )

    print(
        "Rare-address blocks:",
        f"{len(address_index):,}"
    )

    print(
        "Address-number blocks:",
        f"{len(number_index):,}"
    )

    return (
        exact_index,
        name_index,
        address_index,
        number_index
    )


# ============================================================
# PASS 3
# EVALUATE MEMBER 2 ON OUR 100K VALIDATION S1
# ============================================================

def evaluate(
    ground_truth,
    exact_index,
    name_index,
    address_index,
    number_index
):

    print()
    print("=" * 70)
    print("PASS 3: MEMBER 2 VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        VALIDATION_S1,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df = normalize_dataframe(df)

    total_s1 = 0
    s1_with_true_matches = 0
    s1_with_candidates = 0

    total_true_matches = 0
    recovered_true_matches = 0

    total_candidates = 0

    exact_candidates_added = 0
    name_candidates_added = 0
    address_candidates_added = 0
    number_candidates_added = 0

    for row in df.itertuples(index=False):

        s1_id = row.entity_id

        country = row.country.strip().lower()

        name = row.business_name_normalized

        address = row.business_address_normalized

        true_matches = ground_truth.get(
            s1_id,
            set()
        )

        total_s1 += 1

        if true_matches:
            s1_with_true_matches += 1

        total_true_matches += len(true_matches)

        candidates = set()

        # --------------------------------------------
        # 1. EXACT NAME
        # --------------------------------------------

        before = len(candidates)

        if name:

            candidates.update(
                exact_index.get(
                    (country, name),
                    ()
                )
            )

        exact_candidates_added += (
            len(candidates) - before
        )

        # --------------------------------------------
        # 2. RARE NAME TOKENS
        # --------------------------------------------

        before = len(candidates)

        for token in set(get_tokens(name)):

            candidates.update(
                name_index.get(
                    (country, token),
                    ()
                )
            )

        name_candidates_added += (
            len(candidates) - before
        )

        # --------------------------------------------
        # 3. RARE ADDRESS TOKENS
        # --------------------------------------------

        before = len(candidates)

        for token in set(
            get_tokens(address)
        ):

            candidates.update(
                address_index.get(
                    (country, token),
                    ()
                )
            )

        address_candidates_added += (
            len(candidates) - before
        )

        # --------------------------------------------
        # 4. ADDRESS NUMBERS
        # --------------------------------------------

        before = len(candidates)

        for number in get_address_numbers(
            address
        ):

            candidates.update(
                number_index.get(
                    (country, number),
                    ()
                )
            )

        number_candidates_added += (
            len(candidates) - before
        )

        candidates.discard(s1_id)

        if candidates:
            s1_with_candidates += 1

        total_candidates += len(candidates)

        recovered = len(
            true_matches.intersection(candidates)
        )

        recovered_true_matches += recovered

        if total_s1 % 10_000 == 0:

            current_recall = (
                recovered_true_matches
                / total_true_matches
                * 100
                if total_true_matches
                else 0
            )

            avg_candidates = (
                total_candidates / total_s1
            )

            print(
                f"Processed {total_s1:,} S1 | "
                f"Recall = {current_recall:.4f}% | "
                f"Avg candidates = "
                f"{avg_candidates:.2f}"
            )

    candidate_recall = (
        recovered_true_matches
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    average_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0
    )

    print()
    print("=" * 70)
    print("MEMBER 2 RESULTS")
    print("=" * 70)

    print(
        "S1 records evaluated:",
        f"{total_s1:,}"
    )

    print(
        "S1 records with true matches:",
        f"{s1_with_true_matches:,}"
    )

    print(
        "S1 receiving candidates:",
        f"{s1_with_candidates:,}"
    )

    print(
        "Total true matches:",
        f"{total_true_matches:,}"
    )

    print(
        "Recovered true matches:",
        f"{recovered_true_matches:,}"
    )

    print(
        "Candidate Recall:",
        f"{candidate_recall:.4f}%"
    )

    print(
        "Total candidates:",
        f"{total_candidates:,}"
    )

    print(
        "Average candidates per S1:",
        f"{average_candidates:.2f}"
    )

    print()
    print("-" * 70)
    print("CANDIDATE SOURCE DIAGNOSTICS")
    print("-" * 70)

    print(
        "New candidates from exact name:",
        f"{exact_candidates_added:,}"
    )

    print(
        "New candidates from rare name tokens:",
        f"{name_candidates_added:,}"
    )

    print(
        "New candidates from rare address tokens:",
        f"{address_candidates_added:,}"
    )

    print(
        "New candidates from address numbers:",
        f"{number_candidates_added:,}"
    )

    print()
    print("=" * 70)
    print("COMPARISON WITH OUR V4")
    print("=" * 70)

    print()
    print(
        "Our V4:"
    )

    print(
        "  Candidate Recall: "
        "74.6942%"
    )

    print(
        "  Avg candidates/S1: "
        "133.30"
    )

    print()

    print(
        "Member 2:"
    )

    print(
        f"  Candidate Recall: "
        f"{candidate_recall:.4f}%"
    )

    print(
        f"  Avg candidates/S1: "
        f"{average_candidates:.2f}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("MEMBER 2 CANDIDATE COMPARISON")
    print("=" * 70)

    print()
    print("Validation S1:")
    print(VALIDATION_S1)

    print()
    print("Dataset:")
    print(DATASET)

    ground_truth = load_ground_truth()

    (
        name_counts,
        address_counts,
        number_counts
    ) = count_features()

    (
        exact_index,
        name_index,
        address_index,
        number_index
    ) = build_indexes(
        name_counts,
        address_counts,
        number_counts
    )

    evaluate(
        ground_truth,
        exact_index,
        name_index,
        address_index,
        number_index
    )


if __name__ == "__main__":
    main()