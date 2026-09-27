import re
import os
from collections import Counter, defaultdict

import pandas as pd

from normalization import normalize_dataframe


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "data"

SOURCE1 = DATASET / "train_source1.tsv"
SOURCE2 = DATASET / "train_source2.tsv"
SOURCE3 = DATASET / "train_source3.tsv"

OUTPUT = DATASET / "candidate_pairs.tsv"

CHUNK_SIZE = 100_000

MIN_TOKEN_LENGTH = 4

NAME_MIN_FREQ = 1
NAME_MAX_FREQ = 100

ADDRESS_MIN_FREQ = 1
ADDRESS_MAX_FREQ = 100

NUMBER_MIN_FREQ = 2
NUMBER_MAX_FREQ = 500


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


# ---------------------------------------------------------
# PASS 1: COUNT TOKENS AND ADDRESS NUMBERS
# ---------------------------------------------------------

def count_features():

    print()
    print("========== PASS 1: COUNT FEATURES ==========")

    name_counts = Counter()
    address_counts = Counter()
    number_counts = Counter()

    for source_name, path in [
        ("SOURCE2", SOURCE2),
        ("SOURCE3", SOURCE3)
    ]:

        print(f"\nCounting {source_name}...")

        chunk_number = 0

        for df in load_chunks(path):

            chunk_number += 1

            df = normalize_dataframe(df)

            for name in df["business_name_normalized"]:
                tokens = set(get_tokens(name))

                for token in tokens:
                    name_counts[token] += 1

            for address in df["business_address_normalized"]:
                tokens = set(get_tokens(address))

                for token in tokens:
                    address_counts[token] += 1

                numbers = get_address_numbers(address)

                for number in numbers:
                    number_counts[number] += 1

            if chunk_number % 10 == 0:
                print(
                    f"{source_name}: processed "
                    f"{chunk_number * CHUNK_SIZE:,} rows"
                )

    print()
    print("Unique name tokens:", len(name_counts))
    print("Unique address tokens:", len(address_counts))
    print("Unique address numbers:", len(number_counts))

    return name_counts, address_counts, number_counts


# ---------------------------------------------------------
# PASS 2: BUILD INDEXES
# ---------------------------------------------------------

def build_indexes(name_counts, address_counts, number_counts):

    print()
    print("========== PASS 2: BUILD INDEXES ==========")

    exact_index = defaultdict(set)
    name_index = defaultdict(set)
    address_index = defaultdict(set)
    number_index = defaultdict(set)

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

    print("Eligible name tokens:", len(eligible_name_tokens))
    print("Eligible address tokens:", len(eligible_address_tokens))
    print("Eligible address numbers:", len(eligible_numbers))

    for source_name, path in [
        ("SOURCE2", SOURCE2),
        ("SOURCE3", SOURCE3)
    ]:

        print(f"\nBuilding indexes from {source_name}...")

        chunk_number = 0

        for df in load_chunks(path):

            chunk_number += 1

            df = normalize_dataframe(df)

            for _, row in df.iterrows():

                entity_id = row["entity_id"]
                country = row["country"].strip().lower()

                name = row["business_name_normalized"]
                address = row["business_address_normalized"]

                if not country:
                    continue

                # Exact normalized name
                if name:
                    exact_key = (country, name)
                    exact_index[exact_key].add(entity_id)

                # Rare name tokens
                for token in set(get_tokens(name)):

                    if token in eligible_name_tokens:
                        key = (country, token)
                        name_index[key].add(entity_id)

                # Rare address tokens
                for token in set(get_tokens(address)):

                    if token in eligible_address_tokens:
                        key = (country, token)
                        address_index[key].add(entity_id)

                # Address numbers
                for number in get_address_numbers(address):

                    if number in eligible_numbers:
                        key = (country, number)
                        number_index[key].add(entity_id)

            if chunk_number % 10 == 0:
                print(
                    f"{source_name}: indexed "
                    f"{chunk_number * CHUNK_SIZE:,} rows"
                )

    print()
    print("Exact-name blocks:", len(exact_index))
    print("Name-token blocks:", len(name_index))
    print("Address-token blocks:", len(address_index))
    print("Address-number blocks:", len(number_index))

    return (
        exact_index,
        name_index,
        address_index,
        number_index
    )


# ---------------------------------------------------------
# PASS 3: GENERATE CANDIDATES
# ---------------------------------------------------------

def generate_candidates(
    exact_index,
    name_index,
    address_index,
    number_index
):

    print()
    print("========== PASS 3: GENERATE CANDIDATES ==========")

    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    first_write = True

    total_candidates = 0
    total_source1 = 0

    for chunk_number, df in enumerate(load_chunks(SOURCE1), start=1):

        df = normalize_dataframe(df)

        output_rows = []

        for _, row in df.iterrows():

            s1_id = row["entity_id"]
            country = row["country"].strip().lower()

            name = row["business_name_normalized"]
            address = row["business_address_normalized"]

            candidate_ids = set()

            # -----------------------------------------
            # 1. EXACT NORMALIZED NAME + COUNTRY
            # -----------------------------------------

            if name:

                key = (country, name)

                candidate_ids.update(
                    exact_index.get(key, ())
                )

            # -----------------------------------------
            # 2. RARE NAME TOKEN + COUNTRY
            # -----------------------------------------

            for token in set(get_tokens(name)):

                key = (country, token)

                candidate_ids.update(
                    name_index.get(key, ())
                )

            # -----------------------------------------
            # 3. RARE ADDRESS TOKEN + COUNTRY
            # -----------------------------------------

            for token in set(get_tokens(address)):

                key = (country, token)

                candidate_ids.update(
                    address_index.get(key, ())
                )

            # -----------------------------------------
            # 4. ADDRESS NUMBER + COUNTRY
            # -----------------------------------------

            for number in get_address_numbers(address):

                key = (country, number)

                candidate_ids.update(
                    number_index.get(key, ())
                )

            # Remove self-match just in case
            candidate_ids.discard(s1_id)

            for candidate_id in candidate_ids:

                output_rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": candidate_id
                    }
                )

            total_candidates += len(candidate_ids)
            total_source1 += 1

        # Write this chunk immediately
        if output_rows:

            output_df = pd.DataFrame(output_rows)

            output_df.to_csv(
                OUTPUT,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

        if chunk_number % 5 == 0:

            average = (
                total_candidates / total_source1
                if total_source1
                else 0
            )

            print(
                f"Processed Source1: "
                f"{total_source1:,} | "
                f"Candidates: {total_candidates:,} | "
                f"Avg candidates/S1: {average:.2f}"
            )

    print()
    print("========== CANDIDATE GENERATION COMPLETE ==========")

    print("Source1 records:", f"{total_source1:,}")
    print("Candidate pairs:", f"{total_candidates:,}")

    average = (
        total_candidates / total_source1
        if total_source1
        else 0
    )

    print("Average candidates per Source1:", f"{average:.2f}")

    print("Output file:")
    print(OUTPUT)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting final candidate generation...")

    print()
    print("Dataset:")
    print(DATASET)

    # Pass 1
    name_counts, address_counts, number_counts = count_features()

    # Pass 2
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

    # Pass 3
    generate_candidates(
        exact_index,
        name_index,
        address_index,
        number_index
    )