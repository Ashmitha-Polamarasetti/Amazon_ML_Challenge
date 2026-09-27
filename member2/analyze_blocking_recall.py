import re
from collections import Counter

import pandas as pd

from normalization import normalize_dataframe


DATASET = "../data"


def load_source(path, nrows=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )


def get_tokens(text):
    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 4
    }


def get_address_numbers(address):
    if not address:
        return set()

    return set(re.findall(r"\d+", address))


print("Loading Source 1...")
source1 = load_source(
    f"{DATASET}/train_source1.tsv",
    nrows=10000
)

print("Loading Source 2...")
source2 = load_source(
    f"{DATASET}/train_source2.tsv"
)

print("Loading Source 3...")
source3 = load_source(
    f"{DATASET}/train_source3.tsv"
)


print("Normalizing...")

source1 = normalize_dataframe(source1)
source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)


# ---------------------------------------------------------
# COUNT TOKENS / NUMBERS
# ---------------------------------------------------------

print("Counting blocking features...")

name_counts = Counter()
address_counts = Counter()
number_counts = Counter()

for df in [source2, source3]:

    for name in df["business_name_normalized"]:

        for token in get_tokens(name):
            name_counts[token] += 1

    for address in df["business_address_normalized"]:

        for token in get_tokens(address):
            address_counts[token] += 1

        for number in get_address_numbers(address):
            number_counts[number] += 1


# ---------------------------------------------------------
# BUILD INDEXES
# ---------------------------------------------------------

print("Building indexes...")

exact_index = {}
name_index = {}
address_index = {}
number_index = {}


for df in [source2, source3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]
        country = row["country"].strip().lower()

        name = row["business_name_normalized"]
        address = row["business_address_normalized"]


        # Exact name
        if name:

            exact_index.setdefault(
                (country, name),
                set()
            ).add(entity_id)


        # Rare name tokens
        for token in get_tokens(name):

            if 1 <= name_counts[token] <= 100:

                name_index.setdefault(
                    (country, token),
                    set()
                ).add(entity_id)


        # Rare address tokens
        for token in get_tokens(address):

            if 1 <= address_counts[token] <= 100:

                address_index.setdefault(
                    (country, token),
                    set()
                ).add(entity_id)


        # Address numbers
        for number in get_address_numbers(address):

            if 2 <= number_counts[number] <= 500:

                number_index.setdefault(
                    (country, number),
                    set()
                ).add(entity_id)


# ---------------------------------------------------------
# LOAD GROUND TRUTH
# ---------------------------------------------------------

print("Loading ground truth...")

ground_truth = pd.read_csv(
    f"{DATASET}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

test_ids = set(source1["entity_id"])

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(test_ids)
]

source1_lookup = source1.set_index("entity_id")


# ---------------------------------------------------------
# RECALL BY BLOCKING RULE
# ---------------------------------------------------------

total_true_matches = 0

found_exact = set()
found_name = set()
found_address = set()
found_number = set()
found_any = set()


for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    if not matched_ids:
        continue

    true_ids = {
        x.strip()
        for x in matched_ids.split(",")
        if x.strip()
    }

    total_true_matches += len(true_ids)

    s1_row = source1_lookup.loc[s1_id]

    country = s1_row["country"].strip().lower()
    name = s1_row["business_name_normalized"]
    address = s1_row["business_address_normalized"]


    exact_ids = set()

    if name:

        exact_ids = exact_index.get(
            (country, name),
            set()
        )


    name_ids = set()

    for token in get_tokens(name):

        if 1 <= name_counts[token] <= 100:

            name_ids.update(
                name_index.get(
                    (country, token),
                    set()
                )
            )


    address_ids = set()

    for token in get_tokens(address):

        if 1 <= address_counts[token] <= 100:

            address_ids.update(
                address_index.get(
                    (country, token),
                    set()
                )
            )


    number_ids = set()

    for number in get_address_numbers(address):

        if 2 <= number_counts[number] <= 500:

            number_ids.update(
                number_index.get(
                    (country, number),
                    set()
                )
            )


    found_exact.update(
        (s1_id, x)
        for x in true_ids.intersection(exact_ids)
    )

    found_name.update(
        (s1_id, x)
        for x in true_ids.intersection(name_ids)
    )

    found_address.update(
        (s1_id, x)
        for x in true_ids.intersection(address_ids)
    )

    found_number.update(
        (s1_id, x)
        for x in true_ids.intersection(number_ids)
    )

    generated_ids = (
        exact_ids
        | name_ids
        | address_ids
        | number_ids
    )

    found_any.update(
        (s1_id, x)
        for x in true_ids.intersection(generated_ids)
    )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("========== BLOCKING RECALL ANALYSIS ==========")

print("Source 1 records:", len(source1))
print("Total true matches:", total_true_matches)

print()
print("Exact name matches:", len(found_exact))
print("Name-token matches:", len(found_name))
print("Address-token matches:", len(found_address))
print("Address-number matches:", len(found_number))
print("Combined matches:", len(found_any))

print()

if total_true_matches:

    print(
        f"Exact name recall: "
        f"{len(found_exact) / total_true_matches:.4%}"
    )

    print(
        f"Name-token recall: "
        f"{len(found_name) / total_true_matches:.4%}"
    )

    print(
        f"Address-token recall: "
        f"{len(found_address) / total_true_matches:.4%}"
    )

    print(
        f"Address-number recall: "
        f"{len(found_number) / total_true_matches:.4%}"
    )

    print(
        f"Combined recall: "
        f"{len(found_any) / total_true_matches:.4%}"
    )

print()
print("Missing true matches:", total_true_matches - len(found_any))