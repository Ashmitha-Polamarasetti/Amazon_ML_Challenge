import re
import pandas as pd
from collections import Counter

from normalization import normalize_dataframe

DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"


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
    rf"{DATASET}\train\train_source1.tsv",
    nrows=10000
)

print("Loading full Source 2...")

source2 = load_source(
    rf"{DATASET}\train\train_source2.tsv"
)

print("Loading full Source 3...")

source3 = load_source(
    rf"{DATASET}\train\train_source3.tsv"
)


print("Normalizing...")

source1 = normalize_dataframe(source1)
source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)


# =========================================================
# 1. COUNT NAME TOKENS
# =========================================================

print("Counting name tokens...")

name_counts = Counter()

for df in [source2, source3]:

    for name in df["business_name_normalized"]:

        for token in get_tokens(name):
            name_counts[token] += 1


# =========================================================
# 2. COUNT ADDRESS TOKENS
# =========================================================

print("Counting address tokens...")

address_counts = Counter()

for df in [source2, source3]:

    for address in df["business_address_normalized"]:

        for token in get_tokens(address):
            address_counts[token] += 1


# =========================================================
# 3. COUNT ADDRESS NUMBERS
# =========================================================

print("Counting address numbers...")

number_counts = Counter()

for df in [source2, source3]:

    for address in df["business_address_normalized"]:

        for number in get_address_numbers(address):
            number_counts[number] += 1


# =========================================================
# 4. BUILD INDEXES
# =========================================================

print("Building indexes...")

name_index = {}
address_index = {}
exact_index = {}
number_index = {}


for df in [source2, source3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        country = row["country"].strip().lower()

        name = row["business_name_normalized"]

        address = row["business_address_normalized"]


        # -------------------------------------------------
        # Exact normalized name
        # -------------------------------------------------

        if name:

            exact_index.setdefault(
                (country, name),
                set()
            ).add(entity_id)


        # -------------------------------------------------
        # Rare name tokens
        # Frequency: 1 to 100
        # -------------------------------------------------

        for token in get_tokens(name):

            if 1 <= name_counts[token] <= 100:

                key = (country, token)

                name_index.setdefault(
                    key,
                    set()
                ).add(entity_id)


        # -------------------------------------------------
        # Rare address tokens
        # Frequency: 1 to 100
        # -------------------------------------------------

        for token in get_tokens(address):

            if 1 <= address_counts[token] <= 100:

                key = (country, token)

                address_index.setdefault(
                    key,
                    set()
                ).add(entity_id)


        # -------------------------------------------------
        # Address number blocking
        # Frequency: 2 to 500
        # -------------------------------------------------

        for number in get_address_numbers(address):

            if 2 <= number_counts[number] <= 500:

                key = (country, number)

                number_index.setdefault(
                    key,
                    set()
                ).add(entity_id)


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    rf"{DATASET}\train\train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)


test_ids = set(source1["entity_id"])

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(test_ids)
]


source1_lookup = source1.set_index("entity_id")


# =========================================================
# RECALL EVALUATION
# =========================================================

total_true_matches = 0
found_true_matches = 0


for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids = row["matched_entity_ids"]


    if not matched_ids:
        continue


    true_ids = set(
        matched_ids.split(",")
    )


    total_true_matches += len(true_ids)


    s1_row = source1_lookup.loc[s1_id]


    country = s1_row["country"].strip().lower()

    name = s1_row["business_name_normalized"]

    address = s1_row["business_address_normalized"]


    generated_ids = set()


    # =====================================================
    # 1. EXACT NORMALIZED NAME
    # =====================================================

    if name:

        generated_ids.update(
            exact_index.get(
                (country, name),
                set()
            )
        )


    # =====================================================
    # 2. RARE NAME TOKENS
    # Frequency: 1 to 100
    # =====================================================

    for token in get_tokens(name):

        if 1 <= name_counts[token] <= 100:

            generated_ids.update(
                name_index.get(
                    (country, token),
                    set()
                )
            )


    # =====================================================
    # 3. RARE ADDRESS TOKENS
    # Frequency: 1 to 100
    # =====================================================

    for token in get_tokens(address):

        if 1 <= address_counts[token] <= 100:

            generated_ids.update(
                address_index.get(
                    (country, token),
                    set()
                )
            )


    # =====================================================
    # 4. ADDRESS NUMBER BLOCKING
    # Frequency: 2 to 500
    # =====================================================

    for number in get_address_numbers(address):

        if 2 <= number_counts[number] <= 500:

            generated_ids.update(
                number_index.get(
                    (country, number),
                    set()
                )
            )


    # =====================================================
    # COUNT TRUE MATCHES FOUND
    # =====================================================

    found_true_matches += len(
        true_ids.intersection(generated_ids)
    )


# =========================================================
# FINAL RECALL
# =========================================================

recall = (
    found_true_matches / total_true_matches
    if total_true_matches
    else 0
)


print()

print("========== COMBINED BLOCKING RECALL ==========")

print(
    "Test Source 1 records:",
    len(source1)
)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "True matches found:",
    found_true_matches
)

print(
    f"Combined Recall: {recall:.4%}"
)