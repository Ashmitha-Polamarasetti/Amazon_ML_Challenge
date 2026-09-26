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


def get_address_numbers(address):
    if not address:
        return set()

    # Extract numeric components from the normalized address.
    # Examples: 1795, 570, 13, 560001
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


print("Counting address numbers...")

number_counts = Counter()

for df in [source2, source3]:

    for address in df["business_address_normalized"]:

        for number in get_address_numbers(address):
            number_counts[number] += 1


print("Building address-number country index...")

index = {}

for df in [source2, source3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]
        country = row["country"].strip().lower()
        address = row["business_address_normalized"]

        for number in get_address_numbers(address):

            # Ignore extremely common numbers.
            # Keep numbers occurring 2 to 500 times.
            if 2 <= number_counts[number] <= 500:

                key = (country, number)

                index.setdefault(
                    key,
                    set()
                ).add(entity_id)


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


total_true_matches = 0
found_true_matches = 0


for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    if not matched_ids:
        continue

    true_ids = set(matched_ids.split(","))

    total_true_matches += len(true_ids)

    s1_row = source1_lookup.loc[s1_id]

    country = s1_row["country"].strip().lower()
    address = s1_row["business_address_normalized"]

    generated_ids = set()

    for number in get_address_numbers(address):

        if 2 <= number_counts[number] <= 500:

            key = (country, number)

            generated_ids.update(
                index.get(key, set())
            )

    found_true_matches += len(
        true_ids.intersection(generated_ids)
    )


recall = (
    found_true_matches / total_true_matches
    if total_true_matches
    else 0
)


print()
print("========== ADDRESS NUMBER COVERAGE ==========")
print("Test Source 1 records:", len(source1))
print("Total true matches:", total_true_matches)
print("True matches found:", found_true_matches)
print(f"Address-number Recall: {recall:.4%}")