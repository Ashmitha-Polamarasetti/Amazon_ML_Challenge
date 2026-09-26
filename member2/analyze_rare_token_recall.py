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


def get_tokens(name):
    if not name:
        return set()

    return {
        token
        for token in name.split()
        if len(token) >= 4
    }


def build_token_country_index(df, token_counts):
    index = {}

    for _, row in df.iterrows():

        name = row["business_name_normalized"]
        country = row["country"].strip().lower()

        for token in get_tokens(name):

            frequency = token_counts[token]

            if 5 <= frequency <= 100:

                key = (country, token)

                index.setdefault(key, set()).add(
                    row["entity_id"]
                )

    return index


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


print("Counting tokens...")

token_counts = Counter()

for df in [source2, source3]:

    for name in df["business_name_normalized"]:

        for token in get_tokens(name):
            token_counts[token] += 1


print("Building token-country index...")

index = build_token_country_index(
    pd.concat([source2, source3], ignore_index=True),
    token_counts
)


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

    name = s1_row["business_name_normalized"]

    generated_ids = set()

    for token in get_tokens(name):

        frequency = token_counts[token]

        if 5 <= frequency <= 100:

            key = (country, token)

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
print("========== RARE TOKEN COVERAGE ==========")
print("Test Source 1 records:", len(source1))
print("Total true matches:", total_true_matches)
print("True matches found:", found_true_matches)
print(f"Rare-token Recall: {recall:.4%}")