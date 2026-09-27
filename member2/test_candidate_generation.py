import pandas as pd

from normalization import normalize_dataframe


# Load only a small number of rows
source1 = pd.read_csv(
    "../data/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=100
)

source2 = pd.read_csv(
    "../data/train_source2.tsv",
    sep="\t",
    dtype=str,
    nrows=1000
)

source3 = pd.read_csv(
    "../data/train_source3.tsv",
    sep="\t",
    dtype=str,
    nrows=1000
)


# Normalize
source1 = normalize_dataframe(source1)
source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)


print("Source 1 rows:", len(source1))
print("Source 2 rows:", len(source2))
print("Source 3 rows:", len(source3))

print("\nNormalized S1 example:")
print(source1.iloc[0][[
    "entity_id",
    "business_name_normalized",
    "business_address_normalized"
]])


# Simple exact normalized-name + country index
name_index = {}

for _, row in pd.concat([source2, source3]).iterrows():

    name = row["business_name_normalized"]
    country = row["country"].strip().lower()

    if not name or not country:
        continue

    key = (country, name)

    if key not in name_index:
        name_index[key] = set()

    name_index[key].add(row["entity_id"])


# Generate simple candidates
candidate_count = 0

for _, row in source1.iterrows():

    name = row["business_name_normalized"]
    country = row["country"].strip().lower()

    key = (country, name)

    candidates = name_index.get(key, set())

    candidate_count += len(candidates)

    if candidates:
        print("\nS1:", row["entity_id"])
        print("Name:", row["business_name"])
        print("Candidates:", candidates)


print("\nTotal exact-name candidates:", candidate_count)