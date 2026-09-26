import pandas as pd
from collections import Counter

from normalization import normalize_dataframe

DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"


def load_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def get_tokens(name):
    if not name:
        return []

    return {
        token
        for token in name.split()
        if len(token) >= 4
    }


print("Loading Source 2...")
source2 = load_source(
    rf"{DATASET}\train\train_source2.tsv"
)

print("Loading Source 3...")
source3 = load_source(
    rf"{DATASET}\train\train_source3.tsv"
)

print("Normalizing...")

source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)

print("Counting tokens...")

counts = Counter()

for name in source2["business_name_normalized"]:
    for token in get_tokens(name):
        counts[token] += 1

for name in source3["business_name_normalized"]:
    for token in get_tokens(name):
        counts[token] += 1


print()
print("========== RARE TOKEN ANALYSIS ==========")
print("Unique tokens:", len(counts))

for limit in [10, 25, 50, 100, 250, 500, 1000]:

    number_of_tokens = sum(
        1 for count in counts.values()
        if count <= limit
    )

    print(
        f"Tokens with frequency <= {limit}: "
        f"{number_of_tokens}"
    )


print()
print("========== SAMPLE RARE TOKENS ==========")

rare_tokens = [
    (token, count)
    for token, count in counts.items()
    if 5 <= count <= 100
]

rare_tokens.sort(key=lambda x: x[1])

for token, count in rare_tokens[:100]:
    print(f"{token:<35} {count}")