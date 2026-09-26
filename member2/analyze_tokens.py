
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


token_counts = Counter()


print("Counting business-name tokens...")


for name in source2["business_name_normalized"]:

    for token in name.split():

        if len(token) >= 3:
            token_counts[token] += 1


for name in source3["business_name_normalized"]:

    for token in name.split():

        if len(token) >= 3:
            token_counts[token] += 1


print()
print("========== TOKEN FREQUENCY ==========")

print(
    "Unique tokens:",
    len(token_counts)
)

print()
print("Most common tokens:")

for token, count in token_counts.most_common(50):

    print(
        f"{token:30} {count}"
    )
