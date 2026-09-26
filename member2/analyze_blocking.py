import pandas as pd

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


def load_ground_truth(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def prefix_key(name, country, length):
    name = str(name).strip().lower()
    country = str(country).strip().lower()

    if not name:
        return ""

    return f"{country}_{name[:length]}"


print("Loading first 10,000 Source 1 records...")

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

print("Loading ground truth...")

ground_truth = load_ground_truth(
    rf"{DATASET}\train\train_ground_truth.tsv"
)

print("Normalizing...")

source1 = normalize_dataframe(source1)
source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)


# ---------------------------------------------------------
# Build complete lookup of Source 2 + Source 3
# ---------------------------------------------------------

all_sources = pd.concat(
    [source2, source3],
    ignore_index=True
)


id_lookup = all_sources.set_index(
    "entity_id"
)


# ---------------------------------------------------------
# Analyze true matches
# ---------------------------------------------------------

test_ids = set(source1["entity_id"])

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(test_ids)
]


total_matches = 0

prefix_3_matches = 0
prefix_5_matches = 0
prefix_7_matches = 0

country_matches = 0

examples = []


source1_lookup = source1.set_index(
    "entity_id"
)


for _, gt in ground_truth.iterrows():

    s1_id = gt["source1_entity_id"]

    if s1_id not in source1_lookup.index:
        continue

    s1 = source1_lookup.loc[s1_id]

    s1_name = s1["business_name_normalized"]
    s1_country = str(s1["country"]).strip().lower()

    true_ids = gt["matched_entity_ids"].split(",")

    for true_id in true_ids:

        if true_id not in id_lookup.index:
            continue

        true_row = id_lookup.loc[true_id]

        true_name = true_row[
            "business_name_normalized"
        ]

        true_country = str(
            true_row["country"]
        ).strip().lower()

        total_matches += 1

        if s1_country == true_country:
            country_matches += 1

        if (
            prefix_key(s1_name, s1_country, 3)
            == prefix_key(true_name, true_country, 3)
        ):
            prefix_3_matches += 1

        if (
            prefix_key(s1_name, s1_country, 5)
            == prefix_key(true_name, true_country, 5)
        ):
            prefix_5_matches += 1

        if (
            prefix_key(s1_name, s1_country, 7)
            == prefix_key(true_name, true_country, 7)
        ):
            prefix_7_matches += 1

        if len(examples) < 20:
            examples.append(
                {
                    "s1_id": s1_id,
                    "s1_name": s1_name,
                    "true_id": true_id,
                    "true_name": true_name,
                    "country_same":
                        s1_country == true_country,
                    "prefix3_same":
                        prefix_key(
                            s1_name,
                            s1_country,
                            3
                        )
                        ==
                        prefix_key(
                            true_name,
                            true_country,
                            3
                        ),
                    "prefix5_same":
                        prefix_key(
                            s1_name,
                            s1_country,
                            5
                        )
                        ==
                        prefix_key(
                            true_name,
                            true_country,
                            5
                        )
                }
            )


print()
print("========== BLOCKING DIAGNOSTIC ==========")

print(
    "Total true matches available:",
    total_matches
)

print(
    "Same country:",
    country_matches,
    f"({country_matches / total_matches:.4%})"
)

print(
    "Same country + first 3 chars:",
    prefix_3_matches,
    f"({prefix_3_matches / total_matches:.4%})"
)

print(
    "Same country + first 5 chars:",
    prefix_5_matches,
    f"({prefix_5_matches / total_matches:.4%})"
)

print(
    "Same country + first 7 chars:",
    prefix_7_matches,
    f"({prefix_7_matches / total_matches:.4%})"
)

print()
print("========== EXAMPLES ==========")

for example in examples:
    print(example)