
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


# ---------------------------------------------------------
# PASS 1: Exact normalized business name
# ---------------------------------------------------------

def build_exact_name_index(df):
    index = {}

    for _, row in df.iterrows():

        name = row["business_name_normalized"]

        if not name:
            continue

        index.setdefault(name, []).append(
            row["entity_id"]
        )

    return index


# ---------------------------------------------------------
# PASS 2: Country + first 5 characters
# ---------------------------------------------------------

def create_prefix_block_key(name, country):

    name = str(name).strip().lower()
    country = str(country).strip().lower()

    if not name:
        return ""

    prefix = name[:5]

    return f"{country}_{prefix}"


def build_prefix_index(df):

    index = {}

    for _, row in df.iterrows():

        key = create_prefix_block_key(
            row["business_name_normalized"],
            row["country"]
        )

        if not key:
            continue

        index.setdefault(key, []).append(
            row["entity_id"]
        )

    return index


# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------

def generate_candidates(source1, source2, source3):

    print("Building exact-name indexes...")

    source2_exact = build_exact_name_index(source2)
    source3_exact = build_exact_name_index(source3)

    print("Building prefix indexes...")

    source2_prefix = build_prefix_index(source2)
    source3_prefix = build_prefix_index(source3)

    candidates = []

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]

        name = row["business_name_normalized"]

        country = row["country"]

        candidate_ids = set()

        # ---------------------------------------------
        # PASS 1: Exact normalized name
        # ---------------------------------------------

        if name:

            candidate_ids.update(
                source2_exact.get(name, [])
            )

            candidate_ids.update(
                source3_exact.get(name, [])
            )

        # ---------------------------------------------
        # PASS 2: Country + first 5 characters
        # ---------------------------------------------

        prefix_key = create_prefix_block_key(
            name,
            country
        )

        if prefix_key:

            candidate_ids.update(
                source2_prefix.get(prefix_key, [])
            )

            candidate_ids.update(
                source3_prefix.get(prefix_key, [])
            )

        # ---------------------------------------------
        # Store unique candidates
        # ---------------------------------------------

        for candidate_id in candidate_ids:

            candidates.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id
                }
            )

    return pd.DataFrame(candidates)


# ---------------------------------------------------------
# Small test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Loading small test sample...")

    source1 = load_source(
        rf"{DATASET}\train\train_source1.tsv",
        nrows=10000
    )

    source2 = load_source(
        rf"{DATASET}\train\train_source2.tsv",
        nrows=30000
    )

    source3 = load_source(
        rf"{DATASET}\train\train_source3.tsv",
        nrows=30000
    )

    print("Normalizing...")

    source1 = normalize_dataframe(source1)
    source2 = normalize_dataframe(source2)
    source3 = normalize_dataframe(source3)

    print("Generating multi-pass candidates...")

    candidates = generate_candidates(
        source1,
        source2,
        source3
    )

    print()
    print("========== MULTI-PASS TEST RESULT ==========")

    print(
        "Source 1 records:",
        len(source1)
    )

    print(
        "Source 2 records:",
        len(source2)
    )

    print(
        "Source 3 records:",
        len(source3)
    )

    print(
        "Candidate pairs:",
        len(candidates)
    )

    print()

    print("Sample candidates:")

    print(
        candidates.head(20).to_string(index=False)
    )
