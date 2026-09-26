
import pandas as pd
from collections import Counter

from normalization import normalize_dataframe


DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"

# Maximum number of records allowed in one token block.
# Very common tokens are ignored.
MAX_BLOCK_SIZE = 500

# Minimum token length.
MIN_TOKEN_LENGTH = 3


def load_source(path, nrows=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )


def get_tokens(name):
    """
    Get useful tokens from a normalized business name.
    """

    if not name:
        return []

    tokens = set()

    for token in name.split():

        if len(token) >= MIN_TOKEN_LENGTH:
            tokens.add(token)

    return list(tokens)


# ---------------------------------------------------------
# STEP 1: Count token frequency
# ---------------------------------------------------------

def count_tokens(source2, source3):

    counts = Counter()

    for name in source2["business_name_normalized"]:

        for token in get_tokens(name):
            counts[token] += 1

    for name in source3["business_name_normalized"]:

        for token in get_tokens(name):
            counts[token] += 1

    return counts


# ---------------------------------------------------------
# STEP 2: Build exact-name index
# ---------------------------------------------------------

def build_exact_name_index(df):

    index = {}

    for _, row in df.iterrows():

        name = row["business_name_normalized"]

        if not name:
            continue

        index.setdefault(
            name,
            []
        ).append(
            row["entity_id"]
        )

    return index


# ---------------------------------------------------------
# STEP 3: Build selective token index
# ---------------------------------------------------------

def build_token_index(df, token_counts):

    index = {}

    for _, row in df.iterrows():

        name = row["business_name_normalized"]

        for token in get_tokens(name):

            frequency = token_counts[token]

            # Ignore very common tokens.
            if frequency > MAX_BLOCK_SIZE:
                continue

            index.setdefault(
                token,
                []
            ).append(
                row["entity_id"]
            )

    return index


# ---------------------------------------------------------
# STEP 4: Generate candidates
# ---------------------------------------------------------

def generate_candidates(
    source1,
    source2,
    source3
):

    print("Counting token frequencies...")

    token_counts = count_tokens(
        source2,
        source3
    )

    print(
        "Unique tokens:",
        len(token_counts)
    )

    print("Building exact-name indexes...")

    source2_exact = build_exact_name_index(
        source2
    )

    source3_exact = build_exact_name_index(
        source3
    )

    print("Building selective token indexes...")

    source2_token = build_token_index(
        source2,
        token_counts
    )

    source3_token = build_token_index(
        source3,
        token_counts
    )

    candidates = []

    print("Generating candidates...")

    for counter, (_, row) in enumerate(
        source1.iterrows(),
        start=1
    ):

        s1_id = row["entity_id"]

        name = row[
            "business_name_normalized"
        ]

        candidate_ids = set()

        # -------------------------------------------------
        # PASS 1: Exact normalized name
        # -------------------------------------------------

        if name:

            candidate_ids.update(
                source2_exact.get(
                    name,
                    []
                )
            )

            candidate_ids.update(
                source3_exact.get(
                    name,
                    []
                )
            )

        # -------------------------------------------------
        # PASS 2: Selective tokens
        # -------------------------------------------------

        tokens = get_tokens(name)

        for token in tokens:

            # Only use selective tokens.
            if token_counts[token] > MAX_BLOCK_SIZE:
                continue

            candidate_ids.update(
                source2_token.get(
                    token,
                    []
                )
            )

            candidate_ids.update(
                source3_token.get(
                    token,
                    []
                )
            )

        # -------------------------------------------------
        # Store candidates
        # -------------------------------------------------

        for candidate_id in candidate_ids:

            candidates.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id
                }
            )

        # Progress display
        if counter % 1000 == 0:

            print(
                "Processed Source 1:",
                counter
            )

    return pd.DataFrame(
        candidates
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Loading small test datasets..."
    )

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

    source1 = normalize_dataframe(
        source1
    )

    source2 = normalize_dataframe(
        source2
    )

    source3 = normalize_dataframe(
        source3
    )

    candidates = generate_candidates(
        source1,
        source2,
        source3
    )

    print()
    print(
        "========== SELECTIVE TOKEN TEST =========="
    )

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
        candidates.head(20).to_string(
            index=False
        )
    )
