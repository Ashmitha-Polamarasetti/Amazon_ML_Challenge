import random
from pathlib import Path

import joblib
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODEL_PATH = ROOT / "matching_model.joblib"
CANDIDATES = DATA / "candidate_pairs.tsv"
OUTPUT = DATA / "matching_results.tsv"

CHUNK_SIZE = 100_000

# Start conservative.
# We will inspect the number of matches produced.
THRESHOLD = 0.80


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in value
    )

    return " ".join(value.split())


def token_jaccard(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def make_features(row1, row2):

    name1 = normalize_text(row1["business_name"])
    name2 = normalize_text(row2["business_name"])

    address1 = normalize_text(row1["business_address"])
    address2 = normalize_text(row2["business_address"])

    country1 = str(row1["country"]).strip().lower()
    country2 = str(row2["country"]).strip().lower()

    return [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        token_jaccard(name1, name2),

        fuzz.ratio(address1, address2) / 100.0,
        fuzz.token_sort_ratio(address1, address2) / 100.0,
        token_jaccard(address1, address2),

        int(country1 == country2),
    ]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("Loading Source 1...")

source1 = pd.read_csv(
    DATA / "train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Source1:", len(source1))


print("Loading Source 2...")

source2 = pd.read_csv(
    DATA / "train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Source2:", len(source2))


print("Loading Source 3...")

source3 = pd.read_csv(
    DATA / "train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Source3:", len(source3))


# ============================================================
# CREATE LOOKUPS
# ============================================================

print("Creating lookups...")

source1_lookup = source1.set_index("entity_id").to_dict("index")
source2_lookup = source2.set_index("entity_id").to_dict("index")
source3_lookup = source3.set_index("entity_id").to_dict("index")

del source1
del source2
del source3


# ============================================================
# PROCESS CANDIDATES
# ============================================================

print()
print("Starting candidate scoring...")

first_write = True

processed = 0
matched = 0


for chunk in pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE,
):

    feature_rows = []
    pair_ids = []

    for s1_id, candidate_id in zip(
        chunk["source1_entity_id"],
        chunk["candidate_entity_id"]
    ):

        row1 = source1_lookup.get(s1_id)

        if candidate_id.startswith("S2-"):
            row2 = source2_lookup.get(candidate_id)
        else:
            row2 = source3_lookup.get(candidate_id)

        if row1 is None or row2 is None:
            continue

        feature_rows.append(
            make_features(row1, row2)
        )

        pair_ids.append(
            (s1_id, candidate_id)
        )


    if feature_rows:

        probabilities = model.predict_proba(
            feature_rows
        )[:, 1]


        output_rows = []

        for (s1_id, candidate_id), probability in zip(
            pair_ids,
            probabilities
        ):

            if probability >= THRESHOLD:

                output_rows.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_id": candidate_id,
                    "match_probability": probability
                })


        if output_rows:

            result = pd.DataFrame(output_rows)

            result.to_csv(
                OUTPUT,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            matched += len(result)


    processed += len(chunk)

    print(
        f"Processed: {processed:,} | "
        f"Matches: {matched:,}"
    )


print()
print("========== PREDICTION COMPLETE ==========")
print("Candidate pairs processed:", f"{processed:,}")
print("Predicted matches:", f"{matched:,}")
print("Threshold:", THRESHOLD)
print("Output:", OUTPUT)