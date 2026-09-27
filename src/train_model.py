import random
from pathlib import Path

import pandas as pd
import numpy as np

from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

GROUND_TRUTH = DATA / "train_ground_truth.tsv"
CANDIDATES = DATA / "candidate_pairs.tsv"

MODEL_PATH = ROOT / "matching_model.joblib"

POSITIVE_TARGET = 50_000
NEGATIVE_TARGET = 50_000

CHUNK_SIZE = 500_000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Keep letters/numbers/spaces.
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
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

positive_pairs = set()

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if not matched:
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id:
            positive_pairs.add(
                (s1_id, candidate_id)
            )


print(
    f"True matching pairs loaded: "
    f"{len(positive_pairs):,}"
)


# ============================================================
# SAMPLE CANDIDATE PAIRS
# ============================================================

print()
print("Sampling candidate pairs...")

positive_samples = []
negative_samples = []

seen_positive = set()

rng = random.Random(RANDOM_SEED)

processed = 0

for chunk in pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    for s1_id, candidate_id in zip(
        chunk["source1_entity_id"],
        chunk["candidate_entity_id"]
    ):

        pair = (s1_id, candidate_id)

        if pair in positive_pairs:

            if len(positive_samples) < POSITIVE_TARGET:

                positive_samples.append(pair)

        else:

            # Randomly collect negatives instead of taking
            # only the first negatives in the file.
            if len(negative_samples) < NEGATIVE_TARGET:

                negative_samples.append(pair)

            else:

                index = rng.randrange(processed + 1)

                if index < NEGATIVE_TARGET:
                    negative_samples[index] = pair

        processed += 1

    print(
        f"Processed candidates: {processed:,} | "
        f"positives: {len(positive_samples):,} | "
        f"negatives: {len(negative_samples):,}"
    )

    if (
        len(positive_samples) >= POSITIVE_TARGET
        and len(negative_samples) >= NEGATIVE_TARGET
    ):
        break


print()
print("========== SAMPLE ==========")
print("Positive pairs:", len(positive_samples))
print("Negative pairs:", len(negative_samples))


# ============================================================
# COLLECT REQUIRED ENTITY IDS
# ============================================================

needed_s1 = {
    x[0]
    for x in positive_samples + negative_samples
}

needed_candidates = {
    x[1]
    for x in positive_samples + negative_samples
}


print()
print("Required Source1 records:", len(needed_s1))
print("Required candidate records:", len(needed_candidates))


# ============================================================
# LOAD ONLY REQUIRED RECORDS
# ============================================================

def load_required_records(path, needed_ids):

    result = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        selected = chunk[
            chunk["entity_id"].isin(needed_ids)
        ]

        for _, row in selected.iterrows():
            result[row["entity_id"]] = row.to_dict()

        if len(result) >= len(needed_ids):
            break

    return result


print()
print("Loading required Source 1 records...")

source1_lookup = load_required_records(
    DATA / "train_source1.tsv",
    needed_s1
)

print(
    "Source1 loaded:",
    len(source1_lookup)
)


print()
print("Loading required Source 2 records...")

source2_lookup = load_required_records(
    DATA / "train_source2.tsv",
    needed_candidates
)

print(
    "Source2 loaded:",
    len(source2_lookup)
)


remaining_candidates = needed_candidates - set(source2_lookup)

print(
    "Candidate records still needed from Source3:",
    len(remaining_candidates)
)


print()
print("Loading required Source 3 records...")

source3_lookup = load_required_records(
    DATA / "train_source3.tsv",
    remaining_candidates
)

print(
    "Source3 loaded:",
    len(source3_lookup)
)


# Combine candidate lookup.

candidate_lookup = {}

candidate_lookup.update(source2_lookup)
candidate_lookup.update(source3_lookup)


# ============================================================
# BUILD TRAINING DATA
# ============================================================

print()
print("Building training features...")

X = []
y = []

all_samples = [
    (pair, 1)
    for pair in positive_samples
] + [
    (pair, 0)
    for pair in negative_samples
]

random.shuffle(all_samples)

missing = 0

for (s1_id, candidate_id), label in all_samples:

    row1 = source1_lookup.get(s1_id)
    row2 = candidate_lookup.get(candidate_id)

    if row1 is None or row2 is None:
        missing += 1
        continue

    X.append(
        make_features(row1, row2)
    )

    y.append(label)


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)


print()
print("Training rows:", len(X))
print("Positive labels:", int(y.sum()))
print("Negative labels:", int(len(y) - y.sum()))
print("Missing records:", missing)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("Training classifier...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_SEED
)

model.fit(X, y)


# ============================================================
# TRAINING EVALUATION
# ============================================================

pred = model.predict(X)

print()
print("========== TRAINING RESULTS ==========")

print(
    classification_report(
        y,
        pred,
        digits=4
    )
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print()
print("Model saved to:")
print(MODEL_PATH)

print()
print("========== TRAINING COMPLETE ==========")