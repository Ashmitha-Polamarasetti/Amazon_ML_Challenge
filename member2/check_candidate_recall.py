import pandas as pd


GROUND_TRUTH = "../data/train_ground_truth.tsv"
CANDIDATES = "../data/candidate_pairs.tsv"


print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


# Build a set containing every true S1 -> S2/S3 match
true_pairs = set()

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    if not matched_ids:
        continue

    for candidate_id in matched_ids.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id:
            true_pairs.add((s1_id, candidate_id))


print("True matching pairs:", len(true_pairs))


# Stream candidate file
found_pairs = set()
candidate_count = 0

print("\nChecking candidate pairs...")

for chunk in pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=1_000_000
):

    candidate_count += len(chunk)

    for s1_id, candidate_id in zip(
        chunk["source1_entity_id"],
        chunk["candidate_entity_id"]
    ):
        pair = (s1_id, candidate_id)

        if pair in true_pairs:
            found_pairs.add(pair)

    print(
        f"Processed candidates: {candidate_count:,} | "
        f"True matches found: {len(found_pairs):,}"
    )


print("\n========== BLOCKING RECALL ==========")

print("True matching pairs:", len(true_pairs))
print("True matches found:", len(found_pairs))

if true_pairs:
    recall = len(found_pairs) / len(true_pairs)
else:
    recall = 0

print(f"Candidate recall: {recall:.4%}")

missing_pairs = true_pairs - found_pairs

print("Missing true matches:", len(missing_pairs))