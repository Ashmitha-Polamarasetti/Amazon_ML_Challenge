import pandas as pd

from blocking_candidates import (
    load_source,
    generate_candidates
)

from normalization import normalize_dataframe


DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"


def load_ground_truth(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


if __name__ == "__main__":

    print("Loading small test datasets...")

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

    print("Generating candidates...")

    candidates = generate_candidates(
        source1,
        source2,
        source3
    )

    # Build candidate lookup
    candidate_map = {}

    for _, row in candidates.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        candidate_map.setdefault(
            s1_id,
            set()
        ).add(candidate_id)

    print("Loading ground truth...")

    ground_truth = load_ground_truth(
        rf"{DATASET}\train\train_ground_truth.tsv"
    )

    # Only evaluate S1 records present in our small test
    test_s1_ids = set(source1["entity_id"])

    ground_truth = ground_truth[
        ground_truth["source1_entity_id"].isin(test_s1_ids)
    ]

    total_true_matches = 0
    found_true_matches = 0

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        true_ids = set(
            matched_ids.split(",")
        )

        total_true_matches += len(true_ids)

        generated_ids = candidate_map.get(
            s1_id,
            set()
        )

        found_true_matches += len(
            true_ids.intersection(generated_ids)
        )

    recall = (
        found_true_matches / total_true_matches
        if total_true_matches > 0
        else 0
    )

    print()
    print("========== BLOCKING RECALL ==========")
    print("Test Source 1 records:", len(source1))
    print("Candidate pairs:", len(candidates))
    print("Total true matches:", total_true_matches)
    print("True matches found:", found_true_matches)
    print(f"Candidate Recall: {recall:.4%}")
    