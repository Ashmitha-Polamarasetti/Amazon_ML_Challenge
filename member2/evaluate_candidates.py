import pandas as pd

from candidate_generation import (
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

    print("Loading datasets...")

    source1 = load_source(
        rf"{DATASET}\train\train_source1.tsv"
    )

    source2 = load_source(
        rf"{DATASET}\train\train_source2.tsv"
    )

    source3 = load_source(
        rf"{DATASET}\train\train_source3.tsv"
    )

    ground_truth = load_ground_truth(
        rf"{DATASET}\train\train_ground_truth.tsv"
    )

    print("Normalizing datasets...")

    source1 = normalize_dataframe(source1)
    source2 = normalize_dataframe(source2)
    source3 = normalize_dataframe(source3)

    print("Generating candidates...")

    candidates = generate_candidates(
        source1,
        source2,
        source3
    )

    # Create lookup:
    # S1 entity -> set of candidate entity IDs
    candidate_map = {}

    for _, row in candidates.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        candidate_map.setdefault(
            s1_id, set()
        ).add(candidate_id)

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
    print("========== CANDIDATE RECALL ==========")
    print("Total true matches:", total_true_matches)
    print("True matches found:", found_true_matches)
    print(f"Candidate Recall: {recall:.4%}")