import pandas as pd

DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"

GROUND_TRUTH = rf"{DATASET}\train\train_ground_truth.tsv"
CANDIDATES = r"C:\Amazon_ML_Hackathon\member2\final_candidates.tsv"

CHUNK_SIZE = 1_000_000


def load_ground_truth():

    print("Loading ground truth...")

    df = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    ground_truth = {}

    total_true_matches = 0

    for _, row in df.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        true_ids = set(matched_ids.split(","))

        ground_truth[s1_id] = true_ids

        total_true_matches += len(true_ids)

    print("Source1 records with ground truth:", len(ground_truth))
    print("Total true matches:", f"{total_true_matches:,}")

    return ground_truth, total_true_matches


def evaluate_candidates(ground_truth):

    print()
    print("========== EVALUATING CANDIDATES ==========")

    found_matches = set()

    processed_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            CANDIDATES,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        processed_rows += len(chunk)

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id
            candidate_id = row.candidate_entity_id

            true_ids = ground_truth.get(s1_id)

            if true_ids is not None and candidate_id in true_ids:

                found_matches.add(
                    (s1_id, candidate_id)
                )

        if chunk_number % 10 == 0:

            print(
                f"Processed candidate rows: "
                f"{processed_rows:,} | "
                f"True matches found: "
                f"{len(found_matches):,}"
            )

    return len(found_matches)


if __name__ == "__main__":

    print("Starting final candidate recall evaluation...")

    ground_truth, total_true_matches = load_ground_truth()

    found_matches = evaluate_candidates(ground_truth)

    recall = (
        found_matches / total_true_matches
        if total_true_matches > 0
        else 0
    )

    print()
    print("============================================")
    print("       FINAL CANDIDATE RECALL")
    print("============================================")
    print(
        "Total true matches:",
        f"{total_true_matches:,}"
    )
    print(
        "True matches found:",
        f"{found_matches:,}"
    )
    print(
        "True matches missed:",
        f"{total_true_matches - found_matches:,}"
    )
    print(
        f"Candidate Recall: {recall:.4%}"
    )
    print("============================================")