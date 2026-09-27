import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

# Reuse the EXACT candidate generation and feature logic
# from the already-tested final inference script.
import run_final_inference as base


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\AmazonMLChallenge")

MATCHING_OUTPUT = PROJECT / "output" / "matching_results.tsv"
CANDIDATE_OUTPUT = PROJECT / "output" / "candidate_pairs.tsv"

S1_FILE = base.S1_FILE
DB_FILE = base.DB_FILE
MODEL_FILE = base.MODEL_FILE

THRESHOLD = 0.85

# Safe starting point for an ~8 GB RAM machine.
WORKERS = 3

# executor.map sends work in chunks.
MAP_CHUNKSIZE = 100


# ============================================================
# GLOBAL WORKER OBJECTS
# ============================================================

_worker_conn = None
_worker_cursor = None
_worker_model = None


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def initialize_worker():
    """
    Each worker gets its own read-only SQLite connection
    and its own copy of the trained model.
    """

    global _worker_conn
    global _worker_cursor
    global _worker_model

    import sqlite3

    _worker_conn = sqlite3.connect(
        f"file:{DB_FILE}?mode=ro",
        uri=True,
        timeout=60
    )

    # Give SQLite a modest per-process cache.
    _worker_conn.execute(
        "PRAGMA query_only = ON"
    )

    _worker_conn.execute(
        "PRAGMA temp_store = MEMORY"
    )

    _worker_cursor = _worker_conn.cursor()

    saved = joblib.load(MODEL_FILE)

    _worker_model = saved["model"]

    if saved["features"] != base.FEATURE_COLUMNS:
        raise RuntimeError(
            "Model feature order does not match inference features."
        )


# ============================================================
# PROCESS ONE S1 ENTITY
# ============================================================

def process_s1(item):

    global _worker_cursor
    global _worker_model

    sequence_number, row = item

    s1_id = row["entity_id"]

    s1 = {
        "business_name": row.get(
            "business_name",
            ""
        ) or "",

        "business_address": row.get(
            "business_address",
            ""
        ) or "",

        "country": row.get(
            "country",
            ""
        ) or "",
    }

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidate_ids = base.generate_candidates(
        _worker_cursor,
        s1
    )

    candidate_ids = sorted(candidate_ids)

    # --------------------------------------------------------
    # Load candidate records
    # --------------------------------------------------------

    candidate_records = base.load_candidate_records(
        _worker_cursor,
        candidate_ids
    )

    feature_rows = []
    scored_ids = []

    missing_records = 0

    for candidate_id in candidate_ids:

        candidate = candidate_records.get(
            candidate_id
        )

        if candidate is None:
            missing_records += 1
            continue

        feature_rows.append(
            base.calculate_features(
                s1,
                candidate
            )
        )

        scored_ids.append(candidate_id)

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    matched_ids = []

    if feature_rows:

        X = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        probabilities = (
            _worker_model.predict_proba(X)[:, 1]
        )

        matched_ids = [
            candidate_id
            for candidate_id, probability
            in zip(
                scored_ids,
                probabilities
            )
            if probability >= THRESHOLD
        ]

    matched_ids = sorted(
        set(matched_ids)
    )

    return (
        sequence_number,
        s1_id,
        candidate_ids,
        matched_ids,
        missing_records,
    )


# ============================================================
# READ EXISTING CHECKPOINT
# ============================================================

def inspect_existing_output(path):

    line_count = 0
    last_id = None

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.reader(
            f,
            delimiter="\t"
        )

        header = next(reader, None)

        if header is None:
            raise RuntimeError(
                f"Empty output file: {path}"
            )

        for row in reader:

            if not row:
                continue

            line_count += 1
            last_id = row[0]

    return line_count, last_id


# ============================================================
# VERIFY RESUME POSITION
# ============================================================

def verify_checkpoint():

    print("=" * 70)
    print("VERIFYING EXISTING CHECKPOINT")
    print("=" * 70)

    match_count, match_last_id = (
        inspect_existing_output(
            MATCHING_OUTPUT
        )
    )

    candidate_count, candidate_last_id = (
        inspect_existing_output(
            CANDIDATE_OUTPUT
        )
    )

    print(
        "Completed matching rows:",
        f"{match_count:,}"
    )

    print(
        "Completed candidate rows:",
        f"{candidate_count:,}"
    )

    if match_count != candidate_count:

        raise RuntimeError(
            "Output row counts do not match. "
            "Do NOT resume."
        )

    if match_last_id != candidate_last_id:

        raise RuntimeError(
            "Last S1 IDs in the two output files "
            "do not match. Do NOT resume."
        )

    if match_count == 0:

        raise RuntimeError(
            "No completed rows found."
        )

    # Verify that the last saved ID is exactly the
    # corresponding S1 in the official test file.

    expected_last_id = None

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for i, row in enumerate(
            reader,
            start=1
        ):

            if i == match_count:

                expected_last_id = row["entity_id"]
                break

    if expected_last_id is None:

        raise RuntimeError(
            "Checkpoint exceeds number of test S1 records."
        )

    print(
        "Last saved S1 ID:",
        match_last_id
    )

    print(
        "Expected S1 ID:",
        expected_last_id
    )

    if match_last_id != expected_last_id:

        raise RuntimeError(
            "Checkpoint does not align with test_source1.tsv. "
            "Do NOT resume."
        )

    print()
    print("CHECKPOINT VERIFIED SUCCESSFULLY.")
    print(
        f"Resume will begin after S1 #{match_count:,}"
    )

    return match_count


# ============================================================
# STREAM REMAINING S1 RECORDS
# ============================================================

def remaining_records(completed):

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for sequence_number, row in enumerate(
            reader,
            start=1
        ):

            if sequence_number <= completed:
                continue

            yield (
                sequence_number,
                row
            )


# ============================================================
# RESUME INFERENCE
# ============================================================

def run():

    completed = verify_checkpoint()

    print()
    print("=" * 70)
    print("FAST PARALLEL FINAL INFERENCE")
    print("=" * 70)

    print("Workers:", WORKERS)
    print("Threshold:", THRESHOLD)
    print(
        "Already completed:",
        f"{completed:,}"
    )

    total_test = 1_732_544

    remaining = total_test - completed

    print(
        "Remaining:",
        f"{remaining:,}"
    )

    print()

    newly_processed = 0
    new_candidates = 0
    new_matches = 0
    missing_records = 0

    # Append to existing verified files.
    with (
        open(
            MATCHING_OUTPUT,
            "a",
            encoding="utf-8",
            newline=""
        ) as match_f,

        open(
            CANDIDATE_OUTPUT,
            "a",
            encoding="utf-8",
            newline=""
        ) as candidate_f
    ):

        match_writer = csv.writer(
            match_f,
            delimiter="\t",
            lineterminator="\n"
        )

        candidate_writer = csv.writer(
            candidate_f,
            delimiter="\t",
            lineterminator="\n"
        )

        with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=initialize_worker
        ) as executor:

            results = executor.map(
                process_s1,
                remaining_records(completed),
                chunksize=MAP_CHUNKSIZE
            )

            for result in results:

                (
                    sequence_number,
                    s1_id,
                    candidate_ids,
                    matched_ids,
                    missing
                ) = result

                # Results from executor.map are returned
                # in the same order as the input.
                candidate_writer.writerow(
                    [
                        s1_id,
                        ",".join(candidate_ids)
                    ]
                )

                match_writer.writerow(
                    [
                        s1_id,
                        ",".join(matched_ids)
                    ]
                )

                newly_processed += 1
                new_candidates += len(
                    candidate_ids
                )
                new_matches += len(
                    matched_ids
                )
                missing_records += missing

                if newly_processed % 10_000 == 0:

                    total_done = (
                        completed
                        + newly_processed
                    )

                    percent = (
                        total_done
                        / total_test
                        * 100
                    )

                    avg_candidates = (
                        new_candidates
                        / newly_processed
                    )

                    avg_matches = (
                        new_matches
                        / newly_processed
                    )

                    print(
                        f"Processed {total_done:,} / "
                        f"{total_test:,} "
                        f"({percent:.1f}%) | "
                        f"Avg candidates={avg_candidates:.2f} | "
                        f"Avg matches={avg_matches:.2f}"
                    )

                    match_f.flush()
                    candidate_f.flush()

    total_done = (
        completed
        + newly_processed
    )

    print()
    print("=" * 70)
    print("FINAL INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Total S1 completed:",
        f"{total_done:,}"
    )

    print(
        "New S1 processed:",
        f"{newly_processed:,}"
    )

    print(
        "Missing candidate records:",
        f"{missing_records:,}"
    )

    print()
    print("Matching output:")
    print(MATCHING_OUTPUT)

    print()
    print("Candidate output:")
    print(CANDIDATE_OUTPUT)

    print("=" * 70)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Required for multiprocessing on Windows.
    import multiprocessing

    multiprocessing.freeze_support()

    run()