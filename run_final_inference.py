import csv
import re
import sqlite3
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\AmazonMLChallenge")

DATASET = Path(
    r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset"
)

TEST = DATASET / "test"

S1_FILE = TEST / "test_source1.tsv"
S2_FILE = TEST / "test_source2.tsv"
S3_FILE = TEST / "test_source3.tsv"

MODEL_FILE = PROJECT / "matching_model.joblib"

OUTPUT_DIR = PROJECT / "output"

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

DB_FILE = PROJECT / "final_candidate_index.db"

THRESHOLD = 0.85

MAX_ADDRESS_BLOCK = 200
MAX_SIGNATURE_BLOCK = 100
MAX_FUZZY_BLOCK = 300

FUZZY_THRESHOLD = 85
PREFIX_LENGTH = 3

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


LEGAL_WORDS = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "pvt",
    "private",
    "llp",
    "lp",
}


FEATURE_COLUMNS = [
    "name_exact",
    "name_signature_exact",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_length_ratio",

    "address_exact",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "address_length_ratio",

    "address_number_overlap",
    "address_number_count",
    "address_number_exact",
    "address_missing",

    "same_country",
    "candidate_is_s2",
    "candidate_is_s3",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def name_signature(name):

    name = normalize(name)

    if not name:
        return ""

    tokens = []

    for token in name.split():

        if token in LEGAL_WORDS:
            continue

        if len(token) <= 1:
            continue

        tokens.append(token)

    return " ".join(
        sorted(set(tokens))
    )


def fuzzy_block_key(name, country):

    name = normalize(name)

    if not name:
        return ""

    tokens = []

    for token in name.split():

        if token in LEGAL_WORDS:
            continue

        if len(token) < 3:
            continue

        tokens.append(token)

    if not tokens:
        return ""

    tokens.sort()

    prefix = tokens[0][:PREFIX_LENGTH]

    if not prefix:
        return ""

    return f"{country}\x1f{prefix}"


def useful_address_tokens(address):

    address = normalize(address)

    if not address:
        return set()

    result = set()

    for token in address.split():

        if token.isdigit():
            result.add(token)

        elif len(token) >= 5:
            result.add(token)

    return result


# ============================================================
# FEATURE HELPERS
# ============================================================

def text_tokens(text):

    text = normalize(text)

    if not text:
        return set()

    return set(text.split())


def address_numbers(text):

    text = normalize(text)

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


def safe_length_ratio(a, b):

    a = normalize(a)
    b = normalize(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b)
    ) / max(
        len(a),
        len(b)
    )


def jaccard(set1, set2):

    if not set1 and not set2:
        return 0.0

    union = set1 | set2

    if not union:
        return 0.0

    return len(
        set1 & set2
    ) / len(union)


# ============================================================
# DATABASE
# ============================================================

def create_database():

    print()
    print("=" * 70)
    print("STEP 1: BUILDING DISK-BACKED TEST INDEX")
    print("=" * 70)

    if DB_FILE.exists():

        print("Existing database found:")
        print(DB_FILE)

        answer = input(
            "Reuse existing database? (y/n): "
        ).strip().lower()

        if answer == "y":
            return

        DB_FILE.unlink()

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE candidates (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            source TEXT,
            normalized_name TEXT,
            signature TEXT,
            fuzzy_key TEXT
        );

        CREATE TABLE name_blocks (
            block_key TEXT,
            entity_id TEXT
        );

        CREATE TABLE signature_blocks (
            block_key TEXT,
            entity_id TEXT
        );

        CREATE TABLE fuzzy_blocks (
            block_key TEXT,
            entity_id TEXT,
            normalized_name TEXT
        );

        CREATE TABLE address_blocks (
            block_key TEXT,
            entity_id TEXT
        );
        """
    )

    conn.commit()

    for file_path, source in [
        (S2_FILE, "S2"),
        (S3_FILE, "S3"),
    ]:

        print()
        print("Indexing:", file_path.name)

        count = 0

        candidate_batch = []
        name_batch = []
        signature_batch = []
        fuzzy_batch = []
        address_batch = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter="\t"
            )

            for row in reader:

                count += 1

                entity_id = row["entity_id"]

                name = (
                    row["business_name"]
                    or ""
                )

                address = (
                    row["business_address"]
                    or ""
                )

                country = (
                    row["country"]
                    or ""
                )

                norm_name = normalize(name)

                signature = name_signature(name)

                fuzzy_key = fuzzy_block_key(
                    name,
                    country
                )

                candidate_batch.append(
                    (
                        entity_id,
                        name,
                        address,
                        country,
                        source,
                        norm_name,
                        signature,
                        fuzzy_key,
                    )
                )

                if norm_name:

                    name_batch.append(
                        (
                            f"{country}\x1f{norm_name}",
                            entity_id
                        )
                    )

                if signature:

                    signature_batch.append(
                        (
                            f"{country}\x1f{signature}",
                            entity_id
                        )
                    )

                if fuzzy_key:

                    fuzzy_batch.append(
                        (
                            fuzzy_key,
                            entity_id,
                            norm_name
                        )
                    )

                for token in useful_address_tokens(
                    address
                ):

                    address_batch.append(
                        (
                            f"{country}\x1f{token}",
                            entity_id
                        )
                    )

                if len(candidate_batch) >= 10_000:

                    cursor.executemany(
                        """
                        INSERT INTO candidates
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        candidate_batch
                    )

                    cursor.executemany(
                        """
                        INSERT INTO name_blocks
                        VALUES (?, ?)
                        """,
                        name_batch
                    )

                    cursor.executemany(
                        """
                        INSERT INTO signature_blocks
                        VALUES (?, ?)
                        """,
                        signature_batch
                    )

                    cursor.executemany(
                        """
                        INSERT INTO fuzzy_blocks
                        VALUES (?, ?, ?)
                        """,
                        fuzzy_batch
                    )

                    cursor.executemany(
                        """
                        INSERT INTO address_blocks
                        VALUES (?, ?)
                        """,
                        address_batch
                    )

                    conn.commit()

                    candidate_batch.clear()
                    name_batch.clear()
                    signature_batch.clear()
                    fuzzy_batch.clear()
                    address_batch.clear()

                if count % 500_000 == 0:

                    print(
                        f"  processed {count:,}"
                    )

        if candidate_batch:

            cursor.executemany(
                """
                INSERT INTO candidates
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                candidate_batch
            )

            cursor.executemany(
                """
                INSERT INTO name_blocks
                VALUES (?, ?)
                """,
                name_batch
            )

            cursor.executemany(
                """
                INSERT INTO signature_blocks
                VALUES (?, ?)
                """,
                signature_batch
            )

            cursor.executemany(
                """
                INSERT INTO fuzzy_blocks
                VALUES (?, ?, ?)
                """,
                fuzzy_batch
            )

            cursor.executemany(
                """
                INSERT INTO address_blocks
                VALUES (?, ?)
                """,
                address_batch
            )

            conn.commit()

        print(
            f"Finished {source}: {count:,} records"
        )

    print()
    print("Creating SQLite indexes...")

    cursor.execute(
        """
        CREATE INDEX idx_name_blocks
        ON name_blocks(block_key)
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_signature_blocks
        ON signature_blocks(block_key)
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_fuzzy_blocks
        ON fuzzy_blocks(block_key)
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_address_blocks
        ON address_blocks(block_key)
        """
    )

    conn.commit()
    conn.close()

    print("Database complete:")
    print(DB_FILE)


# ============================================================
# BLOCK LOOKUPS
# ============================================================

def get_simple_block(
    cursor,
    table,
    key,
    limit
):

    cursor.execute(
        f"""
        SELECT entity_id
        FROM {table}
        WHERE block_key = ?
        LIMIT ?
        """,
        (
            key,
            limit + 1
        )
    )

    rows = cursor.fetchall()

    if len(rows) > limit:
        return []

    return [
        row[0]
        for row in rows
    ]


def generate_candidates(
    cursor,
    s1
):

    country = (
        s1["country"]
        or ""
    )

    name = (
        s1["business_name"]
        or ""
    )

    address = (
        s1["business_address"]
        or ""
    )

    norm_name = normalize(name)

    candidates = set()

    # --------------------------------------------------------
    # 1. EXACT NORMALIZED NAME
    # --------------------------------------------------------

    if norm_name:

        key = (
            f"{country}\x1f{norm_name}"
        )

        cursor.execute(
            """
            SELECT entity_id
            FROM name_blocks
            WHERE block_key = ?
            """,
            (key,)
        )

        candidates.update(
            row[0]
            for row in cursor.fetchall()
        )

    # --------------------------------------------------------
    # 2. NAME SIGNATURE
    # --------------------------------------------------------

    signature = name_signature(name)

    if signature:

        key = (
            f"{country}\x1f{signature}"
        )

        candidates.update(
            get_simple_block(
                cursor,
                "signature_blocks",
                key,
                MAX_SIGNATURE_BLOCK
            )
        )

    # --------------------------------------------------------
    # 3. FUZZY NAME
    # --------------------------------------------------------

    fuzzy_key = fuzzy_block_key(
        name,
        country
    )

    if fuzzy_key and norm_name:

        cursor.execute(
            """
            SELECT entity_id, normalized_name
            FROM fuzzy_blocks
            WHERE block_key = ?
            LIMIT ?
            """,
            (
                fuzzy_key,
                MAX_FUZZY_BLOCK + 1
            )
        )

        rows = cursor.fetchall()

        if len(rows) <= MAX_FUZZY_BLOCK:

            for (
                entity_id,
                candidate_name
            ) in rows:

                score = token_set_ratio(
                    norm_name,
                    candidate_name
                )

                if score >= FUZZY_THRESHOLD:

                    candidates.add(
                        entity_id
                    )

    # --------------------------------------------------------
    # 4. ADDRESS TOKENS
    # --------------------------------------------------------

    for token in useful_address_tokens(
        address
    ):

        key = (
            f"{country}\x1f{token}"
        )

        candidates.update(
            get_simple_block(
                cursor,
                "address_blocks",
                key,
                MAX_ADDRESS_BLOCK
            )
        )

    return candidates


# ============================================================
# LOAD CANDIDATE RECORDS
# ============================================================

def load_candidate_records(
    cursor,
    candidate_ids
):

    records = {}

    candidate_ids = list(
        candidate_ids
    )

    # SQLite has a variable limit, so query in chunks.
    for start in range(
        0,
        len(candidate_ids),
        500
    ):

        chunk = candidate_ids[
            start:start + 500
        ]

        placeholders = ",".join(
            "?"
            for _ in chunk
        )

        cursor.execute(
            f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country,
                source
            FROM candidates
            WHERE entity_id IN ({placeholders})
            """,
            chunk
        )

        for row in cursor.fetchall():

            records[row[0]] = {
                "business_name": row[1] or "",
                "business_address": row[2] or "",
                "country": row[3] or "",
                "source": row[4],
            }

    return records


# ============================================================
# FEATURES
# ============================================================

def calculate_features(
    s1,
    candidate
):

    name1 = s1["business_name"] or ""
    name2 = candidate["business_name"] or ""

    address1 = (
        s1["business_address"]
        or ""
    )

    address2 = (
        candidate["business_address"]
        or ""
    )

    norm_name1 = normalize(name1)
    norm_name2 = normalize(name2)

    norm_address1 = normalize(address1)
    norm_address2 = normalize(address2)

    name_exact = int(
        bool(norm_name1)
        and norm_name1 == norm_name2
    )

    sig1 = name_signature(name1)
    sig2 = name_signature(name2)

    signature_exact = int(
        bool(sig1)
        and sig1 == sig2
    )

    name_r = ratio(
        norm_name1,
        norm_name2
    )

    name_sort = token_sort_ratio(
        norm_name1,
        norm_name2
    )

    name_set = token_set_ratio(
        norm_name1,
        norm_name2
    )

    name_len = safe_length_ratio(
        norm_name1,
        norm_name2
    )

    address_missing = int(
        not norm_address1
        or not norm_address2
    )

    address_exact = int(
        bool(norm_address1)
        and norm_address1
        == norm_address2
    )

    if norm_address1 and norm_address2:

        address_r = ratio(
            norm_address1,
            norm_address2
        )

        address_sort = token_sort_ratio(
            norm_address1,
            norm_address2
        )

        address_set = token_set_ratio(
            norm_address1,
            norm_address2
        )

    else:

        address_r = 0.0
        address_sort = 0.0
        address_set = 0.0

    tokens1 = text_tokens(
        norm_address1
    )

    tokens2 = text_tokens(
        norm_address2
    )

    address_j = jaccard(
        tokens1,
        tokens2
    )

    address_len = safe_length_ratio(
        norm_address1,
        norm_address2
    )

    numbers1 = address_numbers(
        norm_address1
    )

    numbers2 = address_numbers(
        norm_address2
    )

    common_numbers = (
        numbers1 & numbers2
    )

    number_overlap = int(
        bool(common_numbers)
    )

    number_count = len(
        common_numbers
    )

    number_exact = int(
        bool(numbers1)
        and bool(numbers2)
        and numbers1 == numbers2
    )

    same_country = int(
        s1["country"]
        == candidate["country"]
    )

    source = candidate["source"]

    return [
        name_exact,
        signature_exact,
        name_r,
        name_sort,
        name_set,
        name_len,

        address_exact,
        address_r,
        address_sort,
        address_set,
        address_j,
        address_len,

        number_overlap,
        number_count,
        number_exact,
        address_missing,

        same_country,

        int(source == "S2"),
        int(source == "S3"),
    ]


# ============================================================
# FINAL INFERENCE
# ============================================================

def run_inference():

    print()
    print("=" * 70)
    print("STEP 2: LOADING MODEL")
    print("=" * 70)

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved["model"]

    if saved["features"] != FEATURE_COLUMNS:

        raise RuntimeError(
            "Model feature order mismatch."
        )

    print("Model loaded.")
    print("Threshold:", THRESHOLD)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_FILE
    )

    cursor = conn.cursor()

    print()
    print("=" * 70)
    print("STEP 3: FINAL TEST INFERENCE")
    print("=" * 70)

    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    no_candidate_s1 = 0
    singleton_predictions = 0
    missing_records = 0

    with (
        open(
            S1_FILE,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as s1_f,

        open(
            MATCHING_OUTPUT,
            "w",
            encoding="utf-8",
            newline=""
        ) as match_f,

        open(
            CANDIDATE_OUTPUT,
            "w",
            encoding="utf-8",
            newline=""
        ) as candidate_f
    ):

        reader = csv.DictReader(
            s1_f,
            delimiter="\t"
        )

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

        match_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids"
            ]
        )

        candidate_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids"
            ]
        )

        for row in reader:

            s1_id = row["entity_id"]

            s1 = {
                "business_name":
                    row["business_name"] or "",

                "business_address":
                    row["business_address"] or "",

                "country":
                    row["country"] or "",
            }

            candidate_ids = generate_candidates(
                cursor,
                s1
            )

            total_candidates += len(
                candidate_ids
            )

            if not candidate_ids:
                no_candidate_s1 += 1

            # Stable order makes output reproducible.
            candidate_ids = sorted(
                candidate_ids
            )

            candidate_writer.writerow(
                [
                    s1_id,
                    ",".join(candidate_ids)
                ]
            )

            candidate_records = (
                load_candidate_records(
                    cursor,
                    candidate_ids
                )
            )

            feature_rows = []
            scored_ids = []

            for candidate_id in candidate_ids:

                candidate = (
                    candidate_records.get(
                        candidate_id
                    )
                )

                if candidate is None:

                    missing_records += 1
                    continue

                feature_rows.append(
                    calculate_features(
                        s1,
                        candidate
                    )
                )

                scored_ids.append(
                    candidate_id
                )

            matched_ids = []

            if feature_rows:

                X = np.asarray(
                    feature_rows,
                    dtype=np.float32
                )

                probabilities = (
                    model.predict_proba(
                        X
                    )[:, 1]
                )

                matched_ids = [
                    candidate_id
                    for (
                        candidate_id,
                        probability
                    )
                    in zip(
                        scored_ids,
                        probabilities
                    )
                    if probability >= THRESHOLD
                ]

            matched_ids = sorted(
                set(matched_ids)
            )

            if not matched_ids:
                singleton_predictions += 1

            total_matches += len(
                matched_ids
            )

            match_writer.writerow(
                [
                    s1_id,
                    ",".join(matched_ids)
                ]
            )

            total_s1 += 1

            if total_s1 % 10_000 == 0:

                avg_candidates = (
                    total_candidates
                    / total_s1
                )

                avg_matches = (
                    total_matches
                    / total_s1
                )

                print(
                    f"Processed {total_s1:,} S1 | "
                    f"Avg candidates = "
                    f"{avg_candidates:.2f} | "
                    f"Avg matches = "
                    f"{avg_matches:.2f}"
                )

                # Make sure progress is physically written
                # to disk during the long run.
                match_f.flush()
                candidate_f.flush()

    conn.close()

    print()
    print("=" * 70)
    print("FINAL TEST INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "S1 records:",
        f"{total_s1:,}"
    )

    print(
        "Candidate pairs:",
        f"{total_candidates:,}"
    )

    print(
        "Average candidates/S1:",
        f"{total_candidates / total_s1:.2f}"
    )

    print(
        "Predicted matches:",
        f"{total_matches:,}"
    )

    print(
        "Average matches/S1:",
        f"{total_matches / total_s1:.2f}"
    )

    print(
        "S1 with no candidates:",
        f"{no_candidate_s1:,}"
    )

    print(
        "Predicted singleton S1:",
        f"{singleton_predictions:,}"
    )

    print(
        "Missing candidate records:",
        f"{missing_records:,}"
    )

    print()
    print("Outputs:")
    print(MATCHING_OUTPUT)
    print(CANDIDATE_OUTPUT)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL TEST INFERENCE")
    print("=" * 70)

    print()
    print("Threshold:", THRESHOLD)

    create_database()

    run_inference()


if __name__ == "__main__":
    main()