import csv
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio

# ============================================================
# PATHS
# ============================================================

DATASET = Path(r"D:\6ab10eb3b23ba_student_resource\student_resource\dataset")
TRAIN = DATASET / "train"

S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

VALIDATION_DIR = Path(r"D:\AmazonMLChallenge\validation_sample")
S1_FILE = VALIDATION_DIR / "validation_source1.tsv"
GT_FILE = VALIDATION_DIR / "validation_ground_truth.tsv"

# ============================================================
# SETTINGS
# ============================================================

MAX_SIGNATURE_BLOCK = 150
MAX_ADDRESS_BLOCK = 300
MAX_NAME_TOKEN_BLOCK = 250
MAX_NUMBER_BLOCK = 300
MAX_FUZZY_BLOCK = 500

FUZZY_THRESHOLD = 80
PREFIX_LENGTH = 2

# Require at least this many characters for ordinary name tokens.
MIN_NAME_TOKEN_LEN = 3

LEGAL_WORDS = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp",
    "corporation", "company", "co", "plc", "pvt", "private",
    "llp", "lp",
}

# Very common address words are weak blocking evidence.
ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "lane", "building", "floor",
    "near", "main", "market", "center", "centre", "complex",
    "block", "sector", "plot", "house", "shop", "office",
}

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_signature(name):
    tokens = []
    for token in normalize(name).split():
        if token in LEGAL_WORDS or len(token) <= 1:
            continue
        tokens.append(token)
    return " ".join(sorted(set(tokens)))


def useful_name_tokens(name):
    tokens = set()
    for token in normalize(name).split():
        if token in LEGAL_WORDS:
            continue
        if len(token) >= MIN_NAME_TOKEN_LEN:
            tokens.add(token)
    return tokens


def useful_address_tokens(address):
    result = set()
    for token in normalize(address).split():
        if token.isdigit():
            result.add(token)
        elif len(token) >= 5 and token not in ADDRESS_STOPWORDS:
            result.add(token)
    return result


def address_numbers(address):
    return set(re.findall(r"\d+", normalize(address)))


def fuzzy_keys(name, country):
    """
    V5 uses multiple small fuzzy keys instead of relying only on the
    lexicographically first name token. This makes retrieval more robust
    to token insertion/deletion/reordering.
    """
    tokens = sorted(useful_name_tokens(name))
    keys = set()

    for token in tokens:
        if len(token) >= PREFIX_LENGTH:
            keys.add((country, token[:PREFIX_LENGTH]))

    return keys


# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(file_path):
    print("\n" + "=" * 70)
    print(f"Building V5 index: {file_path.name}")
    print("=" * 70)

    name_index = defaultdict(list)
    signature_index = defaultdict(list)
    name_token_index = defaultdict(list)
    address_index = defaultdict(list)
    number_index = defaultdict(list)
    fuzzy_index = defaultdict(list)

    count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            entity_id = row["entity_id"]
            country = row["country"] or ""
            raw_name = row["business_name"] or ""
            raw_address = row["business_address"] or ""

            norm_name = normalize(raw_name)

            if norm_name:
                name_index[(country, norm_name)].append(entity_id)

            signature = name_signature(raw_name)
            if signature:
                signature_index[(country, signature)].append(entity_id)

            for token in useful_name_tokens(raw_name):
                name_token_index[(country, token)].append(entity_id)

            for token in useful_address_tokens(raw_address):
                address_index[(country, token)].append(entity_id)

            for number in address_numbers(raw_address):
                number_index[(country, number)].append(entity_id)

            if norm_name:
                for key in fuzzy_keys(raw_name, country):
                    fuzzy_index[key].append((entity_id, norm_name))

            count += 1

            if count % 1_000_000 == 0:
                print(f"Processed {count:,} rows...")

    print(f"Finished {file_path.name}: {count:,} rows")
    print(f"Exact-name blocks: {len(name_index):,}")
    print(f"Signature blocks: {len(signature_index):,}")
    print(f"Name-token blocks: {len(name_token_index):,}")
    print(f"Address-token blocks: {len(address_index):,}")
    print(f"Address-number blocks: {len(number_index):,}")
    print(f"Fuzzy blocks: {len(fuzzy_index):,}")

    return {
        "name": name_index,
        "signature": signature_index,
        "name_token": name_token_index,
        "address": address_index,
        "number": number_index,
        "fuzzy": fuzzy_index,
    }


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():
    truth = {}

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]
            text = (row["matched_entity_ids"] or "").strip()

            truth[s1_id] = {
                x.strip()
                for x in text.split(",")
                if x.strip()
            } if text else set()

    print(f"Ground-truth rows loaded: {len(truth):,}")
    return truth


# ============================================================
# SAFE BLOCK ADDITION
# ============================================================

def add_block(candidates, block, maximum):
    if block and len(block) <= maximum:
        candidates.update(block)
        return True
    return False


# ============================================================
# V5 EVALUATION
# ============================================================

def evaluate(s2, s3, truth):
    print("\n" + "=" * 70)
    print("Evaluating Candidate Baseline V5")
    print("=" * 70)

    total_s1 = 0
    total_true = 0
    recovered = 0
    total_candidates = 0
    s1_with_candidates = 0
    s1_with_truth = 0

    strategy_hits = Counter()

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["entity_id"]
            country = row["country"] or ""
            raw_name = row["business_name"] or ""
            raw_address = row["business_address"] or ""

            norm_name = normalize(raw_name)
            signature = name_signature(raw_name)
            name_tokens = useful_name_tokens(raw_name)
            addr_tokens = useful_address_tokens(raw_address)
            numbers = address_numbers(raw_address)

            candidates = set()

            # ----------------------------------------------------
            # 1. Exact normalized business name
            # ----------------------------------------------------
            if norm_name:
                key = (country, norm_name)
                before = len(candidates)
                candidates.update(s2["name"].get(key, []))
                candidates.update(s3["name"].get(key, []))
                if len(candidates) > before:
                    strategy_hits["exact_name"] += 1

            # ----------------------------------------------------
            # 2. Legal-suffix-free sorted name signature
            # ----------------------------------------------------
            if signature:
                key = (country, signature)
                before = len(candidates)

                add_block(
                    candidates,
                    s2["signature"].get(key, []),
                    MAX_SIGNATURE_BLOCK
                )
                add_block(
                    candidates,
                    s3["signature"].get(key, []),
                    MAX_SIGNATURE_BLOCK
                )

                if len(candidates) > before:
                    strategy_hits["signature"] += 1

            # ----------------------------------------------------
            # 3. Rare / informative exact name-token blocks
            # ----------------------------------------------------
            for token in name_tokens:
                key = (country, token)
                before = len(candidates)

                add_block(
                    candidates,
                    s2["name_token"].get(key, []),
                    MAX_NAME_TOKEN_BLOCK
                )
                add_block(
                    candidates,
                    s3["name_token"].get(key, []),
                    MAX_NAME_TOKEN_BLOCK
                )

                if len(candidates) > before:
                    strategy_hits["name_token"] += 1

            # ----------------------------------------------------
            # 4. Informative address tokens
            # ----------------------------------------------------
            for token in addr_tokens:
                key = (country, token)
                before = len(candidates)

                add_block(
                    candidates,
                    s2["address"].get(key, []),
                    MAX_ADDRESS_BLOCK
                )
                add_block(
                    candidates,
                    s3["address"].get(key, []),
                    MAX_ADDRESS_BLOCK
                )

                if len(candidates) > before:
                    strategy_hits["address_token"] += 1

            # ----------------------------------------------------
            # 5. Address-number blocks
            # ----------------------------------------------------
            # Numbers can survive otherwise heavily corrupted addresses.
            for number in numbers:
                key = (country, number)

                # Very short numbers are often too ambiguous.
                if len(number) < 2:
                    continue

                before = len(candidates)

                add_block(
                    candidates,
                    s2["number"].get(key, []),
                    MAX_NUMBER_BLOCK
                )
                add_block(
                    candidates,
                    s3["number"].get(key, []),
                    MAX_NUMBER_BLOCK
                )

                if len(candidates) > before:
                    strategy_hits["address_number"] += 1

            # ----------------------------------------------------
            # 6. Multi-token-prefix fuzzy name retrieval
            # ----------------------------------------------------
            if norm_name:
                fuzzy_records = {}

                for key in fuzzy_keys(raw_name, country):
                    block2 = s2["fuzzy"].get(key, [])
                    block3 = s3["fuzzy"].get(key, [])

                    if len(block2) <= MAX_FUZZY_BLOCK:
                        for entity_id, candidate_name in block2:
                            fuzzy_records[entity_id] = candidate_name

                    if len(block3) <= MAX_FUZZY_BLOCK:
                        for entity_id, candidate_name in block3:
                            fuzzy_records[entity_id] = candidate_name

                before = len(candidates)

                for entity_id, candidate_name in fuzzy_records.items():
                    if token_set_ratio(norm_name, candidate_name) >= FUZZY_THRESHOLD:
                        candidates.add(entity_id)

                if len(candidates) > before:
                    strategy_hits["fuzzy_name"] += 1

            actual = truth.get(s1_id, set())

            total_s1 += 1
            total_true += len(actual)
            recovered += len(actual.intersection(candidates))
            total_candidates += len(candidates)

            if actual:
                s1_with_truth += 1

            if candidates:
                s1_with_candidates += 1

            if total_s1 % 10_000 == 0:
                recall = recovered / total_true if total_true else 0
                avg = total_candidates / total_s1

                print(
                    f"Processed {total_s1:,} S1 | "
                    f"Recall = {recall:.4%} | "
                    f"Avg candidates = {avg:.2f}"
                )

    recall = recovered / total_true if total_true else 0
    avg = total_candidates / total_s1 if total_s1 else 0

    print("\n" + "=" * 70)
    print("BASELINE V5 RESULTS")
    print("=" * 70)
    print(f"S1 records evaluated: {total_s1:,}")
    print(f"S1 records with true matches: {s1_with_truth:,}")
    print(f"S1 receiving candidates: {s1_with_candidates:,}")
    print(f"Total true matches: {total_true:,}")
    print(f"Recovered true matches: {recovered:,}")
    print(f"Candidate Recall: {recall:.4%}")
    print(f"Total candidates: {total_candidates:,}")
    print(f"Average candidates per S1: {avg:.2f}")

    print("\nStrategy-use diagnostics:")
    for key, value in strategy_hits.most_common():
        print(f"  {key}: {value:,}")

    print("=" * 70)

    if recall >= 0.95:
        print("DECISION: STRONG GO - candidate recall reached at least 95%.")
    elif recall >= 0.90:
        print("DECISION: GO / INVESTIGATE - candidate recall reached at least 90%.")
    elif recall >= 0.85:
        print("DECISION: BORDERLINE - improvement is meaningful but needs review.")
    else:
        print("DECISION: STOP - keep the existing V4 final submission.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("CANDIDATE BASELINE V5 - VALIDATION ONLY")
    print("=" * 70)

    print("\nThis script does NOT overwrite final submission outputs.")
    print("Validation S1: existing 100,000-record validation sample")

    print("\nV5 strategies:")
    print("1. Exact normalized business name")
    print("2. Legal-suffix-free sorted name signature")
    print("3. Informative exact name-token blocks")
    print("4. Informative address-token blocks")
    print("5. Address-number blocks")
    print("6. Multi-token-prefix fuzzy name retrieval")
    print("7. Same-country blocking")

    s2 = build_index(S2_FILE)
    s3 = build_index(S3_FILE)
    truth = load_ground_truth()

    evaluate(s2, s3, truth)


if __name__ == "__main__":
    main()
