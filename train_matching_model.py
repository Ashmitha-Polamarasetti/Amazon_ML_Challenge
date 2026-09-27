import csv
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"D:\AmazonMLChallenge")

FEATURE_FILE = PROJECT / "matching_features.tsv"

MODEL_FILE = PROJECT / "matching_model.joblib"

PREDICTION_FILE = (
    PROJECT / "holdout_predictions.tsv"
)


# ============================================================
# FEATURES
# ============================================================

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
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("STEP 1: LOADING MATCHING FEATURES")
    print("=" * 70)

    print("Reading:")
    print(FEATURE_FILE)

    df = pd.read_csv(
        FEATURE_FILE,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "split": str,
        }
    )

    print()
    print(
        "Total rows:",
        f"{len(df):,}"
    )

    print(
        "Positive rows:",
        f"{int(df['label'].sum()):,}"
    )

    print(
        "Negative rows:",
        f"{int((df['label'] == 0).sum()):,}"
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    holdout_df = df[
        df["split"] == "holdout"
    ].copy()

    print()
    print(
        "Train rows:",
        f"{len(train_df):,}"
    )

    print(
        "Holdout rows:",
        f"{len(holdout_df):,}"
    )

    return train_df, holdout_df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_df):

    print()
    print("=" * 70)
    print("STEP 2: TRAINING MATCHING MODEL")
    print("=" * 70)

    X_train = train_df[
        FEATURE_COLUMNS
    ].astype(np.float32)

    y_train = train_df[
        "label"
    ].astype(np.int8)

    print(
        "Feature count:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Training examples:",
        f"{len(X_train):,}"
    )

    print()
    print("Training HistGradientBoostingClassifier...")

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Training complete.")

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        MODEL_FILE
    )

    print()
    print("Model saved:")
    print(MODEL_FILE)

    return model


# ============================================================
# PAIR-LEVEL DIAGNOSTICS
# ============================================================

def pair_level_diagnostics(
    holdout_df,
    probabilities
):

    print()
    print("=" * 70)
    print("STEP 3: PAIR-LEVEL DIAGNOSTICS")
    print("=" * 70)

    y_true = holdout_df[
        "label"
    ].to_numpy()

    try:

        auc = roc_auc_score(
            y_true,
            probabilities
        )

        print(
            f"ROC AUC: {auc:.6f}"
        )

    except ValueError:

        print(
            "ROC AUC could not be calculated."
        )

    print()
    print(
        "These are diagnostics only."
    )

    print(
        "Final threshold selection uses "
        "macro S1-level F0.5."
    )


# ============================================================
# MACRO F0.5
# ============================================================

def calculate_s1_f05(
    holdout_df,
    probabilities,
    threshold
):
    """
    Calculates F0.5 independently for every S1 and
    returns the macro average.

    Ground-truth set here consists of positive candidate
    pairs recovered by candidate generation.

    S1 with no positive labeled pair therefore has an
    empty target set.
    """

    work = holdout_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
    ].copy()

    work["probability"] = probabilities

    total_score = 0.0
    s1_count = 0

    predicted_match_count = 0

    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0

    for s1_id, group in work.groupby(
        "source1_entity_id",
        sort=False
    ):

        true_set = set(
            group.loc[
                group["label"] == 1,
                "candidate_entity_id"
            ]
        )

        predicted_set = set(
            group.loc[
                group["probability"]
                >= threshold,
                "candidate_entity_id"
            ]
        )

        predicted_match_count += len(
            predicted_set
        )

        # ----------------------------------------------------
        # Special singleton / empty-set case
        # ----------------------------------------------------

        if (
            len(true_set) == 0
            and len(predicted_set) == 0
        ):

            score = 1.0

        elif (
            len(true_set) == 0
            and len(predicted_set) > 0
        ):

            score = 0.0

        else:

            tp = len(
                true_set & predicted_set
            )

            fp = len(
                predicted_set - true_set
            )

            fn = len(
                true_set - predicted_set
            )

            true_positive_total += tp
            false_positive_total += fp
            false_negative_total += fn

            if tp == 0:

                score = 0.0

            else:

                precision = (
                    tp / (tp + fp)
                )

                recall = (
                    tp / (tp + fn)
                )

                beta2 = 0.25

                denominator = (
                    beta2 * precision
                    + recall
                )

                if denominator == 0:

                    score = 0.0

                else:

                    score = (
                        (1 + beta2)
                        * precision
                        * recall
                        / denominator
                    )

        total_score += score
        s1_count += 1

    macro_f05 = (
        total_score / s1_count
        if s1_count
        else 0.0
    )

    return {
        "threshold": threshold,
        "macro_f05": macro_f05,
        "s1_count": s1_count,
        "predicted_matches":
            predicted_match_count,
        "tp": true_positive_total,
        "fp": false_positive_total,
        "fn": false_negative_total,
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def threshold_search(
    holdout_df,
    probabilities
):

    print()
    print("=" * 70)
    print("STEP 4: MACRO F0.5 THRESHOLD SEARCH")
    print("=" * 70)

    # Precision is weighted more strongly by F0.5,
    # so include several high thresholds.
    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.88,
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]

    results = []

    best = None

    for threshold in thresholds:

        result = calculate_s1_f05(
            holdout_df,
            probabilities,
            threshold
        )

        results.append(
            result
        )

        print(
            f"Threshold {threshold:>5.2f} | "
            f"Macro F0.5 = "
            f"{result['macro_f05']:.6f} | "
            f"Predicted matches = "
            f"{result['predicted_matches']:,} | "
            f"TP = {result['tp']:,} | "
            f"FP = {result['fp']:,} | "
            f"FN = {result['fn']:,}"
        )

        if (
            best is None
            or result["macro_f05"]
            > best["macro_f05"]
        ):

            best = result

    print()
    print("-" * 70)
    print("BEST THRESHOLD")
    print("-" * 70)

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Macro F0.5: "
        f"{best['macro_f05']:.6f}"
    )

    print(
        f"Predicted matches: "
        f"{best['predicted_matches']:,}"
    )

    print(
        f"TP: {best['tp']:,}"
    )

    print(
        f"FP: {best['fp']:,}"
    )

    print(
        f"FN: {best['fn']:,}"
    )

    return best, results


# ============================================================
# SAVE HOLDOUT PREDICTIONS
# ============================================================

def save_predictions(
    holdout_df,
    probabilities,
    best_threshold
):

    print()
    print("=" * 70)
    print("STEP 5: SAVING HOLDOUT PREDICTIONS")
    print("=" * 70)

    output = holdout_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
    ].copy()

    output[
        "match_probability"
    ] = probabilities

    output[
        "predicted_match"
    ] = (
        probabilities
        >= best_threshold
    ).astype(int)

    output.to_csv(
        PREDICTION_FILE,
        sep="\t",
        index=False
    )

    print(
        "Saved:"
    )

    print(
        PREDICTION_FILE
    )

    print(
        "Rows:",
        f"{len(output):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("TRAIN MATCHING MODEL")
    print("=" * 70)

    train_df, holdout_df = (
        load_data()
    )

    model = train_model(
        train_df
    )

    print()
    print(
        "Generating holdout probabilities..."
    )

    X_holdout = holdout_df[
        FEATURE_COLUMNS
    ].astype(np.float32)

    probabilities = (
        model.predict_proba(
            X_holdout
        )[:, 1]
    )

    pair_level_diagnostics(
        holdout_df,
        probabilities
    )

    best, results = (
        threshold_search(
            holdout_df,
            probabilities
        )
    )

    save_predictions(
        holdout_df,
        probabilities,
        best["threshold"]
    )

    print()
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best threshold:",
        f"{best['threshold']:.2f}"
    )

    print(
        "Holdout macro F0.5:",
        f"{best['macro_f05']:.6f}"
    )

    print()
    print("Model:")
    print(MODEL_FILE)

    print()
    print("Predictions:")
    print(PREDICTION_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()