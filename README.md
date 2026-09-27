@"
# Amazon ML Challenge 2026 - Business Entity Resolution

## Team
Mission Impossible

Team Members:
- Joshika
- Ashmitha
- Pramodini
- Deepthi

## Overview
This solution performs business entity resolution between Source 1 reference
entities and noisy Source 2 / Source 3 business records.

The pipeline consists of:

1. Text normalization
2. Country-aware candidate blocking
3. Exact normalized-name matching
4. Legal-suffix-free name signatures
5. Address-token blocking
6. Prefix-based fuzzy-name candidate retrieval
7. Similarity feature extraction
8. HistGradientBoosting classification
9. Probability thresholding
10. Generation of matching_results.tsv and candidate_pairs.tsv

## Model
The final matching model is a scikit-learn HistGradientBoostingClassifier.

The model uses 19 similarity and metadata features derived from business names,
addresses, country information, address numbers, and source information.

Final probability threshold: 0.85

## Validation
Final V4 candidate-generation validation recall: 74.6942%.

End-to-end holdout macro F0.5: 0.814786.

An experimental V5 candidate generator achieved 84.1224% candidate recall but
increased average candidates from approximately 133 to 289 per Source 1 entity.
The validated V4 pipeline was retained for final inference because of the
substantial computational increase.

## Source Files
src/candidate_baseline_v4.py
    Candidate-generation validation.

src/build_matching_training_data.py
    Builds labeled candidate pairs for model training.

src/build_matching_features.py
    Computes the matching features.

src/train_matching_model.py
    Trains the HistGradientBoosting matching model.

src/evaluate_full_pipeline.py
    Performs end-to-end validation and threshold evaluation.

src/run_final_inference.py
    Main final inference pipeline.

src/resume_final_inference_optimized.py
    Memory/runtime optimized resumable inference implementation.

## Dependencies
Install dependencies with:

pip install -r requirements.txt

## Outputs
The final submission produces:

output/matching_results.tsv
output/candidate_pairs.tsv

The matching threshold used for final inference is 0.85.
"@ | Set-Content final_submission\code\business_entity_resolution\README.md