import pandas as pd

from normalization import normalize_dataframe


DATASET = r"C:\AmazonMLChallenge\student_resource\dataset"

path = rf"{DATASET}\train\train_source1.tsv"

df = pd.read_csv(path, sep="\t", nrows=10)

normalized_df = normalize_dataframe(df)

print(
    normalized_df[
        [
            "business_name",
            "business_name_normalized",
            "business_address",
            "business_address_normalized",
            "country",
        ]
    ].to_string(index=False)
)