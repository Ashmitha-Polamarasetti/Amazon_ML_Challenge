import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features import create_features


source1 = pd.read_csv("data/train_source1.tsv", sep="\t")
source2 = pd.read_csv("data/train_source2.tsv", sep="\t")


# Create TF-IDF vectorizers
name_vectorizer = TfidfVectorizer()
address_vectorizer = TfidfVectorizer()


# Fit the vectorizers using the real training data
name_vectorizer.fit(
    source1["business_name"].fillna("").map(str).tolist()
    + source2["business_name"].fillna("").map(str).tolist()
)

address_vectorizer.fit(
    source1["business_address"].fillna("").map(str).tolist()
    + source2["business_address"].fillna("").map(str).tolist()
)


# Take one real record from each source
record1 = source1.iloc[0].to_dict()
record2 = source2.iloc[0].to_dict()


print("\nS1 record:")
print(record1)

print("\nS2 record:")
print(record2)

print("\nFeatures:")
print(
    create_features(
        record1,
        record2,
        name_vectorizer,
        address_vectorizer
    )
)