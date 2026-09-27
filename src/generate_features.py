import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features import create_features


# Load training data
source1 = pd.read_csv("data/train_source1.tsv", sep="\t")
source2 = pd.read_csv("data/train_source2.tsv", sep="\t")


# Create TF-IDF vectorizers
name_vectorizer = TfidfVectorizer()
address_vectorizer = TfidfVectorizer()


# Fit vectorizers on Source 1
name_vectorizer.fit(
    source1["business_name"].fillna("").astype(str)
)

address_vectorizer.fit(
    source1["business_address"].fillna("").astype(str)
)


# Generate features for the first 5 records
features_list = []

for i in range(5):

    record1 = source1.iloc[i].to_dict()
    record2 = source2.iloc[i].to_dict()

    features = create_features(
        record1,
        record2,
        name_vectorizer,
        address_vectorizer
    )

    features_list.append(features)


# Convert features into a DataFrame
features_df = pd.DataFrame(features_list)


print("\nGenerated Features:")
print(features_df)