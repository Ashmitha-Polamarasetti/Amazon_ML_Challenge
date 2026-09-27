import re
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def name_similarity(name1, name2):
    name1 = normalize_text(name1)
    name2 = normalize_text(name2)

    return fuzz.ratio(name1, name2) / 100.0


def address_similarity(address1, address2):
    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    return fuzz.ratio(address1, address2) / 100.0


def country_same(country1, country2):
    if not isinstance(country1, str) or not isinstance(country2, str):
        return 0

    return int(country1.strip().lower() == country2.strip().lower())

def create_features(record1, record2, name_vectorizer, address_vectorizer):
    return {
        "name_similarity": name_similarity(
            record1["business_name"],
            record2["business_name"]
        ),

        "name_jaccard": jaccard_similarity(
            record1["business_name"],
            record2["business_name"]
        ),

        "name_tfidf": tfidf_similarity(
            record1["business_name"],
            record2["business_name"],
            name_vectorizer
        ),

        "address_similarity": address_similarity(
            record1["business_address"],
            record2["business_address"]
        ),

        "address_jaccard": jaccard_similarity(
            record1["business_address"],
            record2["business_address"]
        ),

        "address_tfidf": tfidf_similarity(
            record1["business_address"],
            record2["business_address"],
            address_vectorizer
        ),

        "country_same": country_same(
            record1["country"],
            record2["country"]
        )
    }

def jaccard_similarity(text1, text2):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 and not words2:
        return 1.0

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

def tfidf_similarity(text1, text2, vectorizer):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    vectors = vectorizer.transform([text1, text2])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return float(similarity)