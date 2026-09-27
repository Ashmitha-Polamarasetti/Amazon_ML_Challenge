import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Normalize business names and addresses while preserving
    multilingual Unicode characters.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)

    # Replace punctuation/symbols with spaces.
    # Unicode letters and numbers are preserved.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text



def normalize_dataframe(df):
    """
    Add normalized business name and address columns.
    Original columns are preserved.
    """

    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_text)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_text)
    )

    return df