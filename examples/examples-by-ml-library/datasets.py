import pandas as pd

from sklearn.datasets import fetch_20newsgroups, load_diabetes
from sklearn.model_selection import train_test_split


def load_diabetes_dataset():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    return X_train, X_test, y_train, y_test


def load_diabetes_dataframe():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["y"] = diabetes.target
    return df


def load_newsgroup_sentences():
    categories = [
        "alt.atheism",
        "soc.religion.christian",
        "comp.graphics",
        "sci.med",
    ]
    print(f"⏳  Fetching the newsgroups data...")
    newsgroups = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )
    return [doc.strip().split() for doc in newsgroups.data]
