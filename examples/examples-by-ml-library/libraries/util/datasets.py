import pandas as pd

import torch
from sklearn.datasets import fetch_20newsgroups, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split


def load_regression_dataset(as_numpy: bool = False):
    print(f"🔍  Loading the diabetes dataset")
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )
    if as_numpy:
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

        y_train = torch.from_numpy(y_train).float().view(-1, 1)
        y_test = torch.from_numpy(y_test).float().view(-1, 1)

        return X_train, X_test, y_train, y_test
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def load_regression_dataframe():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["y"] = diabetes.target
    return df


def load_classification_dataset():
    print(f"🔍  Loading the breast cancer dataset")
    databunch = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        databunch.data, databunch.target, test_size=0.1, random_state=13
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def load_text_dataset():
    print(f"⏳  Fetching the newsgroups data...")
    newsgroups = fetch_20newsgroups(
        subset="train",
        categories=[
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ],
        shuffle=True,
        random_state=42,
    )
    return [doc.strip().split() for doc in newsgroups.data]
