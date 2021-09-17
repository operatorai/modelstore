from collections import namedtuple

import pandas as pd
import torch
from sklearn.datasets import fetch_20newsgroups, load_diabetes
from sklearn.model_selection import train_test_split

Dataset = namedtuple("Dataset", ["X_train", "X_test", "y_train", "y_test"])
_diabetes = None


def load_diabetes_dataset(as_numpy: bool = False):
    print(f"🔍  Loading the diabetes dataset")
    global _diabetes
    if _diabetes is None:
        diabetes = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.1, random_state=13
        )
        _diabetes = Dataset(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
    if as_numpy:
        X_train = torch.from_numpy(_diabetes.X_train).float()
        X_test = torch.from_numpy(_diabetes.X_test).float()

        y_train = torch.from_numpy(_diabetes.y_train).float().view(-1, 1)
        y_test = torch.from_numpy(_diabetes.y_test).float().view(-1, 1)

        return X_train, X_test, y_train, y_test
    return (
        _diabetes.X_train,
        _diabetes.X_test,
        _diabetes.y_train,
        _diabetes.y_test,
    )


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
