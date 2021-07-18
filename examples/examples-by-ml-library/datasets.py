import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_diabetes_dataset():
    diabetes = load_diabetes()
    X_train, _, y_train, _ = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    return X_train, y_train


def load_diabetes_dataframe():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["y"] = diabetes.target
    return df
