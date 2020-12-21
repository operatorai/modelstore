import numpy as np
import pandas as pd
from modelstore.meta import datasets


def test_describe_np_training():
    exp = {"shape": [10, 50]}
    dataset = np.random.rand(10, 50)
    res = datasets.describe_training(dataset)
    assert exp == res


def test_describe_np_labels():
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    labels = np.array([1, 2, 1, 2, 1])
    res = datasets.describe_labels(labels)
    assert exp == res


def test_describe_df_training():
    exp = {"shape": [20, 50]}
    rows = []
    for _ in range(20):
        rows.append({f"col_{j}": j for j in range(50)})
    df = pd.DataFrame(rows)
    res = datasets.describe_training(df)
    assert exp == res


def test_describe_df_labels():
    exp = {"shape": [5], "values": {1: 3, 2: 2}}
    labels = pd.Series([1, 2, 1, 2, 1])
    res = datasets.describe_labels(labels)
    assert exp == res
