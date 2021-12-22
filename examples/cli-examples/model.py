import joblib
import numpy as np
from modelstore.utils.cli import info
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_and_save():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "ls",
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    info(f"📈  Trained a model with RMSE={rmse}.")

    file_name = "model.joblib"
    joblib.dump(model, file_name)
    info(f"✅  Model saved to file={file_name}.")


if __name__ == "__main__":
    train_and_save()
