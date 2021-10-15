import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train(model_type):
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    if model_type == "sklearn":
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "ls",
        }
        model = GradientBoostingRegressor(**params)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"📈  Trained a model with RMSE={rmse}.")
    return model, {"result": rmse}
