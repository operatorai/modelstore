import torch
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error
from torch import nn

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


# pylint: disable=missing-class-docstring
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def _train_example_model() -> ExampleNet:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset(as_numpy=True)

    # Train the model
    model = ExampleNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(5):
        print(f"🤖  Training epoch: {epoch}...")
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    results = mean_squared_error(y_test, model(X_test).detach().numpy())
    print(f"🔍  Fit model MSE={results}.")
    return model, optimizer


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a PyTorch model
    model, optimizer = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the pytorch model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(
        DIABETES_DOMAIN, model=model, optimizer=optimizer
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the pytorch "{DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(DIABETES_DOMAIN, model_id)
    model.eval()

    _, X_test, _, y_test = load_diabetes_dataset(as_numpy=True)
    results = mean_squared_error(y_test, model(X_test).detach().numpy())
    print(f"🔍  Loaded model MSE={results}.")
