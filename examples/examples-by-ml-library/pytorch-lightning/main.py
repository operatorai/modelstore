import json
import os
from tempfile import mkdtemp

import click
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from modelstore import ModelStore
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def create_model_store(backend) -> ModelStore:
    if backend == "filesystem":
        # By default, create a new local model store
        # in our home directory
        home_dir = os.path.expanduser("~")
        return ModelStore.from_file_system(root_directory=home_dir)
    if backend == "gcloud":
        # The modelstore library assumes you have already created
        # a Cloud Storage bucket and will raise an exception if it doesn't exist
        return ModelStore.from_gcloud(
            os.environ["GCP_PROJECT_ID"],
            os.environ["GCP_BUCKET_NAME"],
        )
    if backend == "aws":
        # The modelstore library assumes that you already have
        # created an s3 bucket where you want to store your models, and
        # will raise an exception if it doesn't exist.
        return ModelStore.from_aws_s3(os.environ["AWS_BUCKET_NAME"])
    if backend == "hosted":
        # To use the hosted model store, you need an API key
        return ModelStore.from_api_key(
            os.environ["MODELSTORE_KEY_ID"], os.environ["MODELSTORE_ACCESS_KEY"]
        )
    raise ValueError(f"Unknown model store: {backend}")


class ExampleLightningNet(pl.LightningModule):
    def __init__(self):
        super(ExampleLightningNet, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        train_loss = F.mse_loss(y_hat, y)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        val_loss = F.mse_loss(y_hat, y)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.05
        )
        return [optimizer], [scheduler]


def train():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().view(-1, 1)
    data_set = TensorDataset(X_test, y_test)
    val_dataloader = DataLoader(data_set)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    data_set = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(data_set)

    model = ExampleLightningNet()
    trainer = pl.Trainer(max_epochs=5, default_root_dir=mkdtemp())
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer, model


@click.command()
@click.option(
    "--storage",
    type=click.Choice(
        ["filesystem", "gcloud", "aws", "hosted"], case_sensitive=False
    ),
)
def main(storage):
    model_domain = "diabetes-boosting-demo"

    # Create a model store instance
    model_store = create_model_store(storage)

    # In this demo, we train a single layered net
    # using the sklearn.datasets.load_diabetes dataset
    print(f"🤖  Training a pytorch-lightning model...")
    trainer, model = train()

    print(f"⤴️  Uploading the archive to the {model_domain} domain.")
    meta = model_store.pytorch_lightning.upload(
        model_domain, trainer=trainer, model=model
    )

    # The upload returns meta-data about the model that was uploaded
    # This meta-data has also been sync'ed into the cloud storage
    #  bucket
    print(f"✅  Finished uploading the pytorch-lightning model!")
    print(json.dumps(meta, indent=4))


if __name__ == "__main__":
    main()
