import random

from annoy import AnnoyIndex
from modelstore.model_store import ModelStore

_NUM_DIMENSIONS = 40
_NUM_TREES = 10
_METRIC = "angular"
_DOMAIN_NAME = "example-annoy-index"


def _train_example_model() -> AnnoyIndex:
    # Create an index
    print("🤖  Creating an Annoy index...")
    model = AnnoyIndex(_NUM_DIMENSIONS, _METRIC)
    for i in range(1000):
        vector = [random.gauss(0, 1) for z in range(_NUM_DIMENSIONS)]
        model.add_item(i, vector)
    model.build(_NUM_TREES)

    # Show some nearest neighbours
    results = model.get_nns_by_item(0, 10)
    print(f"🔍  Nearest neighbors = {results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train an Annoy index
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the Annoy model to the "{_DOMAIN_NAME}" domain.')
    meta_data = modelstore.upload(
        _DOMAIN_NAME,
        model=model,
        num_dimensions=_NUM_DIMENSIONS,
        metric=_METRIC,
        num_trees=_NUM_TREES,
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the Annoy "{_DOMAIN_NAME}" domain model={model_id}')
    model = modelstore.load(_DOMAIN_NAME, model_id)

    # Find some nearest neighbours
    results = model.get_nns_by_item(0, 10)
    print(f"🔍  Nearest neighbors = {results}.")
