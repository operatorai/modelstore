from typing import Tuple

from gensim.models import word2vec
from modelstore.model_store import ModelStore

from libraries.util.datasets import load_text_dataset
from libraries.util.domains import NEWSGROUP_EMBEDDINGS_DOMAIN


def _train_example_model() -> word2vec.Word2Vec:
    # Load the data
    sentences = load_text_dataset()

    # Train a word2vec model
    print(f"🤖  Training a word2vec model...")
    model = word2vec.Word2Vec(sentences, min_count=2)

    most_similar = set([k[0] for k in model.wv.most_similar("cool", topn=5)])
    print(f"🤖  Most similar to 'cool': {most_similar}")
    return model


def train_and_upload(modelstore: ModelStore) -> Tuple[str, str]:
    # Train a word2vec model
    model = _train_example_model()

    # Upload the model to the model store
    print(
        f"⤴️  Uploading the word2vec model to the {NEWSGROUP_EMBEDDINGS_DOMAIN} domain."
    )
    meta_data = modelstore.upload(NEWSGROUP_EMBEDDINGS_DOMAIN, model=model)
    return NEWSGROUP_EMBEDDINGS_DOMAIN, meta_data["model"]["model_id"]


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the word2vec "{NEWSGROUP_EMBEDDINGS_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(NEWSGROUP_EMBEDDINGS_DOMAIN, model_id)

    # Find some nearest neighbours
    most_similar = set([k[0] for k in model.wv.most_similar("cool", topn=5)])
    print(f"🤖  Most similar to 'cool': {most_similar}")
