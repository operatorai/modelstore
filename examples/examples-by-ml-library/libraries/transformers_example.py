from modelstore.model_store import ModelStore
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

_DOMAIN_NAME = "example-distilbert-model"


def _train_example_model():
    model_name = "distilbert-base-cased"
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        finetuning_task="mnli",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    # Skipped for brevity!
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=build_compute_metrics_fn(data_args.task_name),
    # )
    # trainer.train()
    return model, tokenizer, config


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model, tokenizer, config = _train_example_model()

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the transformers model to the "{_DOMAIN_NAME}" domain.'
    )
    meta_data = modelstore.upload(
        _DOMAIN_NAME,
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the transformers "{_DOMAIN_NAME}" domain model={model_id}'
    )
    model, tokenizer, config = modelstore.load(_DOMAIN_NAME, model_id)

    # Run some example predictions
    # ...
