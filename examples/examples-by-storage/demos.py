import tempfile

from modelstore.model_store import ModelStore


def list_domains(modelstore: ModelStore):
    print(f"✅  Listing existing domains:")
    domains = modelstore.list_domains()
    for domain in domains:
        print(f"\t  Domain: {domain}")


def list_models_in_domain(modelstore: ModelStore, model_domain: str) -> list:
    print(f"✅  Listing models for {model_domain}:")
    model_ids = modelstore.list_models(domain=model_domain)
    for model_id in model_ids:
        print(f"\t  Domain: {model_domain} has model with id={model_id}")
    return model_ids


def list_models_in_domain_with_state(
    modelstore: ModelStore,
    model_domain: str,
    state_name: str,
    expected_model_id: str = None,
) -> list:
    print(f"✅  Listing models for {model_domain} that are in state={state_name}:")
    model_ids = modelstore.list_models(model_domain, state_name=state_name)
    for model_id in model_ids:
        print(f"\t  Domain: {model_domain} has model with id={model_id}")
    if expected_model_id is not None:
        assert expected_model_id in model_ids
    return model_ids


def load_model(modelstore: ModelStore, model_domain: str, model_id: str):
    print(f"✅  Loading models into memory")
    model = modelstore.load(model_domain, model_id)
    print(f"\t  Loaded a {type(model)} model={model_id}")


def download_model(modelstore: ModelStore, model_domain: str, model_id: str):
    print(f"✅  Downloading models")
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\t  Downloading {model_id}")
        model_path = modelstore.download(tmp_dir, model_domain, model_id)
        print(f"\t  Downloaded a model to: {model_path}")


def download_latest_model(modelstore: ModelStore, model_domain: str):
    print(f"✅  Downloading latest model")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = modelstore.download(tmp_dir, model_domain)
        print(f"\t  Downloaded latest model to: {model_path}")


def list_model_states(modelstore: ModelStore, expected_states: list):
    print("✅  Listing available model states")
    model_state_names = modelstore.list_model_states()
    for model_state in model_state_names:
        print(f"\t  Model state: {model_state}")
    for state_name in expected_states:
        assert state_name in model_state_names


def set_model_state(
    modelstore: ModelStore, model_domain: str, model_id: str, state_name: str
):
    print(f"✅  Setting model_id={model_id} to state={state_name}:")
    modelstore.set_model_state(model_domain, model_id, state_name)


def remove_model_state(
    modelstore: ModelStore, model_domain: str, model_id: str, state_name: str
):
    print(f"✅  Unsetting model_id={model_id} from state={state_name}:")
    modelstore.remove_model_state(model_domain, model_id, state_name)


def delete_model(modelstore: ModelStore, model_domain: str, model_id: str):
    print(f"✅  Deleting model_id={model_id} from the model store.")
    modelstore.delete_model(model_domain, model_id, skip_prompt=True)
