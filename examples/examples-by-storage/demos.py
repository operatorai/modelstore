import tempfile

from modelstore.model_store import ModelStore


def list_domains(modelstore: ModelStore):
    print(f"✅  Listing existing domains:")
    domains = modelstore.list_domains()
    for domain in domains:
        print(f"\t  Domain: {domain}")


def list_models_in_domain(modelstore: ModelStore, model_domain: str):
    print(f"✅  Listing models for {model_domain}:")
    versions = modelstore.list_versions(domain=model_domain)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")


def list_models_in_domain_with_state(
    modelstore: ModelStore, model_domain: str, state_name: str
) -> list:
    print(
        f"✅  Listing models for {model_domain} that are in state={state_name}:"
    )
    model_ids = modelstore.list_versions(model_domain, state_name=state_name)
    for model_id in model_ids:
        print(f"\t  Domain: {model_domain} has model with id={model_id}")
    return model_ids


def load_models(modelstore: ModelStore, model_domain: str, model_ids: dict):
    print(f"💡  Loading models into memory")
    for model_type, model_id in model_ids.items():
        print(f"\t  Loading {model_type}={model_id}")
        model = modelstore.load(model_domain, model_id)
        print(f"\t  Loaded a {type(model)} model")


def download_models(modelstore: ModelStore, model_domain: str, model_ids: dict):
    print(f"💡  Downloading models")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for model_type, model_id in model_ids.items():
            print(f"\t  Downloading {model_type}={model_id}")
            model_path = modelstore.download(tmp_dir, model_domain, model_id)
            print(f"\t  Downloaded a model to: {model_path}")


def download_latest_model(modelstore: ModelStore, model_domain: str):
    print(f"💡  Downloading latest model")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = modelstore.download(tmp_dir, model_domain)
        print(f"\t  Downloaded latest model to: {model_path}")


def create_a_model_state(modelstore: ModelStore, state_name: str):
    print(f"✅  Creating model state={state_name}:")
    modelstore.create_model_state(state_name)


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
