import json
import os
import tempfile
import time

import click

from model import train
from modelstores import create_model_store


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(
        ["aws", "azure", "gcloud", "filesystem", "hosted"], case_sensitive=False
    ),
)
def main(modelstore_in):
    print(f"🆕  Running modelstore example with {modelstore_in} backend.")

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)
    model_domain = "diabetes-boosting-demo"

    # This demo downloads models; we'll store them into a temporary
    # directory
    tmp_dir = tempfile.mkdtemp()

    # # In this demo, we train two models, so that we can demonstrate
    # # how modelstore keeps track of uploaded models for us
    model_ids = {}
    for model_type in ["sklearn", "xgboost"]:
        print(f"🤖  Training a {model_type} model...")
        model, result = train(model_type)

        #  We're going to write the results to a file
        results_file = os.path.join(tmp_dir, "results.json")
        with open(results_file, "w") as out:
            out.write(json.dumps(result))

        print(f"⤴️  Uploading to the {model_domain} domain.")
        meta_data = modelstore.upload(
            model_domain, model=model, extras=results_file
        )

        # Currently, modelstore stores artifacts in a prefix
        #  that has the training timestamp encoded in it. If
        # we upload two models at the exact same second, the
        # second one will overwrite the first. So we add in an
        # artifical sleep to split things out
        time.sleep(1)

        # The upload returns meta-data about the model that was uploaded
        # This meta-data has also been sync'ed into the s3 bucket
        model_id = meta_data["model"]["model_id"]
        print(f"✅  Finished uploading the {model_type} model: {model_id}")
        model_ids[model_type] = model_id

    # We now have push an additional two models into our store
    # How does modelstore enable you to manage them?

    # List all the domains
    print(f"✅  Listing existing domains:")
    domains = modelstore.list_domains()
    for domain in domains:
        print(f"\t  Domain: {domain}")

    # List the models in the diabest-boosting-demo domain
    print(f"✅  Listing models for {model_domain}:")
    versions = modelstore.list_versions(domain=model_domain)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")

    # Download models back
    print(f"⤵️  Downloading {model_domain} models:")
    for model_type, model_id in model_ids.items():
        print(f"\t  Downloading {model_type}={model_id}")
        target = os.path.join(tmp_dir, f"downloaded-{model_type}-model")
        os.makedirs(target, exist_ok=True)

        model_path = modelstore.download(
            local_path=target,
            domain=model_domain,
            model_id=model_id,
        )
        print(f"\t  Downloaded to: {model_path}")

    # You don't need to download models manually, you can
    # also load models straight into memory
    print(f"💡  Loading models into memory")
    for model_type, model_id in model_ids.items():
        print(f"\t  Loading {model_type}={model_id}")
        model = modelstore.load(model_domain, model_id)
        print(f"\t  Loaded a {type(model)} model")

    if modelstore_in == "hosted":
        # The rest is currently not implemented in the 'hosted'
        # modelstore
        return

    # Create a new model state
    state_prod = "production"
    print(f"✅  Creating model state={state_prod}:")
    modelstore.create_model_state(state_prod)

    # Set the first model to the production state
    prod_model = list(model_ids.values())[0]
    print(f"✅  Setting model_id={prod_model} to state={state_prod}:")
    modelstore.set_model_state(model_domain, prod_model, state_prod)

    # List the models that are in production
    print(
        f"✅  Listing models for {model_domain} that are in state={state_prod}:"
    )
    versions = modelstore.list_versions(model_domain, state_name=state_prod)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")


if __name__ == "__main__":
    main()
