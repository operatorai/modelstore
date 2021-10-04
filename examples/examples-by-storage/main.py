import json
import os
import tempfile

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
    print(f"\n🆕  Running modelstore example with {modelstore_in} backend.")

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)

    # This demo downloads models; we'll store them into a temporary
    # directory
    tmp_dir = tempfile.mkdtemp()

    # In this demo, we train two models, so that we can demonstrate
    # how modelstore keeps track of uploaded models for us
    model_ids = []
    for model_type in ["sklearn", "xgboost"]:
        print(f"🤖  Training a {model_type} model...")
        model, result = train(model_type)

        #  We're going to write the results to a file
        results_file = os.path.join(tmp_dir, "results.json")
        with open(results_file, "w") as out:
            out.write(json.dumps(result))

        model_domain = "diabetes-boosting-demo"
        print(f"⤴️  Uploading to the {model_domain} domain.")
        meta_data = modelstore.upload(
            model_domain, model=model, extras=results_file
        )

        # The upload returns meta-data about the model that was uploaded
        # This meta-data has also been sync'ed into the s3 bucket
        print(f"✅  Finished uploading the {model_type} model!")
        model_ids.append(meta_data["model"]["model_id"])

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

    # Download models
    print(f"⤵️  Downloading {model_domain} models:")
    for model_id in model_ids:
        print(f"⤵️  Downloading {model_id}:")
        target = os.path.join(tmp_dir, f"downloaded-{model_type}-model")
        os.makedirs(target, exist_ok=True)

        model_path = modelstore.download(
            local_path=target,
            domain=model_domain,
            model_id=model_id,
        )
        print(f"⤵️  Downloaded to: {model_path}")

    # Load models straight into memory
    for model_id in model_ids:
        model = modelstore.load(model_domain, model_id)
        print(f"⤵️  Loaded a {type(model)} model")

    if modelstore_in == "hosted":
        # The rest is currently not implemented in the 'hosted'
        # modelstore
        return

    # Create a new model state
    state_prod = "production"
    print(f"✅  Creating model state={state_prod}:")
    modelstore.create_model_state(state_prod)

    # Set the first model to the production state
    print(f"✅  Setting model_id={versions[0]} to state={state_prod}:")
    modelstore.set_model_state(model_domain, versions[0], state_prod)

    # List the models that are in production
    print(
        f"✅  Listing models for {model_domain} that are in state={state_prod}:"
    )
    versions = modelstore.list_versions(model_domain, state_name=state_prod)
    for version in versions:
        print(f"\t  Domain: {model_domain} has model with id={version}")


if __name__ == "__main__":
    main()
