import json
import os
import tempfile
import time

import click

import demos
from model import train
from modelstores import create_model_store


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(["aws", "azure", "gcloud", "filesystem"], case_sensitive=False),
)
def main(modelstore_in):
    print(f"🆕  Running modelstore example with {modelstore_in} backend.")

    # Create a model store instance
    modelstore = create_model_store(modelstore_in)
    model_domain = "diabetes-boosting-demo"

    # This demo downloads models; we'll store them into a temporary
    # directory
    tmp_dir = tempfile.mkdtemp()

    # In this demo, we train two models, so that we can demonstrate
    # how modelstore keeps track of uploaded models for us
    model_ids = {}
    for model_type in ["sklearn", "xgboost"]:
        print(f"🤖  Training a {model_type} model...")
        model, result = train(model_type)

        #  We're going to write the results to a file
        results_file = os.path.join(tmp_dir, "results.json")
        with open(results_file, "w") as out:
            out.write(json.dumps(result))

        print(f"⤴️  Uploading to the {model_domain} domain.")
        meta_data = modelstore.upload(model_domain, model=model, extras=results_file)

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

    # Let's demo all the different things you can do!

    # List all of the domains
    demos.list_domains(modelstore)

    # List the models in the diabest-boosting-demo domain
    demos.list_models_in_domain(modelstore, model_domain)

    # Download the models back
    demos.download_models(modelstore, model_domain, model_ids)

    # Download the latest model
    demos.download_latest_model(modelstore, model_domain)

    # Load models back into memory
    demos.load_models(modelstore, model_domain, model_ids)

    # Create a couple of new model states
    state_names = ["staging", "production"]
    for state_name in state_names:
        demos.create_a_model_state(modelstore, state_name)

    # List them back
    model_state_names = demos.list_model_states(modelstore)
    for state_name in state_names:
        assert state_name in model_state_names

    # Set the first model to the production state
    model_id = list(model_ids.values())[0]
    demos.set_model_state(modelstore, model_domain, model_id, "production")

    # List the models that are in production - the new model is there
    model_ids = demos.list_models_in_domain_with_state(
        modelstore, model_domain, "production"
    )
    assert model_id in model_ids

    # Remove a state from a model
    demos.remove_model_state(modelstore, model_domain, model_id, "production")

    # Set the model to a different state
    demos.set_model_state(modelstore, model_domain, model_id, "staging")
    model_ids = demos.list_models_in_domain_with_state(
        modelstore, model_domain, "staging"
    )
    assert model_id in model_ids

    # Delete the model!
    demos.delete_model(model_domain, model_id)
    for state_name in state_names:
        # The model no longer appears in the listing by state
        model_ids = demos.list_models_in_domain_with_state(
            modelstore, model_domain, state_name
        )
        assert model_id not in model_ids

    # The model no longer appears when listing all of them
    model_ids = demos.list_models_in_domain(modelstore, model_domain)
    assert model_id not in model_ids

    # @TODO catch the exception if trying to download a deleted model


if __name__ == "__main__":
    main()
