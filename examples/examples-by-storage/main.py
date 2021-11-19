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

    # Let's demo all the different things you can do!
    demos.list_domains(modelstore)

    # List the models in the diabest-boosting-demo domain
    demos.list_models_in_domain(modelstore, model_domain)

    # Download the models back
    demos.download_models(modelstore, model_domain, model_ids)

    # Download the latest model
    demos.download_latest_model(modelstore, model_domain)

    # Load models back into memory
    demos.load_models(modelstore, model_domain, model_ids)

    # Create a new model state
    state_name = "production"
    demos.create_a_model_state(modelstore, state_name)

    # Set the first model to the production state
    model_id = list(model_ids.values())[0]
    demos.set_model_state(modelstore, model_domain, model_id, state_name)

    # List the models that are in production - the new model is there
    model_ids = demos.list_models_in_domain_with_state(
        modelstore, model_domain, state_name
    )
    assert model_id in model_ids

    # Remove a state from a model
    demos.remove_model_state(modelstore, model_domain, model_id, state_name)

    # List the models that are in production - the removed model is not there
    model_ids = demos.list_models_in_domain_with_state(
        modelstore, model_domain, state_name
    )
    assert model_id not in model_ids


if __name__ == "__main__":
    main()
