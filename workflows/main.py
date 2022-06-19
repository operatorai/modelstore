#    Copyright 2022 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import click

from actions import actions
from fixtures import extra
from fixtures.models import iter_models
from fixtures.modelstores import create_model_store


@click.command()
@click.option(
    "--modelstore-in",
    type=click.Choice(["aws", "azure", "gcloud", "filesystem"]),
)
def main(modelstore_in: str):
    """ Executes all of the integration tests in a given storage type """
    print(f"🆕  Running modelstore example with {modelstore_in} backend.")

    model_store = create_model_store(modelstore_in)
    actions.run_on_storage(model_store)

    extra_files = extra.files()
    for model in iter_models():
        actions.run_with_model(
            model_store,
            model,
            extra_metadata=extra.metadata(),
            extra_files=extra_files,
        )

    # Clean up
    for extra_file in extra_files:
        os.remove(extra_file)


if __name__ == "__main__":
    main()
