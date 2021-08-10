#    Copyright 2020 Neal Lathia
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
from functools import partial

from modelstore.models.modelmanager import ModelManager
from modelstore.storage.storage import CloudStorage

# pylint disable=import-outside-toplevel
MODEL_DIRECTORY = "transformers"


class TransformersManager(ModelManager):

    """
    Model persistence for Transformer models:
    https://huggingface.co/transformers/main_classes/model.html#transformers.TFPreTrainedModel.save_pretrained
    https://github.com/huggingface/transformers/blob/e50a931c118b9f55f77a743bf703f436bf7a7c29/src/transformers/modeling_utils.py#L676
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("transformers", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return ["transformers"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        deps = super().optional_dependencies()
        return deps + ["torch", "tensorflow"]

    def _required_kwargs(self):
        return ["model", "tokenizer"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import transformers

        return (
            isinstance(kwargs.get("config"), transformers.PretrainedConfig)
            and isinstance(kwargs.get("model"), transformers.PreTrainedModel)
            and isinstance(
                kwargs.get("tokenizer"), transformers.PreTrainedTokenizerBase
            )
        )

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError(
                "Model/config/tokenizer is not a transformers.PretrainedConfig!"
            )
        return [
            partial(
                _save_transformers,
                config=kwargs["config"],
                model=kwargs["model"],
                tokenizer=kwargs["tokenizer"],
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing the config for the model
        https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
        """
        return kwargs["config"].to_dict()


def _save_transformers(
    tmp_dir: str,
    config: "transformers.PretrainedConfig",
    model: "transformers.PreTrainedModel",
    tokenizer: "transformers.PreTrainedTokenizerBase",
) -> str:

    model_dir = os.path.join(tmp_dir, MODEL_DIRECTORY)
    model.save_pretrained(model_dir)
    if config is not None:
        config.save_pretrained(model_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(model_dir)
    return model_dir
