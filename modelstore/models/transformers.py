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
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

# pylint disable=import-outside-toplevel
MODEL_DIRECTORY = "transformers"


class TransformersManager(ModelManager):

    """
    Model persistence for Transformer models:
    https://huggingface.co/transformers/main_classes/model.html#transformers.TFPreTrainedModel.save_pretrained
    https://github.com/huggingface/transformers/blob/e50a931c118b9f55f77a743bf703f436bf7a7c29/src/transformers/modeling_utils.py#L676
    """

    NAME = "transformers"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["transformers"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["torch", "tensorflow"]

    def _required_kwargs(self):
        return ["model", "tokenizer", "config"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import transformers

        return (
            isinstance(kwargs.get("model"), transformers.PreTrainedModel)
            and isinstance(kwargs.get("config"), transformers.PretrainedConfig)
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

    def load(self, model_path: str, meta_data: dict) -> Any:
        """
        Loads a model, stored in model_path,
        back into memory
        """
        # pylint: disable=import-outside-toplevel
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        model_dir = _get_model_directory(model_path)
        model = AutoModel.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer, config


def _get_model_directory(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_transformers(
    tmp_dir: str,
    config: "transformers.PretrainedConfig",
    model: "transformers.PreTrainedModel",
    tokenizer: "transformers.PreTrainedTokenizerBase",
) -> str:
    model_dir = _get_model_directory(tmp_dir)
    os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir
