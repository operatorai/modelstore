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
from typing import Any, List

from modelstore.metadata import metadata
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

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
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        if "config" in kwargs:
            from transformers import PretrainedConfig
            if not isinstance(kwargs.get("config"), PretrainedConfig):
                logger.debug("unknown config type: %s", type(processor))
                return False
        if "tokenizer" in kwargs:
            from transformers import PreTrainedTokenizerBase
            if not isinstance(kwargs.get("tokenizer"), PreTrainedTokenizerBase):
                logger.debug("unknown tokenizer type: %s", type(processor))
                return False
        if "processor" in kwargs:
            from transformers import ProcessorMixin
            from transformers.image_processing_utils import BaseImageProcessor
            processor = kwargs.get("processor")
            is_processor = (
                isinstance(processor, ProcessorMixin)
                or isinstance(processor, BaseImageProcessor)
            )
            if not is_processor:
                logger.debug("unknown processor type: %s", type(processor))
                return False

        # The model must be either a PyTorch or TF pretrained model
        try:
            from transformers import TFPreTrainedModel
            if isinstance(kwargs.get("model"), TFPreTrainedModel):
                return True
        except RuntimeError:
            # Cannot import tensorflow things
            pass
        
        from transformers import PreTrainedModel
        return isinstance(kwargs.get("model"), PreTrainedModel)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model not matched with transformers")
        return [
            partial(
                _save_transformers,
                entities=[
                    kwargs["model"],
                    kwargs.get("tokenizer"),
                    kwargs.get("config"),
                    kwargs.get("processor"),
                ],
            ),
        ]
            

    def get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing the config for the model
        """
        if "config" in kwargs:
            return kwargs["config"].to_dict()
        return {}

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)
        model_dir = _get_model_directory(model_path)
        model_files = set(os.listdir(model_dir))
        logger.debug("Loading from: %s...", model_files)

        # pylint: disable=import-outside-toplevel
        # Infer whether a tokenizer was saved
        tokenizer = None
        if "tokenizer.json" in model_files:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(tokenizer))

        # Infer whether a config was saved
        config = None
        if "config.json" in model_files:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(config))

        processor = None
        if "preprocessor_config.json" in model_files:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(processor))

        # Infer whether we're loading a PyTorch or Tensorflow model
        is_pytorch = "pytorch_model.bin" in model_files
        logger.debug("Loading transformers model with pytorch=%s", is_pytorch)

        if is_pytorch:
            from transformers import AutoModel, GPT2LMHeadModel

            # In examples-by-ml-library/libraries/huggingface/gpt2*.py, we want
            # to load a GPT2 model with a language model head. If we just
            # load the model with AutoModel, then it won't have this.
            # This is a hack to get around that, like we did in the XGBoost
            # manager, and currently does not generalise beyond this case
            model_types = {
                "GPT2LMHeadModel": GPT2LMHeadModel,
                # @TODO add other model types
            }
            model_type = meta_data.model_type().type
            if model_type in model_types:
                logger.debug("Loading with %s...", model_type)
                model = model_types[model_type].from_pretrained(model_dir)
            else:
                logger.debug("Loading with AutoModel...")
                model = AutoModel.from_pretrained(model_dir)
        else:
            from transformers import TFAutoModel, TFGPT2LMHeadModel

            # In examples-by-ml-library/libraries/huggingface/gpt2*.py, we want
            # to load a GPT2 model with a language model head. If we just
            # load the model with TFAutoModel, then it won't have this.
            # This is a hack to get around that, like we did in the XGBoost
            # manager, and currently does not generalise beyond this case
            model_types = {
                "TFGPT2LMHeadModel": TFGPT2LMHeadModel,
                # @TODO add other model types
            }
            model_type = meta_data.model_type().type
            if model_type in model_types:
                logger.debug("Loading with %s...", model_type)
                model = model_types[model_type].from_pretrained(model_dir)
            else:
                logger.debug("Loading with TFAutoModel...")
                model = TFAutoModel.from_pretrained(model_dir)
        if tokenizer is not None:
            return model, tokenizer, config
        if processor is not None:
            return model, processor, config
        return model, config


def _get_model_directory(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_transformers(
    tmp_dir: str,
    entities: List,
) -> str:
    model_dir = _get_model_directory(tmp_dir)
    os.makedirs(model_dir)
    for entity in entities:
        if entity is not None:
            entity.save_pretrained(model_dir)
    return model_dir
