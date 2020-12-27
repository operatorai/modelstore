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

# pylint disable=import-outside-toplevel
MODEL_CHECKPOINT = "checkpoint.pt"


class PyTorchLightningManager(ModelManager):

    """
    Model persistence for PyTorch Lightning models:
    https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html#checkpoint-saving

    // @TODO: export as for onnx & torchscript
    https://pytorch-lightning.readthedocs.io/en/latest/new-project.html#predict-or-deploy
    """

    @classmethod
    def required_dependencies(cls) -> list:
        return ["pytorch_lightning"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        deps = super().optional_dependencies()
        return deps + ["torch", "torchvision"]

    def _required_kwargs(self):
        return ["trainer", "model"]

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {
            "library": "pytorch_lightning",
            "type": type(kwargs["model"]).__name__,
        }

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

    def _get_functions(self, **kwargs) -> list:
        return [
            partial(
                _save_lightning_model,
                trainer=kwargs["trainer"],
                model=kwargs["model"],
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Currently empty
        // @TODO: investigate other params we can
        return here
        """
        return {}


def _save_lightning_model(
    tmp_dir: str, trainer: "Trainer", model: "LightningModule"
) -> str:
    from pytorch_lightning import Trainer
    from pytorch_lightning.core.lightning import LightningModule

    if not isinstance(trainer, Trainer):
        raise TypeError("'trainer' is not a pytorch_lightning.Trainer!")
    if not isinstance(model, LightningModule):
        raise TypeError("Model is not a LightningModule!")

    file_path = os.path.join(tmp_dir, MODEL_CHECKPOINT)
    trainer.save_checkpoint(file_path)
    return file_path
