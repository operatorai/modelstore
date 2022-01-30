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
import inspect
import os
import sys
from functools import partial
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

# pylint disable=import-outside-toplevel
MODEL_CHECKPOINT = "checkpoint.pt"


class PyTorchLightningManager(ModelManager):

    """
    Model persistence for PyTorch Lightning models:
    https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#manual-saving

    // @TODO: export as for onnx & torchscript
    https://pytorch-lightning.readthedocs.io/en/latest/new-project.html#predict-or-deploy
    """

    NAME = "pytorch_lightning"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["pytorch_lightning"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["torch", "torchvision"]

    def _required_kwargs(self):
        return ["trainer", "model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from pytorch_lightning import LightningModule, Trainer

        return isinstance(kwargs.get("trainer"), Trainer) and isinstance(
            kwargs.get("model"), LightningModule
        )

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("'trainer' is not from pytorch_lightning!")

        return [
            partial(
                _save_lightning_model,
                trainer=kwargs["trainer"],
            ),
        ]

    @classmethod
    def _find_class(cls, class_name: str):
        modules = sys.modules.copy()
        for module_name in modules:
            try:
                classes = inspect.getmembers(modules[module_name], inspect.isclass)
                classes = [c for c in classes if c[0] == class_name]
                if len(classes) == 1:
                    return classes[0][1]
            except (ImportError, TypeError, ModuleNotFoundError):
                continue
        raise ValueError(f"Please import {class_name} before calling load()")

    def load(self, model_path: str, meta_data: dict) -> Any:
        # The name of the class for the model
        model_class_name = meta_data["model"]["model_type"]["type"]
        model_file = _model_file_path(model_path)

        # We assume that class has already been imported, so it exists
        # in the current module
        model_class = self._find_class(model_class_name)
        return model_class.load_from_checkpoint(model_file)


def _model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_CHECKPOINT)


def _save_lightning_model(tmp_dir: str, trainer: "pytorch_lightning.Trainer") -> str:
    file_path = _model_file_path(tmp_dir)
    trainer.save_checkpoint(file_path)
    return file_path
