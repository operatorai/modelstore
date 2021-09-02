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
from modelstore.models.util import convert_numpy, convert_tensors
from modelstore.storage.storage import CloudStorage

# pylint disable=import-outside-toplevel
MODEL_CHECKPOINT = "checkpoint.pt"
MODEL_PT = "model.pt"


class PyTorchManager(ModelManager):

    """
    Model persistence for PyTorch models:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("pytorch", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return ["torch"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        deps = super().optional_dependencies()
        return deps + ["torchvision"]

    def _required_kwargs(self):
        return ["model", "optimizer"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import torch

        return isinstance(kwargs.get("model"), torch.nn.Module) and isinstance(
            kwargs.get("optimizer"), torch.optim.Optimizer
        )

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("model/optimizer is not from torch!")
        return [
            partial(
                _save_state_dict,
                model=kwargs["model"],
                optimizer=kwargs["optimizer"],
            ),
            partial(_save_model, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary the optimizer's state
        dictionary
        """
        params = kwargs["optimizer"].state_dict()
        params = convert_numpy(params)
        return convert_tensors(params)

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import torch

        file_path = _get_model_path(model_path)
        return torch.load(file_path)


def _get_model_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_PT)


def _save_state_dict(
    tmp_dir: str, model: "nn.Module", optimizer: "optim.Optimizer"
) -> str:
    # pylint: disable=import-outside-toplevel
    import torch

    file_path = os.path.join(tmp_dir, MODEL_CHECKPOINT)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file_path,
    )
    return file_path


def _save_model(tmp_dir: str, model: "nn.Module") -> str:
    # pylint: disable=import-outside-toplevel
    import torch

    file_path = _get_model_path(tmp_dir)
    torch.save(model, file_path)
    return file_path
