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
MODEL_PT = "model.pt"


class PyTorchManager(ModelManager):

    """
    Model persistence for PyTorch models:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
    """

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

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"library": "pytorch"}

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

    def _get_functions(self, **kwargs) -> list:
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
        return _convert_tensors(kwargs["optimizer"].state_dict())


def _convert_tensors(model_params: dict) -> dict:
    import torch

    for k, v in model_params.items():
        if isinstance(v, dict):
            model_params[k] = _convert_tensors(v)
        if isinstance(v, torch.Tensor):
            if hasattr(v, "detach"):
                v = v.detach()
            model_params[k] = v.cpu().numpy()
    return model_params


def _save_state_dict(
    tmp_dir: str, model: "nn.Module", optimizer: "optim.Optimizer"
) -> str:
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Model is not a torch.nn.Module!")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("Optimizer is not a torch.optim.Optimizer!")

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
    import torch

    file_path = os.path.join(tmp_dir, MODEL_PT)
    torch.save(model, file_path)
    return file_path
