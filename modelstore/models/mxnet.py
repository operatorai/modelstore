#    Copyright 2021 Neal Lathia
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

MODEL_FILE = "model-symbol.json"
PARAMS_FILE = "model-{}.params"


class MxnetManager(ModelManager):

    """
    Model persistence for Mxnet (Hybrid) Gluon models
    https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html
    """

    NAME = "mxnet"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["mxnet"]

    def optional_dependencies(self) -> list:
        return super().optional_dependencies() + ["onnx"]

    def _required_kwargs(self):
        return ["model", "epoch"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from mxnet.gluon import nn

        # Using nn.HybridBlock instead of nn.Block because Hybrid
        # blocks can be exported
        return isinstance(kwargs.get("model"), nn.HybridBlock)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not an mxnet nn.HybridBlock!")
        if "epoch" not in kwargs:
            raise ValueError("Mxnet uploads require the 'epoch' kwarg to be set.")

        return [
            partial(
                save_model,
                model=kwargs["model"],
                epoch=kwargs["epoch"],
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        return {
            "epoch": kwargs["epoch"],
        }

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from mxnet.gluon import SymbolBlock

        epoch = int(meta_data["model"]["parameters"]["epoch"])
        return SymbolBlock.imports(
            model_file_path(model_path),
            ["data"],
            params_file_path(model_path, epoch),
        )


def model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_FILE)


def params_file_path(parent_dir: str, epoch: int) -> str:
    return os.path.join(parent_dir, PARAMS_FILE.format(f"{epoch:04d}"))


def save_model(tmp_dir: str, model: "nn.HybridBlock", epoch: int) -> str:
    # model.export() stores files in the current directory, so we chdir()
    # to the target directory where we want the files saved
    cwd = os.getcwd()
    os.chdir(tmp_dir)
    # Two files path-symbol.json and path-xxxx.params will be created,
    # where xxxx is the 4 digits epoch number, in the current directory
    model.export("model", epoch=epoch)
    # Go back to the previous working directory
    os.chdir(cwd)
    return [
        model_file_path(tmp_dir),
        params_file_path(tmp_dir, epoch),
    ]
