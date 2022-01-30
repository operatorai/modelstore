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
from modelstore.utils.log import logger

MODEL_FILE = "model.onnx"


class OnnxManager(ModelManager):

    """
    Model persistence for ONNX models:
    https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
    """

    NAME = "onnx"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["onnx", "onnxruntime"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from onnx import ModelProto

        return isinstance(kwargs.get("model"), ModelProto)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not an onnx.ModelProto!")

        return [
            partial(
                save_model,
                model=kwargs["model"],
            ),
        ]

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import onnxruntime as rt

        model_path = _model_file_path(model_path)
        with open(model_path, "rb") as lines:
            model = lines.read()
        return rt.InferenceSession(model)


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_FILE)


def save_model(tmp_dir: str, model: "onnx.ModelProto") -> str:
    file_path = _model_file_path(tmp_dir)
    logger.debug("Saving onnx model to %s", file_path)
    with open(file_path, "wb") as f:
        f.write(model.SerializeToString())
    return file_path
