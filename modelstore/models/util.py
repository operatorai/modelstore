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


def convert_tensors(model_params):
    import torch

    if isinstance(model_params, torch.Tensor):
        if hasattr(model_params, "detach"):
            model_params = model_params.detach()
        return model_params.cpu().numpy()
    if isinstance(model_params, list):
        return [convert_tensors(c) for c in model_params]
    if isinstance(model_params, dict):
        return {k: convert_tensors(v) for k, v in model_params.items()}

    return model_params


def convert_numpy(model_params):
    import numpy as np

    if isinstance(model_params, np.ndarray):
        return model_params.tolist()

    if isinstance(model_params, list):
        return [convert_numpy(c) for c in model_params]
    if isinstance(model_params, dict):
        return {k: convert_numpy(v) for k, v in model_params.items()}
    return model_params
