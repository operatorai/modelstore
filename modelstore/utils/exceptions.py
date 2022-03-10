#    Copyright 2022 Neal Lathia
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


class FilePullFailedException(Exception):
    def __init__(self, base_exception: Exception):
        super().__init__()
        self.base_exception = base_exception


class ModelDeletedException(Exception):
    def __init__(self, domain: str, model_id: str):
        super().__init__(f"model='{model_id}' has been deleted from domain='{domain}'")


class ModelNotFoundException(Exception):
    def __init__(self, domain: str, model_id: str):
        super().__init__(f"model='{model_id}' does not exist in domain='{domain}'.")


class DomainNotFoundException(Exception):
    def __init__(self, domain: str):
        super().__init__(f"The domain='{domain}' does not exist.")

class ModelExistsException(Exception):
    def __init__(self, domain: str, model_id: str):
        super().__init__(f"model='{model_id}' already exists in this domain={domain}.")
