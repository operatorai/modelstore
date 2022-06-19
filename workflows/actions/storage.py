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
from typing import Callable, List
from modelstore import ModelStore
from modelstore.utils import exceptions


def assert_get_missing_domain_raises(model_store: ModelStore, _: str):
    """ Calling get_domain() with an unknown domain raises an exception """
    try:
        _ = model_store.get_domain("missing-domain")
    except exceptions.DomainNotFoundException:
        print("✅  Raises a DomainNotFoundException if it can't find a domain")
        return
    raise AssertionError("failed to raise DomainNotFoundException")


def assert_get_missing_model_raises(model_store: ModelStore, domain: str):
    """ Calling get_model_info() for a missing model raise an exception """
    try:
        _ = model_store.get_model_info(domain, "missing-model")
    except exceptions.ModelNotFoundException:
        print("✅  Modelstore raises a ModelNotFoundException if it can't find a model")
        return
    raise AssertionError("failed to raise ModelNotFoundException")


def get_actions() -> List[Callable]:
    """ Returns the set of actions that can be run on a model_store """
    return [
        assert_get_missing_domain_raises,
        assert_get_missing_model_raises,
    ]
