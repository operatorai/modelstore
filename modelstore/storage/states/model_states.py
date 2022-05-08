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
import os
from enum import Enum


class ReservedModelStates(Enum):

    """ReservedModelStates are states that are
    created & managed by the modelstore library,
    so users cannot create a state with this name
    """

    DELETED: str = "modelstore-deleted"


def is_valid_state_name(state_name: str) -> bool:
    """Whether a state name is valid for usage"""
    if any(state_name == x for x in [None, ""]):
        return False
    if len(state_name) < 3:
        return False
    if os.path.split(state_name)[1] != state_name:
        return False
    if is_reserved_state(state_name):
        return False
    return True


def is_reserved_state(state_name: str) -> bool:
    """Whether a state name is a reserved state"""
    reserved_state_names = set(x.value for x in ReservedModelStates)
    return state_name in reserved_state_names
