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
from datetime import datetime

import pytest
from modelstore.storage.states.model_states import (
    ReservedModelStates,
    is_valid_state_name,
)

# pylint: disable=protected-access


@pytest.mark.parametrize(
    "state_name,allow_reserved_states,is_valid",
    [
        (None, False, False),
        ("", False, False),
        ("a", False, False),
        ("path/to/place", False, False),
        ("other", False, True),
        (ReservedModelStates.DELETED.value, False, False),
        (ReservedModelStates.DELETED.value, True, True),
    ],
)
def test_is_valid_state_name(state_name, allow_reserved_states, is_valid):
    assert is_valid_state_name(state_name, allow_reserved_states) == is_valid
