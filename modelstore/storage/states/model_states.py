import os
from enum import Enum

from modelstore.utils.log import logger


class ReservedModelStates(Enum):

    DELETED: str = "modelstore-deleted"


def is_valid_state_name(state_name: str) -> bool:
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
    reserved_state_names = set(x.value for x in ReservedModelStates)
    return state_name in reserved_state_names
