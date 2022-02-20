import os
from enum import Enum

from modelstore.utils.log import logger


class ReservedModelStates(Enum):

    DELETED: str = "deleted"


def is_valid_state_name(state_name: str) -> bool:
    if any(state_name == x for x in [None, ""]):
        logger.debug("state_name has invalid value: %s", state_name)
        return False
    if len(state_name) < 3:
        logger.debug("state_name is too short: %s", state_name)
        return False
    if any(state_name == x.value for x in ReservedModelStates):
        logger.debug("state_name cannot be a reserved value: %s", state_name)
        return False
    if os.path.split(state_name)[1] != state_name:
        logger.debug("state_name cannot be a path: %s", state_name)
        return False
    return True
