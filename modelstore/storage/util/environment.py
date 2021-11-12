import os
from typing import Optional


def get_value(
    arg: str, env_key: str, allow_missing: bool = False
) -> Optional[str]:
    """Modelstore storage can optionally be instantiated using
    environment variables. This function is used to decide whether to
    - pull a variable from the user's environment;
    - return the one that was passed in;
    - return None
    """
    if arg is not None:
        # arg has been passed in as non-None, so return it
        return arg
    if env_key not in os.environ and allow_missing:
        # The environment key doesn't exist for a variable that
        # is allowed to be missing, so return None
        return None
    # Return the environment variable; this will KeyError if it
    # is missing
    return os.environ[env_key]
