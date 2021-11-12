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
