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
import uuid
import re

from modelstore.utils.log import logger

# Avoids characters that can't be used on Windows
# https://github.com/operatorai/modelstore/issues/140
_RESERVED_CHARACTERS = [
    "<",
    ">",
    ":",
    '"',
    "/",
    "\\",
    "|",
    "?",
    "*",
    "#",
    "^",
    "`",
    "%",
    "~",
    "{",
    "}",
    "[",
    "]",
]


def new() -> str:
    """Currently returns a uuid4 ID; in the future
    we can support different ID types & lengths
    """
    return str(uuid.uuid4())


def validate(model_id: str) -> bool:
    """Model ids need to comply with various
    conditions so that we can use their ID when storing
    models into the different storage layers
    """
    if re.search(" +", model_id) is not None:
        logger.info("Model id contains one or more spaces")
        return False

    matches = [x for x in _RESERVED_CHARACTERS if x in model_id]
    if len(matches) == 0:
        return True
    logger.info("Model id contains reserved characters: %s", matches)
    return False
