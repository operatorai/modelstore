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
import sys
from unittest.mock import patch

from modelstore.meta import runtime


def test_get_python_version():
    vers = sys.version_info
    expected = ".".join(str(x) for x in [vers.major, vers.minor, vers.micro])
    assert runtime.get_python_version() == expected


@patch("modelstore.meta.runtime.getpass")
def test_get_user(mock_getpass):
    mock_getpass.getuser.return_value = "username"
    assert runtime.get_user() == "username"
