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
import getpass
import sys


def get_python_version() -> str:
    """Returns the current python version"""
    vers = sys.version_info
    version = ".".join(str(x) for x in [vers.major, vers.minor, vers.micro])
    return f"python:{version}"


def get_user() -> str:
    """Returns the current user"""
    return getpass.getuser()
