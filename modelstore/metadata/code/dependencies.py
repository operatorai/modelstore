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
import importlib
import sys

import pkg_resources
from modelstore.utils.log import logger

# pylint: disable=broad-except


def _get_version(modname: str) -> str:
    try:
        if modname == "pickle":
            # pylint: disable=import-outside-toplevel
            import pickle

            return pickle.format_version
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            logger.debug("Trying to import: %s", modname)
            mod = importlib.import_module(modname)
        return mod.__version__
    except AttributeError:
        try:
            #  Annoy does not have a __version__
            return pkg_resources.get_distribution(modname).version
        except Exception:
            logger.debug("Unable to get %s's version", modname)
            return None
    except ImportError:
        logger.debug("%s is not installed.", modname)
        return None
    except Exception:
        logger.error("Error importing: %s.", modname)
        return None


def get_dependency_versions(modnames: list) -> dict:
    """
    This function re-implements the functionality of the 'private' `_get_deps_info()`
    function in sklearn:

    https://github.com/scikit-learn/scikit-learn/blob/a0a76fcfbe1e19c8f9e422b41260471f05d8f560/sklearn/utils/_show_versions.py#L35
    """  # noqa
    return {modname: _get_version(modname) for modname in modnames}


def module_exists(modname: str) -> bool:
    """Returns True if a module has been installed"""
    return _get_version(modname) is not None
