import importlib
import sys

import pkg_resources
from modelstore.models.common import save_json
from modelstore.utils.log import logger

# pylint: disable=broad-except
_PYTHON_INFO_FILE = "python-info.json"
_MODEL_TYPE_FILE = "model-info.json"


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
    return _get_version(modname) is not None


def save_dependencies(tmp_dir: str, deps: list) -> str:
    deps_info = get_dependency_versions(deps)
    deps_info = {k: v for k, v in deps_info.items() if v is not None}
    return save_json(tmp_dir, _PYTHON_INFO_FILE, deps_info)


def save_model_info(tmp_dir, model_info: dict) -> str:
    return save_json(tmp_dir, _MODEL_TYPE_FILE, model_info)
