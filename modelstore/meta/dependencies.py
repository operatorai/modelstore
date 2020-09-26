import importlib
import json
import sys
import tarfile

from modelstore.models.common import save_json
from modelstore.utils.log import logger

# pylint: disable=broad-except
_PYTHON_INFO_FILE = "python-info.json"
_MODEL_TYPE_FILE = "model-info.json"


def _get_version(modname: str) -> str:
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        return mod.__version__
    except ImportError:
        logger.debug("%s is not installed.", modname)
        return None
    except Exception:
        logger.error("Error importing %s.", modname)
        return None


def _get_dependency_versions(modnames: list) -> dict:
    """
    This function re-implements the functionality of the 'private' `_get_deps_info()`
    function in sklearn:

    https://github.com/scikit-learn/scikit-learn/blob/a0a76fcfbe1e19c8f9e422b41260471f05d8f560/sklearn/utils/_show_versions.py#L35
    """  # noqa
    return {modname: _get_version(modname) for modname in modnames}


def module_exists(modname: str) -> bool:
    return _get_version(modname) is not None


def save_dependencies(tmp_dir: str, deps: list) -> str:
    deps_info = _get_dependency_versions(deps)
    deps_info = {k: v for k, v in deps_info.items() if v is not None}
    return save_json(tmp_dir, _PYTHON_INFO_FILE, deps_info)


def save_model_info(tmp_dir, model_name: str, model_info: dict) -> str:
    model_info["name"] = model_name
    return save_json(tmp_dir, _MODEL_TYPE_FILE, model_info)


def extract_dependencies(archive_path: str) -> dict:
    return _extract_json(archive_path, _PYTHON_INFO_FILE)


def extract_model_type(archive_path: str) -> dict:
    return _extract_json(archive_path, _MODEL_TYPE_FILE)


def _extract_json(archive_path: str, file_name: str) -> dict:
    if not archive_path.endswith(".tar.gz"):
        return {}
    with tarfile.open(archive_path, "r:gz") as tar:
        deps_info = tar.extractfile(file_name)
        if deps_info is not None:
            deps = deps_info.read()
            return json.loads(deps)
    return {}
