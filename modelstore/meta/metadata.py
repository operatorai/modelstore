from datetime import datetime

import modelstore
from modelstore.meta import dependencies, revision, runtime


def generate_for_model(
    domain: str,
    model_id: str,
    model_info: dict,
    model_params: dict = None,
    model_data: dict = None,
):
    metadata = {
        "domain": domain,
        "model_id": model_id,
        "model_type": _remove_nones(model_info),
    }
    if model_params is not None:
        metadata["parameters"] = _remove_nones(model_params)
    if model_data is not None:
        metadata["data"] = _remove_nones(model_data)
    return metadata


def generate_for_code(deps_list: dict):
    versioned_deps = dependencies.get_dependency_versions(deps_list)
    metadata = {
        "runtime": f"python:{runtime.get_python_version()}",
        "user": runtime.get_user(),
        "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
        "dependencies": _remove_nones(versioned_deps),
    }
    git_meta = revision.git_meta()
    if git_meta is not None:
        metadata["git"] = git_meta
    return metadata


def generate(
    model_meta: dict,
    storage_meta: dict,
    code_meta: dict,
):
    return {
        "model": model_meta,
        "storage": storage_meta,
        "code": code_meta,
        "modelstore": modelstore.__version__,
    }


def _remove_nones(values) -> dict:
    return {k: v for k, v in values.items() if v is not None}
