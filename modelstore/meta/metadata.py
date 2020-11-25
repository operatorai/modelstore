from datetime import datetime

import modelstore
from modelstore.meta import dependencies, revision, runtime


def generate(
    model_type: str,
    model_id: str,
    domain: str,
    location: dict,
    deps_list: dict,
    model_params: dict,
):
    versioned_deps = dependencies.get_dependency_versions(deps_list)
    meta_data = {
        "model": {
            "domain": domain,
            "model_id": model_id,
            "type": model_type,
            "params": _remove_nones(model_params),
        },
        "storage": location,
        "meta": {
            "runtime": f"python:{runtime.get_python_version()}",
            "user": runtime.get_user(),
            "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
            "dependencies": _remove_nones(versioned_deps),
        },
        "modelstore": modelstore.__version__,
    }
    git_meta = revision.git_meta()
    if git_meta is not None:
        meta_data["meta"]["git"] = git_meta
    return meta_data


def _remove_nones(values) -> dict:
    return {k: v for k, v in values.items() if v is not None}
