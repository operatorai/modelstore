from datetime import datetime

import modelstore
from modelstore.meta import dependencies, revision, runtime


def generate(
    model_type: str,
    model_id: str,
    domain: str,
    location: dict,
    deps_list: dict,
):
    deps = {
        k: v
        for k, v in dependencies.get_dependency_versions(deps_list).items()
        if v is not None
    }
    meta_data = {
        "model": {"domain": domain, "model_id": model_id, "type": model_type,},
        "storage": location,
        "meta": {
            "runtime": f"python:{runtime.get_python_version()}",
            "user": runtime.get_user(),
            "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
            "dependencies": deps,
        },
        "modelstore": modelstore.__version__,
    }
    git_meta = revision.git_meta()
    if git_meta is not None:
        meta_data["meta"]["git"] = git_meta
    return meta_data
