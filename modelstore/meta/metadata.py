import uuid
from datetime import datetime

import modelstore
from modelstore.meta import dependencies, revision, runtime


def generate(domain: str):
    model_id = str(uuid.uuid4())
    # Warning! Mac OS translates ":" in paths to "/"
    upload_time = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    location = self.storage.upload(domain, upload_time, archive_path)
    meta_data = {
        "model": {
            "domain": domain,
            "model_id": model_id,
            "type": dependencies.extract_model_type(archive_path),
        },
        "storage": {"name": self.storage.get_name(), "location": location,},
        "meta": {
            "runtime": f"python:{runtime.get_python_version()}",
            "user": runtime.get_user(),
            "created": upload_time,
            "dependencies": dependencies.extract_dependencies(archive_path),
        },
        "modelstore": modelstore.__version__,
    }

    git_meta = revision.git_meta()
    if git_meta is not None:
        meta_data["meta"]["git"] = git_meta

    self.storage.set_meta_data(domain, model_id, meta_data)
    return meta_data
