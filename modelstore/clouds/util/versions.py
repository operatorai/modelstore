from datetime import datetime


def sort_by_version(meta_data: dict, version: str):
    if not any(version.startswith(x) for x in ["0.0.1", "0.0.2", "0.0.3"]):
        raise NotImplementedError(f"Sort order for version {version}")

    created = datetime.strptime(
        meta_data["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    return created


def sorted_by_created(versions: list):
    return sorted(
        versions,
        key=lambda x: sort_by_version(x, x["modelstore"]),
        reverse=True,
    )
