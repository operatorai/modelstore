from datetime import datetime


def sort_by_version(meta_data: dict, version: str):
    if not version.startswith("0.0.1"):
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
