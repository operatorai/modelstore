from datetime import datetime


def sort_by_version(meta_data: dict):
    version = meta_data["modelstore"]
    if version in ["0.0.4", "0.0.5"]:
        return datetime.strptime(
            meta_data["code"]["created"], "%Y/%m/%d/%H:%M:%S"
        )
    # Earlier versions of modelstore had a different meta-data structure
    return datetime.strptime(meta_data["meta"]["created"], "%Y/%m/%d/%H:%M:%S")


def sorted_by_created(versions: list):
    return sorted(
        versions,
        key=sort_by_version,
        reverse=True,
    )
