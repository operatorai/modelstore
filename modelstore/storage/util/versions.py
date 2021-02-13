from datetime import datetime


def sort_by_version(meta_data: dict):
    created = datetime.strptime(
        meta_data["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
    )
    return created


def sorted_by_created(versions: list):
    return sorted(
        versions,
        key=sort_by_version,
        reverse=True,
    )
