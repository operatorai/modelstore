from datetime import datetime


def sort_by_version(meta_data: dict):
    if "code" in meta_data:
        return datetime.strptime(
            meta_data["code"]["created"], "%Y/%m/%d/%H:%M:%S"
        )
    if "meta" in meta_data:
        return datetime.strptime(
            meta_data["meta"]["created"], "%Y/%m/%d/%H:%M:%S"
        )
    return 1


def sorted_by_created(versions: list):
    return sorted(
        versions,
        key=sort_by_version,
        reverse=True,
    )
