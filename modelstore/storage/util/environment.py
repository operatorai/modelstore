import os


def get_value(arg: str, env_key: str, allow_missing: bool = False) -> str:
    if arg is not None:
        return arg
    if env_key not in os.environ and allow_missing:
        return None
    return os.environ[env_key]
