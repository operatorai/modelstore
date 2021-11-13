from modelstore.utils.log import logger

try:
    import git

    GIT_EXISTS = True
except ImportError:
    logger.info("Warning: no git installation. Will not collect git meta data.")
    GIT_EXISTS = False


def _repo_name(repo: "git.Repo") -> str:
    if not GIT_EXISTS:
        return ""
    # pylint: disable=broad-except
    try:
        repo_url = repo.remotes.origin.url
        return repo_url.split(".git")[0].split("/")[-1]
    except Exception:
        return ""


def git_meta() -> dict:
    if not GIT_EXISTS:
        return {}
    # pylint: disable=broad-except
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "repository": _repo_name(repo),
            "sha": repo.head.object.hexsha,
            "local_changes": repo.is_dirty(),
            "branch": repo.active_branch.name,
        }
    except Exception:
        return None
