import git
from git.exc import InvalidGitRepositoryError

from modelstore.utils.log import logger

# pylint: disable=broad-except


def _repo_name(repo: git.Repo) -> str:
    try:
        repo_url = repo.remotes.origin.url
        return repo_url.split(".git")[0].split("/")[-1]
    except Exception as ex:
        logger.error("Error retrieving repo name %s", str(ex))
        return ""


def git_meta() -> dict:
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "repository": _repo_name(repo),
            "sha": repo.head.object.hexsha,
            "local_changes": repo.is_dirty(),
            "branch": repo.active_branch.name,
        }
    except Exception as ex:
        logger.error("Error retrieving repo details %s", str(ex))
        return None
