import git

# pylint: disable=broad-except


def _repo_name(repo: git.Repo) -> str:
    try:
        repo_url = repo.remotes.origin.url
        return repo_url.split(".git")[0].split("/")[-1]
    except Exception:
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
    except Exception:
        return None
