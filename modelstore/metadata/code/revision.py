#    Copyright 2022 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
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
    except Exception as exc:
        logger.debug("error extracting git repo: %s", str(exc))
        return ""


def git_meta() -> dict:
    """Returns meta data about the current git repo"""
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
    except Exception as exc:
        logger.debug("error generating git meta-data: %s", str(exc))
        return None
