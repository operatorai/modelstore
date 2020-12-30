#    Copyright 2020 Neal Lathia
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
import os
import subprocess

import git
from modelstore.meta import revision

# pylint: disable=protected-access
# pylint: disable=bare-except


def test_repo_name():
    repo = git.Repo(search_parent_directories=True)
    repo_name = revision._repo_name(repo)
    if repo_name == "":
        # Not a git repo
        return
    assert repo_name == "modelstore"


def test_fail_gracefully():
    # Assumes that there is no git repo at /
    current_wd = os.getcwd()
    os.chdir("/")
    assert revision.git_meta() is None
    os.chdir(current_wd)


def test_git_meta():
    try:
        res = subprocess.check_output("git log . | head -n 1", shell=True)
        exp = res.decode("utf-8").strip().split(" ")[1]
    except:
        # Repo is not a git repo
        return

    meta = revision.git_meta()
    assert meta is not None
    assert meta["repository"] == "modelstore"
    if meta["local_changes"] is False:
        assert meta["sha"] == exp
