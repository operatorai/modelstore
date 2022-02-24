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
import os
from pathlib import Path

import pytest
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
    get_archive_path,
)


@pytest.fixture
def mock_blob_storage(tmp_path):
    return FileSystemStorage(str(tmp_path))


@pytest.fixture
def mock_model_file(tmp_path):
    model_file = os.path.join(tmp_path, "test-file.txt")
    Path(model_file).touch()
    return model_file


def test_upload(mock_blob_storage, mock_model_file):
    model_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_archive_path(
            mock_blob_storage.root_prefix,
            "test-domain",
            mock_model_file,
        ),
    )
    rsp = mock_blob_storage.upload("test-domain", mock_model_file)
    assert rsp["type"] == "file_system"
    assert rsp["path"] == model_path
    assert os.path.exists(model_path)


def test_download_latest():
    pass


def test_download():
    pass


def test_delete_model():
    # prefix = self._get_storage_location(meta_data)
    # if not skip_prompt:
    #     message = f"Delete model from domain={domain} with model_id={model_id}?"
    #     if not click.confirm(message):
    #         logger.info("Aborting; not deleting model")

    # # Delete the artifact itself
    # self._remove(prefix)

    # # Set the model as deleted in the meta data by unsetting it from
    # # all custom states, setting it to a reserved state, and then deleting
    # # the main meta-data file
    # for state_name in self.list_model_states():
    #     self.unset_model_state(domain, model_id, state_name)

    # self.set_model_state(domain, model_id, ReservedModelStates.DELETED.value)

    # logger.debug("Deleting meta-data for %s=%s", domain, model_id)
    # remote_path = self._get_metadata_path(domain, model_id)
    # self._remove(remote_path)
    pass
