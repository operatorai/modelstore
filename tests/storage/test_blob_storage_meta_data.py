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
import pytest

from modelstore.metadata import metadata
from modelstore.utils.exceptions import DomainNotFoundException

from modelstore.storage.util.paths import (
    get_domain_path,
    get_model_version_path,
)

# pylint: disable=unused-import
from tests.storage.test_blob_storage import (
    mock_meta_data,
    mock_model_file,
    mock_blob_storage,
    assert_file_contents_equals,
)

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access


def test_list_domains(mock_blob_storage):
    # Create two models in two domains
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    meta_data = mock_meta_data("domain-2", "model-1", inc_time=1)
    mock_blob_storage.set_meta_data("domain-2", "model-1", meta_data)

    # The results should be reverse time sorted
    domains = mock_blob_storage.list_domains()
    assert len(domains) == 2
    assert domains[0] == "domain-2"
    assert domains[1] == "domain-1"


def test_list_models(mock_blob_storage):
    # Create two models in one domain
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    meta_data = mock_meta_data("domain-1", "model-2", inc_time=1)
    mock_blob_storage.set_meta_data("domain-1", "model-2", meta_data)

    # List the models in domain-1; we expect two
    models = mock_blob_storage.list_models("domain-1")
    assert len(models) == 2

    # The results should be reverse time sorted
    assert models[0] == "model-2"
    assert models[1] == "model-1"


def test_set_meta_data(mock_blob_storage):
    # Set the meta data of a fake model
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Expected two files to be created
    # (1) The meta data for the 'latest' model
    domain_meta_data_path = get_domain_path(mock_blob_storage.root_prefix, "domain-1")
    assert_file_contents_equals(domain_meta_data_path, meta_data)

    # (2) The meta data for a specific model
    model_meta_data_path = get_model_version_path(
        mock_blob_storage.root_prefix,
        "domain-1",
        "model-1",
    )
    assert_file_contents_equals(model_meta_data_path, meta_data)


def test_get_meta_data(mock_blob_storage):
    # Set the meta data of a fake model
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Retrieve it back
    # pylint: disable=no-member
    get_result = mock_blob_storage.get_meta_data("domain-1", "model-1")
    result = metadata.Summary.from_dict(get_result)
    assert result == meta_data


@pytest.mark.parametrize(
    "domain,model_id",
    [(None, "model-2"), ("", "model-2"), ("domain-1", None), ("domain-1", "")],
)
def test_get_meta_data_undefined_input(mock_blob_storage, domain, model_id):
    with pytest.raises(DomainNotFoundException):
        mock_blob_storage.get_meta_data(domain, model_id)
