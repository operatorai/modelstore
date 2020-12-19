import modelstore
from modelstore.meta import metadata

# pylint: disable=protected-access


def test_generate_for_model():
    exp = {
        "domain": "test-model",
        "model_id": "test-model-id",
        "model_type": {"library": "model-library", "name": "model-class"},
    }
    res = metadata.generate_for_model(
        "test-model-id",
        {"library": "model-library", "name": "model-class"},
        "test-model",
    )
    assert exp == res


def test_generate_for_code():
    deps_list = ["pytest"]
    res = metadata.generate_for_code(deps_list)
    assert res["runtime"].startswith("python")
    assert all(k in res for k in ["user", "created", "dependencies", "git"])
    assert res["dependencies"]["pytest"] == "6.2.1"
    assert res["git"]["repository"] == "modelstore"


def test_generate():
    res = metadata.generate(model_meta=None, storage_meta=None, code_meta=None)
    assert all(k in res for k in ["model", "storage", "code", "modelstore"])
    assert res["modelstore"] == modelstore.__version__


def test_remove_nones():
    exp = {"a": "value-a"}
    res = metadata._remove_nones({"a": "value-a", "b": None})
    assert exp == res
