import uuid
import pytest

from modelstore.ids import model_ids


def test_new() -> str:
    model_id = model_ids.new()
    assert isinstance(model_id, str)
    assert len(model_id) == len(str(uuid.uuid4()))


@pytest.mark.parametrize(
    "model_id,is_valid",
    [
        ("a-model-id", True),
        ("a model id", False),
    ],
)
def test_validate_no_spaces(model_id: str, is_valid: bool):
    assert model_ids.validate(model_id) == is_valid


def test_validate_no_special_characters():
    for character in model_ids._RESERVED_CHARACTERS:
        model_id = f"an-invalid-{character}-model-id"
        assert not model_ids.validate(model_id)
