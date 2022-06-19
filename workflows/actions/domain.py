from modelstore import ModelStore
from modelstore.utils.exceptions import DomainNotFoundException


def assert_get_missing_domain_raises(modelstore: ModelStore):
    """ Calling get_domain() with an unknown domain raises an exception """
    try:
        _ = modelstore.get_domain("missing-domain")
    except DomainNotFoundException:
        print("✅  Raises a DomainNotFoundException if it can't find a domain")
