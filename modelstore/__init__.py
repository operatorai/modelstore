from pkg_resources import get_distribution

from modelstore.model_store import ModelStore

__version__ = get_distribution("modelstore").version
