from pkg_resources import DistributionNotFound, get_distribution

# pylint: disable=unused-import
from modelstore.model_store import ModelStore

try:
    __version__ = get_distribution("modelstore").version
except DistributionNotFound:
    __version__ = "unavailable"
