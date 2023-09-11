
from .statstable import StatsTable, MetaTable
from .plot import statsplot, vulcanoplot
from .stats import calculate_stats
from .dimred import DimRed

from . import _version
__version__ = _version.get_versions()['version']
