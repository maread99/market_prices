"""Market Prices."""

from .prices.yahoo import PricesYahoo
from .prices.csv import PricesCsv
from .utils.calendar_utils import get_exchange_info

__all__ = [PricesYahoo, PricesCsv, get_exchange_info]

__copyright__ = "Copyright (c) 2022 Marcus Read"


# Resolve version
__version__ = None

from importlib.metadata import version

try:
    # get version from installed package
    __version__ = version("market_prices")
except ImportError:
    pass

if __version__ is None:
    try:
        # if package not installed, get version as set when package built
        from ._version import version
    except Exception:
        # If package not installed and not built, leave __version__ as None
        pass
    else:
        __version__ = version

del version
