"""Market Prices."""

from .prices.yahoo import PricesYahoo
from .utils.calendar_utils import get_exchange_info

__all__ = [
    PricesYahoo,
    get_exchange_info
]

__version__ = "0.8.1"

__title__ = "market_prices"
__description__ = "Meaningful OHLCV datasets"
__url__ = "https://github.com/maread99/market_prices"
__doc__ = __description__ + " <" + __url__ + ">"

__author__ = "Marcus Read"
__email__ = "marcusaread@gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2022 Marcus Read"
