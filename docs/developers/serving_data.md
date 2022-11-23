# Serving Data

## Serving Price Data
For an explanation of how price data is served, including the definition of base intervals, see the data_availability.ipynb tutorial and the 'Serving Price Data' section of `prices.base.PricesBase.__doc__`.

## Data providers
The vast bulk of `market_prices` is data-source agnostic. Price sources can be added by subclassing `PricesBase` and concreting the abstract methods and attributes as described by `PricesBase.__doc__`.

Currently (Nov 2022) the yahoo API (via [yahooquery](https://yahooquery.dpguthrie.com/) is the only price source supported, implemented as the `PricesYahoo` class. This serves as the default 'out-the-box' data source.

On incorporating a further data provider it will be necessary to review `PricesYahoo` for potential common functionality that can be refactored back to either `PricesBase` or a common intermediary subclass. Also, it will be necessary to review `PricesBase` class for functionality that was assumed common although which, given the perspective of other providers, may prove unique to Yahoo or more appropriate to include to an intermediary class.

### What does a provider have to provide? 
`market_prices` requires that the data source provides OHLCV data (open, high, low, close, volume).

A source will provide OHLCV prices at one or more interval. For example [yahooquery](https://yahooquery.dpguthrie.com/) provides for requesting historic prices at one of the following intervals:
'1m', '2m', '5m', '15m', '30m', '60m', '90m'  (in minutes)
'1h'  (in hours)
'1d', '5d'  (in days)
'1wk'  (in weeks)
'1mo', '3mo'  (in months)

See `Serving Price Data` section of `PricesBase.__doc__` for notes on how only a subset of these available intervals are used as base intervals. A subclass's base intervals are defined by concreting the BaseInterval class attribute as an `intervals._BaseInterval` enumerator with members as base intervals (see `PricesYahoo` implementation for an example).

The `BASE_LIMITS` class attribute defines the default earliest dates / datetimes for which data for these base intervals is available. The `base_limits` property then offers availability limits as specifically applicable to the instance's symbols.
