# Tutorials / Documentation

## Tutorials

To get a good appreciation of what `market_prices` offers and how it works it's recommended that the following tutorials are worked through in the order listed.

### **Introduction**

* [**quickstart.ipynb**](./tutorials/quickstart.ipynb)
    * An introduction to basic usage and a peek beyond.

### **`get`**
These tutorials covers all aspects of the Prices classes `get` method which is used to get a prices dataset.

* [**prices.ipynb**](./tutorials/prices.ipynb)
    * Create a prices object from `PricesYahoo` or `PricesCsv`.
* [**intervals.ipynb**](./tutorials/intervals.ipynb)
    * Define the time interval represented by each row of a prices table.
* [**periods.ipynb**](./tutorials/periods.ipynb)
    * Define the period covered by a prices table.
        * The Golden Rule
        * The Silver Rule
    * Define period from `start` through `end`.
    * Define period as a duration bound with `start` or `end`.
        * Durations in terms of trading times with `minutes` and `hours`.
        * Durations in terms of trading sessions with `days`.
        * Durations in calendar terms with `weeks`, `months` and `years`.
    * Multiple symbols trading on different exchanges with differing opening times and in different timezones.
        * `lead_symbol` option to determine the exchange calendar against which to evaluate the period.
    * Timezones (`tzin`)
    * `add_a_row` option to include the row prior to period start.
* [**anchor.ipynb**](./tutorials/anchor.ipynb)
    * How indices are evaluated.
        * "open" and "workback" `anchor`
        * `force` indices to exclude non-trading periods.
        * `openend` to determine the final indice when the period end is an unaligned session close.
        * Overlapping indices warning.
        * Circumstances in which indices are unable to observe session breaks.
    * Interrogating indices with `.pt` accessor methods.
* [**data_availability.ipynb**](./tutorials/data_availability.ipynb)
    * How availabiliy of data at underlying base intervals determines the period over which data is available for any specific interval.
    * `PricesIntradayUnavailableError` and `LastIndiceInaccurateError`.
    * Options when a request can only be partially fufilled; `strict`, `priority` and `composite`.
    * Composite calendars.
* [**other_get_options.ipynb**](./tutorials/other_get_options.ipynb)
    * Post-processing options, including `tzout`, `fill`, `include`, `exclude`, `side`, `close_only` and `lose_single_symbol`.

### **Other Prices methods**

These tutorials cover other methods of Prices classes.

* [**specific_query_methods**](./tutorials/specific_query_methods.ipynb)
    * Covers the following methods that return a single-row `DataFrame` giving prices for a specific session, time or period.
        * `session_prices`
        * `close_at`
        * `price_at`
        * `price_range`
* [**other_prices_methods**](./tutorials/other_prices_methods.ipynb)
    * Covers the following methods:
        * `request_all_prices` to request all available prices from the data provider.
        * `prices_for_symbols` to return a Prices instance for a subset of symbols.

### **Price table accessor**

The `.pt` accessor opens the door to a wealth of functionality to interrogate and operate on `DataFrame` returned by `get`.

* [**pt_accessor.ipynb**](./tutorials/pt_accessor.ipynb)
    * Examples of all the properties and methods offered by the price table accessor classes, from reindexing and downsampling to querying and tidying.

## Other documentation

### Parsing
[parsing.md](./public/parsing.md) offers a explanation of how the [`valimp`](https://github.com/maread99/valimp) library is used to validate and otherwise parse inputs to public functions.

### Typing
[typing.md](./public/typing.md) covers:
* Type annotation.
* Custom types of the `mptypes.py` module.

### Method documentation.

[method_doc.md](./public/method_doc.md) explains what to expect from the documentation of public methods.

### Issues
See the [Issues](https://github.com/maread99/market_prices/issues) page for outstanding issues.

### Developer

Contributions to `market_prices` are certainly welcome. If you are looking to contribute, please do have a look through the following developer docs. 

[serving_data.md](./developers/serving_data.md) :
* The internals around serving price data.
* Data providers.

[typing_doc.md](./developers/typing_doc.md) :
* typing specifications.
* documentation specifications.

[testing.md](./developers/testing.md) :
* Tests for `PricesBase`.
* Data from yahoo API is unreliable for test suite.

[releases.md](./developers/releases.md) :
* Versioning.
* Draft release notes.
* Release workflow.

[other_internals.md](./developers/other_internals.md) :
* Considerations for intervals that are not a factor of (sub)session length.
* `daterange.GetterIntraday.daterange_tight`