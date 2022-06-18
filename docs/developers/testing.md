# Testing

## Tests for `PricesBase`
The tests for the `PricesBase` class are defined on different modules according to whether or not the tested method requests price data.

Tests for methods that do not request price data are defined on `test_base.py`.

Tests for methods that request price data are defined on `test_base_prices.py` where tests get data from locally stored resources.

### **`PricesBaseTst`**

Tests that request data should use fixtures that return an instance of the `PricesBaseTst` class, for example `prices_us`, `prices_us_lon_hg`, `prices_with_break` etc.

`PricesBaseTst` concretes `PricesBase` with the same base intervals and limits as `PricesYahoo` although requests prices from a stored resource. Any test that uses a `PricesBaseTst` fixture will have 'now' mocked to the minute that the locally stored price data was originally requested.

New `PricesBaseTst` fixtures can be defined. The associated price data can be stored to resources by simply passing a corresponding instance of PricesYahoo (for which price data has not been requested) to `tests.utils.save_resource_pbt`. The method requests and stores all available price data together with the minute the data was requested.

Tests should be designed to work if the associated price data were to be changed to data taken at a different time or for different equities (albeit equities listed on the same exchanges).
* Tests should not define static inputs unless they can be reasonably considered durable (for example, `start` and `end` parameters when requesting daily data).

## Tests for `PricesYahoo`

### Data from yahoo API is unreliable for the test suite
The yahoo API starts to fail to return prices, seemingly at all intervals, for certain sessions if it receieves a high frequency of requests from a single IP.

The unavailability is short lived and after, say, 5 minutes or so the API reverts to providing data for these sessions. The issue appears to be at an IP address level - change the IP address that the request is sent from and prices are available for these sessions, change it back and they continue to be unavailable until that short period of time has passed.

Execute the tests individually and this issue does not arise (unless executed immediately following execution of the test suite, i.e. within the period during which prices for these sessions are not available).

All seems to suggest that the API is implemented to change behaviour when the frequency of requests from a specific IP address crosses a threshold, and that change (perhaps to serving requests from an incomplete database?) results in missing data for these sessions.

(NOTE: when data is not available for all indices of a session PricesYahoo fills the missing data with data from the prior or following session and a `PricesMissingWarning` is raised.)

The sessions for which data becomes unavailable are USUALLY the same ones. These are listed in `_flakylist` towards the top of `test_yahoo.py`. **Tests that request data from the API should be written so as to avoid requesting data for a period that starts or ends on any of these flakylisted sessions**. To this end the `test_yahoo.py` module provides the following methods to evaluate sessions that are not flakylisted:
* `get_valid_sessions`
* `get_valid_conforming_sessions`
