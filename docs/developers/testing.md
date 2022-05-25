# Testing

## Data from yahoo API is unreliable for test suite

At some point during execution of the test suite the yahoo API starts to fail to return prices, seemingly at at all intervals, for certain sessions.

The unavailability is short lived and after, say, 5 minutes or so the API reverts to providing data for these sessions. The issue appears to be at an IP address level - change the IP address that the request is sent from and prices are available for these sessions, change it back and they continue to be unavailable until that short period of time has passed.

Execute the tests individually and this issue does not arise (unless executed immediately following execution of the test suite, i.e. within the period during which prices for these sessions are not available).

All seems to suggest that, maybe, the API is implemented to, knowingly or otherwise, temporarily withhold data for these sessions when the frequency of requests from a specific IP address crosses a threshold(?).

(NOTE: when data is not available for all indices of a session the missing data is filled with data from the prior or following session and a `PricesMissingWarning` is raised.)

The sessions for which data becomes unavailable are always the same ones. They are listed in `_flakylist` towards the top of `test_yahoo.py`. **Tests should be written so as to avoid requesting data for a period that starts or ends on any of these flakylisted sessions**. To this end the `test_yahoo.py` module provides the following methods to evaluate sessions that are not flakylisted:
* `get_valid_sessions`
* `get_valid_conforming_sessions`