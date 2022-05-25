"""Tests for market_prices.prices.yahoo module."""

from __future__ import annotations

from collections import abc
import datetime
import functools
import itertools
import inspect
import typing
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pydantic
import pytest
import pytz
import yahooquery as yq

import market_prices.prices.yahoo as m
from market_prices.prices.base import PricesBase
from market_prices import data, daterange, helpers, intervals, mptypes, errors, pt
from market_prices.intervals import TDInterval, DOInterval
from market_prices.mptypes import Anchor, OpenEnd, Priority
from market_prices.support import tutorial_helpers as th
from market_prices.utils import calendar_utils as calutils


# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, no-self-use, unused-argument, invalid-name
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway.
#   invalid-name: names in tests not expected to strictly conform with snake_case.

# Any flake8 disabled violations handled via per-file-ignores on .flake8

# pylint: disable=too-many-lines

UTC = pytz.UTC

# NOTE See ../docs/developers/testing.md...
# ...sessions that yahoo temporarily fails to return prices for if (seemingly)
# send a high frequency of requests for prices from the same IP address.
_blacklist = (
    pd.Timestamp("2022-05-24"),
    pd.Timestamp("2022-05-23"),
    pd.Timestamp("2022-05-10"),
    pd.Timestamp("2022-04-27"),
    pd.Timestamp("2022-04-22"),
    pd.Timestamp("2022-04-14"),
    pd.Timestamp("2022-04-01"),
    pd.Timestamp("2022-03-29"),
    pd.Timestamp("2022-03-28"),
    pd.Timestamp("2022-02-15"),
    pd.Timestamp("2022-02-07"),
    pd.Timestamp("2022-01-24"),
    pd.Timestamp("2022-01-14"),
    pd.Timestamp("2022-01-13"),
    pd.Timestamp("2022-01-12"),
    pd.Timestamp("2022-01-11"),
    pd.Timestamp("2021-10-18"),
    pd.Timestamp("2020-06-25"),
    pd.Timestamp("2020-05-07"),
)


def current_session_in_blacklist(calendars: list[xcals.ExchangeCalendar]) -> bool:
    """Query if current session is in the blacklist.

    Will return True if most recent session of any of `calendars` is
    blacklisted.

    Parameters
    ----------
    calendars
        Calendars against which to evaluate current session.
    """
    for cal in calendars:
        # TODO xcals 4.0 lose wrapper
        today = helpers.to_tz_naive(cal.minute_to_session(helpers.now(), "previous"))
        if today in _blacklist:
            return True
    return False


class skip_if_fails_and_today_blacklisted:
    """Decorator to skip test if fails due to today being blacklisted.

    Skips test if test raises errors.PricesUnavailableFromSourceError and
    today is in the blacklist.

    Parameters
    ----------
    cal_names
        Names of calendars against which to evaluate 'today'. Test will be
        skipped if a defined exception is raised and the blacklist includes
        'today' as evaluated against any of these calendars.

    exceptions
        Additional exception types. In addition to
        `errors.PricesUnavailableFromSourceError` test will also be skipped
        if exception of any of these types is raised and 'today' is
        blacklisted.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        cal_names: list[str],
        exceptions: list[type[Exception]] | None = None,
    ):
        self.cals = [xcals.get_calendar(name) for name in cal_names]

        permitted_exceptions = [errors.PricesUnavailableFromSourceError]
        if exceptions is not None:
            permitted_exceptions += exceptions
        self.permitted_exceptions = tuple(permitted_exceptions)

    def __call__(self, f) -> abc.Callable:
        @functools.wraps(f)
        def wrapped_test(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except self.permitted_exceptions:
                if current_session_in_blacklist(self.cals):
                    pytest.skip(f"Skipping {f.__name__}: today in blacklist.")
                raise

        return wrapped_test


class skip_if_prices_unavailable_for_blacklisted_session:
    """Decorator to skip test if fails due to unavailable prices.

    Skips test if raises `errors.PricesUnavailableFromSourceError` for a
    period bounded on either side by a minute or date that corresponds with
    a _blacklisted session.

    Parameters
    ----------
    cal_names
        Names of calendars against which to evaluate sessions corresponding
        with period bounds.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, cal_names: list[str]):
        cals = [xcals.get_calendar(name) for name in cal_names]
        self.cc = calutils.CompositeCalendar(cals)

    def _black_sessions(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[pd.Timestamp]:
        rtrn = []
        for bound in (start, end):
            if helpers.is_date(bound):
                if bound in _blacklist:
                    rtrn.append(bound)
            elif self.cc.minute_to_sessions(bound).isin(_blacklist).any():
                rtrn.append(bound)
        return rtrn

    def __call__(self, f) -> abc.Callable:
        @functools.wraps(f)
        def wrapped_test(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except errors.PricesUnavailableFromSourceError as err:
                sessions = self._black_sessions(err.params["start"], err.params["end"])
                if sessions:
                    pytest.skip(
                        f"Skipping {f.__name__}: prices unavailable for period bound"
                        f" with blacklisted session(s) {sessions}."
                    )
                raise

        return wrapped_test


# NOTE: Leave commented out. Uncomment to test decorator locally.
# @skip_if_prices_unavailable_for_blacklisted_session(["XLON"])
# def test_skip_if_prices_unavailable_for_blacklisted_session_decorator():
#     xlon = xcals.get_calendar("XLON", side="left")
#     params = {
#         'interval': '5m',
#         'start': xlon.session_open(_blacklist[2]),
#         'end': xlon.session_close(_blacklist[1]), # helpers.now() to test single bound
#     }
#     raise errors.PricesUnavailableFromSourceError(params, None)

# @skip_if_prices_unavailable_for_blacklisted_session(["XLON"])
# def test_skip_if_prices_unavailable_for_blacklisted_session_decorator2():
#     params = {
#         'interval': '5m',
#         'start': _blacklist[2],
#         'end': _blacklist[1],
#     }
#     raise errors.PricesUnavailableFromSourceError(params, None)


class DataUnavailableForTestError(Exception):
    """Base error class for unavailable test data."""

    def __str__(self) -> str:
        return getattr(self, "_msg", "Test Data unavailable.")

    def __unicode__(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()


def skip_if_data_unavailable(f: abc.Callable) -> abc.Callable:
    """Decorator to skip test if fails on `TestDataUnavailableError`."""

    @functools.wraps(f)
    def wrapped_test(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except DataUnavailableForTestError:
            pytest.skip(f"Skipping {f.__name__}: valid test inputs unavailable.")

    return wrapped_test


class ValidSessionUnavailableError(DataUnavailableForTestError):
    """No valid session available for requested restrictions.

    Parameters as for `get_valid_session`.
    """

    def __init__(self, session: pd.Timestamp, limit: pd.Timestamp):
        # pylint: disable=super-init-not-called
        self._msg = (
            f"There are no valid sessions from {session} and with limit as {limit}."
        )


def get_valid_session(
    session: pd.Timestamp,
    calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
    direction: typing.Literal["next", "previous"],
    limit: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return session that is not in blacklist.

    Returns `session` if `session` is not in blacklist, otherwise returns
    nearest session to `session`, in the direction of `direction`, that is
    not in blacklist. Sessions evaluated against `calendar`.

    Raises `ValidSessionUnavailableError` if session would be `limit` or 'beyond'.
    """
    session_received = session
    while session in _blacklist:
        # xcals 4.0 lose wrappers within clause
        if direction == "next":
            session = helpers.to_tz_naive(calendar.next_session(session))
            if limit is not None and session >= limit:
                raise ValidSessionUnavailableError(session_received, limit)
        else:
            session = helpers.to_tz_naive(calendar.previous_session(session))
            if limit is not None and session <= limit:
                raise ValidSessionUnavailableError(session_received, limit)
    return session


def get_valid_consecutive_sessions(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return two valid consecutive sessions.

    Sessions will be sessions of a given `calendar` for which intraday data
    is available at a given `bi`.

    Raises `ValidSessionUnavailableError` if no such sessions available.
    """
    start, limit = th.get_sessions_range_for_bi(prices, bi)
    start = get_valid_session(start, calendar, "next", limit)
    end = prices.cc.next_session(start)
    while end in _blacklist:
        start = get_valid_session(end, calendar, "next", limit)
        end = prices.cc.next_session(start)
        if end > limit:
            raise ValidSessionUnavailableError(start, limit)
    return start, end


def test__adjust_high_low():
    """Verify staticmethod PricesYahoo._adjust_high_low."""
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    ohlcv = (
        [100, 103, 98, 103.4, 0],  # close higher than high
        [104, 109, 104, 107, 0],
        [106, 108, 104, 107, 0],
        [106, 110, 107, 109, 0],  # open lower than low
        [108, 112, 108, 112, 0],
        [112, 114, 107, 106.4, 0],  # close lower than low
        [112, 108, 104, 105, 0],  # open higher than high
    )
    index = pd.date_range(
        start=pd.Timestamp("2022-01-01"), freq="D", periods=len(ohlcv)
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = m.PricesYahoo._adjust_high_low(df)

    ohlcv_expected = (
        [100, 103.4, 98, 103.4, 0],  # close was higher than high
        [104, 109, 104, 107, 0],
        [106, 108, 104, 107, 0],
        [107, 110, 107, 109, 0],  # open was lower than low
        [108, 112, 108, 112, 0],
        [112, 114, 106.4, 106.4, 0],  # close was lower than low
        [108, 108, 104, 105, 0],  # open was higher than high
    )
    expected = pd.DataFrame(ohlcv_expected, index=index, columns=columns)
    assert (expected.open >= expected.low).all()
    assert (expected.low <= expected.high).all()
    assert (expected.high >= expected.close).all()

    assert_frame_equal(rtrn, expected)


def test__fill_reindexed():
    """Verify staticmethod PricesYahoo._fill_reindexed."""
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    mock_bi = intervals.BI_ONE_MIN
    mock_symbol = "SYMB"

    def f(df: pd.DataFrame, cal: xcals.ExchangeCalendar) -> pd.DataFrame:
        return m.PricesYahoo._fill_reindexed(df, cal, mock_bi, mock_symbol)

    ohlcv = (
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [8.4, 8.8, 8.2, 8.6, 88],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [10.4, 10.8, 10.2, 10.6, 101],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
    )
    index = pd.DatetimeIndex(
        [
            "2022-01-01 08:00",
            "2022-01-01 09:00",
            "2022-01-01 10:00",
            "2022-01-02 08:00",
            "2022-01-02 09:00",
            "2022-01-02 10:00",
            "2022-01-03 08:00",
            "2022-01-03 09:00",
            "2022-01-03 10:00",
            "2022-01-04 08:00",
            "2022-01-04 09:00",
            "2022-01-04 10:00",
        ]
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    # method ignores calendar save for confirming no session straddles UTC dateline.
    xlon = xcals.get_calendar("XLON", start="2021-12-21")
    rtrn = f(df, xlon)

    ohlcv_expected = (
        [1.4, 1.4, 1.4, 1.4, 0],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [8.4, 8.4, 8.4, 8.4, 0],
        [8.4, 8.4, 8.4, 8.4, 0],
        [8.4, 8.8, 8.2, 8.6, 88],
        [10.4, 10.4, 10.4, 10.4, 0],
        [10.4, 10.8, 10.2, 10.6, 101],
        [10.6, 10.6, 10.6, 10.6, 0],
    )

    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )
    assert_frame_equal(rtrn, expected)

    # When session' indices straddle the UTC dataline, verify indices grouped
    # by session rather than calendar day.
    xasx = xcals.get_calendar("XASX", start="2021-01-01")
    session = "2022-01-05"
    next_session = "2022-01-06"
    assert xasx.is_session(session) and xasx.is_session(next_session)

    open_, close = xasx.session_open_close(session)
    next_open, next_close = xasx.session_open_close(next_session)

    index = pd.DatetimeIndex(
        [
            "2022-01-05 02:00",
            "2022-01-05 03:00",
            "2022-01-05 04:00",
            "2022-01-05 23:00",
            "2022-01-06 00:00",
        ],
        tz=pytz.UTC,
    )

    assert (
        ((open_ <= index) & (index < close))
        | ((next_open <= index) & (index < next_close))
    ).all()

    ohlcv = (
        [0.4, 0.8, 0.2, 0.6, 1],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [2.4, 2.8, 2.2, 2.6, 22],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [4.4, 4.8, 4.2, 4.6, 44],
    )

    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = f(df, xasx)

    ohlcv_expected = (
        [0.4, 0.8, 0.2, 0.6, 1],
        [0.6, 0.6, 0.6, 0.6, 0],
        [2.4, 2.8, 2.2, 2.6, 22],
        [4.4, 4.4, 4.4, 4.4, 0],
        [4.4, 4.8, 4.2, 4.6, 44],
    )

    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )

    # verify raises warning and fills forward when prices for a full day
    # are missing.
    ohlcv = (
        [0.4, 0.8, 0.2, 0.6, 1],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    index = pd.DatetimeIndex(
        [
            "2022-01-01 08:00",
            "2022-01-01 09:00",
            "2022-01-01 10:00",
            "2022-01-02 08:00",
            "2022-01-02 09:00",
            "2022-01-02 10:00",
            "2022-01-03 08:00",
            "2022-01-03 09:00",
            "2022-01-03 10:00",
        ]
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: ['2022-01-02']"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        rtrn = f(df, xlon)

    ohlcv_expected = (
        [0.4, 0.8, 0.2, 0.6, 1],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [2.6, 2.6, 2.6, 2.6, 0],
        [2.6, 2.6, 2.6, 2.6, 0],
        [2.6, 2.6, 2.6, 2.6, 0],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )

    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )
    assert_frame_equal(rtrn, expected)

    # when prices for a first day are missing, verify raises warning and
    # fills back from next open.
    ohlcv = (
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: ['2022-01-01']"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        rtrn = f(df, xlon)

    ohlcv_expected = (
        [3.4, 3.4, 3.4, 3.4, 0],
        [3.4, 3.4, 3.4, 3.4, 0],
        [3.4, 3.4, 3.4, 3.4, 0],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )

    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )
    assert_frame_equal(rtrn, expected)

    # verify raises warning and fills both ways (as noted above) for both xlon and
    # xasx (i.e. crossing UTC midnight, which involves different code path).
    ohlcv = (
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
    )
    index = pd.DatetimeIndex(
        [
            "2022-01-05 01:00",
            "2022-01-05 02:00",
            "2022-01-05 03:00",
            "2022-01-06 01:00",
            "2022-01-06 02:00",
            "2022-01-06 03:00",
            "2022-01-07 01:00",
            "2022-01-07 02:00",
            "2022-01-07 03:00",
            "2022-01-10 01:00",
            "2022-01-10 02:00",
            "2022-01-10 03:00",
            "2022-01-11 01:00",
            "2022-01-11 02:00",
            "2022-01-11 03:00",
            "2022-01-12 01:00",
            "2022-01-12 02:00",
            "2022-01-12 03:00",
        ],
        tz=pytz.UTC,
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    match_sessions = ["2022-01-05", "2022-01-07", "2022-01-10", "2022-01-12"]
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: {match_sessions}"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        rtrn = f(df, xlon)

    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    with pytest.warns(errors.PricesMissingWarning, match=match):
        rtrn = f(df, xasx)

    ohlcv_expected = (
        [1.4, 1.4, 1.4, 1.4, 0],
        [1.4, 1.4, 1.4, 1.4, 0],
        [1.4, 1.4, 1.4, 1.4, 0],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [3.6, 3.6, 3.6, 3.6, 0],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
        [8.6, 8.6, 8.6, 8.6, 0],
        [8.6, 8.6, 8.6, 8.6, 0],
        [8.6, 8.6, 8.6, 8.6, 0],
    )

    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )
    assert_frame_equal(rtrn, expected)


def test__fill_reindexed_daily(one_min, monkeypatch):
    """Verify staticmethod PricesYahoo._fill_reindexed_daily."""
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    symbol = "AZN.L"
    xlon = xcals.get_calendar("XLON", start="1990-01-01", side="left")
    delay = pd.Timedelta(15, "T")
    prices = m.PricesYahoo(symbol, calendars=xlon, delays=delay.components.minutes)

    def f(df: pd.DataFrame, cal: xcals.ExchangeCalendar) -> pd.DataFrame:
        return prices._fill_reindexed_daily(df, cal, symbol)

    def match(sessions: pd.DatetimeIndex) -> str:
        return re.escape(
            f"Prices from Yahoo are missing for '{symbol}' at the base"
            f" interval '{prices.bis.D1}' for the following sessions:"
            f" {sessions}."
        )

    index = pd.DatetimeIndex(
        [
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-11",
            "2021-01-12",
            "2021-01-13",
            "2021-01-14",
        ]
    )
    ohlcv = (
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [5.4, 5.8, 5.2, 5.6, 55],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [7.4, 7.8, 7.2, 7.6, 77],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
    )

    df = pd.DataFrame(ohlcv, index=index, columns=columns)

    # verify when now is after session open + delay missing values are filled
    # and a missing prices warning is raised.
    now = xlon.session_open(index[-1]) + delay + one_min
    mock_now(monkeypatch, now)
    missing_sessions = pd.DatetimeIndex(
        [
            "2021-01-04",
            "2021-01-07",
            "2021-01-08",
            "2021-01-12",
            "2021-01-14",
        ]
    ).format(date_format="%Y-%m-%d")

    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        rtrn = f(df.copy(), xlon)

    ohlcv_expected = (
        [1.4, 1.4, 1.4, 1.4, 0],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [2.6, 2.6, 2.6, 2.6, 0],
        [2.6, 2.6, 2.6, 2.6, 0],
        [5.4, 5.8, 5.2, 5.6, 55],
        [5.6, 5.6, 5.6, 5.6, 0],
        [7.4, 7.8, 7.2, 7.6, 77],
        [7.6, 7.6, 7.6, 7.6, 0],
    )
    expected = pd.DataFrame(
        ohlcv_expected, index=index, columns=columns, dtype="float64"
    )
    assert_frame_equal(rtrn, expected)

    # verify when now is on limit of session open + delay missing values for
    # last row are included and raised warning does not include last row.
    mock_now(monkeypatch, now - one_min)
    missing_sessions = missing_sessions[:-1]

    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        rtrn = f(df.copy(), xlon)

    missing_row = pd.DataFrame(
        [[np.NaN] * 5], index=index[-1:], columns=columns, dtype="float64"
    )
    expected = pd.concat([expected[:-1], missing_row])
    assert_frame_equal(rtrn, expected)

    # verify as expected when no missing values to last row
    mock_now(monkeypatch, now)
    last_row = pd.DataFrame(
        [[8.4, 8.8, 8.2, 8.6, 88]], index=index[-1:], columns=columns, dtype="float64"
    )
    df = pd.concat([df[:-1], last_row])
    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        rtrn = f(df.copy(), xlon)
    expected = pd.concat([expected[:-1], last_row])
    assert_frame_equal(rtrn, expected)

    # verify returns unchanged and without raising error when no other missing
    # prices except last row
    mock_now(monkeypatch, now - one_min)
    ohlcv = (
        [0.4, 0.8, 0.2, 0.6, 10],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = f(df.copy(), xlon)

    assert_frame_equal(rtrn, df)

    # verify returns unchanged when no missing prices
    ohlcv = (
        [0.4, 0.8, 0.2, 0.6, 10],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = f(df.copy(), xlon)
    assert_frame_equal(rtrn, df)

    # verify that missing prices before first trade date are not filled and
    # that no warning raised. Mocks first trade date...
    prices._first_trade_dates[symbol] = pd.Timestamp("2021-01-06")
    ohlcv = (
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [5.4, 5.8, 5.2, 5.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = f(df.copy(), xlon)
    assert_frame_equal(rtrn, df)


class TestConstructor:
    """Tests for attributes defined by PricesYahoo constructor.

    Only tests those attributes set by PricesYahoo's extension of the base
    constructor.
    """

    @pytest.fixture(scope="class")
    def symbols(self) -> abc.Iterator[str]:
        yield (
            "GOOG IT4500.MI ^IBEX ^FTMC 9988.HK GBPEUR=X GC=F BTC-GBP CL=F"
            " ES=F ZB=F HG=F GEN.L QAN.AX CA.PA BAS.DE FER.MC"
        )

    def test_invalid_symbol(self, symbols):
        invalid_symbol = "INVALIDSYMB"
        match = f"Symbol '{invalid_symbol}' is not recognised by the yahoo API."
        with pytest.raises(ValueError, match=match):
            _ = m.PricesYahoo(invalid_symbol)
        symbols_ = symbols.split()
        symbols_ = symbols_[:4] + [invalid_symbol] + symbols_[4:]
        with pytest.raises(ValueError, match=match):
            _ = m.PricesYahoo(symbols_)

    @pytest.fixture(scope="class")
    def prices(self, symbols) -> abc.Iterator[m.PricesYahoo]:
        yield m.PricesYahoo(symbols)

    def test_ticker(self, prices, symbols):
        """Verify sets yq.Ticker."""
        assert isinstance(prices._ticker, yq.Ticker)
        assert prices._ticker.symbols == helpers.symbols_to_list(symbols)

    def test_daily_limit(self, prices):
        """Verify constructor setting a daily base limit."""
        assert isinstance(prices.base_limits[prices.BaseInterval.D1], pd.Timestamp)

    def test_calendars(self, prices):
        expected_calendars = {
            "GOOG": "XNYS",
            "GEN.L": "XLON",
            "9988.HK": "XHKG",
            "QAN.AX": "XASX",
            "CA.PA": "XPAR",
            "BAS.DE": "XFRA",
            "FER.MC": "XMAD",
            "IT4500.MI": "XMIL",
            "^IBEX": "XMAD",
            "^FTMC": "XLON",
            "GBPEUR=X": "24/5",
            "BTC-GBP": "24/7",
            "GC=F": "CMES",
            "CL=F": "us_futures",
            "ES=F": "CMES",
            "ZB=F": "CMES",
            "HG=F": "CMES",
        }

        for k, cal in prices.calendars.items():
            assert isinstance(cal, xcals.ExchangeCalendar)
            assert expected_calendars[k] == cal.name

    def test_delays(self, prices):
        expected_delays = {
            "GOOG": 0,
            "GEN.L": 15,
            "9988.HK": 15,
            "QAN.AX": 20,
            "CA.PA": 15,
            "BAS.DE": 15,
            "FER.MC": 15,
            "IT4500.MI": 15,
            "^IBEX": 15,
            "^FTMC": 15,
            "GBPEUR=X": 0,
            "BTC-GBP": 0,
            "GC=F": 10,
            "CL=F": 10,
            "ES=F": 10,
            "ZB=F": 10,
            "HG=F": 10,
        }

        for k, delay in prices.delays.items():
            assert isinstance(delay, pd.Timedelta)
            assert delay == pd.Timedelta(expected_delays[k], "T")


@pytest.fixture
def session_length_xnys() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "H")


@pytest.fixture
def session_length_xhkg() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "H")


@pytest.fixture
def session_length_xlon() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(8.5, "H")


@pytest.fixture(scope="module")
def pricess() -> abc.Iterator[dict[str, m.PricesYahoo]]:
    yield dict(
        us=m.PricesYahoo(["MSFT"]),
        with_break=m.PricesYahoo(["9988.HK"]),
        us_lon=m.PricesYahoo(["MSFT", "AZN.L"]),
        us_oz=m.PricesYahoo(["MSFT", "QAN.AX"]),
        inc_245=m.PricesYahoo(["AZN.L", "GBPEUR=X"]),
        inc_247=m.PricesYahoo(["AZN.L", "BTC-GBP"]),
        only_247=m.PricesYahoo(["BTC-GBP"]),
    )


@pytest.fixture
def prices_us(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = pricess["us"].symbols
    yield m.PricesYahoo(symbols)


@pytest.fixture
def prices_us_lon(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = pricess["us_lon"].symbols
    yield m.PricesYahoo(symbols)


@pytest.fixture
def prices_us_lon_hkg() -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = "MSFT, AZN.L, 9988.HK"
    yield m.PricesYahoo(symbols, lead_symbol="MSFT")


@pytest.fixture
def prices_with_break(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = pricess["with_break"].symbols
    yield m.PricesYahoo(symbols)


@pytest.fixture
def prices_only_247(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = pricess["only_247"].symbols
    yield m.PricesYahoo(symbols)


def assert_price_table(df: pd.DataFrame, prices: m.PricesYahoo):
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert not df.isna().all(axis=1).any()

    symbols = prices.symbols
    assert isinstance(df.columns, pd.MultiIndex)
    expected_columns = pd.MultiIndex.from_product(
        (symbols, ["open", "high", "low", "close", "volume"]), names=("symbol", "")
    )
    assert_index_equal(df.columns, expected_columns, check_order=False)


def assert_daily(df: pd.DataFrame, prices: m.PricesYahoo):
    """Assert `df` represents daily data."""
    assert_price_table(df, prices)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.pt.interval == intervals.ONE_DAY


def assertions_daily(
    df: pd.DataFrame,
    prices: m.PricesYahoo,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    cc = prices.cc
    start_ = cc.sessions.get_slice_bound(start, "left")
    end_ = cc.sessions.get_slice_bound(end, "right")
    slc = slice(start_, end_)
    expected = cc.sessions[slc]
    assert_index_equal(df.index, expected)

    symbols = prices.symbols
    for s in symbols:
        subset = df[s].dropna()
        cal = prices.calendars[s]
        if subset.empty:
            assert cal.sessions_in_range(df.pt.first_ts, df.pt.last_ts).empty
            continue
        equiv = df[s].pt.reindex_to_calendar(cal)
        if equiv.iloc[-1].isna().all() and equiv.index[-1] == end:
            # no prices yet for live session
            equiv = equiv[:-1]
        assert_frame_equal(subset, equiv, check_freq=False)
        assert (subset.close <= subset.high).all()
        assert (subset.low <= subset.close).all()


def assert_prices_table_ii(df: pd.DataFrame, prices: m.PricesYahoo):
    """Assert `df` is a price table with interval index."""
    assert df.index.is_non_overlapping_monotonic
    assert isinstance(df.index, pd.IntervalIndex)
    assert_price_table(df, prices)


def assert_multiple_days(
    df: pd.DataFrame, prices: m.PricesYahoo, interval: intervals.PTInterval
):
    """Assert `df` is price table with interval as a multiple of 1D."""
    assert df.pt.interval == interval
    assert_prices_table_ii(df, prices)


class TestRequestDataDaily:
    """Verify implementation of abstract _request_data for daily interval."""

    def test_prices(self, pricess):
        """Verify standard assertions for all prices fixtures."""
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-12-31")
        for prices in pricess.values():
            interval = prices.BaseInterval.D1
            rtrn = prices._request_data(interval, start, end)
            assertions_daily(rtrn, prices, start, end)

    def test_data_unavailable_for_symbol(self, pricess):
        """Verify as expected when data not available for one or more symbols."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2022-01-02")
        rtrn = prices._request_data(interval, start, end)
        all_missing = []
        for s in prices.symbols:
            all_missing.append(rtrn[s].isna().all(axis=None))
        assert any(all_missing)
        assertions_daily(rtrn, prices, start, end)

    def test_start_none(self, pricess):
        """Verify as expected when start is None."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = None
        end = pd.Timestamp("2021-12-31")
        rtrn = prices._request_data(interval, start, end)
        assert rtrn.pt.first_ts == prices.base_limits[prices.BaseInterval.D1]
        assert rtrn.pt.last_ts == end
        # assertions not expected to hold for dataframe that includes dates prior
        # to first date that any of the symbols first traded.
        start = rtrn["BTC-GBP"].first_valid_index()
        rtrn_ss = rtrn[start:]
        assertions_daily(rtrn_ss, prices, start, end)

    def test_live_indice(self, pricess):
        """Verify return with live indice as expected."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = pd.Timestamp("2022-01-01")
        end = helpers.now(intervals.ONE_DAY)
        rtrn = prices._request_data(interval, start, end)
        assert rtrn.pt.first_ts == start
        assert rtrn.pt.last_ts == end
        assertions_daily(rtrn, prices, start, end)


def get_data_bounds(
    prices: m.PricesYahoo,
    interval: intervals.BI,
    limits: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], slice, slice]:
    """Get start and end bounds for price data.

    Parameters
    ----------
    prices
        Prices instance against which to evaluate bounds.

    interval
        Base interval for which bounds required.

    limits : optional
        Define limits within which to get bounds. By default, will
        evaluate bounds for limits of data availability at `interval`.

    Returns
    -------
    tuple[]

        tuple[0]
            [0]: start bound. Open of session following session
            corresopnding to `limits`[0]. Session corresponding to
            `limits`[0] will be session of which `limits`[0] is a
            minute, or otherwise the next session.

            [1]: end bound. Open of session corresopnding to
            `limits`[1]. Session corresponding to `limits`[1] will be
            session of which `limits`[1] is a minute, or otherwise the
            previous session. Setting end bound to a session open
            ensure all the prior session is included (regardless of
            end alignment).

        tuple[1] : slice
            slice from start to end of sessions falling within bounds.
            Does not include session corresponding with `limits`[1].
            Can be used to index a composite calendar. slice dates are
            tz-naive.

        tuple[2] : slice
            slice from start to end of sessions falling within bounds.
            Does not include session corresponding with `limits`[1].
            Can be used be index an ExchangeCalendar. slice dates have
            same tz as calendar sessions.
    """
    ll, rl = prices.limits[interval] if limits is None else limits
    cc = prices.cc
    first_available_session = cc.minute_to_sessions(ll, "next")[0]
    start_session = cc.next_session(first_available_session)
    last_available_session = cc.minute_to_sessions(rl, "previous")[-1]
    end_session = cc.previous_session(last_available_session)
    slc = slice(start_session, end_session)
    start = cc.session_open(start_session)
    # set end as open of next session to ensure include all of end_session
    end = cc.session_open(last_available_session)

    # TODO xcals 4.0 del clause, remove slc_cal from return, then
    # revise client calls and replace all references to slc_cal from
    # `slc_cal` to `slc`. Also, revise method documentation 'Returns'.
    if cc.calendars[0].sessions.tz is pytz.UTC:
        slc_cal = slice(
            slc.start.tz_localize(pytz.UTC),  # pylint: disable=no-member
            slc.stop.tz_localize(pytz.UTC),  # pylint: disable=no-member
        )
    else:
        slc_cal = slc

    return (start, end), slc, slc_cal


def _get_sessions_daterange_for_bi_cal(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    cal: xcals.ExchangeCalendar,
    length_end_session: pd.Timedelta | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_limit, end_limit = prices.limits[bi]
    if not bi.is_one_minute:
        prev_bi = bi.previous
        assert prev_bi is not None
        prev_start_limit, _ = prices.limits[prev_bi]
        assert prev_start_limit is not None
        end_limit = prev_start_limit
    limit_session = cal.minute_to_past_session(end_limit, 2)

    start_session = cal.minute_to_future_session(start_limit, 2)
    end_session = cal.session_offset(start_session, 5)
    assert end_session < limit_session
    if length_end_session is not None:
        duration = None
        while duration != length_end_session:
            open_, close = cal.session_open_close(end_session)
            duration = close - open_
            end_session = cal.next_session(end_session)
            if end_session > limit_session:
                raise ValueError(
                    "Assumed test session length not found!"
                    f" Length assumed as {length_end_session}."
                )

    return start_session, end_session


def _get_sessions_daterange_for_bi_cc(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    length_end_session: pd.Timedelta | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_limit, end_limit = prices.limits[bi]
    if not bi.is_one_minute:
        prev_bi = bi.previous
        assert prev_bi is not None
        prev_start_limit, _ = prices.limits[prev_bi]
        assert prev_start_limit is not None
        end_limit = prev_start_limit
    cc = prices.cc

    first_session = cc.minute_to_sessions(start_limit, "next")[-1]
    last_session = cc.minute_to_sessions(end_limit, "previous")[0]
    sessions = prices.cc.sessions_in_range(first_session, last_session)
    start_session = sessions[1]
    end_session = sessions[5]
    limit_session = sessions[-1]
    assert end_session < limit_session
    if length_end_session is not None:
        avail_range = (end_session, limit_session)
        bv = cc.sessions_length(*avail_range) == length_end_session
        sessions = prices.cc.sessions_in_range(*avail_range)[bv]
        try:
            end_session = sessions[0]
        except IndexError:
            raise ValueError(
                "Assumed test session length not found!"
                f" Length assumed as {length_end_session}."
            ) from None

    return start_session, end_session


def get_sessions_daterange_for_bi(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    cal: xcals.ExchangeCalendar | None = None,
    length_end_session: pd.Timedelta | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get daterange that cannot be served by a base interval lower than `bi`.

    Parameters
    ----------
    prices
        prices against which to evaluate price availability limits.

    bi
        Intraday base interval that will be the lowest base interval
        that can serve prices over the returned daterange.

    cal : default: `prices`.cc
        Calendar against which to evalute sessions and session length. By
        default will evalute these against the `prices` composite calendar.

    length_end_session
        If passed, session representing end of daterange will have this
        length. If no session has this length then a ValueError will be
        raised.

    Returns
    -------
    tuple
        [0] session representing start of daterange.
        [1] session representing end of daterange.
    """
    if cal is not None:
        start_session, end_session = _get_sessions_daterange_for_bi_cal(
            prices, bi, cal, length_end_session
        )
    else:
        start_session, end_session = _get_sessions_daterange_for_bi_cc(
            prices, bi, length_end_session
        )

    start_session = helpers.to_tz_naive(start_session)  # TODO xcals 4.0 lose line
    end_session = helpers.to_tz_naive(end_session)  # TODO xcals 4.0 lose line

    return start_session, end_session


def assert_interval(df: pd.DataFrame, interval: intervals.TDInterval | intervals.BI):
    """Assert `df` has intraday interval(s) `interval`."""
    value_counts = df.index.length.value_counts()
    assert len(value_counts) == 1
    assert interval == value_counts.index[0]


def assert_most_common_interval(
    df: pd.DataFrame, interval: intervals.TDInterval | intervals.BI
):
    """Assert intraday table `df` has most common interval as `interval`."""
    value_counts = df.index.length.value_counts()
    assert value_counts.index[0] == interval


def assertions_monthly(
    df: pd.DataFrame,
    prices: m.PricesYahoo,
    interval: intervals.DOInterval | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """Assertions for a price table with a monthly interval.

    Asserts interval (only if not None) and bounds.
    """
    assert_prices_table_ii(df, prices)
    if interval is not None:
        assert df.pt.interval is interval
    assert_bounds(df, (start, end))


def assertions_intraday_common(
    df: pd.DataFrame, prices: m.PricesYahoo, interval: intervals.TDInterval
):
    """Assert `df` is an intraday price table with regular `interval`.

    Makes following assertions:
        Index is a non-overlapping, monotonic pd.IntervalIndex.
        Index length is regular and equal to `interval`.

        Columns are MulitIndex with expected indices for `symbols`.

        No row has missing data for all columns.
    """
    assert_prices_table_ii(df, prices)
    assert_interval(df, interval)


def assert_bounds(df: pd.DataFrame, bounds: tuple[pd.Timestamp, pd.Timestamp]):
    """Assert `df` starts / ends on `bounds`."""
    start, end = bounds
    assert df.pt.first_ts == start
    assert df.pt.last_ts == end
    assert not (df.index.left < start).any()
    assert not (df.index.right > end).any()


def assertions_intraday(
    df: pd.DataFrame,
    interval: pd.TDInterval,
    prices: m.PricesYahoo,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_num_rows: int,
):
    """Assert that `df` is an intraday price table.

    Asserts:
        as `assertions_intraday_common`
        `df` bounds are `start` and `end`
        `df`.length is `expected_num_rows`,
        by-symbol reindex match
        congruence of open, high, low, close.
    """
    symbols = prices.symbols
    assertions_intraday_common(df, prices, interval)
    # verify index as expected
    assert_bounds(df, (start, end))
    assert len(df) == expected_num_rows

    for s in symbols:
        subset = df[s].dropna()
        cal = prices.calendars[s]
        equiv = df[s].pt.reindex_to_calendar(cal)
        # Don't check dtype as a volume column dtype can be float if includes
        # missing values.
        assert_frame_equal(subset, equiv, check_freq=False, check_dtype=False)
        assert (subset.close <= subset.high).all()
        assert (subset.low <= subset.close).all()


def expected_table_structure_us(
    prices: m.PricesYahoo,
    interval: intervals.BI | intervals.TDInterval,
    slc: slice,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], int, pd.Series]:
    """Return expected aspects of intraday price table of `prices`["us"].

    Parameters
    ----------
    prices
        `prices`["us"]

    interval
        Expected table base interval. Table will be assumed to maintain a
        regular interval.

    slc
        Slice of `prices`.cc.sessions that `table` is expected to cover.
        Table will be assumed to fully cover all sessions of this slice
        and only those sessions.

    Returns
    -------
    tuple
        [0] tuple[pd.Timestamp, pd.Timestamp]:
            First and last timestamps represented by table

        [1] int
            Expected table length.

        [2] pd.Series
            Expected last timestamp of each session.
    """
    cc = prices.cc
    sessions_durations = cc.closes[slc] - cc.opens[slc]
    sessions_rows = np.ceil(sessions_durations / interval)
    expected_num_rows = int(sessions_rows.sum())
    sessions_end = cc.opens[slc] + (interval * sessions_rows)

    start = cc.opens[slc][0]
    end = sessions_end[-1]
    return (start, end), expected_num_rows, sessions_end


def assertions_intraday_us(
    prices: m.PricesYahoo,
    table: pd.DataFrame,
    interval: intervals.BI | intervals.TDInterval,
    bounds: tuple[pd.Timestamp, pd.Timestamp],
    expected_num_rows: int,
    slc: slice,
    sessions_end: pd.Series,
):
    """Assert intraday prices `table` as expected for `prices`["us"].

    Parameters
    ----------
    prices
        `prices`["us"]

    table
        table to be asserted as corresponding with `prices`["us].

    interval
        Expected `table` base interval.

    bounds
        (first, last) timestamps covered by table.

    expected_num_rows
        Expected `table` length.

    slc
        Slice of `prices`.cc.sessions that `table` is expected to cover.
        Table will be assumed to fully cover all sessions of this slice
        and only those sessions.

    sessions_end
        Expected last indice of each session covered.
    """
    start, end = bounds
    assertions_intraday(table, interval, prices, start, end, expected_num_rows)

    cc = prices.cc
    assert cc.opens[slc].isin(table.index.left).all()
    assert not (cc.opens[slc] - interval).isin(table.index.left).any()
    assert sessions_end.isin(table.index.right).all()
    assert not (sessions_end + interval).isin(table.index.right).any()


class TestRequestDataIntraday:
    """Verify implementation of abstract _request_data for intraday intervals."""

    def test_prices_us(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us"]
        for interval in prices.BaseInterval[:-1]:
            (start, end), slc, _ = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            bounds, num_rows, sessions_end = expected_table_structure_us(
                prices, interval, slc
            )
            assertions_intraday_us(
                prices, df, interval, bounds, num_rows, slc, sessions_end
            )

    def test_prices_with_break(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["with_break"]
        cc = prices.cc
        symbols = prices.symbols
        symbol = symbols[0]
        cal = prices.calendars[symbol]

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc, _ = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)
            assertions_intraday_common(df, prices, interval)

            # verify index as expected
            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())

            exclude_break = prices._subsessions_synced(cal, interval)
            if exclude_break:
                # TODO xcals 4.0 can lose clauses, use cal.break_starts / cal.break_ends
                if cal.break_starts.index.tz is not None:
                    break_starts = cal.break_starts.tz_convert(None)
                    break_ends = cal.break_ends.tz_convert(None)
                else:
                    break_starts = cal.break_starts
                    break_ends = cal.break_ends
                break_durations = break_ends[slc] - break_starts[slc]
                break_rows = break_durations // interval
                expected_num_rows -= int(break_rows.sum())

            assert len(df) == expected_num_rows
            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            sessions_last_indice = cc.opens[slc] + (interval * sessions_rows)
            assert sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

            subset = df[symbol].dropna()
            ignore_breaks = not exclude_break
            equiv = df[symbol].pt.reindex_to_calendar(cal, ignore_breaks=ignore_breaks)
            # Don't check dtype as a volume column dtype can be float if includes
            # missing values.
            assert_frame_equal(subset, equiv, check_freq=False, check_dtype=False)
            assert (subset.close <= subset.high).all()
            assert (subset.low <= subset.close).all()

    @staticmethod
    def get_expected_num_rows_us_lon(
        interval: intervals.TDInterval,
        cc: calutils.CompositeCalendar,
        slc: slice,
        slc_cal: slice,
    ) -> tuple[int, pd.Series]:
        """Get expected number of rows of return for 'us_lon' prices fixture.

        Returns
        -------
        tuple(int, pd.Series)
            tuple[0] : int
                Expected number of rows.
            tuple[1] : pd.Series
                Expected number of rows per session, gross of any gaps, for
                any session, between close of xlon and subsequent xnys open.
        """
        sessions_durations = cc.closes[slc] - cc.opens[slc]
        sessions_rows_gross = np.ceil(sessions_durations / interval)

        xlon, xnys = cc.calendars[0], cc.calendars[1]
        if xlon.name != "XLON":
            xlon, xnys = xnys, xlon
        common_index = xlon.opens[slc_cal].index.union(xnys.opens[slc_cal].index)
        opens_xlon = xlon.opens[slc_cal].reindex(common_index)
        closes_xlon = xlon.closes[slc_cal].reindex(common_index)
        opens_xnys = xnys.opens[slc_cal].reindex(common_index)
        closes_xnys = xnys.closes[slc_cal].reindex(common_index)
        bv = closes_xlon < opens_xnys
        xlon_rows = np.ceil((closes_xlon[bv] - opens_xlon[bv]) / interval)
        xnys_rows = np.ceil((closes_xnys[bv] - opens_xnys[bv]) / interval)
        new_session_rows = xlon_rows + xnys_rows
        # TODO xcals 4.0 del if clause
        if new_session_rows.index.tz is not None:
            new_session_rows = new_session_rows.tz_convert(None)
        # TODO xcals 4.0 del if clause
        if bv.index.tz is not None:
            bv = bv.tz_convert(None)

        sessions_rows = sessions_rows_gross.copy()
        sessions_rows[bv.index[bv]] = new_session_rows

        expected_num_rows = int(sessions_rows.sum())
        return expected_num_rows, sessions_rows_gross

    def test_prices_us_lon(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us_lon"]
        cc = prices.cc

        # xnys and xlon indexes are out of phase for 1H
        for interval in prices.BaseInterval[:-2]:
            (start, end), slc, slc_cal = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            expected_num_rows, sessions_rows_gross = self.get_expected_num_rows_us_lon(
                interval, cc, slc, slc_cal
            )
            sessions_last_indice = cc.opens[slc] + (interval * sessions_rows_gross)
            start = cc.opens[slc][0]
            end = sessions_last_indice[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            assert sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

    def test_prices_us_oz(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us_oz"]
        cc = prices.cc
        symbols = prices.symbols

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc, slc_cal = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)
            assertions_intraday_common(df, prices, interval)

            # verify index as expected
            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)

            cal0, cal1 = prices.calendars["QAN.AX"], prices.calendars["MSFT"]
            common_index = cal0.opens[slc_cal].index.union(cal1.opens[slc_cal].index)
            opens_0 = cal0.opens[slc_cal].reindex(common_index)
            closes_0 = cal0.closes[slc_cal].reindex(common_index)
            opens_1 = cal1.opens[slc_cal].reindex(common_index)
            closes_1 = cal1.closes[slc_cal].reindex(common_index)
            bv = closes_0 < opens_1
            cal0_rows = np.ceil((closes_0[bv] - opens_0[bv]) / interval)
            cal1_rows = np.ceil((closes_1[bv] - opens_1[bv]) / interval)
            new_session_rows = cal0_rows + cal1_rows
            # TODO xcals 4.0 del if clause
            if new_session_rows.index.tz is not None:
                new_session_rows = new_session_rows.tz_convert(None)
            # TODO xcals 4.0 del if clause
            if bv.index.tz is not None:
                bv = bv.tz_convert(None)

            sessions_rows_adj = sessions_rows.copy()
            sessions_rows_adj[bv.index[bv]] = new_session_rows

            expected_num_rows = int(sessions_rows_adj.sum())

            assert len(df) == expected_num_rows
            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            sessions_last_indice = cc.opens[slc] + (interval * sessions_rows)
            sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

            for s in symbols:
                if interval == prices.BaseInterval.H1:
                    # unable to reindex as cals open times are not synced at 1H.
                    continue
                subset = df[s].dropna()
                cal = prices.calendars[s]
                equiv = df[s].pt.reindex_to_calendar(cal)
                # Don't check dtype as a volume column dtype can be float if includes
                # missing values.
                assert_frame_equal(subset, equiv, check_freq=False, check_dtype=False)
                assert (subset.close <= subset.high).all()
                assert (subset.low <= subset.close).all()

    def test_prices_inc_245(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["inc_245"]
        cc = prices.cc

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc, _ = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())
            sessions_last_indice = cc.opens[slc] + (interval * sessions_rows)

            start = cc.opens[slc][0]
            end = sessions_last_indice[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            bv = cc.opens[slc].index.weekday == 0  # pylint: disable=compare-to-zero
            assert not (cc.opens[slc][bv] - interval).isin(df.index.left).any()
            assert (cc.opens[slc][~bv] - interval)[1:].isin(df.index.left).all()

            assert sessions_last_indice.isin(df.index.right).all()
            bv = sessions_last_indice.index.weekday == 4
            assert not (sessions_last_indice[bv] + interval).isin(df.index.right).any()
            right_of_last_indice = sessions_last_indice[~bv] + interval
            assert right_of_last_indice[:-1].isin(df.index.right).all()

    def test_prices_inc_247(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["inc_247"]
        cc = prices.cc

        for interval in prices.BaseInterval[:-1]:
            limits = None
            # prices at 2min for BTC only available for same period as 1min data.
            if interval == prices.BaseInterval.T2:
                limits = prices.limits[prices.BaseInterval.T1]

            (start, end), slc, _ = get_data_bounds(prices, interval, limits)
            df = prices._request_data(interval, start, end)

            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())
            sessions_last_indice = cc.opens[slc] + (interval * sessions_rows)

            start = cc.opens[slc][0]
            end = sessions_last_indice[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            assert (cc.opens[slc] - interval)[1:].isin(df.index.left).all()
            assert (cc.opens[slc] + interval).isin(df.index.left).all()

            assert sessions_last_indice.isin(df.index.right).all()
            assert (sessions_last_indice + interval)[:-1].isin(df.index.right).all()
            assert (sessions_last_indice - interval).isin(df.index.right).all()

    def test_data_unavailable_for_symbol(self, pricess):
        """Verify where data for at least one symbol unavailable."""
        prices = pricess["us_lon"]
        _, _, slc_cal = get_data_bounds(prices, prices.BaseInterval.T5)

        symbol_us = "MSFT"
        xnys = prices.calendars[symbol_us]
        xlon = prices.calendars["AZN.L"]

        common_index = xnys.opens[slc_cal].index.intersection(xlon.opens[slc_cal].index)
        delta = pd.Timedelta(2, "H")

        # don't use first index
        for i in reversed(range(len(common_index) - 1)):
            session = common_index[i]
            # TODO xcals 4.0 lose wrapper
            if helpers.to_tz_naive(session) in _blacklist:
                continue
            start = xlon.opens[session]
            end = start + delta
            if end < xnys.opens[session]:
                break

        start = helpers.to_utc(start)  # TODO xcals 4.0 lose line
        end = helpers.to_utc(end)  # TODO xcals 4.0 lose line

        for interval in prices.BaseInterval[:-1]:
            df = prices._request_data(interval, start, end)
            assert df[symbol_us].isna().all(axis=None)
            expected_num_rows = delta // interval
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

    def test_start_end_session_minutes(self, pricess, one_min):
        """Verify where start and end are session minutes, not open or close."""
        prices = pricess["us_lon"]
        cc = prices.cc
        interval = prices.BaseInterval.T5
        _, slc, slc_cal = get_data_bounds(prices, interval)

        delta = pd.Timedelta(20, "T")
        start = cc.opens[slc][0] + delta
        end = cc.closes[slc][-1] - delta

        expected_num_rows, _ = self.get_expected_num_rows_us_lon(
            interval, cc, slc, slc_cal
        )
        expected_num_rows -= (delta // interval) * 2

        start = helpers.to_utc(start)  # TODO xcals 4.0 lose line
        end = helpers.to_utc(end)  # TODO xcals 4.0 lose line

        df = prices._request_data(interval, start, end)
        assertions_intraday(df, interval, prices, start, end, expected_num_rows)

        # verify limit of same return
        start_limit = start - (interval - one_min)
        end_limit = end + (interval - one_min)
        df = prices._request_data(interval, start_limit, end_limit)
        assertions_intraday(df, interval, prices, start, end, expected_num_rows)

        # verify beyond limit
        start_over_limit = start_limit - one_min
        df = prices._request_data(interval, start_over_limit, end_limit)
        assertions_intraday(
            df, interval, prices, start_over_limit, end, expected_num_rows + 1
        )

        end_over_limit = end_limit + one_min
        df = prices._request_data(interval, start_limit, end_over_limit)
        assertions_intraday(
            df, interval, prices, start, end_over_limit, expected_num_rows + 1
        )

        df = prices._request_data(interval, start_over_limit, end_over_limit)
        assertions_intraday(
            df,
            interval,
            prices,
            start_over_limit,
            end_over_limit,
            expected_num_rows + 2,
        )

    def test_start_end_non_session_minutes(
        self, pricess, session_length_xnys, session_length_xlon
    ):
        """Verify where start and end are not session minutes."""
        prices = pricess["us_lon"]
        interval = prices.BaseInterval.T5
        xnys = prices.calendars["MSFT"]
        xlon = prices.calendars["AZN.L"]

        session_start, _, session_end = get_valid_conforming_sessions(
            prices,
            interval,
            [xnys, xlon],
            [session_length_xnys, session_length_xlon],
            3,
        )
        start = xlon.session_open(session_start)
        end = xnys.session_close(session_end)
        df = prices._request_data(interval, start, end)
        delta = pd.Timedelta(20, "T")
        start_ = start - delta
        end_ = end + delta
        df_not_mins = prices._request_data(interval, start_, end_)
        # rtol for rare inconsistency in data returned by yahoo API when hit with
        # high frequency of requests, e.g. such as executing the test suite
        assert_frame_equal(df, df_not_mins, rtol=0.03)

    def test_start_none(self, pricess):
        """Verify as expected when start is None."""
        prices = pricess["inc_247"]
        end = pd.Timestamp.now().floor("T")
        start = None
        for interval in prices.BaseInterval[:-1]:
            match = (
                "`start` cannot be None if `interval` is intraday. `interval`"
                f"receieved as 'f{interval}'."
            )
            with pytest.raises(ValueError, match=match):
                prices._request_data(interval, start, end)

    # Will fail if today, as evaluated against XLON, is blacklisted. Fails
    # on AssertionError as opposed to PricesUnavailableFromSource given that prices
    # are seemingly always available for BTC-GBP (if today if blacklisted then
    # fails on comparing checking `subset` and `equiv` for equality for "AZN.L").
    @skip_if_fails_and_today_blacklisted(["XLON"], [AssertionError])
    def test_live_indice(self, pricess):
        """Verify return with live indice as expected."""
        prices = pricess["inc_247"]
        start = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(2, "D")
        for interval in prices.BaseInterval[:-1]:
            now = pd.Timestamp.now(tz="UTC").floor("T")
            end = now + interval
            df = prices._request_data(interval, start, end)
            num_rows = (now - start) / interval
            num_rows = np.ceil(num_rows) if num_rows % 1 else num_rows + 1
            expected_end = start + (num_rows * interval)
            assertions_intraday(df, interval, prices, start, expected_end, num_rows)

        symbol = "BTC-GBP"
        delay_mins = 15
        prices = m.PricesYahoo(symbol, delays=[delay_mins])
        cal = prices.calendars[symbol]
        cc = calutils.CompositeCalendar([cal])
        delay = pd.Timedelta(delay_mins, "T")
        start = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(2, "D")
        pp = {
            "minutes": 0,
            "hours": 0,
            "days": 0,
            "weeks": 0,
            "months": 0,
            "years": 0,
            "start": start,
            "end": None,
            "add_a_row": False,
        }
        for interval in prices.BaseInterval[:-1]:
            drg = daterange.GetterIntraday(
                cal, cc, delay, prices.limits[interval][0], False, pp.copy(), interval
            )
            (_, end), _ = drg.daterange
            df = prices._request_data(interval, start, end)
            now = pd.Timestamp.now(tz="UTC").floor("T")
            num_rows = (now - delay - start) / interval
            num_rows = np.ceil(num_rows) if num_rows % 1 else num_rows + 1
            expected_end = start + (num_rows * interval)
            assertions_intraday(df, interval, prices, start, expected_end, num_rows)

    def test_one_min_interval(self, pricess):
        """Verify one minute interval as expected.

        Verifies return on and either side of loops that get and consolidate
        chunks of T1 data.
        """
        prices = pricess["only_247"]
        interval = prices.BaseInterval.T1
        for days in [5, 6, 7, 11, 12, 13]:
            end = pd.Timestamp.now(tz=pytz.UTC).ceil("T")
            start = end - pd.Timedelta(days, "D")
            df = prices._request_data(interval, start, end)
            num_expected_rows = days * 24 * 60
            assertions_intraday(df, interval, prices, start, end, num_expected_rows)

    def test_volume_glitch(self, pricess):
        """Verify volume glitch fix being applied where can be."""

        def assertions(
            prices: m.PricesYahoo,
            start: pd.Timestamp,
            end: pd.Timestamp,
            prev_close: pd.Timestamp | None = None,
        ):
            symbol = prices.symbols[0]
            for interval in prices.bis_intraday:
                interval_ = prices._bi_to_source_key(interval)
                hist = prices._yq_history(
                    interval=interval_, start=start, end=end, adj_timezone=False
                )
                assert isinstance(hist, pd.DataFrame)
                hist_s = hist.loc[symbol].copy()
                if hist_s.iloc[0].isna().any() and hist_s.index[0] < start:
                    # because v rare bug can introduce an initial row with missing
                    # values and indice < start
                    hist_s = hist_s[1:].copy()
                if not hist_s.index[-2] + interval == hist_s.index[-1]:
                    # yahoo can return live indice at end of table
                    hist_s = hist_s[:-1].copy()

                hist0 = hist_s.iloc[0]
                indice = hist0.name

                df = prices._request_data(interval, start, end)[symbol]
                df0_vol = df[indice:indice].volume[0]

                # verify glitch in hist not present in df
                if prev_close is None:
                    assert df0_vol
                assert (not hist0.volume) | (hist0.volume <= df0_vol)

                # verify volume for df indice
                if prev_close is not None:
                    start_alt = prev_close - interval
                else:
                    start_alt = start - interval
                hist_alt = prices._yq_history(
                    interval=interval_, start=start_alt, end=end, adj_timezone=False
                )
                assert isinstance(hist_alt, pd.DataFrame)
                hist_alt_s = hist_alt.loc[symbol].copy()

                if not hist_alt_s.index[-2] + interval == hist_alt_s.index[-1]:
                    # yahoo can return live indice at end of table
                    hist_alt_s = hist_alt_s[:-1].copy()

                assert hist_alt_s.loc[indice].name == df[indice:indice].index[0].left
                assert hist_alt_s.loc[indice].volume == df0_vol

                # verify volume for rest of hist is as df
                # hist will be missing any row for which at least one tick not registered
                hist_vol = hist_s[1:].volume
                df_vol = df.pt.indexed_left.loc[hist_vol.index].volume
                # rtol for rare inconsistency in yahoo data when receives high freq of
                # requests, e.g. under execution of test suite.
                assert_series_equal(hist_vol, df_vol, rtol=0.1, check_dtype=False)

                not_in_hist = df[1:].pt.indexed_left.index.difference(hist_vol.index)
                bv = df.loc[not_in_hist].volume == 0  # pylint: disable=compare-to-zero
                assert bv.all()

        # Verify for us prices
        prices = pricess["us"]
        symbol = prices.symbols[0]
        cal = prices.calendars[symbol]
        now = pd.Timestamp.now(pytz.UTC).floor("T")
        # TODO xcals 4.0 lose wrapper
        session = helpers.to_tz_naive(cal.minute_to_past_session(now, 2))
        session = get_valid_session(session, cal, "previous")
        # extra 30T to cover unaligned end of 1H interval
        end = cal.session_close(session) + pd.Timedelta(30, "T")
        start = cal.session_open(session) + pd.Timedelta(1, "H")
        assertions(prices, start, end)

        # Verify for lon prices
        prices = m.PricesYahoo(["AZN.L"])
        cal = prices.calendars["AZN.L"]
        now = pd.Timestamp.now(pytz.UTC).floor("T")
        # TODO xcals 4.0 lose wrapper
        session = helpers.to_tz_naive(cal.minute_to_past_session(now, 2))
        session = get_valid_session(session, cal, "previous")
        # extra 30T to cover unaligned end of 1H interval
        end = cal.session_close(session) + pd.Timedelta(30, "T")
        start = cal.session_open(session)
        prev_close = cal.previous_close(session)
        assertions(prices, start, end, prev_close)


def test_prices_for_symbols():
    """Verify implementation of abstract `prices_for_symbols`.

    Notes
    -----
    H1 interval not tested as not synchronised for xnys/xlon calendars.
    """
    # pylint: disable=too-complex
    symb_us = "MSFT"
    symb_lon = "AZN.L"
    symbols = [symb_us, symb_lon]
    prices = m.PricesYahoo([symb_us, symb_lon])
    f = prices.prices_for_symbols

    _ = prices.get("1d", start="2021-12-31", end="2022-01-05")

    # set up inraday data as period within a single session during which
    # us and lon calendars overlap.
    cal_us = prices.calendars[symb_us]
    cal_lon = prices.calendars[symb_lon]
    now = pd.Timestamp.now(tz=pytz.UTC)
    end_session = cal_us.minute_to_past_session(now, 2)
    start_session = cal_us.minute_to_past_session(now, 12)

    sessions_us = cal_us.opens[start_session:end_session].index
    sessions_lon = cal_lon.opens[start_session:end_session].index

    common_sessions = sessions_us.intersection(sessions_lon)
    for session in reversed(common_sessions):
        if helpers.to_tz_naive(session) in _blacklist:  # TODO 4.0 xcals lose wrapper
            continue
        lon_close = cal_lon.closes[session]
        us_open = cal_us.opens[session]
        # ensure overlap
        if us_open < lon_close - pd.Timedelta(1, "H"):
            start = us_open - pd.Timedelta(2, "H")
            end = lon_close + pd.Timedelta(2, "H")
            # xcals 4.0 del clause
            if start.tz is not pytz.UTC:
                start = start.tz_localize(pytz.UTC)
                end = end.tz_localize(pytz.UTC)
                us_open = us_open.tz_localize(pytz.UTC)
                lon_close = lon_close.tz_localize(pytz.UTC)
            break

    _ = prices.get("5T", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("2T", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("1T", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("5T", us_open, end)
    _ = prices.get("2T", us_open, end)
    _ = prices.get("1T", us_open, end)

    def assertions(
        pdata: data.Data,
        symb: str,
        interval: intervals.BI,
        expect_missing: bool = True,
    ):
        orig = prices._pdata[interval]
        assert pdata.ranges == orig.ranges

        orig_table = orig._table[symb]
        if expect_missing:
            # Assert that at least one row of original data should be missing
            # from new table
            assert orig_table.isna().all(axis=1).any()

        table = pdata._table
        assert table is not None
        assert table.pt.symbols == [symb]
        assert table.notna().all(axis=1).all()
        assert_frame_equal(table.droplevel(0, axis=1), orig_table.dropna())

    # Verify prices for us symb only
    us = f(symb_us)
    assert us.symbols == [symb_us]
    assert us.calendars_unique == [prices.calendars[symb_us]]

    interval = us.BaseInterval.D1
    pdata = us._pdata[interval]
    assertions(pdata, symb_us, interval, expect_missing=False)
    for interval in us.BaseInterval[:-2]:
        pdata = us._pdata[interval]
        assert pdata._table.pt.first_ts == us_open
        assertions(pdata, symb_us, interval)

    # Verify prices for lon symb only
    lon = f(symb_lon)
    assert lon.symbols == [symb_lon]
    assert lon.calendars_unique == [prices.calendars[symb_lon]]

    for interval in lon.BaseInterval[:-2]:
        if interval == intervals.TDInterval.H1:
            continue
        pdata = lon._pdata[interval]
        assert pdata._table.pt.last_ts == lon_close
        assertions(pdata, symb_lon, interval)

    # Verify prices when symbols as original
    both = f(prices.symbols)
    assert both.symbols == prices.symbols
    assert both.calendars_unique == prices.calendars_unique

    for interval in both.BaseInterval:
        if interval == intervals.TDInterval.H1:
            continue
        pdata = both._pdata[interval]
        table = pdata._table
        orig = prices._pdata[interval]
        assert pdata.ranges == orig.ranges
        assert table.pt.symbols == symbols
        assert not table.isna().all(axis=1).any()
        # verify columns same length an order identically to compare
        assert len(table.columns) == len(orig._table.columns)
        assert_frame_equal(table[orig._table.columns], orig._table)


# =========================================================================
# Following are PricesBase methods that rely on default PricesYahoo
# implementation for testing.


def test__get_bi_table(pricess):
    """Test `_get_bi_table`.

    Test limited to getting an intraday table and daily table for specific
    dateranges.
    """
    prices = pricess["us"]
    interval = prices.bis.T5
    to = pd.Timestamp.now()
    from_ = to - pd.Timedelta(21, "D")
    (start, _), slc, _ = get_data_bounds(prices, interval, (from_, to))
    end = prices.cc.closes[slc][-1]
    table = prices._get_bi_table(interval, (start, end))

    bounds, num_rows, sessions_end = expected_table_structure_us(prices, interval, slc)
    assertions_intraday_us(prices, table, interval, bounds, num_rows, slc, sessions_end)

    delta = interval * 3
    start += delta
    end -= delta
    table = prices._get_bi_table(interval, (start, end))
    num_rows -= 6
    assertions_intraday(table, interval, prices, start, end, num_rows)

    sessions = prices.cc.opens[slc].index
    daterange = sessions[0], sessions[-1]
    table = prices._get_bi_table(prices.bi_daily, daterange)
    assertions_daily(table, prices, *daterange)


def set_get_prices_params(
    prices: m.PricesYahoo,
    pp_raw: mptypes.PP,
    ds_interval: intervals.PTInterval | None,
    lead_symbol: str | None = None,
    anchor: Anchor = Anchor.OPEN,
    openend: OpenEnd = OpenEnd.MAINTAIN,
    strict: bool = True,
    priority: mptypes.Priority = Priority.END,
) -> m.PricesYahoo:
    """Return `prices` with gpp set.

    prices.gpp set to instance of prices.GetPricesParams with arguments
    as received.
    """
    if lead_symbol is None:
        lead_symbol = prices._lead_symbol_default
    gpp = prices.GetPricesParams(
        prices, pp_raw, ds_interval, lead_symbol, anchor, openend, strict, priority
    )
    prices._gpp = gpp
    return prices


def get_pp(
    minutes=0,
    hours=0,
    days=0,
    weeks=0,
    months=0,
    years=0,
    start=None,
    end=None,
    include_prior_close=False,
) -> mptypes.PP:
    """Return dict of default pp, updated for receieved arguments."""
    return dict(
        minutes=minutes,
        hours=hours,
        days=days,
        weeks=weeks,
        months=months,
        years=years,
        start=start,
        end=end,
        add_a_row=include_prior_close,
    )


def get_prices_limit_mock(
    prices: m.PricesYahoo, bi: intervals.BI, limit: pd.Timestamp
) -> m.PricesYahoo:
    """Return instance of m.PricesYahoo with `limit` for `bi` set to limit."""
    limits = m.PricesYahoo.BASE_LIMITS.copy()
    limits[bi] = limit

    class PricesMock(m.PricesYahoo):
        """Mock PricesYahoo class."""

        BASE_LIMITS = limits

    return PricesMock(prices.symbols)


@pytest.fixture
def stricts() -> abc.Iterator[list[bool]]:
    """List of possible values for `strict` parameter."""
    yield [True, False]


@pytest.fixture
def priorities() -> abc.Iterator[list[Priority]]:
    """List of possible values for `priority` parameter."""
    yield [Priority.END, Priority.PERIOD]


class TestTableIntraday:
    """Tests `_get_bi_table_intraday`, `_get_table_intraday`.

    Also independently tests dependency `_downsample_bi_table`.

    Tests limited to functionality provided by method. Only considers
    different values of `get` parameters where those parameters affect the
    return of the method under test.

    Tests for `_get_table_intraday` verify that method returns downsampled
    data but does in not in any way comprehensively test downsampling.
    Testing of downsampling is covered by testing of the underlying
    `pt.PTIntraday.downsample` method.

    Effect of `strict` and `priority` verified in tests for
    `_get_bi_table_intraday`, although not  `_get_table_intraday` (which
    only passes these parameters on to `_get_bi_table_intraday`)
    """

    @pytest.fixture
    def freeze_now(self, monkeypatch):
        """Freeze now during test.

        Used to ensure evaluated error msg matches raised error.
        """
        now = pd.Timestamp.now()

        def mock_now(*_, tz=None, **__) -> pd.Timestamp:
            return pd.Timestamp(now, tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)

    @staticmethod
    def match_liie(
        prices: m.PricesYahoo,
        bis_period: list[intervals.BI],
        bis_accuracy: list[intervals.BI],
        anchor: Anchor = Anchor.OPEN,
    ) -> str:
        """Return error message for LastIndiceIntervalError."""
        bi_accurate = bis_accuracy[0]
        limit = prices.limits[bi_accurate][0]
        cal = prices.gpp.calendar
        earliest_minute = cal.minute_to_trading_minute(limit, "next")
        drg = prices.gpp.drg_intraday_no_limit
        drg.interval = bi_accurate
        would_be_period = drg.daterange[0]

        s = (
            "Full period available at the following intraday base intervals"
            " although these do not allow for representing the end indice with"
            f" the greatest possible accuracy:\n\t{bis_period}.\n"
            "The following base intervals could represent the end indice with the"
            " greatest possible accuracy although have insufficient data available to"
            f" cover the full period:\n\t{bis_accuracy}.\n"
            f"The earliest minute from which data is available at {bi_accurate}"
            f" is {earliest_minute}, although at this base interval the"
            f" requested period evaluates to {would_be_period}.\n"
            f"Period evaluated from parameters: {prices.gpp.pp_raw}."
            "\nData that can express the period end with the greatest possible"
            f" accuracy is available from {earliest_minute}. Pass `strict` as False"
            " to return prices for this part of the period."
            "\nAlternatively, consider"
        )
        if anchor is mptypes.Anchor.OPEN:
            s += " creating a composite table (pass `composite` as True) or"
        s += " passing `priority` as 'period'."
        return re.escape(s)

    def test__get_bi_table_intraday_interval_bi(
        self, prices_us, stricts, priorities, freeze_now
    ):
        """Test gets bi table where ds_interval is a bi or None.

        Verifies effect of `strict` and `priority`.
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices._get_bi_table_intraday

        # Verify return for ds_interval as bi T5
        ds_interval = prices.bis.T5

        # set up for period for which 5T data available, although data not available
        # for lower base intervals.
        start_session, end_session = get_sessions_daterange_for_bi(prices, ds_interval)
        start = cal.session_open(start_session)
        end_session_open = cal.session_open(end_session)
        # set end to time that can not be served accurately with > 5T base data
        end = end_session_open + pd.Timedelta(hours=1, minutes=55)
        pp_inb = get_pp(start=start, end=end)

        # Verify unaffected by strict / priority
        for strict, priority in itertools.product(stricts, priorities):
            prices = set_get_prices_params(
                prices, pp_inb, ds_interval, strict=strict, priority=priority
            )
            table_T5, bi = f()
            assert bi is prices.bis.T5
            assert_bounds(table_T5, (start, end))
            assert_interval(table_T5, ds_interval)

        # make explicit that taking from last loop iteration.
        table_T5 = table_T5  # pylint: disable=self-assigning-variable

        # verify prices unavailable if period extends beyond limit of availability
        prices = get_prices_limit_mock(prices, prices.bis.T5, start)
        f = prices._get_bi_table_intraday
        start_session = cal.session_offset(start_session, -1)
        start = cal.session_open(start_session)
        pp_oob = get_pp(start=start, end=end)
        for priority in priorities:
            prices = set_get_prices_params(
                prices, pp_oob, ds_interval, strict=True, priority=priority
            )
            # error msg not matched (msg verified in test where error originates).
            with pytest.raises(errors.PricesIntradayUnavailableError):
                f()

        # verify if `strict` False then returns prices that are available
        for priority in priorities:
            prices = set_get_prices_params(
                prices, pp_oob, ds_interval, strict=False, priority=priority
            )
            table_, bi = f()
            assert bi is prices.bis.T5
            assert_index_equal(table_.index, table_T5.index)

        # Verify return for ds_interval as None
        ds_interval = None
        # Verify returns as table_5T when data available for full period (5T is the
        # only bi that can serve full period and represent end accurately).
        for strict, priority in itertools.product(stricts, priorities):
            prices = set_get_prices_params(
                prices, pp_inb, ds_interval, strict=strict, priority=priority
            )
            table, bi = f()
            assert bi is prices.bis.T5
            assert_index_equal(table.index, table_T5.index)

        # verify that when data is not available for full period, by default
        # a LastIndiceInaccurateError is raised advising conflict between end
        # accuracy and period length.
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, strict=True, priority=Priority.END
        )
        # NB T1 not accurate as period ends before T1 availability starts.
        msg = self.match_liie(prices, [prices.bis.H1], [prices.bis.T5])
        with pytest.raises(errors.LastIndiceInaccurateError, match=msg):
            f()

        # verify if strict False, will get what's available at an interval that
        # can maintain the end accuracy.
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, strict=False, priority=Priority.END
        )
        table, bi = f()
        assert bi is prices.bis.T5
        assert_index_equal(table.index, table_T5.index)

        # verify that if priority PERIOD then period end will not be maintained
        # in favour of ensuring the full period can be served, even if that means
        # being served at a higher interval. Verify strict makes no difference.
        for strict in stricts:
            prices = set_get_prices_params(
                prices, pp_oob, ds_interval, strict=strict, priority=Priority.PERIOD
            )
            table_H1, bi = f()
            assert bi is prices.bis.H1
            assert_interval(table_H1, prices.bis.H1)
            assert_bounds(table_H1, (start, end_session_open + pd.Timedelta(1, "H")))

    def test__get_bi_table_intraday_interval_non_bi(
        self, prices_us, stricts, priorities, one_min, freeze_now
    ):
        """Test gets bi table where ds_interval not a bi and anchor is workback.

        Verifies effect of `strict` and `priority`.
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices._get_bi_table_intraday

        # set up with:
        # ds_interval that can be served by T1 or T2 only,
        # period end that can only be represented by T1,
        # period that can be fully served from T1 data.
        ds_interval = intervals.TDInterval.T4
        start_limit, _ = prices.limits[prices.bis.T1]
        start = cal.minute_to_trading_minute(start_limit, "next")
        start_session = cal.minute_to_session(start)
        end_session = cal.session_offset(start_session, 10)
        end_session_open = cal.session_open(end_session)
        end = end_session_open + pd.Timedelta(hours=1, minutes=3)
        pp_inb = get_pp(start=start, end=end)

        anchor = Anchor.WORKBACK

        for strict, priority in itertools.product(stricts, priorities):
            prices = set_get_prices_params(
                prices,
                pp_inb,
                ds_interval,
                anchor=anchor,
                strict=strict,
                priority=priority,
            )
            # verify table reflects full requested period
            table_T1, bi = f()
            assert bi is prices.bis.T1
            assert_interval(table_T1, prices.bis.T1)
            assert_bounds(table_T1, (start, end))

        # make explicit that taking from last loop iteration.
        table_T1 = table_T1  # pylint: disable=self-assigning-variable

        # verify prices unavailable if period extends beyond limit of availability
        start_session = cal.minute_to_past_session(start_limit, 2)
        start = cal.session_open(start_session)
        pp_oob = get_pp(start=start, end=end)

        prices = set_get_prices_params(prices, pp_oob, ds_interval, anchor=anchor)
        # of those available, only T2 bi can serve T4 ds_interval
        msg = self.match_liie(prices, [prices.bis.T2], [prices.bis.T1], Anchor.WORKBACK)
        with pytest.raises(errors.LastIndiceInaccurateError, match=msg):
            f()

        # verify if strict False then get the data that was available that could
        # represent end
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, anchor=anchor, strict=False
        )
        table, bi = f()
        assert_frame_equal(table, table_T1)

        # verify if priority PERIOD then will get table covering full period using
        # T2 base data, such that end is a minute off.  Verify strict irrelevant.
        for strict in stricts:
            prices = set_get_prices_params(
                prices,
                pp_oob,
                ds_interval,
                anchor=anchor,
                strict=strict,
                priority=Priority.PERIOD,
            )
            table, bi = f()
            assert bi is prices.bis.T2
            assert_interval(table, prices.bis.T2)
            assert_bounds(table, (start, end - one_min))

    def test__get_table_intraday_interval_bi(self, prices_us):
        """Test returns base interval table as is.

        Verifies that passing on data to `_get_bi_table_intraday` and
        returning unaffected.
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices._get_table_intraday

        ds_interval = prices.bis.T5
        start_session, end_session = get_sessions_daterange_for_bi(prices, ds_interval)
        start = cal.session_open(start_session)
        end_session_open = cal.session_open(end_session)
        end = end_session_open + pd.Timedelta(hours=1, minutes=55)
        pp_inb = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices, pp_inb, ds_interval, priority=Priority.PERIOD
        )
        table = f()
        assert_bounds(table, (start, end))
        assert_interval(table, ds_interval)

    def test__get_table_intraday_interval_T4(self, prices_us, one_min):
        """Test `_get_table_intraday` for T4 ds_interval.

        Verifies that method has downsampled data to T4.

        Verifies effect of 'priority' and 'strict' to ensure both being
        passed through to `_get_bi_table_interval`.
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices._get_table_intraday

        # set up as:
        # a ds_interval that can be served by T1 or T2 only, i.e. T4.
        # period end that can only be represented by T1.
        # period that can be fully served from available T1 data.
        ds_interval = intervals.TDInterval.T4
        start_limit, _ = prices.limits[prices.bis.T1]
        start = cal.minute_to_trading_minute(start_limit, "next")
        start_session = cal.minute_to_session(start)
        end_session = cal.session_offset(start_session, 10)
        end_session_open = cal.session_open(end_session)
        end = end_session_open + pd.Timedelta(hours=1, minutes=3)
        pp_inb = get_pp(start=start, end=end)
        anchor = Anchor.WORKBACK

        prices = set_get_prices_params(prices, pp_inb, ds_interval, anchor=anchor)
        # verify table reflects full requested period
        table_T4_T1 = f()  # T4 interval from T1 base data
        assert_most_common_interval(table_T4_T1, ds_interval)
        assert table_T4_T1.pt.last_ts == end
        assert table_T4_T1.pt.first_ts in pd.date_range(
            start=start, periods=4, freq="T"
        )

        # verify prices unavailable if period extends beyond limit of availability
        start_session = cal.minute_to_past_session(start_limit, 2)
        start = cal.session_open(start_session)
        pp_oob = get_pp(start=start, end=end)
        prices = set_get_prices_params(prices, pp_oob, ds_interval, anchor=anchor)
        # msg not matched, matched in get_bi_table_intraday where raised.
        with pytest.raises(errors.LastIndiceInaccurateError):
            f()

        # verify strict being passed through. If False then get the data that
        # was available that could represent end.
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, anchor=anchor, strict=False
        )
        table = f()
        try:
            assert_frame_equal(table, table_T4_T1)
        except AssertionError:
            # Not impossible that, as time has moved on, the limit hasn't moved to
            # the next minute, in which case first minute of the original table is no
            # longer available.
            assert_frame_equal(table, table_T4_T1[1:])

        # verify `priority` being passed through. If  PERIOD then will get table
        # covering full period using T2 base data, such that end is a minute off.
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, anchor=anchor, priority=Priority.PERIOD
        )
        table = f()
        assert_most_common_interval(table, ds_interval)
        assert table.pt.last_ts == end - one_min
        assert table.pt.first_ts in pd.date_range(start=start, periods=4, freq="T")

    def test__get_table_intraday_interval_H2(self, prices_us):
        """Test `_get_table_intraday` for H2 ds_interval.

        Verifies that method has downsampled data to H2 via
        `_downsample_bi_table`.

        Verifies final indice is shortened by `openend` as "shorten".
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices._get_table_intraday

        # set up as:
        # ds_interval that can be served by H1.
        # period that can only be served by H1.
        REG_LENGTH = pd.Timedelta(hours=6, minutes=30)
        start_s, end_s = get_sessions_daterange_for_bi(
            prices, prices.bis.H1, cal, REG_LENGTH
        )
        pp = get_pp(start=start_s, end=end_s)
        start = cal.session_open(start_s)
        end_close = cal.session_close(end_s)
        for ds_interval in [prices.bis.H1, intervals.TDInterval.H2]:
            # verify when openend is SHORTEN
            prices = set_get_prices_params(
                prices, pp, ds_interval, openend=OpenEnd.SHORTEN
            )
            table = f()
            assert_bounds(table, (start, end_close))
            assert table.index[-1].length == pd.Timedelta(30, "T")
            assert_most_common_interval(table, ds_interval)

            # verify when openend is MAINTAIN
            prices = set_get_prices_params(
                prices, pp, ds_interval, openend=OpenEnd.MAINTAIN
            )
            delta = 30 if ds_interval is prices.bis.H1 else 90
            end_maintain = end_close + pd.Timedelta(delta, "T")
            table = f()
            assert_bounds(table, (start, end_maintain))
            assert_interval(table, ds_interval)

    def test__get_table_intraday_interval_H1(self, prices_us_lon):
        """Test `_get_table_intraday` for H1 ds_interval.

        Verifies that method has downsampled T5 data to H1 via
        table.`pt.downsample`.

        Verifies final indice is shortened by `openend` as "shorten".
        """
        prices = prices_us_lon
        # ds_interval H1 which can only be served by T5 data (cals not synced at H1)
        interval = prices.bis.T5
        range_start, _ = th.get_sessions_range_for_bi(prices, interval)
        _, range_end = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        length = pd.Timedelta(13, "H")
        start, end = th.get_conforming_cc_sessions(
            prices.cc, length, range_start, range_end, 2
        )
        while start in _blacklist or end in _blacklist:
            range_start = prices.cc.next_session(range_start)
            assert range_start < range_end, "Unable to find valid test sessions!"
            start, end = th.get_conforming_cc_sessions(
                prices.cc, length, range_start, range_end, 2
            )

        pp = get_pp(start=start, end=end)

        # Verify for xlon lead
        lead = "AZN.L"
        cal = prices.calendars[lead]
        start_open = cal.session_open(start)
        end_close = cal.session_close(end)

        for ds_interval in [prices.bis.H1, intervals.TDInterval.H2]:
            prices_kwargs = {
                "priority": Priority.PERIOD,
                "pp_raw": pp,
                "ds_interval": ds_interval,
                "lead_symbol": lead,
            }

            # verify when openend is MAINTAIN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.MAINTAIN, **prices_kwargs
            )
            table = prices._get_table_intraday()
            # from knowledge of how sessions relate when length 13H. End is 30 minutes
            # before close as nyse continues to trade after the close, hence to maintain
            # interval end on the last full interval prior to the close. Same effect for
            # H1 and H2.
            end_maintain = end_close - pd.Timedelta(30, "T")
            assert_bounds(table, (start_open, end_maintain))
            assert_interval(table, ds_interval)

            # verify when openend is SHORTEN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.SHORTEN, **prices_kwargs
            )
            table = prices._get_table_intraday()
            assert_bounds(table, (start_open, end_close))
            assert table.index[-1].length == pd.Timedelta(30, "T")
            assert_most_common_interval(table, ds_interval)

        # Verify for xnys lead
        lead = "MSFT"
        cal = prices.calendars[lead]
        start_open = cal.session_open(start)
        end_close = cal.session_close(end)

        for ds_interval in [prices.bis.H1, intervals.TDInterval.H2]:
            prices_kwargs = {
                "priority": Priority.PERIOD,
                "pp_raw": pp,
                "ds_interval": ds_interval,
                "lead_symbol": lead,
            }

            # verify when openend is MAINTAIN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.MAINTAIN, **prices_kwargs
            )
            table = prices._get_table_intraday()

            # from knowledge of how sessions relate when length 13H. H1/H2 unaligned by
            # 30/90 minutes although nothing trades after the close, hence maintains the
            # interval by extending the right of the last indice to beyond the close.
            delta = 30 if ds_interval is prices.bis.H1 else 90
            end_maintain = end_close + pd.Timedelta(delta, "T")
            assert_bounds(table, (start_open, end_maintain))
            assert_interval(table, ds_interval)

            # verify when openend is SHORTEN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.SHORTEN, **prices_kwargs
            )
            table = prices._get_table_intraday()
            assert_bounds(table, (start_open, end_close))
            assert table.index[-1].length == pd.Timedelta(30, "T")
            assert_most_common_interval(table, ds_interval)

    def test__downsample_bi_table(
        self, prices_us_lon, prices_with_break, one_sec, one_min
    ):
        """Tests `_downsample_bi_table`.

        Test inputs limited to circumstances in which method called by
        `_get_table_intraday`.
        """
        # pylint: disable=too-complex

        def assertions(
            prices: m.PricesYahoo,
            factors: tuple[int],
            bi: intervals.BI,
            starts: list[pd.Timestamp] | pd.Timestamp,
            end: pd.Timestamp,
            expected_starts: list[pd.Timestamp] | pd.Timestamp | None = None,
            expected_end: pd.Timestamp | None = None,
            lead_symbol: str | None = None,
            ignore_breaks: bool = False,
        ):
            """Make assertions.

            Parameters
            ----------
            starts: list[pd.Timestamp] | pd.Timestamp
                Start(s) as would be received by `get`.

            expected_starts : list[pd.Timestamp] | pd.Timestamp | None, default: `start`
                Start of expected indexes, same length as and in same order
                as `starts`.

            expected_end : pd.Timestamp | None, default: `end`
                End of expected index.

            lead_symbol : str | None, default: default lead symbol
                Lead symbol to pass to get_prices_params.
            """
            if isinstance(starts, pd.Timestamp):
                starts = [starts]

            if isinstance(expected_starts, pd.Timestamp):
                expected_starts = [expected_starts]
            elif expected_starts is None:
                expected_starts = starts

            exp_end = end if expected_end is None else expected_end

            for start, exp_start in zip(starts, expected_starts):
                for factor in factors:
                    ds_interval = intervals.to_ptinterval(bi * factor)
                    pp = get_pp(start=start, end=end)
                    kwargs = dict(ds_interval=ds_interval, openend=OpenEnd.MAINTAIN)
                    if lead_symbol is not None:
                        kwargs["lead_symbol"] = lead_symbol
                    prices = set_get_prices_params(prices, pp, **kwargs)
                    df_bi, bi_ = prices._get_bi_table_intraday()
                    assert bi_ is bi
                    assert isinstance(ds_interval, TDInterval)
                    expected_index = prices.cc.trading_index(
                        ds_interval,
                        exp_start,
                        exp_end,
                        ignore_breaks,
                        utc=True,
                        curtail_calendar_overlaps=True,
                    )
                    rtrn = prices._downsample_bi_table(df_bi, bi)
                    assert_index_equal(rtrn.index, expected_index)

                    for row in rtrn.iterrows():
                        row = rtrn.iloc[0]
                        left = row.name.left
                        right = row.name.right - one_sec
                        subset = df_bi[left:right]
                        for s in prices.symbols:
                            subset_s, row_s = subset[s], row[s]
                            if subset_s.isna().all(axis=None):
                                assert row_s.isna().all(axis=None)
                                continue
                            assert subset_s.volume.sum() == row_s.volume
                            assert subset_s.high.max() == row_s.high
                            assert subset_s.low.min() == row_s.low
                            assert subset_s.bfill().open[0] == row_s.open
                            assert subset_s.ffill().close[-1] == row_s.close

        prices = prices_us_lon
        lead = "AZN.L"
        xlon = prices.calendars[lead]
        bi = prices.bis.T5

        start, end = get_valid_consecutive_sessions(prices, bi, xlon)
        # test start as both session and time
        starts = (start, prices.cc.session_open(start) + one_min)
        xlon_close = xlon.session_close(end)
        assertions(prices, (2, 6), bi, starts, end, None, xlon_close, lead)

        prices = prices_with_break
        bi = prices.bis.T5
        start, end = get_valid_consecutive_sessions(prices, bi, prices.cc)
        assertions(prices, (2, 5, 7, 12), bi, start, end)

        # Test for interval where breaks are ignored
        bi = prices.bis.H1
        start, end = get_valid_consecutive_sessions(prices, bi, prices.cc)
        # test start as both session and time
        starts = (start, prices.cc.session_open(start) + one_min)
        factors = (2, 5)
        assertions(prices, factors, bi, starts, end, None, None, None, True)

        # Verify for detached calendars.
        prices = m.PricesYahoo(["PETR3.SA", "9988.HK"])
        bi = prices.bis.T5
        bvmf = prices.calendars["PETR3.SA"]
        xhkg = prices.calendars["9988.HK"]
        session_length = [pd.Timedelta(hours=6, minutes=30), pd.Timedelta(8, "H")]
        start, end = get_valid_conforming_sessions(
            prices, bi, [xhkg, bvmf], session_length, 2
        )
        bvmf_open = bvmf.session_open(start)
        # NB 21 is limit before last indice of hkg am session overlaps with pm session
        assertions(
            prices, (2, 5, 7, 12, 21), bi, start, end, bvmf_open, None, "PETR3.SA"
        )
        with pytest.warns(errors.IntervalIrregularWarning):
            assertions(prices, (22,), bi, start, end, bvmf_open, None, "PETR3.SA")

        # Test for interval where breaks are ignored
        bi = prices.bis.H1
        start, end = get_valid_conforming_sessions(
            prices, bi, [xhkg, bvmf], session_length, 2
        )
        bvmf_open = bvmf.session_open(start)
        # test start as both session and time
        starts = (start, bvmf.session_open(start) + one_min)
        exp_starts = (bvmf_open, bvmf_open + one_min)
        assertions(prices, (2, 5), bi, starts, end, exp_starts, None, "PETR3.SA", True)


def test__get_table_daily(prices_us, one_day, monkeypatch):
    """Test `_get_table_daily`.

    Verifies that method returns downsampled data but does in not in any
    way comprehensively test downsampling. Testing of downsampling is
    covered by testing of the underlying `pt.PTDaily.downsample` method.
    """
    prices = prices_us
    f = prices._get_table_daily
    # set up as period as covering 10 sessions (from knowledge of schedule)
    start = pd.Timestamp("2022-02-07")
    end = pd.Timestamp("2022-02-18")
    pp = get_pp(start=start, end=end)

    # test ds_interval as daily
    ds_interval = prices.bi_daily
    prices = set_get_prices_params(prices, pp, ds_interval)
    table_1D = f(force_ds_daily=False)
    assertions_daily(table_1D, prices, start, end)

    # test ds_interval as multiple of days, D2.
    ds_interval = intervals.TDInterval.D2
    prices = set_get_prices_params(prices, pp, ds_interval)
    table = f(force_ds_daily=False)
    assert_most_common_interval(table, intervals.TDInterval.D2)
    assert len(table) == 5
    assert table.pt.last_ts == end + one_day
    assert table.pt.first_ts == start

    # verify effect of force_ds_daily
    table = f(force_ds_daily=True)
    assert_frame_equal(table, table_1D)

    # test ds_interval as monthly.
    # set up as a year where last indice is complete
    start = pd.Timestamp("2021")
    end = pd.Timestamp("2021-12-31")
    pp = get_pp(start=start, end=end)
    ds_interval = intervals.DOInterval.M1
    prices = set_get_prices_params(prices, pp, ds_interval)
    table = f(force_ds_daily=False)
    assert_index_equal(
        table.index.left, pd.date_range("2021-01-01", freq="MS", periods=12)
    )
    assert_index_equal(
        table.index.right, pd.date_range("2021-02-01", freq="MS", periods=12)
    )
    # set up as a year where last indice is not complete
    start = pd.Timestamp("2021")
    end = pd.Timestamp("2021-12-15")
    pp = get_pp(start=start, end=end)
    ds_interval = intervals.DOInterval.M1
    prices = set_get_prices_params(prices, pp, ds_interval)
    table = f(force_ds_daily=False)
    assert_index_equal(
        table.index.left, pd.date_range("2021-01-01", freq="MS", periods=11)
    )
    assert_index_equal(
        table.index.right, pd.date_range("2021-02-01", freq="MS", periods=11)
    )

    # verify that last 'incomplete' indice included if pp to now.
    with monkeypatch.context() as m:
        now = pd.Timestamp("2021-12-15 15:00", tz=UTC)
        mock_now(m, now)
        prices = set_get_prices_params(prices, pp, ds_interval)
        table = f(force_ds_daily=False)
        assert_index_equal(
            table.index.left, pd.date_range("2021-01-01", freq="MS", periods=12)
        )
        assert_index_equal(
            table.index.right, pd.date_range("2021-02-01", freq="MS", periods=12)
        )

    # verify force_ds_daily=True returns daily table
    table = f(force_ds_daily=True)
    cal = prices.calendar_default
    expected = cal.sessions_in_range(start, end)
    if expected.tz is not None:
        expected = expected.tz_convert(None)
    assert_index_equal(table.index, expected)


class TestGetComposite:
    """Tests methods to get composite tables.

    Tests:
        `_get_bi_table_intraday`
        `_get_daily_intraday_composite`

    Tests for `_get_bi_table_intraday` ensure returns a composite table
    although do not comprehensively test the underlying `create_composite`,
    which is tested independently.
    """

    @pytest.fixture
    def lead(self) -> abc.Iterator[str]:
        yield "AZN.L"

    def test__get_daily_intraday_composite(self, prices_us_lon, lead, one_min):
        """Test returns expected daily / intraday composite table."""
        prices = prices_us_lon
        # get table_t of T1 data
        start, end = get_sessions_daterange_for_bi(prices, prices.bis.T1)
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices, pp, ds_interval=prices.bis.T1, lead_symbol=lead
        )
        table_t, bi = prices._get_bi_table_intraday()
        assert bi == prices.bis.T1

        start = pd.Timestamp("2021-01-04")
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None)

        table = prices._get_daily_intraday_composite(table_t)
        start_table_t = table_t.index[0].left
        intraday_part = table_t[start_table_t:]
        assert_frame_equal(intraday_part, table_t)
        daily_part = table[:start_table_t][:-1]

        assert isinstance(daily_part.index, pd.IntervalIndex)

        last_daily_sessions = prices.cc.minute_to_sessions(
            start_table_t - one_min, "previous"
        )
        last_daily_session = last_daily_sessions[-1]
        sessions = prices.cc.sessions_in_range(start, last_daily_session)
        sessions = sessions.tz_localize(pytz.UTC)
        assert_index_equal(daily_part.index.left, sessions)
        assert_index_equal(daily_part.index.right, sessions)
        # verify not missing anything inbetween
        assert_frame_equal(table, pd.concat([daily_part, intraday_part]))

    def test__get_daily_intraday_composite_error(self):
        """Test raises error when calendars overlap."""
        prices = m.PricesYahoo(["BTC-USD", "ES=F"])
        lead = "ES=F"

        cal = prices.calendars[lead]
        session_limit = cal.minute_to_past_session(pd.Timestamp.now(), 3)

        bv = prices.cc.sessions_overlap(end=session_limit)
        start_intraday = prices.cc.sessions_in_range(end=session_limit)[bv][-1]

        pp = get_pp(start=start_intraday)
        prices = set_get_prices_params(
            prices, pp, ds_interval=prices.bis.T1, lead_symbol=lead
        )
        table_t, bi = prices._get_bi_table_intraday()
        assert bi == prices.bis.T1

        start = pd.Timestamp("2021-01-04")
        pp = get_pp(start=start)
        prices = set_get_prices_params(prices, pp, ds_interval=None)

        match = re.escape(
            "Unable to create a composite table from a daily and an intraday table"
            " due to a calendar conflict. Exchange times of consecutive sessions of"
            " different calendars overlap (the end of a session for one calendar"
            " falls after the start of the next session of another calendar.)"
        )
        with pytest.raises(errors.CompositePricesCalendarError, match=match):
            prices._get_daily_intraday_composite(table_t)

    def test__get_table_composite_intraday(
        self, prices_us_lon, lead, session_length_xnys, session_length_xlon, one_min
    ):
        """Test returns intraday composite table.

        Tests standard and edge cases.
        """
        prices = prices_us_lon
        symbols = prices.symbols
        # Test standard case. Set up as:
        # end available from T1 data and can only be reflected accurately by T1 data.
        # start available for intervals >= T5.
        calendars = prices.calendars["MSFT"], prices.calendars["AZN.L"]
        session_length = [session_length_xnys, session_length_xlon]
        _, end_session = get_valid_conforming_sessions(
            prices, prices.bis.T1, calendars, session_length, 2
        )
        end_session_open = prices.cc.session_open(end_session)
        end = end_session_open + pd.Timedelta(7, "T")
        _, start_session = get_valid_conforming_sessions(
            prices, prices.bis.T5, calendars, session_length, 2
        )
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table = prices._get_table_composite()
        # in terms of table1 interval, table2 should cover final incomplete indice and
        # one prior full indice, i.e. in this case 5 + 2 rows
        assert (table[-7:].index.length == prices.bis.T1).all()
        assert (table[:-7].index.length == prices.bis.T5).all()
        assert table.pt.first_ts == prices.calendars[lead].session_open(start_session)
        assert table.pt.last_ts == end

        # Test edge case 1.
        limit_T1_mock = end_session_open - pd.Timedelta(1, "H")
        prices = get_prices_limit_mock(prices, prices.bis.T1, limit_T1_mock)
        end = end_session_open + pd.Timedelta(3, "T")
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        # Horrible bug that appears to be on the other side of the yahoo api.
        # Occasionally, and seemingly randomly, fails to reteive the small amount
        # of data. Try 7 times and skip if it doesn't get it.
        for i in range(7):
            try:
                table_edgecase1 = prices._get_table_composite()
            except AssertionError:
                if i == 6:
                    pytest.skip("Unable to get data from Yahoo to test edge case 1.")
                continue
            else:
                break

        # verify T5 parts of composite tables are the same
        # NB doesn't check the frame as chances are the volume data won't match due to
        # being unable to fix the yahoo volume glitch given lack of availability (with
        # the mock) of prior data
        assert_index_equal(table_edgecase1[:-3].index, table[:-7].index)

        # Test edge case 2.
        limit_T1_mock = end_session_open + pd.Timedelta(6, "T")
        prices = get_prices_limit_mock(prices, prices.bis.T1, limit_T1_mock)
        end = end_session_open + pd.Timedelta(9, "T")
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table_edgecase2 = prices._get_table_composite()
        # verify T5 parts of composite tables are the same
        assert_index_equal(table_edgecase2[:-4].index, table[:-7].index)
        # verify other part is being met from T2 data, and table ends one minute short
        # of what would be max accuaracy
        T2_data = table_edgecase2[-4:]
        assert_interval(T2_data, prices.bis.T2)
        assert len(T2_data) == 4
        assert T2_data.pt.first_ts == end_session_open
        assert T2_data.pt.last_ts == end - one_min

        # Verify raises error under edge case 2 when 'next table' has same interval
        # as table1.
        limit_T1_mock = end_session_open + pd.Timedelta(20, "T")  # unable to serve
        # T2 unable to serve from start of day
        limit_T2_mock = end_session_open + pd.Timedelta(6, "T")
        limits = m.PricesYahoo.BASE_LIMITS.copy()
        limits[prices.bis.T1] = limit_T1_mock
        limits[prices.bis.T2] = limit_T2_mock

        class PricesMock(m.PricesYahoo):
            """Mock PricesYahoo class."""

            BASE_LIMITS = limits

        prices = PricesMock(symbols)

        end = end_session_open + pd.Timedelta(8, "T")
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)

        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices._get_table_composite()

    def test__get_table_composite_daily_intraday(
        self, prices_us_lon, lead, one_min, one_day, stricts, priorities
    ):
        """Test returns daily / intraday composite table.

        Tests standard and edge case.

        Verifies returns composite daily / intraday table when start of table
        can only be served by daily data.

        Verifies raises expected errors.
        """
        prices = prices_us_lon
        # set up as:
        # end avaialble from T2 data and can be reflected accurately by T2 data.
        # start available only for daily data.
        length = pd.Timedelta(13, "H")
        try:
            _start_session, end_session = get_sessions_daterange_for_bi(
                prices, prices.bis.T2, length_end_session=length
            )
        except ValueError as e:
            error_msg = e.args[0]
            if "Assumed test session length not found!" in error_msg:
                pytest.skip(error_msg)
            else:
                raise

        while (
            end_session in _blacklist
            or not (prices.cc.sessions_length(end_session, end_session) == length)[0]
        ):
            end_session = prices.cc.previous_session(end_session)
            if end_session == _start_session:
                error_msg = (
                    "Assumed test session length not found for a 'T2' session that's"
                    " not blacklisted."
                )
                pytest.skip(error_msg)

        end_session_open = prices.cc.session_open(end_session)
        end = end_session_open + pd.Timedelta(6, "T")

        start_H1, _ = get_sessions_daterange_for_bi(prices, prices.bis.H1)
        calendar = prices.calendars[lead]
        start_session = calendar.session_offset(start_H1, -50)

        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table = prices._get_table_composite()

        # verify strict and priority have no effect
        for strict, priority in itertools.product(stricts, priorities):
            prices = set_get_prices_params(
                prices, pp, None, lead, strict=strict, priority=priority
            )
            table_ = prices._get_table_composite()
            assert_frame_equal(table_, table)

        # table2 should cover intraday part of last session
        assert (table[-3:].index.length == prices.bis.T2).all()
        # table 1 daily indices have no length
        assert (table[:-3].index.length == pd.Timedelta(0)).all()
        assert table.pt.first_ts == start_session
        assert table.pt.last_ts == end

        # Verify raises error when would otherwise return daily/intraday composite
        # although period defined in terms of an intraday duration.
        # set hours to value that will ensure start is prior to first date for which
        # intraday prices may be available.
        hours = (end_session - start_H1).days * 8  # more than enough.
        pp = get_pp(hours=hours, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)

        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices._get_table_composite()

        # Verify returns daily/intraday composite table under intraday/daily edge case
        # set up so that T2 availability starts between session open and period end.
        limit_T2_mock = end_session_open + pd.Timedelta(1, "T")
        prices = get_prices_limit_mock(prices, prices.bis.T2, limit_T2_mock)
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table = prices._get_table_composite()

        # Verify that as T2 data not available from session open, fulfilled with T5 data
        # to the extent possible (i.e. to max accuracy available)
        assert (table[-1:].index.length == prices.bis.T5).all()
        # table 1 daily indices have no length
        assert (table[:-1].index.length == pd.Timedelta(0)).all()
        assert table.pt.first_ts == start_session
        assert table.pt.last_ts == end - one_min

        # Verify effect of strict when start prior to left limit for daily data
        start = prices.limit_daily - one_day
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(prices, pp, None, lead, strict=True)
        with pytest.raises(errors.StartTooEarlyError):
            prices._get_table_composite()
        prices = set_get_prices_params(prices, pp, None, lead, strict=False)
        table = prices._get_table_composite()
        assert table.pt.first_ts.tz_convert(None) == prices.limit_daily

    def test__get_table_composite_error(self, prices_us_lon, lead):
        """Test raises error when end prior to earliest available intraday data."""
        prices = prices_us_lon
        start_H1, _ = get_sessions_daterange_for_bi(prices, prices.bis.H1)
        calendar = prices.calendars[lead]
        end_session = calendar.session_offset(start_H1, -50)
        end_session_open = prices.cc.session_open(end_session)
        end = end_session_open + pd.Timedelta(1, "H")
        start_session = calendar.session_offset(end_session, -20)

        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)

        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices._get_table_composite()


class TestForcePartialIndices:
    """Tests for `_force_parital_indices`."""

    @staticmethod
    def get_slc(
        calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> slice:
        """Get slice for `calendar` to represent sessions `start` through `end`.

        Method only required to get slc start / stop in tersm of same tz as
        the `calendar` sessions,

        # TODO: xcals 4.0 lose method and amends clients to assume sessions are
        tz naive everywhere.
        """
        if isinstance(calendar, xcals.ExchangeCalendar):
            if calendar.sessions.tz is not None:
                return slice(start.tz_localize(pytz.UTC), end.tz_localize(pytz.UTC))
            else:
                return slice(start, end)
        else:
            if start.tz is not None:
                start = start.tz_convert(None)
            if end.tz is not None:
                end = end.tz_convert(None)
            return slice(start, end)

    def test_us(self, prices_us):
        """Test for single calendar."""
        prices = prices_us
        start, end = get_sessions_daterange_for_bi(prices, prices.bis.H1)

        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices, pp, ds_interval=prices.bis.H1, priority=Priority.PERIOD
        )
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        # assert left side of all indices is unchanged
        assert (index.left == index_forced.left).all()

        cal = prices.calendar_default
        slc = self.get_slc(cal, start, end)
        closes = pd.DatetimeIndex(cal.closes[slc], tz=pytz.UTC)

        bv = index.right != index_forced.right
        # assert all indices with right side changed now reflect a close
        assert index_forced.right[bv].isin(closes).all()
        # assert that all changed table indices had right > corresponding session close
        changed_indices_sessions = table.pt.sessions(cal)[bv]
        for indice, session in changed_indices_sessions.iteritems():
            assert (indice.left < cal.session_close(session)) & (
                indice.right > cal.session_close(session)
            )
        # check all session closes are represented
        assert closes.isin(index_forced.right).all()

    def test_us_lon(self, prices_us_lon, session_length_xnys):
        """Test for two overlapping calendars."""
        prices = prices_us_lon
        xnys = prices.calendars["MSFT"]
        # use T5 data so that can get H1 data out for cals that are unsynced at H1
        lead = "MSFT"
        start, end = get_sessions_daterange_for_bi(
            prices, prices.bis.T5, xnys, session_length_xnys
        )
        start = get_valid_session(start, xnys, "next", end)
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices, pp, prices.bis.H1, lead, openend=OpenEnd.SHORTEN
        )
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        slc = self.get_slc(prices.cc, start, end)
        closes = pd.DatetimeIndex(prices.cc.closes[slc], tz=pytz.UTC)
        cal_us = prices.calendars["MSFT"]
        cal_uk = prices.calendars["AZN.L"]

        # Verify left side of forced index as expected
        bv = index.left != index_forced.left
        # assert all indices with left side changed now reflect an open
        opens = pd.DatetimeIndex(prices.cc.opens[slc], tz=pytz.UTC)
        assert index_forced.left[bv].isin(opens).all()

        # assert all changed table indices had left < corresponding lon session open
        changed_indices_sessions = table.pt.sessions(prices.cc, direction="next")[bv]
        for indice, session in changed_indices_sessions.iteritems():
            assert (indice.left < cal_uk.session_open(session)) & (
                indice.right > cal_uk.session_open(session)
            )

        # check all cc opens are represented
        first_session = prices.gpp.calendar.date_to_session(start, "next")
        first_open = prices.gpp.calendar.session_open(first_session)

        assert opens[opens >= first_open].isin(index_forced.left).all()

        # Verify right side of forced index as expected
        bv = index.right != index_forced.right
        # assert all indices with right side changed now reflect a close
        assert index_forced.right[bv].isin(closes).all()
        # assert that all changed table indices had right > corresponding session close
        changed_indices_sessions = table.pt.sessions(prices.cc)[bv]
        for indice, session in changed_indices_sessions.iteritems():
            if cal_us.is_session(session):
                us_close = (indice.left < cal_us.session_close(session)) & (
                    indice.right > cal_us.session_close(session)
                )
            else:
                us_close = False
            if cal_uk.is_session(session):
                uk_close = (indice.left < cal_uk.session_close(session)) & (
                    indice.right > cal_uk.session_close(session)
                )
            else:
                uk_close = False
            assert us_close or uk_close

        # check all cc closes are represented
        last_session = prices.gpp.calendar.date_to_session(end, "previous")
        last_close = prices.gpp.calendar.session_close(last_session)
        assert closes[closes <= last_close].isin(index_forced.right).all()

        # Verify that if AZN.L lead then no left side of forced index is unchanged
        lead = "AZN.L"
        prices = set_get_prices_params(
            prices, pp, prices.bis.H1, lead, openend=OpenEnd.SHORTEN
        )
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        # assert left side of all indices is unchanged
        assert (index.left == index_forced.left).all()

    def test_with_break(self, prices_with_break):
        """Test for single calendar with breaks."""
        prices = prices_with_break
        start, end = th.get_sessions_range_for_bi(prices, prices.bis.T5)
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=prices.bis.H1)
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        cal = prices.calendar_default
        slc = self.get_slc(cal, start, end)
        am_closes = pd.DatetimeIndex(cal.break_starts[slc], tz=pytz.UTC)
        pm_opens = pd.DatetimeIndex(cal.break_ends[slc], tz=pytz.UTC)
        pm_closes = pd.DatetimeIndex(cal.closes[slc], tz=pytz.UTC)
        all_closes = pm_closes.union(am_closes)

        bv_left = index.left != index_forced.left
        # assert all indices with left side changed now reflect a pm_subsession open.
        assert index_forced.left[bv_left].isin(pm_opens).all()

        bv_right = index.right != index_forced.right
        # assert all indices with right side changed now reflect a (sub)session close.
        assert index_forced.right[bv_right].isin(all_closes).all()

        # assert all table indices with changed left side had left < corresponding
        # pm subsession open.
        changed_indices_sessions = table.pt.sessions(cal)[bv_left]
        for indice, session in changed_indices_sessions.iteritems():
            break_end = cal.session_break_end(session)
            assert (indice.left < break_end) & (indice.right > break_end)

        # assert all table indices with changed right side had right > corresponding
        # (sub)session close.
        changed_indices_sessions = table.pt.sessions(cal)[bv_right]
        for indice, session in changed_indices_sessions.iteritems():
            if cal.session_has_break(session):
                break_start = cal.session_break_start(session)
                am_close = (indice.left < break_start) & (indice.right > break_start)
            else:
                am_close = False
            close = cal.session_close(session)
            pm_close = (indice.left < close) & (indice.right > close)
            assert am_close or pm_close

        # check all pm subsession opens are represented
        assert pm_opens.dropna().isin(index_forced.left).all()
        # check all (sub)session closes are represented
        assert all_closes.dropna().isin(index_forced.right).all()

    def test_detached_calendars(self):
        """Test for two detached calendars, one of which has breaks."""
        prices = m.PricesYahoo(["MSFT", "9988.HK"])
        lead = "9988.HK"
        # verify for both native H1 data and downsampled from T5
        interval = prices.bis.T5
        start, end = get_sessions_daterange_for_bi(prices, interval)
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices,
            pp,
            ds_interval=prices.bis.H1,
            lead_symbol=lead,
            openend=OpenEnd.MAINTAIN,
        )
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        xhkg = prices.calendars[lead]
        xnys = prices.calendars["MSFT"]
        slc = self.get_slc(xnys, start, end)
        us_closes = pd.DatetimeIndex(xnys.closes[slc], tz=pytz.UTC)
        hk_am_closes = pd.DatetimeIndex(xhkg.break_starts[slc], tz=pytz.UTC)
        hk_pm_opens = pd.DatetimeIndex(xhkg.break_ends[slc], tz=pytz.UTC)
        hk_pm_closes = pd.DatetimeIndex(xhkg.closes[slc], tz=pytz.UTC)
        all_closes = us_closes.union(hk_pm_closes).union(hk_am_closes)

        # pylint: disable=comparison-with-callable  # there's no such comparison
        bv_left = index.left != index_forced.left
        # assert all indices with left side changed now reflect a xhkg pm open.
        assert index_forced.left[bv_left].isin(hk_pm_opens).all()

        bv_right = index.right != index_forced.right
        # assert all indices with right side changed now reflect a (sub)session close.
        assert index_forced.right[bv_right].isin(all_closes).all()

        # assert all table indices with changed left side had left < corresponding
        # xhkg pm open.
        changed_indices_sessions = table.pt.sessions(prices.cc)[bv_left]
        for indice, session in changed_indices_sessions.iteritems():
            break_end = xhkg.session_break_end(session)
            assert (indice.left < break_end) & (indice.right > break_end)

        # assert all changed table indices had right > corresponding (sub)session close.
        changed_indices_sessions = table.pt.sessions(prices.cc)[bv_right]
        for indice, session in changed_indices_sessions.iteritems():
            if xnys.is_session(session):
                xnys_close = xnys.session_close(session)
                us_close = (indice.left < xnys_close) & (indice.right > xnys_close)
            else:
                us_close = False
            if xhkg.is_session(session):
                if xhkg.session_has_break(session):
                    break_start = xhkg.session_break_start(session)
                    hk_am_close = (indice.left < break_start) & (
                        indice.right > break_start
                    )
                else:
                    hk_am_close = False
                xhkg_close = xhkg.session_close(session)
                hk_pm_close = (indice.left < xhkg_close) & (indice.right > xhkg_close)
            else:
                hk_am_close = hk_pm_close = False
            assert us_close or hk_am_close or hk_pm_close

        # check all xhkg pm subsession opens are represented
        assert hk_pm_opens.dropna().isin(index_forced.left).all()

        # check all (sub)session closes are represented
        # only consider those us_closes that fall before the last hk close
        # (as lead symbol is hk, table will end before the end session's us close,
        # and this can be more than one session before if hk has holidays towards
        # the end session).
        bv = (us_closes > hk_pm_closes[0]) & (us_closes < hk_pm_closes[-1])
        us_closes_ = us_closes[bv]
        for closes in (us_closes_, hk_pm_closes, hk_am_closes.dropna()):
            assert closes.isin(index_forced.right).all()


def mock_now(monkeypatch, now: pd.Timestamp):
    """Use `monkeypatch` to mock pd.Timestamp.now to return `now`."""

    def mock_now_(*_, tz=None, **__) -> pd.Timestamp:
        return pd.Timestamp(now.tz_convert(None), tz=tz)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now_)


class ValidSessionsUnavailableError(DataUnavailableForTestError):
    """Test data unavailable to 'get_valid_conforming_sessions'.

    There are an insufficient number of consecutive sessions of the
    requested lengths between the requested limits.

    Parameters
    ----------
    Parameters as receieved to `get_valid_conforming_sessions` except:

    start
        Start limit from which can evaluate valid sessions.

    end
        End limit to which can evaluate valid sessions.
    """

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calendars: list[xcals.ExchangeCalendar],
        session_length: list[pd.Timedelta],
        num_sessions: int,
    ):
        # pylint: disable=super-init-not-called
        self._msg = (
            f"{num_sessions} valid consecutive sessions of the requested lengths"
            f" are not available from {start} through {end}."
            f"\n`calendars` receieved as {calendars}."
            f"\n`session_length` receieved as {session_length}."
        )


def get_valid_conforming_sessions(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    calendars: list[xcals.ExchangeCalendar],
    session_length: list[pd.Timedelta],
    num_sessions: int = 2,
) -> pd.DatetimeIndex:
    """Get conforming sessions for which prices available.

    Prices will be available for all sessions in return.

    Prices will be available at bi for at least one session (evaluated
    against CompositeCalendar) prior to first session.

    Raises `ValidSessionsUnavailableError` if there no sessions conform
    with the requirements.
    """
    # get sessions for which price data available at all base intervals
    session_start, session_end = th.get_sessions_range_for_bi(prices, bi)
    session_first = session_start

    # margin allowing for prices at bi to be available over at least one prior session
    session_start = prices.cc.next_session(session_start)
    session_start = get_valid_session(session_start, prices.cc, "next", session_end)

    try:
        sessions = th.get_conforming_sessions(
            calendars, session_length, session_start, session_end, num_sessions
        )
    except errors.TutorialDataUnavailableError:
        raise ValidSessionsUnavailableError(
            session_first, session_end, calendars, session_length, num_sessions
        ) from None
    while (
        any(sessions.isin(_blacklist))
        # make sure prices also available for session prior to conforming sessions
        or prices.cc.previous_session(sessions[0]) in _blacklist
    ):
        # TODO xcals 4.0 lose wrapper
        session_start = helpers.to_tz_naive(prices.cc.next_session(session_start))
        session_start = get_valid_session(session_start, prices.cc, "next", session_end)
        if session_start >= session_end:
            raise ValidSessionsUnavailableError(
                session_first, session_end, calendars, session_length, num_sessions
            )
        try:
            sessions = th.get_conforming_sessions(
                calendars, session_length, session_start, session_end, num_sessions
            )
        except errors.TutorialDataUnavailableError:
            continue
    return sessions


def get_sessions_xnys_xhkg_xlon(bi: TDInterval, num: int = 2) -> pd.DatetimeIndex:
    """Get multiple consecutive regular sessions available at bi.

    Sessions are consecutive sessions of xnys, xhkg and xlon.
    """
    prices = m.PricesYahoo("MSFT, 9988.HK, AZN.L")
    session_length = [
        pd.Timedelta(hours=6, minutes=30),
        pd.Timedelta(hours=6, minutes=30),
        pd.Timedelta(hours=8, minutes=30),
    ]
    xnys = prices.calendars["MSFT"]
    xhkg = prices.calendars["9988.HK"]
    xlon = prices.calendars["AZN.L"]
    calendars = [xnys, xhkg, xlon]

    for bi_ in prices.bis_intraday:
        # convert `bi` from TDInteval to a BI
        if bi_ == bi:
            bi_req = bi_
            break

    return get_valid_conforming_sessions(prices, bi_req, calendars, session_length, num)


@pytest.fixture(scope="class")
def sessions_xnys_xhkg_xlon_T1() -> abc.Iterator[pd.DatetimeIndex]:
    """Two consecutive regular sessions available with T1 data.

    Sessions are consecutive sessions of xnys, xhkg and xlon.
    """
    yield get_sessions_xnys_xhkg_xlon(TDInterval.T1, 2)


@pytest.fixture(scope="class")
def sessions_xnys_xhkg_xlon_H1() -> abc.Iterator[pd.DatetimeIndex]:
    """Two consecutive regular sessions available with H1 data.

    Sessions are consecutive sessions of xnys, xhkg and xlon.
    """
    yield get_sessions_xnys_xhkg_xlon(TDInterval.H1, 2)


def get_sessions_xnys_xhkg(bi: TDInterval, num: int = 2) -> pd.DatetimeIndex:
    """Get multiple consecutive regular sessions available at bi.

    Sessions are consecutive sessions of xnys and xhkg.
    """
    prices = m.PricesYahoo("MSFT, 9988.HK")
    session_length = [
        pd.Timedelta(hours=6, minutes=30),
        pd.Timedelta(hours=6, minutes=30),
    ]
    xnys = prices.calendars["MSFT"]
    xhkg = prices.calendars["9988.HK"]
    calendars = [xnys, xhkg]

    for bi_ in prices.bis_intraday:
        # convert `bi` from TDInteval to a BI
        if bi_ == bi:
            bi_req = bi_
            break

    return get_valid_conforming_sessions(prices, bi_req, calendars, session_length, num)


class TestGet:
    """Tests for the `get` method.

    These tests are not designed to be exhaustive, rather relies on the
    comprehensive testing of underlying methods.

    Tests here serve:
        to verify errors raised directly by `get`.

        to verify `get` is calling the expected underlying methods for the
        parameters passed.

        to verify parameters are being passed through correctly to the
        underlying methods.

        as integration tests on the underlying parts.

    Tests broadly follow the tutorials.
    """

    @skip_if_fails_and_today_blacklisted(["XLON", "XNYS"])
    def test_gpp(self, prices_us_lon):
        """Test `PricesBase.GetPricesParams` is being initialised.

        Tests parameters passed through as expected from arguments passed
        to `get`.
        """
        prices = prices_us_lon
        assert not prices.has_data

        # verify options default values
        prices.get("5T", minutes=30)
        assert prices.has_data
        assert isinstance(prices.gpp, PricesBase.GetPricesParams)
        assert prices.gpp.anchor is Anchor.OPEN
        assert prices.gpp.openend is OpenEnd.MAINTAIN
        assert prices.gpp.prices is prices
        assert prices.gpp.ds_interval is TDInterval.T5
        assert prices.gpp.lead_symbol == "MSFT"
        assert prices.gpp.strict is True

        # passing as default
        lead, strict = "MSFT", True
        prices.get(
            "10T",
            minutes=30,
            anchor="open",
            openend="maintain",
            lead_symbol=lead,
            strict=strict,
        )
        assert prices.has_data
        assert prices.gpp.anchor is Anchor.OPEN
        assert prices.gpp.openend is OpenEnd.MAINTAIN
        assert prices.gpp.prices is prices
        assert prices.gpp.ds_interval is TDInterval.T10
        assert prices.gpp.lead_symbol == lead
        assert prices.gpp.strict is strict

        # passing as non-default
        lead, strict = "AZN.L", False
        prices.get(
            "3T",
            minutes=30,
            anchor="workback",
            openend="shorten",
            lead_symbol=lead,
            strict=strict,
        )
        assert prices.has_data
        assert prices.gpp.anchor is Anchor.WORKBACK
        assert prices.gpp.openend is OpenEnd.SHORTEN
        assert prices.gpp.prices is prices
        assert prices.gpp.ds_interval is TDInterval.T3
        assert prices.gpp.lead_symbol == lead
        assert prices.gpp.strict is strict

    def test_params_errors(self, prices_us, one_min, session_length_xnys):
        """Test errors raised for invalid parameters.

        Tests that errors are raised for invalid parameters and invalid
        combinations of parameters
        """
        prices = prices_us
        cal = prices.calendar_default
        interval = pd.Timedelta(5, "T")

        anchor = "workback"
        msg = "Cannot force close when anchor is 'workback'."
        with pytest.raises(ValueError, match=msg):
            prices.get(interval, force=True, anchor=anchor)

        msg = "Cannot create a composite table when anchor is 'workback'."
        with pytest.raises(ValueError, match=msg):
            prices.get(composite=True, anchor=anchor)

        msg = (
            "Cannot pass an interval for a composite table, although"
            f" receieved interval as {interval}."
        )
        with pytest.raises(ValueError, match=msg):
            prices.get(interval, composite=True)

        msg = "unexpected keyword argument: 'minute'"
        with pytest.raises(pydantic.ValidationError, match=msg):
            prices.get(minute=3)

        # Verify that a parameter that takes a Literal raises exception if pass
        # non-valid value. Only test for one such parameter.
        msg = re.escape(
            "1 validation error for Get\nanchor\n  unexpected value; permitted:"
            " 'workback', 'open' (type=value_error.const; given=wrkback;"
            " permitted=('workback', 'open'))"
        )
        with pytest.raises(pydantic.ValidationError, match=msg):
            prices.get("30T", anchor="wrkback")

        # verify period parameters being verified by `verify_period_parameters`
        msg = "If pass start and end then cannot pass a duration component."
        with pytest.raises(ValueError, match=msg):
            prices.get("1D", "2021", "2021-02-01", days=20)

        # verify raising `PricesDateRangeEmpty`
        # verify parsing raises error when no indice between `start` and `end`

        # define start and end as sessions for which intraday data available
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        start_session, end_session = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, *sessions_range, 2
        )
        start = cal.session_open(start_session) + one_min
        end = cal.session_close(end_session) - one_min
        msg = (
            f"Calendar '{cal.name}' has no sessions from '{helpers.fts(start)}'"
            f" through '{helpers.fts(end)}'."
        )
        with pytest.raises(errors.PricesDateRangeEmpty, match=msg):
            prices.get("1D", start, end)

        # verify for consecutive dates that do not represent sessions
        msg = (
            f"Calendar '{cal.name}' has no sessions from '2021-01-01' through"
            " '2021-01-03'."
        )
        with pytest.raises(errors.PricesDateRangeEmpty, match=msg):
            prices.get("1D", "2021-01-01", "2021-01-03")

    # ----- Tests related to `interval` parameter -----

    def test_interval_validation(self, prices_us):
        """Test pydantic validating `interval`.

        Only tests that `interval` being validated by `to_ptinterval`.
        `to_ptinterval` is comprehensively tested elsewhere.
        """
        # Verify invalid interval unit
        match = re.escape(
            "`interval` unit must by one of ['MIN', 'T', 'H', 'D', 'M']"
            " (or lower-case) although evaluated to 'G'."
        )
        with pytest.raises(pydantic.ValidationError, match=match):
            prices_us.get("3G")

    @skip_if_fails_and_today_blacklisted(["XLON"])
    def test_interval_only_param(self, prices_us, one_day, one_min):
        """Test passing interval as only parameter.

        Tests base intervals and intervals that require downsampling.
        """
        prices = prices_us
        f = prices.get

        cal = prices.calendar_default
        for bi in prices.bis_intraday:
            df = f(bi)
            assertions_intraday_common(df, prices, bi)
            limit_left, limit_right = prices.limits[bi]
            first_from = cal.minute_to_trading_minute(limit_left, "next")
            first_to = cal.minute_offset_by_sessions(first_from, 1)
            last_to = cal.minute_to_trading_minute(limit_right, "previous")
            last_from = cal.minute_offset_by_sessions(last_to, -1)
            if not cal.is_trading_minute(limit_right):
                last_to += one_min
                last_from += one_min
            if bi is prices.bis.H1:
                last_to += pd.Timedelta(30, "T")  # provide for possibly unaligned end
            assert first_from <= df.pt.first_ts <= first_to
            assert last_from <= df.pt.last_ts <= last_to
            # check for interval that requires downsampling
            for factor in (3, 7):
                interval = bi * factor
                df = f(interval)
                assertions_intraday_common(df, prices, interval)
                last_to_ = last_to + (factor * bi)
                assert first_from <= df.pt.first_ts <= first_to
                assert last_from <= df.pt.last_ts <= last_to_

        df_daily = f(prices.bi_daily)
        assert_daily(df_daily, prices)
        limit_left, limit_right = prices.limits[prices.bis.D1]
        last_session = cal.date_to_session(limit_right, "previous")
        if pd.Timestamp.now(tz=pytz.UTC) < cal.session_open(last_session):
            last_session = cal.previous_session(last_session)
        last_session = helpers.to_tz_naive(last_session)  # TODO xcals 4.0 lose line
        assert df_daily.pt.first_ts == limit_left
        assert df_daily.pt.last_ts == last_session

        df_mult_days = f("3D")
        assert_prices_table_ii(df_mult_days, prices)
        # TODO xcals 4.0 lose wrapper
        first_to = helpers.to_tz_naive(cal.session_offset(limit_left, 3))
        assert limit_left <= df_mult_days.pt.first_ts <= first_to
        assert df_mult_days.pt.last_ts == last_session + one_day

        df_monthly = f("2M")
        assert_multiple_days(df_monthly, prices, intervals.DOInterval.M2)
        pd_offset = pd.tseries.frequencies.to_offset("2MS")
        start_ms = pd_offset.rollforward(limit_left)
        # if not on freq
        next_start_ms = pd_offset.rollforward(limit_left + pd.DateOffset(months=1))
        assert df_monthly.pt.first_ts in (start_ms, next_start_ms)
        end_ms = pd_offset.rollforward(last_session)
        next_end_ms = pd_offset.rollforward(end_ms + one_day)
        assert df_monthly.pt.last_ts in (end_ms, next_end_ms)

    @skip_if_fails_and_today_blacklisted(["XNYS"])
    def test_intervals_inferred(self, prices_us):
        """Test intervals inferred as expected."""
        prices = prices_us
        cal = prices.calendar_default

        bi = prices.bis.T1
        rng_start, rng_end = th.get_sessions_range_for_bi(prices, bi)
        rng_start = get_valid_session(rng_start, cal, "next", rng_end)
        rng_end = get_valid_session(rng_end, cal, "previous", rng_start)
        assert rng_start < rng_end
        rng_start_open = cal.session_open(rng_start)
        rng_end_close = cal.session_close(rng_end)

        # Verify return at intraday intervals.

        # verify end as time. Given end T5 is highest intraday interval that can fulfil.
        end = cal.session_open(rng_end) + pd.Timedelta(5, "T")
        df = prices.get(start=rng_start, end=end)
        assertions_intraday_common(df, prices, prices.bis.T5)
        assert_bounds(df, (rng_start_open, end))

        # verify start as time.
        start = cal.session_open(rng_start) + pd.Timedelta(5, "T")
        df = prices.get(start=start, end=rng_end)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert df.pt.first_ts == (rng_start_open + prices.bis.H1)
        assert rng_end_close in df.index[-1]

        # verify start and end as times.
        df = prices.get(start=start, end=end)
        assertions_intraday_common(df, prices, prices.bis.T5)
        assert_bounds(df, (start, end))

        # verify minutes
        df = prices.get(start=rng_start, minutes=4)
        assertions_intraday_common(df, prices, prices.bis.T2)
        assert_bounds(df, (rng_start_open, rng_start_open + pd.Timedelta(4, "T")))
        df = prices.get(minutes=4)
        assert df.pt.interval in (prices.bis.T1, prices.bis.T2)

        # verify hours
        df = prices.get(start=rng_start, hours=4)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert_bounds(df, (rng_start_open, rng_start_open + pd.Timedelta(4, "H")))
        df = prices.get(start=rng_start, hours=50)
        assert df.pt.interval in prices.bis_intraday

        # verify hours and minutes
        df = prices.get(hours=4, minutes=33)
        assert df.pt.interval in prices.bis_intraday

        # verify days
        df = prices.get(start=rng_start, days=5)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert df.pt.first_ts == rng_start_open
        expected_end = cal.session_close(cal.session_offset(rng_start, 4))
        assert expected_end in df.index[-1]
        df = prices.get(days=5)
        assert df.pt.interval in prices.bis_intraday

        # verify start and end describing 5 sessions
        # TODO xcals 4.0 lose wrapper
        end = helpers.to_tz_naive(cal.session_offset(rng_start, 4))
        df = prices.get(start=rng_start, end=end)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert df.pt.first_ts == rng_start_open
        assert expected_end in df.index[-1]

        # verify start only, 5 sessions prior to last or current session
        limit_left, limit_right = prices.limits[prices.bis.D1]
        last_session = cal.date_to_session(limit_right, "previous")
        if pd.Timestamp.now(tz=pytz.UTC) < cal.session_open(last_session):
            last_session = cal.previous_session(last_session)
        last_session = helpers.to_tz_naive(last_session)  # TODO xcals 4.0 lose line
        start = cal.session_offset(last_session, -4)
        start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
        df = prices.get(start=start)
        assert df.pt.interval in prices.bis_intraday
        assert df.pt.first_ts == cal.session_open(start)

        # Verify return at daily interval

        # verify days
        df = prices.get(days=6)
        assert_daily(df, prices)
        assert df.pt.last_ts == last_session
        expected_start = cal.session_offset(last_session, -5)
        # TODO xcals 4.0 lose if clause
        if expected_start.tz is not None:
            expected_start = expected_start.tz_convert(None)
        assert df.pt.first_ts == expected_start

        # verify start and end describing > 5 sessions
        # TODO xcals 4.0 lose wrapper
        end = helpers.to_tz_naive(cal.session_offset(rng_start, 5))
        df = prices.get(start=rng_start, end=end)
        assertions_daily(df, prices, rng_start, end)

        # verify start only, 6 sessions prior to last/current_session
        # 6 rather than 5 as if a live session then won't be counted
        # TODO xcals 4.0 lose wrapper
        start = helpers.to_tz_naive(cal.session_offset(last_session, -6))
        df = prices.get(start=start)
        assertions_daily(df, prices, start, last_session)

        # verify no parameters
        df = prices.get()
        assertions_daily(df, prices, limit_left, last_session)

    @skip_if_fails_and_today_blacklisted(["XNYS"])
    def test_interval_invalid(self, prices_us, session_length_xnys, one_min):
        """Tests errors raised when interval invalid.

        Verifies raises:
            `PricesUnavailableIntervalDurationConflict`
            `PricesUnavailableIntervalPeriodError`
            `PricesUnavailableIntervalDurationError`
        """
        prices = prices_us
        cal = prices.calendar_default
        f = prices.get

        # Verify raises PricesUnavailableIntervalDurationError
        f("2H", hours=2, anchor="workback")
        with pytest.raises(errors.PricesUnavailableIntervalDurationError):
            f("2H", hours=1, minutes=59)

        # Verify raises PricesUnvailableDurationConflict (raised directly by `get``)
        match = (
            "Duration cannot be defined in terms of hours and/or"
            " minutes when interval is daily or higher."
        )
        with pytest.raises(errors.PricesUnvailableDurationConflict, match=match):
            prices.get("1d", hours=3)
        with pytest.raises(errors.PricesUnvailableDurationConflict, match=match):
            prices.get("3d", minutes=40)
        with pytest.raises(errors.PricesUnvailableDurationConflict, match=match):
            prices.get("1m", minutes=40, hours=3)

        # Verify raises PricesUnavailableIntervalPeriodError
        interval, days_limit = "3d", 3
        f(interval, days=days_limit)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f(interval, days=days_limit - 1)

        days_limit = 63
        kwargs = dict(interval="3M", end="2021-12-15")
        f(**kwargs, days=days_limit)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f(**kwargs, days=days_limit - 1)

        rng_start, rng_end = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        sessions = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, rng_start, rng_end, 4
        )
        i = 0
        session = sessions[i]
        while session in _blacklist:
            i += 1
            assert i < len(sessions) + 1, "cannot get session for test"
            session = sessions[i]

        kwargs = dict(start=session, days=1, anchor="workback")
        # also verify can pass interval as pd.Timedelta
        f(session_length_xnys, **kwargs)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f(session_length_xnys + one_min, **kwargs)

        # Verify raises PricesUnavailableIntervalPeriodError
        start = cal.session_close(session) - pd.Timedelta(2, "T")
        end = cal.session_open(cal.next_session(session)) + pd.Timedelta(2, "T")
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f("5T", start=start, end=end)

        start = cal.session_open(session) + prices.bis.T5
        end = start + pd.Timedelta(4, "T")
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f("5T", start=start, end=end)

    def test_intervals_overlapping(self, prices_only_247, prices_with_break, one_min):
        """Test raises warning when downsampled intervals would overlap.

        Verifies raises `errors.IntervalIrregularWarning`.
        """
        prices = prices_only_247
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices.get("7T", days=2)
        assert_most_common_interval(df, intervals.TDInterval.T7)

        prices = prices_with_break
        rng_start, rng_end = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session_length = pd.Timedelta(hours=6, minutes=30)
        sessions = th.get_conforming_cc_sessions(
            prices.cc, session_length, rng_start, rng_end, 4
        )
        i = 0
        session = sessions[i]
        while session in _blacklist:
            i += 1
            assert i < len(sessions) + 1
            session = sessions[i]

        interval_limit = pd.Timedelta(hours=1, minutes=45)
        prices.get(interval_limit, session, session)
        with pytest.warns(errors.IntervalIrregularWarning):
            prices.get(interval_limit + one_min, session, session)

    # ----- Tests related to period parameters -----

    def test_start_end_inputs(self, prices_us_lon):
        """Test expected valid inputs for `start` and `end`.

        Also tests `tzin`.
        """
        # pylint: disable=undefined-loop-variable
        prices = prices_us_lon
        cal = prices.calendar_default

        regular_session_length = pd.Timedelta(hours=13)
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        sessions = th.get_conforming_cc_sessions(
            prices.cc, regular_session_length, *sessions_range, 5
        )
        for session in sessions:
            if session not in _blacklist:
                break

        start = cal.session_open(session)
        start_utc = start
        start_local = sl = start.astimezone(prices.tz_default)
        start_str = sl.strftime("%Y-%m-%d %H:%M")
        start_str2 = sl.strftime("%Y-%m-%d %H")
        start_int = pd.Timestamp(start_str).value
        start_dt = datetime.datetime(sl.year, sl.month, sl.day, sl.hour, sl.minute)

        end = cal.session_close(session)
        end -= pd.Timedelta(30, "T")
        end_utc = end
        end_local = el = end.astimezone(prices.tz_default)
        end_str = el.strftime("%Y-%m-%d %H:%M")
        end_str2 = el.strftime("%Y-%m-%d %H:%M")
        end_int = pd.Timestamp(end_str).value
        end_dt = datetime.datetime(el.year, el.month, el.day, el.hour, el.minute)

        starts = (start_utc, start_local, start_str, start_str2, start_int, start_dt)
        ends = (end_utc, end_local, end_str, end_str2, end_int, end_dt)

        df_base = prices.get("2H", starts[0], ends[0])
        for start, end in zip(starts[1:], ends[1:]):
            df = prices.get("2H", start, end)
            assert_frame_equal(df, df_base)

        # Verify `tzin`
        for tzin in ("America/New_York", pytz.timezone("America/New_York"), "MSFT"):
            df = prices.get("2H", start_utc, end_str, tzin=tzin)
            assert_frame_equal(df, df_base)

        # verify can pass as non-default symbol
        start_lon_tz = start_utc.astimezone(prices.timezones["AZN.L"])
        start_lon_str = start_lon_tz.strftime("%Y-%m-%d %H:%M")
        end_lon_tz = end_utc.astimezone(prices.timezones["AZN.L"])
        end_lon_str = end_lon_tz.strftime("%Y-%m-%d %H:%M")
        df = prices.get("2H", start_lon_str, end_lon_str, tzin="AZN.L", tzout="MSFT")
        assert_frame_equal(df, df_base)

    def test_start_end_parsing(self, prices_us, session_length_xnys, one_min, one_sec):
        """Test `start` and `end` being parsed as expected.

        Tests `start` and `end` being passed as dates and times that are
        aligned and unaligned with indices anchored on open.
        """
        prices = prices_us
        cal = prices.calendar_default

        # Verify passing `start` and `end` as dates that represent sessions
        start = pd.Timestamp("2021-12-06")
        end = pd.Timestamp("2021-12-10")
        # verify getting daily data
        df = prices.get("1D", start, end)

        assertions_daily(df, prices, start, end)

        # verify passing `start` and `end` as dates that do not represent sessions
        df = prices.get("1D", start="2021-12-04", end="2021-12-11")
        df = prices.get("1D", start, end)

        # define start and end as sessions for which intraday data available
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        num_sessions = 3
        start_session, _, end_session = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, *sessions_range, num_sessions
        )

        # verify returning intaday data, also verifies can pass `start` and `end`
        # as positional arguments
        df = prices.get("5T", start_session, end_session)
        start = cal.session_open(start_session)
        end = cal.session_close(end_session)
        num_rows = int((session_length_xnys / prices.bis.T5)) * num_sessions
        assertions_intraday(df, TDInterval.T5, prices, start, end, num_rows)

        # verify a random multiple day interval
        df = prices.get("2M", "2021-01-02", "2021-06-30")
        expected_start = pd.Timestamp("2021-03-01")
        expected_end = pd.Timestamp("2021-07-01")
        assertions_monthly(df, prices, DOInterval.M2, expected_start, expected_end)

        # Verify passing `start` and `end` as trading times
        # verify return of daily data
        df_daily = prices.get("1D", start, end)
        assertions_daily(df_daily, prices, start_session, end_session)

        # verify passing `start` and `end` as non-trading times
        assert_frame_equal(df_daily, prices.get("1D", start - one_min, end + one_min))

        # verify passing `start` and `end` one min inside sessions excludes them
        assert_frame_equal(
            df_daily[1:-1], prices.get("1D", start + one_min, end - one_min)
        )

        # verify return of intraday data.
        df = prices.get("10T", start, end)
        assertions_intraday(df, TDInterval.T10, prices, start, end, num_rows // 2)
        df = prices.get("5T", start, end)
        assertions_intraday(df, TDInterval.T5, prices, start, end, num_rows)

        # verify passing `start` and `end` as non-trading times
        assert_frame_equal(df, prices.get("5T", start - one_min, end + one_min))
        delta = pd.Timedelta(45, "T")
        assert_frame_equal(df, prices.get("5T", start - delta, end + delta))

        # verify passing `start` and `end` 6 min inside session bounds knocks 2 indices
        # off each end
        delta = pd.Timedelta(6, "T")
        assert_frame_equal(df[2:-2], prices.get("5T", start + delta, end - delta))

        # verify just one second inside the session bounds will knock off the
        # first/last indices
        assert_frame_equal(df[1:-1], prices.get("5T", start + one_sec, end - one_sec))

        # Verify passing `start` and `end` as mix of data and time
        assert_frame_equal(df_daily, prices.get("1D", start_session, end))
        assert_frame_equal(df_daily, prices.get("1D", start, end_session))
        assert_frame_equal(df_daily[1:], prices.get("1D", start + one_sec, end_session))

        assert_frame_equal(df, prices.get("5T", start_session, end))
        assert_frame_equal(df[:-1], prices.get("5T", start_session, end - one_sec))
        assert_frame_equal(df, prices.get("5T", start, end_session))

    @skip_if_fails_and_today_blacklisted(["XNYS"])
    def test_start_end_none(self, prices_us, session_length_xnys, one_sec, monkeypatch):
        """Test `start` and `end` as None."""
        prices = prices_us
        cal = prices.calendar_default
        for bi in prices.bis_intraday:
            limit_left = prices.limits[bi][0]
            first_from = cal.minute_to_trading_minute(limit_left, "next")
            for ds_interval in (bi, bi * 3):
                df = prices.get(ds_interval, openend="shorten")
                assert first_from <= df.pt.first_ts <= first_from + ds_interval

        # define start and end as sessions for which intraday data available
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, *sessions_range, 2
        )[-1]
        prices = m.PricesYahoo("MSFT")
        now_session_open = cal.session_open(sessions_range[-1])
        for bi in prices.bis_intraday:
            for ds_interval in (bi, bi * 3):
                mock_now(monkeypatch, now_session_open + ds_interval + one_sec)
                df = prices.get(ds_interval, start=session, openend="shorten")
                assert df.pt.last_ts == now_session_open + (ds_interval * 2)

    def test_trading_time_duration(self, prices_us, monkeypatch):
        """Test for periods defined as duration in terms of trading time."""
        prices = prices_us
        cal = prices.calendar_default

        # Verify durations in trading terms
        session, limit_session = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session = get_valid_session(session, cal, "next", limit_session)

        # verify bounding with a session
        df = prices.get("5T", session, minutes=20)
        open_, close = cal.session_open_close(session)
        assertions_intraday(
            df, TDInterval.T5, prices, open_, open_ + pd.Timedelta(20, "T"), 4
        )

        df = prices.get("15T", end=session, hours=1)
        assertions_intraday(
            df, TDInterval.T15, prices, close - pd.Timedelta(1, "H"), close, 4
        )

        # verify bounding with a time
        bound = open_ + pd.Timedelta(30, "T")
        df = prices.get("15T", bound, hours=1, minutes=15)
        delta = pd.Timedelta(hours=1, minutes=15)
        assertions_intraday(df, TDInterval.T15, prices, bound, bound + delta, 5)

        # verify crossing sessions
        df = prices.get("15T", end=bound, hours=1, minutes=15)
        prev_close = cal.session_close(cal.previous_session(session))
        exp_start = prev_close - pd.Timedelta(45, "T")
        assertions_intraday(df, TDInterval.T15, prices, exp_start, bound, 5)

        # verify Silver Rule (if bound unaligned then start from next aligned indice)
        df = prices.get("25T", start=bound, hours=1, minutes=40)
        exp_start = bound + pd.Timedelta(20, "T")
        exp_end = exp_start + pd.Timedelta(hours=1, minutes=40)
        assertions_intraday(df, TDInterval.T25, prices, exp_start, exp_end, 4)

        # verify limit before next indice would be included
        assert_frame_equal(df, prices.get("25T", start=bound, hours=2, minutes=4))
        df = prices.get("25T", start=bound, hours=2, minutes=5)
        exp_end += pd.Timedelta(25, "T")
        assertions_intraday(df, TDInterval.T25, prices, exp_start, exp_end, 5)

        # verify default end with now as trading time
        now = open_ + pd.Timedelta(32, "T")
        mock_now(monkeypatch, now)
        df = prices.get("5T", minutes=20)
        exp_end = open_ + pd.Timedelta(35, "T")
        exp_start = exp_end - pd.Timedelta(20, "T")
        assertions_intraday(df, TDInterval.T5, prices, exp_start, exp_end, 4)

        # verify default end with now as non-trading time
        now = close + pd.Timedelta(2, "H")
        mock_now(monkeypatch, now)
        df = prices.get("2T", minutes=41)
        assertions_intraday(
            df, TDInterval.T2, prices, close - pd.Timedelta(40, "T"), close, 20
        )

    def test_trading_sessions_duration(
        self, prices_us, session_length_xnys, monkeypatch, one_min, one_sec
    ):
        """Test for periods defined as duration in terms of trading sessions."""
        prices = prices_us
        cal = prices.calendar_default

        # verify to bound as session
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        num_sessions = 6
        sessions = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, *sessions_range, num_sessions
        )
        start_session, end_session = sessions[0], sessions[-1]
        # getting daily data
        df_daily = prices.get("D", start=start_session, days=num_sessions)
        assertions_daily(df_daily, prices, start_session, end_session)
        assert_frame_equal(
            df_daily, prices.get("D", end=end_session, days=num_sessions)
        )
        # getting intraday data
        open_ = cal.session_open(start_session)
        close = cal.session_close(end_session)
        num_rows = int((session_length_xnys * num_sessions) / pd.Timedelta(30, "T"))
        df = prices.get("30T", start=start_session, days=num_sessions)
        assertions_intraday(df, TDInterval.T30, prices, open_, close, num_rows)
        assert_frame_equal(df, prices.get("30T", end=end_session, days=num_sessions))

        # verify to bound as time
        # getting intraday data
        start = open_ + pd.Timedelta(88, "T")
        df_intra = prices.get("30T", start, days=num_sessions - 1)
        exp_start = open_ + pd.Timedelta(90, "T")
        exp_end = cal.session_open(end_session) + pd.Timedelta(90, "T")
        num_rows -= 13
        if exp_start.time() != exp_end.time():
            # adjust for different DST observance
            if exp_end.time() > exp_start.time():
                exp_end -= pd.Timedelta(1, "H")
                num_rows -= 2
            else:
                exp_end += pd.Timedelta(1, "H")
                num_rows += 2
        assertions_intraday(
            df_intra, TDInterval.T30, prices, exp_start, exp_end, num_rows
        )
        end = exp_end + one_min  # verify aligning as expected
        assert_frame_equal(df_intra, prices.get("30T", end=end, days=num_sessions - 1))
        # getting daily data
        df = prices.get("1D", start, days=num_sessions - 1)
        assertions_daily(df, prices, sessions[1], end_session)
        df = prices.get("1D", end=end, days=num_sessions - 1)
        assertions_daily(df, prices, start_session, sessions[-2])

        # verify to now as trading time
        # one sec short of 30 min post last session open
        mock_now(monkeypatch, exp_end - one_sec)
        df = prices.get("30T", days=num_sessions - 1)
        assert_frame_equal(df_intra, df)

        # verify to now as non-trading time
        mock_now(monkeypatch, close + pd.Timedelta(2, "H"))
        assert_frame_equal(
            df_daily, prices.get("D", end=end_session, days=num_sessions)
        )

    def test_calendar_time_duration(
        self, prices_us, session_length_xnys, monkeypatch, one_sec, one_day
    ):
        """Test for periods defined as duration in terms of calendar time."""
        prices = prices_us
        cal = prices.calendar_default

        start_range, end_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        num_sessions = 6

        sessions = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, start_range, end_range, num_sessions
        )
        # make sure all sessions have same DST observance
        while sessions.isin(_blacklist).any() or (
            prices.tz_default.dst(sessions[0]) != prices.tz_default.dst(sessions[-1])
        ):
            start_range += one_day
            sessions = th.get_conforming_cc_sessions(
                prices.cc, session_length_xnys, start_range, end_range, num_sessions
            )
        start_session = sessions[0]

        num_sessions_in_week = cal.sessions_distance(
            start_session, start_session + pd.Timedelta(6, "D")
        )

        # verify getting intraday data with bound as date
        end_session = cal.session_offset(start_session, num_sessions_in_week - 1)
        end_session = helpers.to_tz_naive(end_session)  # TODO xcals 4.0 lose line
        df = prices.get("30T", start_session, weeks=1)
        # Verifications piggy backs on separate testing of durations in terms of
        # trading sessions
        expected = prices.get("30T", start_session, days=num_sessions_in_week)
        assert_frame_equal(df, expected)
        expected = prices.get("30T", end=end_session, days=num_sessions_in_week)
        assert_frame_equal(df, expected)

        # verify getting intraday data with bound as time
        # end_session will now be one session later
        end_session2 = cal.session_offset(start_session, num_sessions_in_week)
        end_session2 = helpers.to_tz_naive(end_session2)  # TODO xcals 4.0 lose line
        open_ = cal.session_open(start_session)
        start = open_ + pd.Timedelta(28, "T")
        df = prices.get("30T", start, weeks=1)
        # Verification piggy backs on separate testing of durations in terms of
        # trading sessions
        expected = prices.get("30T", start, days=num_sessions_in_week)
        assert_frame_equal(df, expected)
        end = cal.session_open(end_session2) + pd.Timedelta(32, "T")
        df_intra = prices.get("30T", end=end, weeks=1)
        assert_frame_equal(df_intra, expected)

        # verify getting daily data
        # verify with session bound
        df_daily = prices.get("D", start_session, weeks=1)
        assertions_daily(df_daily, prices, start_session, end_session)
        # verify with time bound
        df = prices.get("D", start, weeks=1)
        assertions_daily(df, prices, sessions[1], end_session2)

        # verify examples where expected return static and known
        df = prices.get(start="2021-01-01", weeks=1)
        first_exp = pd.Timestamp("2021-01-04")
        last_exp = pd.Timestamp("2021-01-08")
        assertions_daily(df, prices, first_exp, last_exp)

        df = prices.get("2M", start="2020", years=2, months=2)
        first_exp = pd.Timestamp("2020-01-01")
        last_exp = pd.Timestamp("2022-03-01")
        assertions_monthly(df, prices, None, first_exp, last_exp)

        # verify to now as trading time
        now = cal.session_open(end_session2) + pd.Timedelta(30, "T") - one_sec
        # one sec short of 30 min post last session open
        mock_now(monkeypatch, now - one_sec)
        df = prices.get("30T", weeks=1)
        assert_frame_equal(df_intra, df)

        # verify to now as non-trading time
        mock_now(monkeypatch, cal.session_open(end_session) + pd.Timedelta(2, "H"))
        assert_frame_equal(df_daily, prices.get("D", weeks=1), check_freq=False)

    def test_lead_symbol(self):
        """Test effect of `lead_symbol`."""
        prices = m.PricesYahoo("MSFT, BTC-USD")
        xnys = prices.calendars["MSFT"]

        # verify effect of lead_symbol on known expected daily returns
        df = prices.get("D", end="2022-01-24", days=3, lead_symbol="BTC-USD")
        exp_start, exp_end = pd.Timestamp("2022-01-22"), pd.Timestamp("2022-01-24")
        assertions_daily(df, prices, exp_start, exp_end)

        df = prices.get("D", end="2022-01-24", days=3, lead_symbol="MSFT")
        exp_start = pd.Timestamp("2022-01-20")
        assertions_daily(df, prices, exp_start, exp_end)

        # verify effect of lead_symbol on intraday data
        session, limit_session = th.get_sessions_range_for_bi(
            prices, prices.bis.T1, xnys
        )
        session = get_valid_session(session, xnys, "next", limit_session)
        # start as one hour prior to "MSFT" open
        half_hour = pd.Timedelta(30, "T")
        xnys_open = xnys.session_open(session)
        start = xnys_open - pd.Timedelta(1, "H")
        args, kwargs = ("6T", start), {"minutes": 30}
        # verify start rolls forward to MSFT open
        df = prices.get(*args, **kwargs, lead_symbol="MSFT")
        assertions_intraday(
            df, TDInterval.T6, prices, xnys_open, xnys_open + half_hour, 5
        )
        # verify start as start given BTC is open (24/7)
        df = prices.get(*args, **kwargs, lead_symbol="BTC-USD")
        assertions_intraday(df, TDInterval.T6, prices, start, start + half_hour, 5)

    def test_add_a_row(self, prices_us):
        """Test effect of `add_a_row`."""
        prices = prices_us
        cal = prices.calendar_default

        # verify for daily interval
        end = pd.Timestamp("2022-01-24")
        interval = "D"
        kwargs = {"end": end, "days": 3}
        base_df = prices.get(interval, **kwargs)
        # from knowledge of calednar
        assertions_daily(base_df, prices, pd.Timestamp("2022-01-20"), end)
        df = prices.get(interval, **kwargs, add_a_row=True)
        assert_frame_equal(df[1:], base_df, check_freq=False)
        assertions_daily(df, prices, pd.Timestamp("2022-01-19"), end)

        # verify for intraday interval, with add_a_row causing cross in sessions
        session, limit_session = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session = get_valid_session(session, cal, "next", limit_session)
        start = cal.session_open(session)
        args = ("10T", start)
        kwargs = {"minutes": 30}
        base_df = prices.get(*args, **kwargs)
        exp_end = start + pd.Timedelta(30, "T")
        assertions_intraday(base_df, TDInterval.T10, prices, start, exp_end, 3)

        df = prices.get(*args, **kwargs, add_a_row=True)
        assert_frame_equal(df[1:], base_df, check_freq=False)
        exp_start = cal.previous_close(start) - pd.Timedelta(10, "T")
        assertions_intraday(df, TDInterval.T10, prices, exp_start, exp_end, 4)

    # ---------------------- Tests related to anchor ----------------------

    @staticmethod
    def create_single_index(
        start: pd.Timestamp, end: pd.Timestamp, interval: TDInterval
    ) -> pd.IntervalIndex:
        """Return continuous index from `start` through `end` at interval."""
        index = pd.date_range(start, end, freq=interval.as_pdfreq)
        return pd.IntervalIndex.from_arrays(index[:-1], index[1:], closed="left")

    def create_index(
        self,
        starts: list[pd.Timestamp],
        ends: list[pd.Timestamp],
        interval: TDInterval,
    ) -> pd.IntervalIndex:
        """Return expected index.

        Expected index created from regular indices between each start and end
        of `starts` and `end`.
        """
        index = pd.IntervalIndex([])
        index = self.create_single_index(starts[0], ends[0], interval)
        for start, end in zip(starts[1:], ends[1:]):
            session_index = self.create_single_index(start, end, interval)
            index = index.union(session_index)
        return index

    def test_open_indices(
        self, prices_us, prices_with_break, sessions_xnys_xhkg_xlon_T1, monkeypatch
    ):
        """Test indices as expected when `anchor` is "open".

        Tests symbol with break and sybmol without break, each at an interval aligned
        with (sub)session closes and an interval that is unaligned.

        Tests over two consecutive regular sessions and, separately, to a mock now.
        """
        start_session, end_session = sessions_xnys_xhkg_xlon_T1

        # Verify for calendar without break
        prices = prices_us
        xnys = prices.calendar_default

        df = prices.get(
            "30T", start_session, end_session, anchor="open", tzout=pytz.UTC
        )
        assert df.pt.has_regular_interval
        assert df.pt.indices_have_regular_trading_minutes(xnys)

        start = xnys.session_open(start_session)
        end_start_session = xnys.session_close(start_session)
        start_end_session = xnys.session_open(end_session)
        end = xnys.session_close(end_session)
        starts = [start, start_end_session]
        ends = [end_start_session, end]
        interval = TDInterval.T30
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

        # verify when unaligned with session close
        df = prices.get(
            "90T", start_session, end_session, anchor="open", tzout=pytz.UTC
        )
        assert df.pt.has_regular_interval
        assert not df.pt.indices_have_regular_trading_minutes(xnys)

        misalignment = pd.Timedelta(1, "H")
        start = xnys.session_open(start_session)
        end_start_session = xnys.session_close(start_session) + misalignment
        start_end_session = xnys.session_open(end_session)
        end = xnys.session_close(end_session) + misalignment
        starts = [start, start_end_session]
        ends = [end_start_session, end]
        interval = TDInterval.T90
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

        # Verify for calendar with break
        prices_hk = prices_with_break
        xhkg = prices_hk.calendar_default

        df = prices_hk.get(
            "30T", start_session, end_session, anchor="open", tzout=pytz.UTC
        )
        assert df.pt.has_regular_interval
        assert df.pt.indices_have_regular_trading_minutes(xhkg)

        starts, ends = [], []
        for session in (start_session, end_session):
            starts.append(xhkg.session_open(session))
            ends.append(xhkg.session_break_start(session))
            starts.append(xhkg.session_break_end(session))
            ends.append(xhkg.session_close(session))
        interval = TDInterval.T30
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

        # verify when unaligned with both am and pm subsession closes
        df = prices_hk.get(
            "40T", start_session, end_session, anchor="open", tzout=pytz.UTC
        )
        assert df.pt.has_regular_interval
        assert not df.pt.indices_have_regular_trading_minutes(xhkg)

        starts, ends = [], []
        for session in (start_session, end_session):
            starts.append(xhkg.session_open(session))
            ends.append(xhkg.session_break_start(session) + pd.Timedelta(10, "T"))
            starts.append(xhkg.session_break_end(session))
            ends.append(xhkg.session_close(session) + pd.Timedelta(20, "T"))
        interval = TDInterval.T40
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

        # # verify to now includes live indice and nothing beyond
        now = ends[-1] - pd.Timedelta(50, "T")
        mock_now(monkeypatch, now)
        # should lose last indice
        df_ = prices_hk.get(
            "40T", start_session, end_session, anchor="open", tzout=pytz.UTC
        )
        # last indice be the same although not values as now is 10 minutes short
        # of the full indice
        assert_frame_equal(df_[:-1], df[:-2])
        assert df_.index[-1] == df.index[-2]

    def test_workback_indices(
        self,
        prices_us,
        prices_with_break,
        sessions_xnys_xhkg_xlon_T1,
        monkeypatch,
        one_min,
    ):
        """Test indices as expected when `anchor` is "workback".

        Tests symbol with break and symbol without break.
        Tests to session end and to minute of a session.
        """
        start_session, end_session = sessions_xnys_xhkg_xlon_T1

        # Verify for symbol without a break
        prices = prices_us
        xnys = prices.calendar_default

        df = prices.get(
            "90T", start_session, end_session, anchor="workback", tzout=pytz.UTC
        )
        assert len(df.pt.indices_length) == 2
        assert len(df.pt.indices_partial_trading(xnys)) == 1
        assert df.pt.indices_have_regular_trading_minutes(xnys)

        interval = TDInterval.T90
        end = xnys.session_close(end_session)
        start = end - (interval * 4)
        index_end = self.create_single_index(start, end, interval)

        end = xnys.session_close(start_session) - pd.Timedelta(1, "H")
        start = end - (interval * 3)
        index_start = self.create_single_index(start, end, interval)

        indice_cross_sessions = pd.Interval(
            index_start[-1].right, index_end[0].left, closed="left"
        )
        index_middle = pd.IntervalIndex([indice_cross_sessions])

        index = index_start.union(index_middle).union(index_end)
        assert_index_equal(df.index, index)

        # Verify for symbol with a break
        prices_hk = prices_with_break
        xhkg = prices_hk.calendar_default

        session = start_session
        df = prices_hk.get("40T", session, session, anchor="workback", tzout=pytz.UTC)
        assert len(df.pt.indices_length) == 2
        assert len(df.pt.indices_partial_trading(xhkg)) == 1
        assert df.pt.indices_have_regular_trading_minutes(xhkg)

        interval = TDInterval.T40
        end = xhkg.session_close(session)
        start = end - (interval * 4)
        index_end = self.create_single_index(start, end, interval)

        end = xhkg.session_break_start(session) - pd.Timedelta(20, "T")
        start = end - (interval * 3)
        index_start = self.create_single_index(start, end, interval)

        indice_cross_sessions = pd.Interval(
            index_start[-1].right, index_end[0].left, closed="left"
        )
        index_middle = pd.IntervalIndex([indice_cross_sessions])

        index = index_start.union(index_middle).union(index_end)
        assert_index_equal(df.index, index)

        # Verify to specific minute
        end = xnys.session_close(start_session) - pd.Timedelta(43, "T")
        df = prices.get("30T", end=end, hours=4, anchor="workback", tzout=pytz.UTC)
        interval = TDInterval.T30
        start = end - (8 * interval)
        index = self.create_single_index(start, end, interval)
        assert_index_equal(df.index, index)

        # Verify to now
        mock_now(monkeypatch, end - one_min)
        prices = m.PricesYahoo("MSFT")
        df_ = prices.get("30T", hours=4, anchor="workback", tzout=pytz.UTC)
        assert_frame_equal(df_, df)

    def test_mult_cal_indices(self, prices_us_lon, sessions_xnys_xhkg_xlon_T1):
        """Test indices as expected when evaluted against multiple calendars.

        Tests expected output over two sessions for symbols trading on two
        different calendars.
        Tests for indices anchored both on "open" and "workback".
        """
        start_session, end_session = sessions_xnys_xhkg_xlon_T1

        prices = prices_us_lon
        xnys = prices.calendars["MSFT"]
        xlon = prices.calendars["AZN.L"]

        kwargs = {
            "interval": "1H",
            "start": start_session,
            "end": end_session,
            "tzout": pytz.UTC,
        }
        interval = TDInterval.H1
        half_hour = pd.Timedelta(30, "T")

        # Verify for indices anchored "open" with xlon as lead cal
        df = prices.get(**kwargs, anchor="open", lead_symbol="AZN.L")
        assert df.pt.has_regular_interval
        assert (df.index.left.minute == 0).all()  # pylint: disable=compare-to-zero
        assert (df.index.right.minute == 0).all()  # pylint: disable=compare-to-zero

        starts, ends = [], []
        starts.append(xlon.session_open(start_session))
        ends.append(xnys.session_close(start_session))
        starts.append(xlon.session_open(end_session))
        # last indice excluded as xnys open after end within unaligned indice
        ends.append(xlon.session_close(end_session) - half_hour)
        index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, index)

        # Verify for indices anchored "open" with xnys as lead cal
        df = prices.get(**kwargs, anchor="open", lead_symbol="MSFT")
        assert df.pt.has_regular_interval
        assert (df.index.left.minute == 30).all()
        assert (df.index.right.minute == 30).all()

        starts, ends = [], []
        starts.append(xnys.session_open(start_session))
        # last indice included as xlon closed after xnys close
        ends.append(xnys.session_close(start_session) + half_hour)
        # first xlon indice starts half hour prior to open to maintain interval
        # based on xnys open
        starts.append(xlon.session_open(end_session) - half_hour)
        ends.append(xnys.session_close(end_session) + half_hour)
        index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, index)

        # Verify for indices anchored "workback" with xlon lead cal.
        df = prices.get(**kwargs, anchor="workback", lead_symbol="AZN.L")

        end = xlon.session_close(end_session)
        # prior half hour gets consumed into indice that crosses sessions
        start = xlon.session_open(end_session) + half_hour
        index_end = self.create_single_index(start, end, interval)

        # following half hour gets consumed into indice that crosses sessions.
        end = xnys.session_close(start_session) - half_hour
        # start unaligned, so excludes first half hour
        start = xlon.session_open(start_session) + half_hour
        index_start = self.create_single_index(start, end, interval)

        indice_cross_sessions = pd.Interval(
            index_start[-1].right, index_end[0].left, closed="left"
        )
        index_middle = pd.IntervalIndex([indice_cross_sessions])

        index = index_start.union(index_middle).union(index_end)
        assert_index_equal(df.index, index)

        # Verify for indices anchored "workback" with xnys lead cal
        df = prices.get(**kwargs, anchor="workback", lead_symbol="MSFT")

        # aligns neatly as both xnys closes and xlon opens on the hour
        end = xnys.session_close(end_session)
        start = xlon.session_open(end_session)
        index_end = self.create_single_index(start, end, interval)

        end = xnys.session_close(start_session)
        # start unaligned, so excludes first half hour
        start = xnys.session_open(start_session) + half_hour
        index_start = self.create_single_index(start, end, interval)

        index = index_start.union(index_end)
        assert_index_equal(df.index, index)

    def test_force(self, prices_with_break, sessions_xnys_xhkg_xlon_T1):
        """Test effect of `force`.

        Testing limited to ensuring option results in force being
        implemented. Does not test the actual implementation of force.
        """
        session = sessions_xnys_xhkg_xlon_T1[0]
        prices_hk = prices_with_break
        xhkg = prices_hk.calendar_default

        # verify not forced
        df = prices_hk.get("40T", session, session, anchor="open", force=False)
        assert df.pt.has_regular_interval
        assert xhkg.session_break_start(session) not in df.index.right
        assert df.pt.last_ts != xhkg.session_close(session)
        assert not df.pt.indices_all_trading(xhkg)
        assert len(df.pt.indices_partial_trading(xhkg)) == 2

        # verify not forced by default
        df_ = prices_hk.get("40T", session, session, anchor="open")
        assert_frame_equal(df_, df)

        # verify forced
        df_f = prices_hk.get("40T", session, session, anchor="open", force=True)
        assert not df_f.pt.has_regular_interval
        assert xhkg.session_break_start(session) in df_f.index.right
        assert df_f.pt.last_ts == xhkg.session_close(session)
        assert df_f.pt.indices_all_trading(xhkg)
        assert df_f.pt.indices_partial_trading(xhkg).empty

    def test_overlapping(
        self, sessions_xnys_xhkg_xlon_T1, prices_only_247, prices_with_break, one_day
    ):
        """Test raises warning when intervals overlap.

        Tests raises IntervalIrregularWarning when:
            last indice of session overlaps with first indice of next session.

            last indice of a morning subsession overlaps with first indice of
            afternoon subsession.
        """
        start_session, end_session = sessions_xnys_xhkg_xlon_T1
        i = 3
        while start_session + one_day != end_session:
            assert i < 15, "No consecutive days as valid sessions for xnys_xhkg_xlon."
            sessions = get_sessions_xnys_xhkg_xlon(TDInterval.T1, i)
            start_session, end_session = sessions[-2], sessions[-1]
            i += 1

        prices_247 = prices_only_247
        x247 = prices_247.calendar_default

        # verify does not overlap on limit
        start_session_close = x247.session_close(start_session)
        end_session_open = x247.session_open(end_session)
        assert start_session_close == end_session_open
        df = prices_247.get("8H", start_session, end_session, anchor="open")
        assert df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 1
        assert start_session_close in df.index.left
        assert start_session_close in df.index.right

        # verify warns
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices_247.get("7H", start_session, end_session, anchor="open")
        assert not df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 2
        assert start_session_close in df.index.left
        assert start_session_close in df.index.right

        prices_hk = prices_with_break
        xhkg = prices_hk.calendar_default

        session = start_session
        # verify am/pm indices do not overlap on limit
        df = prices_hk.get("105T", session, session, anchor="open")
        assert df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 1
        assert xhkg.session_break_end(session) in df.index.left
        assert xhkg.session_break_end(session) in df.index.right

        # verify warns
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices_hk.get("106T", session, session, anchor="open")
        assert not df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 2
        assert xhkg.session_break_end(session) in df.index.left
        assert xhkg.session_break_end(session) in df.index.right

    def test_cal_break_ignored(
        self, sessions_xnys_xhkg_xlon_H1, prices_with_break, sessions_xnys_xhkg_xlon_T1
    ):
        """Test circumstances when indices not aligned to pm open.

        Tests:
            When source does not anchor prices on afternoon open and data not
            available at a lower interval from which indices respecting break
            can be evaluated.

            When indice of a trading index evaluated against the calendar that
            observes the break *partially* overlaps an indice of a trading index
            evaluated against another calendar.

        """
        session_h1, _ = sessions_xnys_xhkg_xlon_H1
        start_session, end_session = sessions_xnys_xhkg_xlon_T1

        # Verify indices not aligned with pm open when source data does not
        # anchor afternoon indices on the pm open.
        prices_hk = prices_with_break
        xhkg = prices_hk.calendar_default
        interval = TDInterval.H1

        df = prices_hk.get("1H", session_h1, session_h1, anchor="open", tzout=pytz.UTC)
        assert df.pt.has_regular_interval
        assert (df.index.left.minute == 30).all()
        assert (df.index.right.minute == 30).all()
        start = xhkg.session_open(session_h1)
        end = xhkg.session_close(session_h1) + pd.Timedelta(30, "T")
        index = self.create_single_index(start, end, interval)
        assert_index_equal(df.index, index)

        # Verify indices not aligned with pm open when union of trading calendars
        # of associated calendars has at least one overlapping indice
        prices_hk_lon = m.PricesYahoo("9988.HK, AZN.L")
        interval = TDInterval.T50
        df = prices_hk_lon.get(
            "50T",
            start_session,
            end_session,
            lead_symbol="9988.HK",
            anchor="open",
            tzout=pytz.UTC,
        )[
            :9
        ]  # only take what's necessary to prove the point
        assert df.pt.last_ts > xhkg.session_break_end(start_session)
        starts, ends = [], []
        start = xhkg.session_open(start_session)
        starts.append(start)
        end = start + (interval * 3)
        ends.append(end)
        start = end + interval
        starts.append(start)
        end = start + (interval * 6)
        ends.append(end)
        index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, index)

    def test_openend(self, prices_us, session_length_xnys, session_length_xlon):
        """Test effect of "openend" option.

        Tests "maintain" and "shorten" where symbols both trade and do not
        trade after an unaligned session close.
        Tests "shorten" acts as "maintain" if data unavailable to evalute
        shorter final indice.
        """
        prices = prices_us
        xnys = prices.calendar_default

        session = get_valid_conforming_sessions(
            prices, prices.bis.T5, [xnys], [session_length_xnys], 1
        )[0]
        start, end = xnys.session_open_close(session)

        # verify maintain when no symbol trades after unaligned close
        df = prices.get("1H", start, end, openend="maintain")
        assert df.pt.has_regular_interval
        half_hour = pd.Timedelta(30, "T")
        exp_end = end + half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 7)

        # verify maintain is default
        df_ = prices.get("1H", start, end)
        assert_frame_equal(df, df_)

        # verify shorten
        df = prices.get("1H", start, end, openend="shorten")
        assert not df.pt.has_regular_interval
        last_indice = df.index[-1]
        assert last_indice.right == end
        assert last_indice.left == end - half_hour
        assert last_indice.length == half_hour
        # verify rest of table as expected
        exp_end = end - half_hour
        assertions_intraday(df[:-1], prices.bis.H1, prices, start, exp_end, 6)

        # verify maintain when no symbol trades after unaligned close
        prices = m.PricesYahoo(["AZN.L", "BTC-USD"], lead_symbol="AZN.L")
        xlon = prices.calendars["AZN.L"]
        session = get_valid_conforming_sessions(
            prices, prices.bis.T5, [xnys], [session_length_xnys], 1
        )[0]
        start, end = xlon.session_open_close(session)

        # verify maintain when symbol trades after unaligned close
        df = prices.get("1H", start, end, openend="maintain")
        assert df.pt.has_regular_interval
        exp_end = end - half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 8)

        # verify shorten
        df = prices.get("1H", start, end, openend="shorten")
        assert not df.pt.has_regular_interval
        last_indice = df.index[-1]
        assert last_indice.right == end
        assert last_indice.left == end - half_hour
        assert last_indice.length == half_hour
        # verify rest of table as expected
        exp_end = end - half_hour
        assertions_intraday(df[:-1], prices.bis.H1, prices, start, exp_end, 8)

        # verify shorten as maintain when data not available to crete short indice
        session = get_valid_conforming_sessions(
            prices, prices.bis.H1, [xnys], [session_length_xnys], 1
        )[0]
        start, end = xlon.session_open_close(session)
        df = prices.get("1H", start, end, openend="shorten")
        assert df.pt.has_regular_interval
        exp_end = end - half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 8)

    def test_daily(self, prices_us, one_day):
        """Test indices where interval is a multiple of days.

        Tests for symbols associated with single and multiple calendars.
        """
        prices = prices_us
        prices_us_247 = m.PricesYahoo("MSFT BTC-USD")
        xnys = prices.calendar_default

        period_start = pd.Timestamp("2021-12-01")
        period_end = pd.Timestamp("2021-12-31")
        assert xnys.is_session(period_end)
        exp_end = period_end + one_day

        def assertions(prices: m.PricesYahoo, lead: str):
            for interval in (13, 8, 7, 5, 4, 3, 2):
                interval_str = str(interval) + "D"
                df = prices.get(
                    interval_str,
                    period_start,
                    period_end,
                    lead_symbol=lead,
                )
                cal = prices.calendars[lead]
                assert df.pt.last_ts == exp_end

                for i in reversed(range(len(df))):
                    start, end = df.index[i].left, df.index[i].right
                    if not cal.is_session(end):
                        # TODO xcals 4.0 lose wrapper
                        end = helpers.to_tz_naive(cal.date_to_session(end, "next"))
                    assert start == helpers.to_tz_naive(
                        cal.session_offset(end, -interval)
                    )

                # verify sessions missing to left of table start are fewer in number
                # than one interval
                excess = len(cal.sessions_in_range(period_start, start - one_day))
                assert excess < interval

                # verify indices contiguous
                assert_index_equal(df.index.left[1:], df.index.right[:-1])

        assertions(prices, "MSFT")
        assertions(prices_us_247, "MSFT")
        assertions(prices_us_247, "BTC-USD")

    def test_monthly(self, prices_us, prices_us_lon, monkeypatch):
        """Test indices where interval is a multiple of months.

        Tests for symbols associated with single and multiple calendars.
        """
        for prices in (prices_us, prices_us_lon):
            # Verify with `start`
            df = prices.get("3M", start="2021-03-01", months=7)
            assert len(df) == 2
            start = pd.Timestamp("2021-03-01")
            end = pd.Timestamp("2021-09-01")
            index = self.create_single_index(start, end, DOInterval.M3)
            assert_index_equal(df.index, index)

            df_ = prices.get("3M", start="2021-03-01", months=6)
            assert_frame_equal(df, df_)

            df_ = prices.get("3M", start="2021-02-02", months=7)
            assert_frame_equal(df, df_)

            # Verify with `end`
            df = prices.get("2M", end="2021-12-31", months=7)
            assert len(df) == 3
            start = pd.Timestamp("2021-07-01")
            end = pd.Timestamp("2022-01-01")
            index = self.create_single_index(start, end, DOInterval.M2)
            assert_index_equal(df.index, index)

            df_ = prices.get("2M", end="2021-12-31", months=6)
            assert_frame_equal(df, df_)

            df_ = prices.get("2M", end="2022-01-30", months=7)
            assert_frame_equal(df, df_)

            # Verify `start` and `end`
            df_ = prices.get("2M", start="2021-05-02", end="2022-01-30")
            assert_frame_equal(df, df_)

            df_ = prices.get("2M", start="2021-05-02", end="2022-01-31")
            assert len(df_) == 4
            df_ = prices.get("2M", start="2021-05-01", end="2022-01-30")
            assert len(df_) == 4

            with monkeypatch.context() as m:
                # Verify now includes live index
                now = pd.Timestamp("2021-12-22 13:22", tz=pytz.UTC)
                mock_now(m, now)
                df_now = prices.get("2M", months=7)
                assert_index_equal(df_now.index, df.index)
                # now = pd.Timestamp("2021-12-15 15:00")
                # m.setattr("pandas.Timestamp.now", lambda *_, **__: now)

    def test_raises_PricesUnavailableIntervalPeriodError(self, prices_us, one_sec):
        """Test raises `errors.PricesUnavailableIntervalPeriodError`.

        Tests raises when expected, with IntradayPricesUnavailable expected
        to take precendence.

        Tests when `start` and `end` represent period less than one interval.

        Tests 'open' and 'workback' anchors independently.
        """
        prices = prices_us
        cal = prices.calendar_default

        start_session_T5, end_session_T5 = th.get_sessions_range_for_bi(
            prices, prices.bis.T5
        )
        anchors = ("open", "workback")

        prev_session = start_session_T5
        # TODO xcals 4.0 lose wrapper
        session = helpers.to_tz_naive(cal.next_session(prev_session))
        while prev_session in _blacklist or session in _blacklist:
            # TODO xcals 4.0 lose wrappers from each of following two lines
            prev_session = helpers.to_tz_naive(cal.next_session(prev_session))
            session = helpers.to_tz_naive(cal.next_session(prev_session))
            assert session < end_session_T5, "cannot get sessions for test"

        session_open = cal.session_open(session)
        prev_session_close = cal.session_close(prev_session)

        bi = TDInterval.T5
        dsi = TDInterval.T15

        # Verify for an interval of a session
        start = session_open + dsi
        end_bi = start + bi
        end_ds = start + dsi

        for anchor in anchors:
            # at bi limit
            df = prices.get(bi, start, end_bi, anchor=anchor)
            assertions_intraday(df, bi, prices, start, end_bi, 1)

            # at ds limit
            df = prices.get(dsi, start, end_ds, anchor=anchor)
            assertions_intraday(df, dsi, prices, start, end_ds, 1)

            # period less than limits
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
                prices.get(bi, start, end_bi - one_sec, anchor=anchor)
                prices.get(dsi, start, end_ds - one_sec, anchor=anchor)

        # Verify for an interval crossing sessions
        def assertions_limit(
            interval: TDInterval, start: pd.Timestamp, end: pd.Timestamp, anchor: str
        ):
            # limit for last interval of prev session and first interval of session
            df = prices.get(interval, start, end, anchor=anchor)
            assertions_intraday(df, interval, prices, start, end, 2)

            # limits for last interval / first interval only
            df = prices.get(interval, start + one_sec, end, anchor=anchor)
            assertions_intraday(df, interval, prices, session_open, end, 1)
            df = prices.get(interval, start, end - one_sec, anchor=anchor)
            assertions_intraday(df, interval, prices, start, prev_session_close, 1)

        def assert_raises(
            interval: TDInterval, start: pd.Timestamp, end: pd.Timestamp, anchor: str
        ):
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
                prices.get(interval, start + one_sec, end - one_sec, anchor=anchor)

        def assertions(
            interval: TDInterval, start: pd.Timestamp, end: pd.Timestamp, anchor: str
        ):
            assertions_limit(interval, start, end, anchor)
            assert_raises(interval, start, end, anchor)

        start = prev_session_close - bi
        end = session_open + bi
        start_ds = prev_session_close - dsi
        end_ds = session_open + dsi

        assertions(bi, start, end, "open")
        assertions(dsi, start_ds, end_ds, "open")
        assertions_limit(bi, start, end, "workback")

        # When could be PricesUnavailable or IntervalPeriod, PricesUnavailable
        # takes precedence
        with pytest.raises(errors.PricesIntradayUnavailableError):
            df = prices.get(bi, start + one_sec, end - one_sec, anchor="workback")

        # verify for workback with dsi
        # on limit
        start = prev_session_close - dsi + bi
        end = session_open + bi
        anchor = "workback"
        df = prices.get(dsi, start, end, anchor=anchor)
        assert_bounds(df, (start, end))
        assert_prices_table_ii(df, prices)
        assert len(df) == 1

        # within limit
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(dsi, start + one_sec, end, anchor=anchor)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(dsi, start, end - one_sec, anchor=anchor)

        # verify when single interval would have comprise minutes from both end of
        # previous session and start of session
        delta = pd.Timedelta(2, "T")
        start = prev_session_close - bi + delta
        end = session_open + delta

        # when on limit, could do it but only by downsampling T1 data which
        # isn't available.
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(bi, start, end, anchor="workback")

        # when within limit, no question, IntervalPeriodError, which it would be
        # even at T1.
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(bi, start + one_sec, end, anchor="workback")

        # Verify raises PeriodTooShort error on crossing session at T1
        start_session_T1, end_session_T1 = th.get_sessions_range_for_bi(
            prices, prices.bis.T1
        )
        prev_session = start_session_T1
        # TODO xcals 4.0 lose wrapper
        session = helpers.to_tz_naive(cal.next_session(prev_session))
        while prev_session in _blacklist or session in _blacklist:
            # TODO xcals 4.0 lose wrappers from each of following two lines
            prev_session = helpers.to_tz_naive(cal.next_session(prev_session))
            session = helpers.to_tz_naive(cal.next_session(prev_session))
            assert session < end_session_T1, "cannot get sessions for test"

        session_open = cal.session_open(session)
        prev_session_close = cal.session_close(prev_session)

        delta = pd.Timedelta(2, "T")
        start = prev_session_close - bi + delta
        end = session_open + delta

        # verify at limit returns as expected
        df = prices.get(bi, start, end, anchor="workback")

        assert_bounds(df, (start, end))
        assert_prices_table_ii(df, prices)
        assert len(df) == 1

        # verify that within limit raises IntervalPeriodError
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(bi, start + one_sec, end, anchor="workback")
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(bi, start, end - one_sec, anchor="workback")

    @skip_if_data_unavailable
    def test_raises_PricesUnavailableIntervalPeriodError2(
        self, prices_us, session_length_xnys, one_min
    ):
        """Test raises errors when period does not span at least one full indice.

        As for `test_raises_PricesUnavailableIntervalPeriodError` although tests
        for when parameters define period as a duration in terms of
        'minutes' and 'hours' or 'days'. When duration in terms of 'minutes' and
        'hours' tests raises `PricesUnavailableIntervalDurationError`

        Also tests effect of strict on error raised when requesting prices
        at the boundardy of data availability.
        """
        # hours/minutes
        prices = prices_us
        cal = prices.calendar_default

        session = get_valid_conforming_sessions(
            prices, prices.bis.T5, [cal], [session_length_xnys], 4
        )[-2]
        session_open = cal.session_open(session)

        _, prev_session_T1, session_T1, _ = get_valid_conforming_sessions(
            prices, prices.bis.T1, [cal], [session_length_xnys], 4
        )
        session_open_T1 = cal.session_open(session_T1)
        prev_session_close_T1 = cal.session_close(prev_session_T1)

        anchors = ("open", "workback")

        bi = TDInterval.T5
        dsi = TDInterval.T15

        # Verify for an interval of a session
        start = session_open + dsi

        for anchor in anchors:
            df = prices.get(bi, start, minutes=5, anchor=anchor)
            assertions_intraday(df, bi, prices, start, start + bi, 1)
            df = prices.get(dsi, start, minutes=15, anchor=anchor)
            assertions_intraday(df, dsi, prices, start, start + dsi, 1)

            with pytest.raises(errors.PricesUnavailableIntervalDurationError):
                prices.get(bi, start, minutes=4, anchor=anchor)
                prices.get(dsi, start, minutes=14, anchor=anchor)

        delta = pd.Timedelta(2, "T")
        end = session_open_T1 + delta

        anchor = "workback"
        # on limit
        df = prices.get(bi, end=end, minutes=5, anchor=anchor)
        assert_bounds(df, (prev_session_close_T1 - bi + delta, end))
        assert_prices_table_ii(df, prices)
        assert len(df) == 1

        with pytest.raises(errors.PricesUnavailableIntervalDurationError):
            prices.get(bi, end=end, minutes=4, anchor=anchor)

        # days
        interval = session_length_xnys
        session_close = cal.session_close(session_T1)

        # verify that when anchor 'open'
        anchor = "open"
        start = session_open_T1
        df = prices.get(interval, start, days=1, anchor=anchor)
        assertions_intraday(df, interval, prices, start, session_close, 1)
        df_ = prices.get(
            session_length_xnys + one_min,
            start,
            days=1,
            openend="shorten",
            anchor=anchor,
        )
        try:
            assert_frame_equal(df_, df)
        except AssertionError as err:
            # Let it pass if it's an issue with the volume column (yahoo API can send
            # slightly different volume data for calls with different bounds (only
            # happens when send high frequency of requests, should pass otherwise).
            if "volume" not in err.args[0]:
                raise
            print(
                "\ntest_raises_PricesUnavailableIntervalPeriodError2: volume column was"
                " not equal. Skipping check.\n"
            )

        anchor = "workback"
        end = start + dsi
        df = prices.get(interval, end=end, days=1, anchor=anchor)
        prev_session_open = cal.session_open(prev_session_T1)
        assert_bounds(df, (prev_session_open + dsi, end))
        assert_prices_table_ii(df, prices)
        assert len(df) == 1
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(session_length_xnys + one_min, end, days=1, anchor=anchor)

        # Test effect of strict on error raised
        l_limit = prices.limits[prices.bis.T5][0]
        l_limit = cal.minute_to_trading_minute(l_limit, "next")
        end = l_limit + pd.Timedelta(2, "H")
        # error depends on strict
        get_kwargs = dict(interval="3H", end=end, hours=6, anchor="workback")
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(**get_kwargs, strict=True)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(**get_kwargs, strict=False)

    # ------------------- Tests related to data availability ------------------

    def test_raises_PricesIntradayUnavailableError(self, prices_us):
        prices = prices_us
        start_T5 = th.get_sessions_range_for_bi(prices, prices.bis.T5)[0]
        session_T5 = start_T5
        stricts = (True, False)
        priorities = ("end", "period")
        anchors = ("open", "workback")

        # verify raises when base interval data unavailable over all period.

        msg_pass_strict = "Consider passing `strict`"
        # wrap in AssertionError expectation as should not match the 'Consider passing
        # `strict`' when data unavailable over full period.
        with pytest.raises(AssertionError):
            # `strict` and `priority` should have no effect
            for strict, priority, anchor in itertools.product(
                stricts, priorities, anchors
            ):
                with pytest.raises(
                    errors.PricesIntradayUnavailableError, match=msg_pass_strict
                ):
                    prices.get(
                        "3T",
                        session_T5,
                        session_T5,
                        strict=strict,
                        priority=priority,
                        anchor=anchor,
                    )

        # verify raises when base interval only available over end of period.
        with pytest.raises(
            errors.PricesIntradayUnavailableError, match=msg_pass_strict
        ):
            # `priority` should make no difference
            for priority in priorities:
                prices.get("3T", session_T5, priority=priority)

        # although returns data from limit if strict False
        df = prices.get("3T", session_T5, strict=False)
        assert df.pt.first_ts >= prices.limits[prices.bis.T1][0]
        assert df.pt.interval == TDInterval.T3

    def test_raises_LastIndiceInaccurateError(self, prices_us):
        prices = prices_us
        xnys = prices.calendar_default

        start_T1, end_T1 = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        start_T1 = get_valid_session(start_T1, xnys, "next", end_T1)
        end_T1 = get_valid_session(end_T1, xnys, "previous", start_T1)

        start_T5_, end_T5 = th.get_sessions_range_for_bi(prices, prices.bis.T5)
        start_T5 = get_valid_session(start_T5_, xnys, "next", end_T5)

        start_H1_, _ = th.get_sessions_range_for_bi(prices, prices.bis.H1)

        # TODO xcals 4.0 lose the wrappers from following two lines
        start_T5_oob = helpers.to_tz_naive(xnys.session_offset(start_T5_, -2))
        start_H1_oob = helpers.to_tz_naive(xnys.session_offset(start_H1_, -2))

        limit_T1 = prices.limits[prices.bis.T1][0]
        limit_T5 = prices.limits[prices.bis.T5][0]
        limit_H1 = prices.limits[prices.bis.H1][0]

        # period end that can only be represented with T1 or T5 data
        end = xnys.session_close(end_T1) - pd.Timedelta(15, "T")

        # verify data available if requested period falls within bounds of
        # available T5 data
        df = prices.get(start=start_T5, end=end)
        assert df.pt.first_ts >= limit_T5
        assert df.pt.interval == prices.bis.T5

        kwargs = dict(start=start_T5_oob, end=end)
        # although raises error if start is before 5T limit
        with pytest.raises(errors.LastIndiceInaccurateError):
            df = prices.get(**kwargs)

        # but will return if only want end of data
        df = prices.get(**kwargs, strict=False)
        assert df.pt.first_ts >= limit_T5
        assert df.pt.interval == prices.bis.T5
        assert df.pt.last_ts == end

        # or if not bothered with best-representing the period end
        df = prices.get(**kwargs, priority="period")
        assert df.pt.first_ts < limit_T5
        assert df.pt.interval == prices.bis.H1
        assert df.pt.last_ts < end

        # verify returns composite table / that composite overrides priority
        df = prices.get(**kwargs, composite=True, priority="period")
        intervals = df.pt.indices_length.index
        assert len(intervals) == 2
        assert prices.bis.H1 in intervals
        assert prices.bis.T5 in intervals
        assert df.pt.first_ts < limit_T5
        assert df.pt.last_ts == end

        # verify can daily / intraday composite table if intraday data only
        # available at period end.
        df = prices.get(start=start_H1_oob, end=end, composite=True)
        assert not df.pt.daily_part.empty
        assert not df.pt.intraday_part.empty
        assert df.pt.first_ts < limit_H1
        assert df.pt.last_ts == end

        # verify 'greatest possible accuracy' concerned with greatest accuracy
        # given the data that's available.
        # set end to time that can only be represented by T1, although for which
        # smallest interval for which data is available is T5
        end = xnys.session_close(start_T5) - pd.Timedelta(3, "T")
        df = prices.get(start=start_T5, end=end)
        assert df.pt.interval == prices.bis.T5
        assert df.pt.last_ts == end - pd.Timedelta(2, "T")

        # verify error not raised when interval passed and anchor "open",
        # regardless of final indice not representing period end
        df_ = prices.get("5T", start=start_T5, end=end)
        assert_frame_equal(df, df_)

        # verify that raises errors when anchor="workback"
        # set period such that T1 data only available over period end and
        # period end can only be served with T1 data.
        end = xnys.session_close(start_T1) - pd.Timedelta(3, "T")
        # whilst prices available when anchor "open"
        df = prices.get("10T", start=start_T5, end=end, anchor="open")
        assert df.pt.interval == TDInterval.T10
        # verify not when anchor is "workback"
        with pytest.raises(errors.LastIndiceInaccurateError):
            prices.get("10T", start=start_T5, end=end, anchor="workback")

        # verify will return later part of period if strict False
        df = prices.get("10T", start=start_T5, end=end, anchor="workback", strict=False)
        assert df.pt.last_ts == end
        assert df.pt.first_ts >= limit_T1
        assert df.index[0].length == TDInterval.T10
        assert df.index[-1].length == TDInterval.T10

        # verify will return full period if priority "period", althrough with
        # lesser end accuracy
        df = prices.get(
            "10T", start=start_T5, end=end, anchor="workback", priority="period"
        )
        assert df.pt.last_ts < end
        assert df.pt.first_ts < limit_T1
        assert df.index[0].length == TDInterval.T10
        assert df.index[-1].length == TDInterval.T10

    def test_raises_direct_errors(
        self,
        prices_us,
    ):
        """Test get() directly raises expected errors.

        Tests expected errors that are raised directly by get().

        `LastIndiceInaccurateError`
        Tests that when period end can be represented with intraday
        data, although period start can only be met with daily data,
        get() directly raises `LastIndiceInaccurateError`. Tests
        message mathes expected given that the version of message
        raised is unique to this circumstance.

        `PricesUnavailableInferredIntervalError`
        Tests raised when expected. Tests mesasge matches expected
        given that error only raised by get().
        """
        prices = prices_us
        xnys = prices.calendar_default
        start_H1 = th.get_sessions_range_for_bi(prices, prices.bis.H1)[0]
        # TODO xcals 4.0 lose wrapper
        start_H1_oob = helpers.to_tz_naive(xnys.session_offset(start_H1, -2))
        start_T5 = th.get_sessions_range_for_bi(prices, prices.bis.T5)[0]

        end = xnys.session_close(start_T5) - pd.Timedelta(3, "T")
        end = end.astimezone(prices.tz_default)

        with pytest.raises(errors.LastIndiceInaccurateError):
            prices.get(start=start_H1_oob, end=end)

        # now prices.gpp available...match message
        bi = prices.bis.T5
        bi_limit = prices.limits[bi][0]
        earliest_minute = xnys.minute_to_trading_minute(bi_limit, "next")
        drg = prices.gpp.drg_intraday_no_limit
        drg.interval = bi

        msg = re.escape(
            "Full period not available at any synchronised intraday base interval."
            " The following base intervals could represent the end indice with the"
            " greatest possible accuracy although have insufficient data available"
            f" to cover the full period:\n\t{[bi]}.\nThe earliest minute from which"
            f" data is available at 0 days 00:05:00 is {earliest_minute}, although"
            " at this base interval the requested period evaluates to"
            f" {drg.daterange[0]}."
            f"\nPeriod evaluated from parameters: {prices.gpp.pp_raw}."
            "\nData that can express the period end with the greatest possible accuracy"
            f" is available from {earliest_minute}. Pass `strict` as False to return"
            " prices for this part of the period."
            f"\nAlternatively, consider creating a composite table (pass `composite`"
            " as True) or passing `priority` as 'period'."
        )
        with pytest.raises(errors.LastIndiceInaccurateError, match=msg):
            prices.get(start=start_H1_oob, end=end)

        # Verify raises `PricesUnvailableInferredIntervalError` when interval
        # inferred as intraday although only daily data available.

        # Verify returns daily data on limit (6 sessions different)
        start = helpers.to_tz_naive(xnys.session_offset(start_H1_oob, -5))
        df = prices.get(start=start, end=start_H1_oob)
        assertions_daily(df, prices, start, start_H1_oob)

        # verify raises error when interval will be inferred as intraday
        start = helpers.to_tz_naive(xnys.session_offset(start_H1_oob, -4))
        pp = get_pp(start=start, end=start_H1_oob)
        msg = re.escape(
            "Given the parameters receieved, the interval was inferred as"
            " intraday although the request can only be met with daily data."
            " To return daily prices pass `interval` as a daily interval,"
            " for example '1D'.\nNB. The interval will only be inferred as"
            " daily if `end` and `start` are defined (if passed) as sessions"
            " (timezone naive and with time component as 00:00) and any"
            " duration is defined in terms of either `days` or `weeks`,"
            " `months` and `years`. Also, if both `start` and `end` are"
            " passed then the distance between them should be no less than"
            f" 6 sessions.\nPeriod parameters were evaluted as {pp}"
        )
        with pytest.raises(errors.PricesUnvailableInferredIntervalError, match=msg):
            prices.get(start=start, end=start_H1_oob)

    # -------------- Tests related to post-processing options -------------

    def test_tzout(self, prices_us_lon_hkg, sessions_xnys_xhkg_xlon_T1):
        """Test `tzout` option.

        Tests for expected default behaviour and overriding default.
        """
        prices = prices_us_lon_hkg
        session = sessions_xnys_xhkg_xlon_T1[-1]

        tzhkg = prices.timezones["9988.HK"]
        tzny = prices.timezones["MSFT"]
        tzlon = prices.timezones["AZN.L"]

        kwargs_daily = dict(interval="1D", end=session, days=10)
        # verify tzout for daily data returns as tz-naive
        assert prices.get(**kwargs_daily).index.tz is None
        assert prices.get(**kwargs_daily, tzout=tzhkg).index.tz is None
        # unless tz is UTC
        assert prices.get(**kwargs_daily, tzout="UTC").index.tz is pytz.UTC

        # verify `tzout` defaults to timezone that `tzin` evaluates to
        kwargs_intraday = dict(end=session, days=2)
        df = prices.get(**kwargs_intraday)
        assert df.index.left.tz == df.index.right.tz == df.pt.tz == tzny
        assert prices.get(**kwargs_intraday, lead_symbol="9988.HK").pt.tz == tzhkg
        # verify tzin overrides `lead_symbol`
        assert (
            prices.get(**kwargs_intraday, lead_symbol="9988.HK", tzin="AZN.L").pt.tz
            == tzlon
        )

        # verify passing `tzout` overrides default
        kwargs_intraday.update({"lead_symbol": "MSFT", "tzin": "AZN.L"})
        assert prices.get(**kwargs_intraday, tzout="9988.HK").pt.tz == tzhkg
        assert prices.get(**kwargs_intraday, tzout="Asia/Hong_Kong").pt.tz == tzhkg
        assert prices.get(**kwargs_intraday, tzout=tzhkg).pt.tz == tzhkg

    def test_post_processing_params(
        self, sessions_xnys_xhkg_xlon_T1, prices_us_lon_hkg, prices_us
    ):
        """Test post-processing params implemented via .pt.operate."""
        prices = prices_us_lon_hkg
        session = sessions_xnys_xhkg_xlon_T1[-1]

        kwargs = dict(end=session, days=2)
        df = prices.get(**kwargs)

        def assertion(**options):
            rtrn = prices.get(**kwargs, **options)
            assert_frame_equal(rtrn, df.pt.operate(**options))

        # assert last row has a missing value (which should be the case as
        # lead is MSFT and xnys and xhkg do not overlap).
        assert df.iloc[-1].isna().any()
        assertion(fill="ffill")
        assertion(include="MSFT")
        assertion(exclude=["MSFT, AZN.L"])
        assertion(side="right")
        assertion(close_only=True)
        # verify for passing multiple options
        assertion(close_only=True, fill="bfill", side="left", exclude="9988.HK")

        prices = prices_us
        df = prices.get(**kwargs)
        rtrn = prices.get(**kwargs, lose_single_symbol=True)
        assert_frame_equal(rtrn, df.pt.operate(lose_single_symbol=True))


def get_current_session(calendar: xcals.ExchangeCalendar) -> pd.Timestamp:
    today = helpers.now(intervals.BI_ONE_DAY)
    if calendar.is_session(today):
        if helpers.now() >= calendar.session_open(today):
            return today
        else:
            # TODO xcals 4.0 lose wrapper
            return helpers.to_tz_naive(calendar.previous_session(today))
    else:
        # TODO xcals 4.0 lose wrapper
        return helpers.to_tz_naive(calendar.date_to_session(today, "previous"))


def test_request_all_prices(prices_us_lon, one_min):
    prices = prices_us_lon
    xnys = prices.calendar_default

    now = helpers.now()
    limits = prices.limits
    prices.request_all_prices()

    # verify local tables as expected
    for bi in prices.bis_intraday:
        table = prices._pdata[bi]._table

        if bi is prices.bis.H1:
            # indices are not aligned at H1
            assert table is None
            continue

        first_ts = table.pt.first_ts
        limit = limits[bi][0]
        earliest_minute = xnys.minute_to_trading_minute(limit, "next")
        # extra minute for processing between getting limit and requesting prices
        assert earliest_minute <= first_ts <= earliest_minute + bi + one_min

        last_ts = table.pt.last_ts
        if xnys.is_trading_minute(now):
            latest_minute = now
        else:
            latest_minute = xnys.previous_close(now)
        # extra min for processing between getting latest_minute and requesting prices
        assert latest_minute - bi <= last_ts <= latest_minute + bi + one_min

    bi = prices.bis.D1
    table = prices._pdata[bi]._table

    first_ts = table.pt.first_ts
    limit = limits[bi][0]
    assert limit == first_ts

    last_ts = table.pt.last_ts
    latest_session = get_current_session(xnys)
    try:
        assert last_ts == latest_session
    except AssertionError:
        if current_session_in_blacklist(prices.calendars_unique):
            latest_session = get_valid_session(latest_session, xnys, "previous")
            assert last_ts == latest_session
        else:
            raise


@skip_if_fails_and_today_blacklisted(["XLON", "XNYS"])
def test_session_prices(prices_us_lon, one_day):
    prices = prices_us_lon
    f = prices.session_prices
    xlon = prices.calendars["AZN.L"]
    xnys = prices.calendars["MSFT"]

    def assertions(rtrn: pd.DataFrame, session: pd.Timestamp):
        assert_price_table(rtrn, prices)
        assert len(rtrn) == 1
        assert rtrn.pt.first_ts == rtrn.pt.last_ts == session

    # verify for known session
    session = pd.Timestamp("2021-03-22")
    rtrn = f(session)
    assertions(rtrn, session)
    # verify returns stacked
    assert_frame_equal(rtrn.pt.stacked, f(session, stack=True))

    # verify for session of only some symbols
    xnys_sessions = xnys.sessions_in_range("2020", "2022")
    xlon_session = xlon.sessions_in_range("2020", "2022")
    # TODO xcals 4.0 lose wrapper
    session = helpers.to_tz_naive(xnys_sessions.difference(xlon_session))[0]
    rtrn = f(session)
    assert rtrn.isna().any(axis=None)
    assertions(rtrn, session)

    # verify raises error when date passed is not a session of any cal
    date = pd.Timestamp("2022-04-24")
    match = f"{date} is not a session of any associated calendar."
    with pytest.raises(ValueError, match=match):
        f(date)  # a sunday

    # verify raises errors if `session` oob
    today = prices.cc.minute_to_sessions(helpers.now(), "previous")[-1]
    limit_right = prices.cc.date_to_session(today, "previous")
    df = f(limit_right)  # at limit
    assert df.index[0] == limit_right
    oob_right = limit_right + one_day
    with pytest.raises(errors.DatetimeTooLateError):
        f(oob_right)

    limit_left = prices.limits[prices.bis.D1][0]
    df = f(limit_left)  # at limit
    assert df.index[0] == limit_left
    oob_left = limit_left - one_day
    with pytest.raises(errors.DatetimeTooEarlyError):
        f(oob_left)

    # verify for current session.
    # NB placed at end as will fail if today blacklisted - test all can before skips.
    current_sessions = [get_current_session(cal) for cal in prices.calendars_unique]
    current_session = max(current_sessions)
    rtrn = f(None)
    assertions(rtrn, current_session)
    # verify returns current session stacked
    rtrn_stacked = f(stack=True)
    # don't verify values as will vary when trading open (values verified for static
    # return later).
    assert_index_equal(rtrn.pt.stacked.columns, rtrn_stacked.columns)
    assert_index_equal(rtrn.pt.stacked.index.levels[1], rtrn_stacked.index.levels[1])


def test__date_to_session(prices_us_lon_hkg):
    prices = prices_us_lon_hkg
    f = prices._date_to_session
    # from knowledge of schedule
    date = pd.Timestamp("2021-12-25")
    assert f(date, "earliest", "next") == pd.Timestamp("2021-12-27")
    assert f(date, "latest", "next") == pd.Timestamp("2021-12-29")
    assert f(date, "earliest", "previous") == pd.Timestamp("2021-12-23")
    assert f(date, "latest", "previous") == pd.Timestamp("2021-12-24")


@skip_if_fails_and_today_blacklisted(["XLON", "XNYS", "XHKG"])
def test_close_at(prices_us_lon_hkg, one_day, monkeypatch):
    prices = prices_us_lon_hkg
    xhkg = prices.calendars["9988.HK"]

    df = prices.get("1D", "2021-12-23", "2021-12-27", fill="ffill", close_only=True)

    # all following from kwowledge of schedules...

    # verify for 'day' on which all calendars open
    rtrn = prices.close_at("2021-12-23")
    assert_frame_equal(rtrn, df.iloc[[0]])

    # verify for 'day' for which only one calendar open
    rtrn = prices.close_at("2021-12-27")
    assert_frame_equal(rtrn, df.iloc[[-1]])

    # verify for 'day' on which all calendars closed
    rtrn = prices.close_at("2021-12-25")
    expected = df["2021-12-24":"2021-12-24"]
    assert_frame_equal(rtrn, expected)

    # verify raises errors if `date` oob
    today = prices.cc.minute_to_sessions(helpers.now(), "previous")[-1]
    limit_right = prices.cc.date_to_session(today, "previous")
    df_limit_right = prices.close_at(limit_right)  # at limit
    assert df_limit_right.index[0] == limit_right
    oob_right = limit_right + one_day
    with pytest.raises(errors.DatetimeTooLateError):
        prices.close_at(oob_right)

    limit_left = prices.limits[prices.bis.D1][0]
    df_limit_left = prices.close_at(limit_left)  # at limit
    assert df_limit_left.index[0] == limit_left
    oob_left = limit_left - one_day
    with pytest.raises(errors.DatetimeTooEarlyError):
        prices.close_at(oob_left)

    # verify considers the latest session of all symbols, NOT just the lead_symbol
    # reset prices
    with monkeypatch.context() as monkey:
        mock_today = pd.Timestamp("2021-12-23")
        now = xhkg.session_close(mock_today) - pd.Timedelta(1, "H")
        mock_now(monkey, now)

        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        # now during XHKG open and MSFT lead symbol so daily table to 'now'
        # should not inlcude session for 23rd
        assert prices.get("1D", days=10).pt.last_ts == pd.Timestamp("2021-12-22")
        # verify that close_at DOES include the 23rd as HKG is open
        rtrn = prices.close_at()
        # given that, notwithstanding time before XNYS open, data is available for
        # all for the 23rd, return should be as if end of 23rd.
        assert_frame_equal(rtrn, df.iloc[[0]], check_freq=False)

    # verify not passing a session returns as at most recent date.
    # NB placed at end as will fail if today blacklisted - test all can before skips.
    current_sessions = [get_current_session(cal) for cal in prices.calendars_unique]
    current_session = max(current_sessions)
    assert prices.close_at().index[0] == current_session


class TestPriceAt:
    """Tests for `price_at` and immediate dependencies."""

    # pylint: disable=self-assigning-variable

    @staticmethod
    def assert_price_at_rtrn_format(table: pd.DataFrame, df: pd.DataFrame):
        expected_columns = pd.Index(table.pt.symbols, name="symbol")
        assert len(df) == 1
        assert_index_equal(expected_columns, df.columns, check_order=False)

    @staticmethod
    def get_cell(
        table: pd.DataFrame,
        s: str,
        session: str | pd.Timestamp,
        column: typing.Literal["open", "close"],
    ) -> float:
        return table.loc[session, (s, column)]

    def assertions(
        self,
        table: pd.DataFrame,
        df: pd.DataFrame,
        indice: pd.Timestamp,
        values: dict[str, tuple[pd.Timestamp, typing.Literal["open", "close"]]],
        tz=UTC,
    ):
        self.assert_price_at_rtrn_format(table, df)
        assert df.index[0] == indice
        assert df.index.tz is tz
        for s, (session, col) in values.items():
            try:
                assert df[s][0] == self.get_cell(table, s, session, col)
            except AssertionError:
                # Can be due to inconsistency of return from yahoo API. Ignore if
                # occurrences are rare.
                val_df, val_table = df[s][0], self.get_cell(table, s, session, col)
                diff = abs((val_table - val_df) / val_df)
                if not diff < 0.03:
                    raise
                caller = inspect.stack()[1].function
                print(
                    f"\n{caller}: letting df_val == table_val assertion pass with"
                    f" rel diff {diff:.2f}%.\n"
                )

    def test_oob(self, prices_us_lon_hkg, one_min):
        """Test raises errors when minute out-of-bounds.

        Also tests does return at limits.
        """
        prices = prices_us_lon_hkg

        # verify raises error if `minute` oob
        limit_left_session = prices.limits[prices.bis.D1][0]
        limit_left = max(
            [c.session_open(limit_left_session) for c in prices.calendars_unique]
        )
        df_left_limit = prices.price_at(limit_left)  # at limit
        assert df_left_limit.index[0] == limit_left
        with pytest.raises(errors.DatetimeTooEarlyError):
            prices.price_at(limit_left - one_min)

        limit_right = now = helpers.now()
        if not current_session_in_blacklist(prices.calendars_unique):
            df_limit_right = prices.price_at(limit_right, UTC)  # at limit
            current_minute = prices.cc.minute_to_trading_minute(now, "previous")
            if not prices.cc.is_open_on_minute(now):
                current_minute += one_min
            assert df_limit_right.index[0] == current_minute
        with pytest.raises(errors.DatetimeTooLateError):
            limit_right = now = helpers.now()  # reset
            prices.price_at(limit_right + one_min)

    def test__price_at_from_daily(self, prices_us_lon_hkg, one_min, monkeypatch):
        prices = prices_us_lon_hkg
        msft, astra, ali = "MSFT", "AZN.L", "9988.HK"
        xnys = prices.calendars[msft]
        xlon = prices.calendars[astra]
        xhkg = prices.calendars[ali]

        table = prices.get("1D", "2021-12-22", "2021-12-29")

        # reset prices
        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        f = prices._price_at_from_daily

        # following assertions from knowledge of schedule

        # two consecutive sessions
        session = pd.Timestamp("2021-12-23")
        prev_session = pd.Timestamp("2021-12-22")

        assert xhkg.session_close(session) == xlon.session_open(session)

        # prior to xhkg open
        minute = xhkg.session_open(session) - one_min
        df = f(minute, UTC)
        indice = xnys.session_close(prev_session)
        values = {
            msft: (prev_session, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg.session_open(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (prev_session, "close"),
            astra: (prev_session, "close"),
            ali: (session, "open"),
        }
        self.assertions(table, df, indice, values)

        # prior to xlon open and xhkg close
        minute = xlon.session_open(session) - one_min
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

        # xlon open
        minute = xlon.session_open(session)
        df = f(minute, UTC)
        indice = xlon.session_open(session)
        values = {
            msft: (prev_session, "close"),
            astra: (session, "open"),
            ali: (session, "close"),
        }
        self.assertions(table, df, indice, values)

        # prior to xnys open
        minute = xnys.session_open(session) - one_min
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys.session_open(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (session, "open"),
            astra: (session, "open"),
            ali: (session, "close"),
        }
        self.assertions(table, df, indice, values)

        # prior to xnys close
        minute = xnys.session_close(session) - one_min
        df = f(minute, UTC)
        indice = xlon.session_close(session)
        values = {
            msft: (session, "open"),
            astra: (session, "close"),
            ali: (session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys.session_close(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (session, "close"),
            astra: (session, "close"),
            ali: (session, "close"),
        }
        self.assertions(table, df, indice, values)

        # verify for times on 27 when only xnys open
        # prev_session for msft is 23, for the others 24
        session = pd.Timestamp("2021-12-27")
        prev_session_msft = pd.Timestamp("2021-12-23")
        prev_session = pd.Timestamp("2021-12-24")

        # prior to xnys open
        minute = xnys.session_open(session) - one_min
        df = f(minute, UTC)
        indice = xlon.session_close(prev_session)
        values = {
            msft: (prev_session_msft, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys.session_open(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (session, "open"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys.session_close(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (session, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # verify for a time on 25 when all calendars closed
        # also verify changing tz for other value
        minute = pd.Timestamp("2021-12-25 13:00", tz=UTC)
        tz = prices.tz_default
        df = f(minute, tz)
        indice = xlon.session_close(prev_session)
        values = {
            msft: (prev_session_msft, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values, tz)

        # verify for minute as None. Repeat example for 27
        # reset prices
        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        f = prices._price_at_from_daily

        # now prior to xnys open
        now = xnys.session_open(session) - one_min
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = xlon.session_close(prev_session)
        values = {
            msft: (prev_session_msft, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # now as xnys open
        now = xnys.session_open(session)
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            msft: (session, "close"),  # close as latest price
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # now as after xnys open, verify indice reflects 'now' as live session
        now = xnys.session_open(session) + pd.Timedelta(22, "T")
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            msft: (session, "close"),  # close as latest price
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        now = xnys.session_close(session)
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            msft: (session, "close"),
            astra: (prev_session, "close"),
            ali: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

    def test_timezone(self, one_min):
        """Test tz parameter processed as mptypes.PricesTimezone."""
        prices = m.PricesYahoo("MSFT, 9988.HK", lead_symbol="MSFT")
        f = prices.price_at
        xnys = prices.calendars["MSFT"]
        xhkg = prices.calendars["9988.HK"]

        table = prices.get("1D", "2019-12-11", "2019-12-12")

        # following assertions from knowledge of schedule
        # two consecutive sessions (of both calendars) that can only be
        # served with daily data
        session = pd.Timestamp("2019-12-12")
        prev_session = pd.Timestamp("2019-12-11")

        minute = xhkg.session_open(session) - one_min
        indice = xnys.session_close(prev_session)
        values = {
            "MSFT": (prev_session, "close"),
            "9988.HK": (prev_session, "close"),
        }

        # verify tz be default is tz associated with lead symbol
        df = f(minute)
        self.assertions(table, df, indice, values, prices.tz_default)

        df = f(minute, tz=UTC)
        self.assertions(table, df, indice, values, UTC)

        df = f(minute, tz="9988.HK")
        self.assertions(table, df, indice, values, xhkg.tz)

        df = f(minute, tz="MSFT")
        self.assertions(table, df, indice, values, xnys.tz)

        df = f(minute, tz="Europe/London")
        self.assertions(table, df, indice, values, pytz.timezone("Europe/London"))

        # verify tz also defines tz of a timezone naive minute
        minute = minute.astimezone(None) + xhkg.tz.utcoffset(session)
        # assert should return different values if minute treated as UTC
        assert minute.tz_localize(UTC) > xhkg.session_close(session)

        df = f(minute, tz=xhkg.tz)
        # although verify same return as above given that tz-naive minute
        # treated as having tz as `tz`
        self.assertions(table, df, indice, values, xhkg.tz)

    @skip_if_prices_unavailable_for_blacklisted_session(["XLON", "XNYS", "XHKG"])
    def test_daily(self, prices_us_lon_hkg, monkeypatch):
        """Test returns prices from daily as expected.

        Tests returns from daily when now

        test_single_symbol_T1_and_now
        """
        prices = prices_us_lon_hkg
        msft, astra, ali = "MSFT", "AZN.L", "9988.HK"
        xnys = prices.calendars[msft]
        xlon = prices.calendars[astra]
        xhkg = prices.calendars[ali]

        # verify minute after intraday limit returns via intraday data
        limit_id = prices.limit_intraday()
        minute = limit_id + pd.Timedelta(1, "H")
        df = prices.price_at(minute)
        assert df.notna().all(axis=None)
        assert prices._pdata[prices.bis.D1]._table is None
        # only required by assert method for symbols...
        table_ = prices._pdata[prices.bis.T5]._table
        self.assert_price_at_rtrn_format(table_, df)

        # verify minute prior to intraday limit returns via _price_at_from_daily
        minute = xnys.previous_close(limit_id) - pd.Timedelta(1, "H")
        session_xnys = helpers.to_tz_naive(xnys.minute_to_session(minute))
        session_xhkg = helpers.to_tz_naive(xhkg.minute_to_session(minute, "previous"))
        session_xlon = helpers.to_tz_naive(xlon.minute_to_session(minute, "previous"))
        table = prices.get("1D", end=session_xnys, days=10)

        prices = m.PricesYahoo(
            "MSFT, AZN.L, 9988.HK", lead_symbol="MSFT"
        )  # reset prices
        f = prices.price_at
        df = f(minute, UTC)
        self.assert_price_at_rtrn_format(table, df)
        indice = max(
            [
                xlon.previous_close(minute),
                xhkg.previous_close(minute),
                xnys.session_open(session_xnys),
            ]
        )
        values = {
            msft: (session_xnys, "open"),
            astra: (session_xlon, "close"),
            ali: (session_xhkg, "close"),
        }
        self.assertions(table, df, indice, values)

        # verify None served via `_prices_at_from_daily` (use data frm previous example)
        mock_now(monkeypatch, minute)
        df = f(None, UTC)
        self.assert_price_at_rtrn_format(table, df)
        indice = minute
        values = {
            msft: (session_xnys, "close"),
            astra: (session_xlon, "close"),
            ali: (session_xhkg, "close"),
        }
        self.assertions(table, df, indice, values)
        assert df.index[0] == minute  # as cal open, minute will be 'now'

    def test_single_symbol_T1_and_now(self, session_length_xlon, monkeypatch, one_min):
        """Tests single symbol with for data available at all bi.

        Tests both sides of edges of notable changes in expected values.

        Also tests that `now` and within min_delay of now returns daily
        close, whilst immediately prior to min_delay returns intraday data.
        """
        prices = m.PricesYahoo("AZN.L")
        xlon = prices.calendar_default
        astra = "AZN.L"

        delta = pd.Timedelta(33, "T")

        bi = prices.bis.T1
        session_prev, session = get_valid_conforming_sessions(
            prices, bi, [xlon], [session_length_xlon], 2
        )
        session_before = prices.cc.previous_session(session_prev)

        prev_close = xlon.session_close(session_prev)
        open_ = xlon.session_open(session)
        close = xlon.session_close(session)
        open_next = xlon.session_open(xlon.next_session(session))

        table = prices.get("1T", session_before, session, tzout=pytz.UTC)
        tableD1 = prices.get("1D", session_before, session)

        # reset prices
        delay = 20
        prices = m.PricesYahoo("AZN.L", delays=delay)
        f = prices.price_at

        # prior to xlon open
        minute = open_ - one_min
        df = f(minute, UTC)
        indice = prev_close
        values = {astra: (prev_close - bi, "close")}
        self.assertions(table, df, indice, values)

        # xlon open to before close
        for minute in (open_, open_ + one_min, open_ + delta, close - one_min):
            df = f(minute, UTC)
            values = {astra: (minute, "open")}
            self.assertions(table, df, minute, values)

        # xlon close
        minute = close
        df = f(minute, UTC)
        indice = minute
        values = {astra: (close - bi, "close")}
        self.assertions(table, df, indice, values)

        # from close to before open
        for minute in (
            close,
            close + one_min,
            close + delta,
            open_next - delta,
            open_next - one_min,
        ):
            df = f(minute, UTC)
            values = {astra: (close - bi, "close")}
            self.assertions(table, df, close, values)

        # verify that from (now - min_delay) through now returns day close
        now = close - delta
        mock_now(monkeypatch, now)

        for minute in (None, now):
            df = f(minute, UTC)
            indice = now
            values = {astra: (session, "close")}
            self.assertions(tableD1, df, indice, values)

        for i in range(delay):
            minute = now - pd.Timedelta(i, "T")
            df = f(minute, UTC)
            indice = minute  # indices as requested minute
            values = {astra: (session, "close")}
            self.assertions(tableD1, df, indice, values)

        # verify that on delay limit prices return for intraday data
        minute = now - pd.Timedelta(delay, "T")
        df = f(minute, UTC)
        indice = minute
        values = {astra: (minute, "open")}
        self.assertions(table, df, indice, values)

    def test_when_all_data_available(self, one_min):
        """Test for session over which data available at all bi."""
        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        msft, astra, ali = "MSFT", "AZN.L", "9988.HK"
        xnys = prices.calendars[msft]
        xlon = prices.calendars[astra]
        xhkg = prices.calendars[ali]

        bi = prices.bis.T1
        delta = pd.Timedelta(43, "T")

        # consecutive sessions
        sessions = get_sessions_xnys_xhkg_xlon(bi, 4)
        session_before, session_prev, session, session_after = sessions

        xnys_prev_close = xnys.session_close(session_prev)
        xhkg_prev_close = xhkg.session_close(session_prev)
        xlon_prev_close = xlon.session_close(session_prev)

        xhkg_open = xhkg.session_open(session)
        xlon_open = xlon.session_open(session)
        xnys_open = xnys.session_open(session)
        xhkg_close = xhkg.session_close(session)
        xlon_close = xlon.session_close(session)
        xnys_close = xnys.session_close(session)
        xhkg_break_start = xhkg.session_break_start(session)
        xhkg_break_end = xhkg.session_break_end(session)

        # following assertions from knowledge of standard sessions
        # xlon open is one hour prior to xlon close during DST in UK
        # otherwise xlon open is same as xhkg close
        xhkg_xlon_touch = xlon_open == xhkg_close  # touch as opposed to overlap

        table = prices.get("1T", session_before, session_after, tzout=UTC)

        # reset prices
        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_prev_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg am subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg break
        minute += delta
        df = f(minute, UTC)
        indice = xhkg_break_start  # as during break not a trading minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break end
        minute = xhkg_break_end
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_end, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg pm subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # prior to xlon open
        minute = xlon_open - one_min
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # xlon open (possible xhkg close)
        minute = xlon_open
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (minute, "open"),
            ali: ali_value,
        }
        self.assertions(table, df, indice, values)

        # after xlon open
        minute += delta
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (minute, "open"),
            ali: ali_value,
        }
        self.assertions(table, df, indice, values)

        if not xhkg_xlon_touch:
            # at xhkg close
            minute = xhkg_close
            df = f(minute, UTC)
            indice = minute
            values = {
                msft: (xnys_prev_close - bi, "close"),
                astra: (minute, "open"),
                ali: (xhkg_close - bi, "close"),
            }
            self.assertions(table, df, indice, values)

        # prior to xnys open
        minute = xnys_open - one_min
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (minute, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys_open
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_open, "open"),
            astra: (minute, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xlon close
        minute = xlon_close
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (minute, "open"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys_close
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_close - bi, "close"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xnys close
        minute += delta
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

    @skip_if_data_unavailable
    def test_data_available_from_T5(self, prices_us_lon_hkg, one_min):
        """Test with minutes for which data only available at bi >= T5."""
        prices = prices_us_lon_hkg
        msft, astra, ali = "MSFT", "AZN.L", "9988.HK"
        xnys = prices.calendars[msft]
        xlon = prices.calendars[astra]
        xhkg = prices.calendars[ali]

        delta = pd.Timedelta(43, "T")
        offset = pd.Timedelta(3, "T")
        delta_reg = delta - offset

        # following assertions from knowledge of standard sessions
        # xlon open is one hour prior to xlon close during DST in UK
        # otherwise xlon open is same as xhkg close

        bi = prices.bis.T5
        bi_less_one_min = bi - one_min
        # consecutive sessions
        sessions = get_sessions_xnys_xhkg_xlon(bi, 4)
        session_before, session_prev, session, session_after = sessions

        xnys_prev_close = xnys.session_close(session_prev)
        xhkg_prev_close = xhkg.session_close(session_prev)
        xlon_prev_close = xlon.session_close(session_prev)

        xhkg_open = xhkg.session_open(session)
        xlon_open = xlon.session_open(session)
        xnys_open = xnys.session_open(session)
        xhkg_close = xhkg.session_close(session)
        xlon_close = xlon.session_close(session)
        xnys_close = xnys.session_close(session)
        xhkg_break_start = xhkg.session_break_start(session)
        xhkg_break_end = xhkg.session_break_end(session)

        xhkg_xlon_touch = xlon_open == xhkg_close  # touch as opposed to overlap

        table = prices.get("5T", session_before, session_after, tzout=UTC)

        # reset prices
        prices = m.PricesYahoo("MSFT, AZN.L, 9988.HK", lead_symbol="MSFT")
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_prev_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify every minute expected to return indice
        minute += one_min
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (indice, "open"),
        }
        for i in range(bi.as_minutes):
            if i:
                minute += one_min
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # verify one minute after right bound.
        minute += one_min
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg break
        minute += delta
        df = f(minute, UTC)
        indice = xhkg_break_start  # as during break not a trading minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break end
        minute = xhkg_break_end
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (xhkg_break_end, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg pm subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute - offset
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (indice, "open"),
        }
        self.assertions(table, df, indice, values)

        # prior to xlon open
        minute = xlon_open - one_min
        df = f(minute, UTC)
        indice = minute - bi_less_one_min
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (xlon_prev_close - bi, "close"),
            ali: (indice, "open"),
        }
        self.assertions(table, df, indice, values)

        # xlon open (possible xhkg close)
        minute = xlon_open
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (minute, "open"),
            ali: ali_value,
        }
        self.assertions(table, df, indice, values)

        # after xlon open
        minute += delta
        df = f(minute, UTC)
        indice = minute - offset
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (indice, "open")
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (indice, "open"),
            ali: ali_value,
        }
        self.assertions(table, df, indice, values)

        if not xhkg_xlon_touch:
            # at xhkg close
            minute = xhkg_close
            df = f(minute, UTC)
            indice = minute
            values = {
                msft: (xnys_prev_close - bi, "close"),
                astra: (minute, "open"),
                ali: (xhkg_close - bi, "close"),
            }
            self.assertions(table, df, indice, values)

        # verifying all minutes that would return expected indice
        # prior to xnys open
        minute = xnys_open - bi
        indice = minute
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (minute, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        for i in range(bi.as_minutes):
            if i:
                minute += one_min
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # check prior minute to ensure other side of edge
        minute = xnys_open - bi - one_min
        indice = minute - bi_less_one_min
        values = {
            msft: (xnys_prev_close - bi, "close"),
            astra: (indice, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys_open
        indice = minute
        values = {
            msft: (xnys_open, "open"),
            astra: (minute, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        for i in range(bi.as_minutes):
            if i:
                minute += one_min
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # check next minute other side of edge
        minute += one_min
        indice = minute
        values = {
            msft: (minute, "open"),
            astra: (minute, "open"),
            ali: (xhkg_close - bi, "close"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xlon close
        minute = xlon_close
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (minute, "open"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xlon close
        minute = xlon_close + one_min
        df = f(minute, UTC)
        indice = xlon_close
        values = {
            msft: (indice, "open"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # one indice after xlon close
        minute = xlon_close + bi
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (indice, "open"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys_close
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: (xnys_close - bi, "close"),
            astra: (xlon_close - bi, "close"),
            ali: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xnys close
        minute += delta
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

    def test_data_available_from_H1(self, one_min):
        """Test with minutes for which intraday data only available at H1.

        Also tests indices unaligned with subsession am close, pm open and
        pm close with multiple calendars.
        """
        prices = m.PricesYahoo("MSFT, 9988.HK", lead_symbol="MSFT")
        msft, ali = "MSFT", "9988.HK"
        xnys = prices.calendars[msft]
        xhkg = prices.calendars[ali]

        delta = pd.Timedelta(63, "T")
        offset = pd.Timedelta(3, "T")
        delta_reg = delta - offset

        bi = prices.bis.H1
        bi_less_one_min = bi - one_min
        half_hour = pd.Timedelta(30, "T")

        # consecutive sessions
        sessions = get_sessions_xnys_xhkg(bi, 4)
        session_before, session_prev, session, session_after = sessions

        xnys_prev_close = xnys.session_close(session_prev)
        xhkg_prev_close = xhkg.session_close(session_prev)

        xhkg_open = xhkg.session_open(session)
        xnys_open = xnys.session_open(session)
        xhkg_close = xhkg.session_close(session)
        xnys_close = xnys.session_close(session)
        xhkg_break_start = xhkg.session_break_start(session)
        xhkg_break_end = xhkg.session_break_end(session)

        table = prices.get("1H", session_before, session_after, tzout=UTC)

        # reset prices
        prices = m.PricesYahoo("MSFT, 9988.HK", lead_symbol="MSFT")
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        msft_prev_close_value = (xnys_prev_close - half_hour, "close")
        values = {
            msft: msft_prev_close_value,
            ali: (xhkg_prev_close - half_hour, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: msft_prev_close_value,
            ali: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice
        # and one minute other side of either bound.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {
            msft: msft_prev_close_value,
            ali: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify minutes either side of bounds
        minute += one_min
        indice = minute
        values = {
            msft: msft_prev_close_value,
            ali: (indice, "open"),
        }
        for i in (0, 1, 2, bi.as_minutes - 2, bi.as_minutes - 1):
            minute_ = minute + pd.Timedelta(i, "T")
            df = f(minute_, UTC)
            self.assertions(table, df, indice, values)

        # verify one minute after right bound.
        minute = minute_ + one_min
        indice = minute
        values = {
            msft: msft_prev_close_value,
            ali: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            msft: msft_prev_close_value,
            ali: (xhkg_break_start - half_hour, "close"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg break
        # select minutes represending start and end of break and minutes either
        # side of unaligned final indice of am session.
        minutes = (
            xhkg_break_start,
            xhkg_break_start + one_min,
            xhkg_break_start + half_hour - one_min,
            xhkg_break_start + half_hour,
            xhkg_break_start + half_hour + one_min,
            xhkg_break_start + bi - one_min,
        )
        indice = indice  # unchanged
        values = values  # unchanged
        for minute_ in minutes:
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # on xhkg break end through first half hour of pm session
        indice = xhkg_break_end
        values = {
            msft: msft_prev_close_value,
            ali: (xhkg_break_end - half_hour, "open"),
        }
        for minute in (
            xhkg_break_end,
            xhkg_break_end + one_min,
            xhkg_break_end + half_hour - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # during xhkg pm subsession
        # minutes served as open of second indice of pm subsession
        indice = xhkg_break_end + half_hour
        values = {
            msft: msft_prev_close_value,
            ali: (indice, "open"),
        }
        for minute in (
            xhkg_break_end + half_hour,
            xhkg_break_end + half_hour + one_min,
            xhkg_break_end + half_hour + bi - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # verify next minute
        minute = xhkg_break_end + half_hour + bi
        indice = minute
        values = {
            msft: msft_prev_close_value,
            ali: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xhkg close and through and post unaligned final interval
        indice = xhkg_close
        ali_close_value = (xhkg_close - half_hour, "close")
        values = {
            msft: msft_prev_close_value,
            ali: ali_close_value,
        }
        for minute in (
            xhkg_close,
            xhkg_close + one_min,
            xhkg_close + half_hour - one_min,
            xhkg_close + half_hour,
            xhkg_close + half_hour + one_min,
            xnys_open - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # xnys open through first indice
        indice = xnys_open
        values = {
            msft: (xnys_open, "open"),
            ali: ali_close_value,
        }
        for minute in (
            xnys_open,
            xnys_open + one_min,
            xnys_open + bi - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # minutes expected to return as open of second xnys indice
        indice = xnys_open + bi
        values = {
            msft: (indice, "open"),
            ali: ali_close_value,
        }
        for minute in (
            xnys_open + bi,
            xnys_open + bi + one_min,
            xnys_open + bi + bi - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # last minute to return as open of penultimate xnys session indice
        minute = xnys_close - half_hour - one_min
        indice = xnys_close - half_hour - bi
        values = {
            msft: (indice, "open"),
            ali: ali_close_value,
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # minutes expected to return as open of final indice of xnys session
        indice = xnys_close - half_hour
        values = {
            msft: (xnys_close - half_hour, "open"),
            ali: ali_close_value,
        }
        for minute in (
            xnys_close - half_hour,
            xnys_close - half_hour + one_min,
            xnys_close - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # minutes expected to return as close of final indice of xnys session
        indice = xnys_close
        values = {
            msft: (xnys_close - half_hour, "close"),
            ali: ali_close_value,
        }
        final_minute = prices.cc.minute_to_trading_minute(xnys_close, "next") - one_min
        for minute in (
            xnys_close,
            xnys_close + one_min,
            xnys_close + half_hour - one_min,
            xnys_close + half_hour,
            xnys_close + half_hour + one_min,
            final_minute,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

    def test_unaligned_single_symbol(
        self, prices_with_break, session_length_xhkg, one_min
    ):
        """Tests single symbol (xhkg) with indices unaligned with close.

        Indices unaligned with am subsession close, pm subsession open
        and pm close.
        """
        prices = prices_with_break
        ali = "9988.HK"
        xhkg = prices.calendars[ali]

        delta = pd.Timedelta(63, "T")
        offset = pd.Timedelta(3, "T")
        delta_reg = delta - offset

        bi = prices.bis.H1
        bi_less_one_min = bi - one_min
        half_hour = pd.Timedelta(30, "T")
        # four consecutive sessions
        sessions = get_valid_conforming_sessions(
            prices, bi, [xhkg], [session_length_xhkg], 4
        )
        session_before, session_prev, session, session_after = sessions

        prev_close = xhkg.session_close(session_prev)
        open_ = xhkg.session_open(session)
        close = xhkg.session_close(session)
        break_start = xhkg.session_break_start(session)
        break_end = xhkg.session_break_end(session)

        table = prices.get("1H", session_before, session_after, tzout=UTC)

        # reset prices
        prices = m.PricesYahoo("9988.HK")
        f = prices.price_at

        # prior to xhkg open
        minute = open_ - one_min
        df = f(minute, UTC)
        indice = prev_close
        values = {ali: (prev_close - half_hour, "close")}
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = open_
        indice = minute
        values = {ali: (open_, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice
        # and one minute other side of either bound.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {ali: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify minutes either side of bounds
        minute += one_min
        indice = minute
        values = {ali: (indice, "open")}
        for i in (0, 1, 2, bi.as_minutes - 2, bi.as_minutes - 1):
            minute_ = minute + pd.Timedelta(i, "T")
            df = f(minute_, UTC)
            self.assertions(table, df, indice, values)

        # verify one minute after right bound.
        minute = minute_ + one_min
        indice = minute
        values = {ali: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on break start and during break
        # select minutes represending start and end of break and minutes either
        # side of unaligned final indice of am session.
        indice = break_start
        values = {ali: (break_start - half_hour, "close")}
        for minute in (
            break_start,
            break_start + one_min,
            break_start + half_hour - one_min,
            break_start + half_hour,
            break_start + half_hour + one_min,
            break_start + bi - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # on xhkg break end through first half hour of pm session
        indice = break_end
        values = {ali: (break_end - half_hour, "open")}
        for minute in (
            break_end,
            break_end + one_min,
            break_end + half_hour - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # during pm subsession
        # minutes served as open of second indice of pm subsession
        indice = break_end + half_hour
        values = {ali: (indice, "open")}
        for minute in (
            break_end + half_hour,
            break_end + half_hour + one_min,
            break_end + half_hour + bi - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # verify next minute
        minute = break_end + half_hour + bi
        indice = minute
        values = {ali: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify returns open of second to last indice
        minute = close - half_hour - one_min
        indice = close - half_hour - bi
        values = {ali: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify prior to close returns open of last unaligned indice
        indice = close - half_hour
        values = {ali: (indice, "open")}
        for minute in (
            close - half_hour,
            close - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # xhkg close and through and post unaligned final interval
        indice = close
        values = {ali: (close - half_hour, "close")}
        next_open = xhkg.session_open(xhkg.next_session(session))
        for minute in (
            close,
            close + one_min,
            close + half_hour - one_min,
            close + half_hour,
            close + half_hour + one_min,
            next_open - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # first minute of following session
        minute = next_open
        indice = minute
        values = {ali: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)


def test_price_range(prices_us_lon_hkg, one_day, monkeypatch):
    """Test `price_range`.

    Test does not test return from `get`, rather limited to testing
    added functionality.

    Tests all parameters are being passed to `get`.
    Tests return as expected.
    Tests effect of `strict`.
    Tests returns `underlying` table.
    Tests output in expected timezone given passed `tzout`, `tzin` and
    `lead_symbol`.
    Tests output can be stacked. Tests for each type of underlying PT.
    """
    prices = prices_us_lon_hkg
    f = prices.price_range

    xnys = prices.calendars["MSFT"]
    xhkg = prices.calendars["9988.HK"]
    xlon = prices.calendars["AZN.L"]

    _, session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 2)
    minute = xlon.session_close(session) - pd.Timedelta(43, "T")
    session_T5 = get_sessions_xnys_xhkg_xlon(prices.bis.T5, 2)[-1]
    minute_T5 = xhkg.session_close(session_T5) - pd.Timedelta(78, "T")
    # TODO xcals 4.0 lose wrapper
    session_D1 = helpers.to_tz_naive(xnys.session_offset(session_T5, -20))
    session_D1 = get_valid_session(session_D1, prices.cc, "previous")

    def get(kwargs, **others) -> pd.DataFrame:
        return prices.get(**kwargs, **others, composite=True, openend="shorten")

    def assertions(
        rng: pd.DataFrame,
        table: pd.DataFrame,
        tz: pytz.BaseTzInfo = prices.tz_default,
        to_now: bool = False,
    ):
        assert_prices_table_ii(rng, prices)
        assert len(rng.index) == 1
        assert rng.pt.tz == tz

        start = table.pt.first_ts
        end = table.pt.last_ts
        if isinstance(table.pt, (pt.PTDailyIntradayComposite, pt.PTDaily)):
            # TODO xcals wrappers STAY STAY STAY
            start = prices.cc.session_open(helpers.to_tz_naive(start))
            if isinstance(table.pt, (pt.PTDaily)):
                end = prices.cc.session_close(helpers.to_tz_naive(end))

        if to_now:
            end = min(end, helpers.now())

        indice = rng.index[0]
        assert indice.left == start
        assert indice.right == end

        table = table.pt.fillna("both")
        for s in prices.symbols:
            assert rng.loc[indice, (s, "open")] == table.iloc[0][(s, "open")]
            assert rng.loc[indice, (s, "close")] == table.iloc[-1][(s, "close")]
            assert rng.loc[indice, (s, "high")] == table[(s, "high")].max()
            assert rng.loc[indice, (s, "low")] == table[(s, "low")].min()
            assert rng.loc[indice, (s, "volume")] == table[(s, "volume")].sum()

    def test_it(kwargs: dict, pt_type: pt._PT | None = None, to_now: bool = False):
        """Standard test for `kwargs`."""
        table = get(kwargs)
        if pt_type is not None:
            assert isinstance(table.pt, pt_type)
        rng = f(**kwargs)
        assertions(rng, table, to_now=to_now)

    # verify for passing `hours` and `minutes` and `end`
    kwargs = dict(end=minute, hours=40, minutes=30)
    table = get(kwargs)
    rng = f(**kwargs)
    assertions(rng, table)

    # verify underlying
    _, df = f(**kwargs, underlying=True)
    assert_frame_equal(df, table)

    # verify strict has no effect (will only have effect if start earlier than
    # limit for which daily data avaiable - tested for later).
    assert_frame_equal(f(**kwargs, strict=True), f(**kwargs, strict=False))

    # Verify for passing `days`
    # verify for when underlying table is intraday
    kwargs = dict(end=minute, days=4)
    table = get(kwargs)
    assert isinstance(table.pt, pt.PTIntraday)
    rng = f(**kwargs)
    assertions(rng, table)
    assert_frame_equal(rng.pt.stacked, f(**kwargs, stack=True))

    # Sidetrack to verify `tzout`
    tz = pytz.UTC
    rng = f(**kwargs, tzout=tz)
    assertions(rng, table, tz)

    tz = pytz.timezone("Australia/Perth")
    rng = f(**kwargs, tzout=tz)
    assertions(rng, table, tz)

    rng = f(**kwargs, tzout="9988.HK")
    assertions(rng, table, xhkg.tz)

    # verify output in terms of `tzin` if tzout not otherwise passed.
    tzin = pytz.timezone("Australia/Perth")
    rng = f(**kwargs, tzin=tzin)
    assertions(rng, table, tzin)

    lead = "9988.HK"
    table_ = get(kwargs, lead_symbol=lead)
    rng = f(**kwargs, lead_symbol=lead)
    assertions(rng, table_, xhkg.tz)

    # but not if `tzout` passed
    tzout = pytz.timezone("Europe/Rome")
    rng = f(**kwargs, tzin=tzin, tzout=tzout)
    assertions(rng, table, tzout)

    rng = f(**kwargs, lead_symbol=lead, tzout=tzout)
    assertions(rng, table_, tzout)

    # (verify `lead_symbol` being passed through whilst at it)
    assert table.index[0] != table_.index[0]

    # back to, verify for when underlying table is daily
    kwargs = dict(end=session, days=8)
    table = get(kwargs)
    assert isinstance(table.pt, pt.PTDaily)
    rng = f(**kwargs)
    assertions(rng, table)
    assert_frame_equal(rng.pt.stacked, f(**kwargs, stack=True))

    # while on a quick rtrn one, verify include and exclude being passed through
    include = ["9988.HK", "AZN.L"]
    rng_include = f(**kwargs, include=include)
    assert_frame_equal(rng_include, rng[include])
    rng_exclude = f(**kwargs, exclude="MSFT")
    assert_frame_equal(rng_include, rng_exclude)

    # verify for when underlying table is daily/intraday composite
    kwargs = dict(end=minute_T5, days=20)
    table = get(kwargs)
    assert isinstance(table.pt, pt.PTDailyIntradayComposite)
    rng = f(**kwargs)
    assertions(rng, table)
    assert_frame_equal(rng.pt.stacked, f(**kwargs, stack=True))

    # verify for passing `weeks`, `months` and `years`.
    kwargs = dict(end=minute_T5, weeks=1, months=1, years=1)
    test_it(kwargs)

    # verify effect of `strict` when start prior to first day for which data available
    kwargs = dict(start=prices.limit_daily - one_day, end=minute)
    table = get(kwargs, strict=False)
    rng = f(**kwargs, strict=False)
    assertions(rng, table)
    with pytest.raises(errors.StartTooEarlyError):
        f(**kwargs, strict=True)

    # verify period right never > now
    rng = f()
    assert rng.index.right <= helpers.now()

    # mock now to ensure same return for `price_range` and `get``
    mock_now(monkeypatch, pd.Timestamp.now(tz=pytz.UTC) - pd.Timedelta(5, "D"))
    # verify for passing `start` and for requesting to now
    kwargs = dict(start=minute)
    test_it(kwargs, to_now=True)

    # verify for no arguments
    test_it({}, to_now=True)
