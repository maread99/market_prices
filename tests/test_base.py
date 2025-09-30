"""Tests for market_prices.prices.base module.

Tests that do not require prices to be requested.

Notes
-----
Tests for the base module that require price data to be requested are on
`test_base_prices`.
"""

import dataclasses
import itertools
import re
import typing
import warnings
from collections import abc

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

import market_prices.prices.base as m
from market_prices import daterange, errors, helpers, intervals, mptypes
from market_prices.helpers import UTC
from market_prices.prices.yahoo import PricesYahoo
from market_prices.utils import calendar_utils as calutils

from .utils import get_resource

# ruff: noqa: FBT003  boolean-positional-value-in-call  # Happy to ignore here
# ruff: noqa: B028  no-explicit-stack-level  # Happy to ignore here
# ruff: noqa: N801  invalid-class-name  # Doesn't like _ at the end


@pytest.fixture
def t1_us_lon() -> abc.Iterator[pd.DataFrame]:
    """'T1' price table for us and lon symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L"]
    > df = prices.get(
        "1min",
        start=pd.Timestamp("2022-02"),
        end=pd.Timestamp("2022-02-09"),
        anchor="open",
    )
    """
    yield get_resource("t1_us_lon")


@pytest.fixture
def t5_us_lon() -> abc.Iterator[pd.DataFrame]:
    """'T5' price table for us and lon symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L"]
    > df = prices.get(
        "5min",
        start=pd.Timestamp("2022-01"),
        end=pd.Timestamp("2022-02-07"),
        anchor="open",
    )
    """
    yield get_resource("t5_us_lon")


def test_fill_reindexed():
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    mock_bi = intervals.BI_ONE_MIN
    mock_symbol = "SYMB"

    def f(
        df: pd.DataFrame, cal: xcals.ExchangeCalendar
    ) -> tuple[pd.DataFrame, list[errors.PricesMissingWarning]]:
        return m.fill_reindexed(df, cal, mock_bi, mock_symbol, "Yahoo")

    ohlcv = (
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [8.4, 8.8, 8.2, 8.6, 88],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [10.4, 10.8, 10.2, 10.6, 101],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
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
    rtrn, warnings_ = f(df, xlon)

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
        tz=UTC,
    )

    assert (
        ((open_ <= index) & (index < close))
        | ((next_open <= index) & (index < next_close))
    ).all()

    ohlcv = (
        [0.4, 0.8, 0.2, 0.6, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2.4, 2.8, 2.2, 2.6, 22],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [4.4, 4.8, 4.2, 4.6, 44],
    )

    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn, warnings_ = f(df, xasx)

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
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
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
    rtrn, warnings_ = f(df, xlon)
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: ['2022-01-02']"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        warnings.warn(warnings_[0])

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
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn, warnings_ = f(df, xlon)
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: ['2022-01-01']"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        warnings.warn(warnings_[0])

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
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [6.4, 6.8, 6.2, 6.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
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
        tz=UTC,
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    match_sessions = ["2022-01-05", "2022-01-07", "2022-01-10", "2022-01-12"]
    rtrn, warnings_ = f(df, xlon)
    match = re.escape(
        f"Prices from Yahoo are missing for '{mock_symbol}' at the base interval"
        f" '{mock_bi}' for the following sessions: {match_sessions}"
    )
    with pytest.warns(errors.PricesMissingWarning, match=match):
        warnings.warn(warnings_[0])

    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn, warnings_ = f(df, xasx)
    with pytest.warns(errors.PricesMissingWarning, match=match):
        warnings.warn(warnings_[0])

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


def mock_now(monkeypatch, now: pd.Timestamp):
    """Use `monkeypatch` to mock pd.Timestamp.now to return `now`."""

    def mock_now_(*_, tz=None, **__) -> pd.Timestamp:
        return pd.Timestamp(now.tz_convert(None), tz=tz)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now_)


def test_fill_reindexed_daily(one_min, monkeypatch):
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    symbol = "AZN.L"
    xlon = xcals.get_calendar("XLON", start="1990-01-01", side="left")
    delay = pd.Timedelta(15, "min")

    def f(
        df: pd.DataFrame,
        cal: xcals.ExchangeCalendar,
        mindate: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, list[errors.PricesMissingWarning]]:
        mindate = cal.first_session if mindate is None else mindate
        return m.fill_reindexed_daily(df, cal, mindate, delay, symbol, "Yahoo")

    def match(sessions: pd.DatetimeIndex | list[str]) -> str:
        return re.escape(
            f"Prices from Yahoo are missing for '{symbol}' at the base"
            f" interval '{intervals.TDInterval.D1}' for the following sessions:"
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
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [1.4, 1.8, 1.2, 1.6, 11],
        [2.4, 2.8, 2.2, 2.6, 22],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [5.4, 5.8, 5.2, 5.6, 55],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [7.4, 7.8, 7.2, 7.6, 77],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    )

    df = pd.DataFrame(ohlcv, index=index, columns=columns)

    # verify when now is after session open + delay missing values are filled
    # and a missing prices warning is raised.
    now = xlon.session_open(index[-1]) + delay + one_min
    mock_now(monkeypatch, now)
    missing_sessions = [
        "2021-01-04",
        "2021-01-07",
        "2021-01-08",
        "2021-01-12",
        "2021-01-14",
    ]

    rtrn, warnings_ = f(df.copy(), xlon)
    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        warnings.warn(warnings_[0])

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
    rtrn, warnings_ = f(df.copy(), xlon)
    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        warnings.warn(warnings_[0])

    missing_row = pd.DataFrame(
        [[np.nan] * 5], index=index[-1:], columns=columns, dtype="float64"
    )
    expected = pd.concat([expected[:-1], missing_row])
    assert_frame_equal(rtrn, expected)

    # verify as expected when no missing values to last row
    mock_now(monkeypatch, now)
    last_row = pd.DataFrame(
        [[8.4, 8.8, 8.2, 8.6, 88]], index=index[-1:], columns=columns, dtype="float64"
    )
    df = pd.concat([df[:-1], last_row])
    rtrn, warnings_ = f(df.copy(), xlon)
    with pytest.warns(errors.PricesMissingWarning, match=match(missing_sessions)):
        warnings.warn(warnings_[0])
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
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn, _ = f(df.copy(), xlon)

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
    rtrn, _ = f(df.copy(), xlon)
    assert_frame_equal(rtrn, df)

    # verify that missing prices before mindate are not filled and no warning raised
    ohlcv = (
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [2.4, 2.8, 2.2, 2.6, 22],
        [3.4, 3.8, 3.2, 3.6, 33],
        [4.4, 4.8, 4.2, 4.6, 44],
        [5.4, 5.8, 5.2, 5.6, 55],
        [5.4, 5.8, 5.2, 5.6, 66],
        [7.4, 7.8, 7.2, 7.6, 77],
        [8.4, 8.8, 8.2, 8.6, 88],
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn, _ = f(df.copy(), xlon, mindate=pd.Timestamp("2021-01-06"))
    assert_frame_equal(rtrn, df)


def test_adjust_high_low():
    columns = pd.Index(["open", "high", "low", "close", "volume"])
    ohlcv = (
        [100.0, 103.0, 98.0, 103.4, 0],  # close higher than high
        [104.0, 109.0, 104.0, 107.0, 0],
        [106.0, 108.0, 104.0, 107.0, 0],
        [106.0, 110.0, 107.0, 109.0, 0],  # open lower than low
        [108.0, 112.0, 108.0, 112.0, 0],
        [112.0, 114.0, 107.0, 106.4, 0],  # close lower than low
        [112.0, 108.0, 104.0, 105.0, 0],  # open higher than high
    )
    index = pd.date_range(
        start=pd.Timestamp("2022-01-01"), freq="D", periods=len(ohlcv)
    )
    df = pd.DataFrame(ohlcv, index=index, columns=columns)
    rtrn = m.adjust_high_low(df)

    ohlcv_expected = (
        [100.0, 103.4, 98, 103.4, 0],  # close was higher than high
        [104.0, 109.0, 104.0, 107.0, 0],
        [106.0, 108.0, 104.0, 107.0, 0],
        [107.0, 110.0, 107.0, 109.0, 0],  # open was lower than low
        [108.0, 112.0, 108.0, 112.0, 0],
        [112.0, 114.0, 106.4, 106.4, 0],  # close was lower than low
        [108.0, 108.0, 104.0, 105.0, 0],  # open was higher than high
    )
    expected = pd.DataFrame(ohlcv_expected, index=index, columns=columns)
    assert (expected.open >= expected.low).all()
    assert (expected.low <= expected.high).all()
    assert (expected.high >= expected.close).all()

    assert_frame_equal(rtrn, expected)


def test_get_columns_multiindex():
    symb = "SYMB"
    rtrn = m.get_columns_multiindex(symb)
    assert isinstance(rtrn, pd.MultiIndex)
    expected = pd.MultiIndex(
        (
            [symb],
            ["open", "high", "low", "close", "volume"],
        ),
        codes=([0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
        names=["symbol", ""],
    )
    assert_index_equal(rtrn, expected)

    existing_cols = pd.Index(["close", "low", "high", "open", "vol"])
    rtrn = m.get_columns_multiindex(symb, existing_cols)
    assert isinstance(rtrn, pd.MultiIndex)
    expected = pd.MultiIndex(
        (
            [symb],
            ["close", "low", "high", "open", "vol"],
        ),
        codes=([0, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
        names=["symbol", ""],
    )
    assert_index_equal(rtrn, expected)


def test_create_composite(t1_us_lon, t5_us_lon, one_day):
    f = m.create_composite

    first_df = t5_us_lon
    start = pd.Timestamp("2022-02-03 14:00", tz=UTC)
    stop = pd.Timestamp("2022-02-09 15:32", tz=UTC)
    second_df = t1_us_lon[start:stop]

    start_indice = first_df.index[33]
    end_indice = second_df.index[-6]
    assert end_indice.right == pd.Timestamp("2022-02-09 15:28", tz=UTC)
    first = (first_df, start_indice)
    second = (second_df, end_indice)

    rtrn = f(first, second)
    assert not rtrn.index.has_duplicates
    assert not rtrn.index.is_overlapping

    assert rtrn.pt.first_ts == start_indice.left
    assert rtrn.pt.last_ts == end_indice.right

    bv_T5 = rtrn.index.length == intervals.TDInterval.T5
    bv_T1 = rtrn.index.length == intervals.TDInterval.T1

    rtrn_T5 = rtrn[bv_T5]
    rtrn_T1 = rtrn[bv_T1]

    assert rtrn_T5.pt.last_ts == rtrn_T1.pt.first_ts
    split = rtrn_T5.pt.last_ts

    assert_frame_equal(first_df.loc[start_indice.left : split][:-1], rtrn_T5)
    assert_frame_equal(second_df.loc[split : end_indice.left], rtrn_T1)

    match = "`first` table must preceed and partially overlap `second`."
    # verify raises error when 'first' does not preceed 'second'
    with pytest.raises(ValueError, match=match):
        _ = f(second, first)

    # verify raises error when 'first' overlaps 'second'
    with pytest.raises(ValueError, match=match):
        subset = second_df[: first_df.index[-1].left - one_day]
        f(first, (subset, subset.index[-5]))


def test_inferred_intraday_interval(calendars_extended, one_min, monkeypatch):
    cal = calendars_extended
    default_kwargs = dict(minutes=0, hours=0, days=0, start=None, end=None)

    def f(**kwargs) -> bool:
        pp = {**default_kwargs, **kwargs}
        return m.PricesBase._inferred_intraday_interval(cal, pp)

    def assert_intraday(**kwargs):
        assert f(**kwargs)

    def assert_daily(**kwargs):
        assert not f(**kwargs)

    # start and end always > 5 trading days apart when combined
    session = start_date = cal.date_to_session(pd.Timestamp("2021-10-01"), "next")
    session = cal.session_offset(session, 7)
    time = session.replace(minute=33, hour=11)
    session = cal.session_offset(session, 7)
    midnight = helpers.to_utc(session)
    end_date = cal.session_offset(session, 7)
    # start or/and end with at least one as a time
    for tm in [time, midnight]:
        assert_intraday(start=tm)
        assert_intraday(end=tm)
        assert_intraday(start=tm, end=end_date)
        assert_intraday(start=start_date, end=tm)

    assert_intraday(start=time, end=midnight)

    # start or/and end always as a date(s)

    assert_daily(start=start_date)  # end now, > 5 days diff
    assert_daily(end=end_date)
    assert_daily(start=start_date, end=end_date)  # diff > 5 days

    # testing 5 day diff
    assert_intraday(start=start_date, end=start_date)  # same date

    end = cal.session_offset(start_date, 3)
    assert_intraday(start=start_date, end=end)  # 4 days diff

    end = cal.session_offset(end, 1)
    assert_intraday(start=start_date, end=end)  # 5 days diff, on limit

    end = cal.session_offset(end, 1)
    assert_daily(start=start_date, end=end)  # 6 days diff, over limit

    # testing 5 day diff with end as None

    last_session = cal.minute_to_past_session(helpers.now())
    last_session_close = cal.session_close(last_session)

    def mock_now(*_, **__) -> pd.Timestamp:
        return last_session_close - (5 * one_min)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now)

    assert_intraday(start=last_session)  # same date

    # as now open, 5 days will count to, and inclusive of, session prior to
    # last_session
    start = cal.session_offset(last_session, -4)
    assert_intraday(start=start)  # 4 days diff to prior session

    start = cal.session_offset(start, -1)
    assert_intraday(start=start)  # 5 days diff, on limit

    start = cal.session_offset(start, -1)
    assert_daily(start=start)  # 6 days diff, over limit

    def mock_now_after_close(*_, **__) -> pd.Timestamp:
        return last_session_close + (5 * one_min)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now_after_close)

    assert_intraday(start=last_session)  # same date

    # as now after close, 5 days will count to, and inclusive of last_session
    start = cal.session_offset(last_session, -3)
    assert_intraday(start=start)  # 4 days diff

    start = cal.session_offset(start, -1)
    assert_intraday(start=start)  # 5 days diff, on limit

    start = cal.session_offset(start, -1)
    assert_daily(start=start)  # 6 days diff, over limit

    # testing duration components
    trading_kwargs = []
    for minutes, hours in itertools.product([0, 1], [0, 1]):
        if minutes or hours:
            trading_kwargs.append(dict(minutes=minutes, hours=hours))

    for kwargs in trading_kwargs:
        assert_intraday(**kwargs)
        assert_intraday(**{**kwargs, **dict(start=start_date)})
        assert_intraday(**{**kwargs, **dict(end=end_date)})

    calendar_kwargs = []
    for weeks, months, years in itertools.product([0, 1], [0, 1], [0, 1]):
        if weeks or months or years:
            calendar_kwargs.append(dict(weeks=weeks, months=months, years=years))

    for kwargs in calendar_kwargs:
        assert_daily(**kwargs)
        assert_daily(**{**kwargs, **dict(start=start_date)})
        assert_daily(**{**kwargs, **dict(end=end_date)})

    for days in [1, 4, 5]:
        assert_intraday(days=days)

    assert_daily(days=6)


class TestSubClasses:
    """Verify subclasses concrete required class attributes.

    Notes
    -----
    Abstract methods should be tested no a dedicated test module for each
    subclass.
    """

    @pytest.fixture(
        scope="class",
        params=[
            PricesYahoo,
        ],
    )
    def subclasses(self, request) -> abc.Iterator[type[m.PricesBase]]:
        """Parameterized fixture of subclasses of PricesBase."""
        yield request.param

    def test_abstract_attributes(self, subclasses):
        Cls = subclasses

        assert isinstance(Cls.BaseInterval, intervals._BaseIntervalMeta)
        assert isinstance(Cls.BASE_LIMITS, dict)

        for k, v in Cls.BASE_LIMITS.items():
            assert v is None or isinstance(v, (pd.Timedelta, pd.Timestamp))
            assert k in Cls.BaseInterval

        for bi in Cls.BaseInterval:
            assert bi in Cls.BASE_LIMITS


@pytest.fixture(scope="class")
def daily_limit() -> abc.Iterator[pd.Timestamp]:
    """Limit of daily price availability for a mock prices class."""
    yield pd.Timestamp("1990-08-27")


@pytest.fixture(scope="class")
def cal_start(daily_limit) -> abc.Iterator[pd.Timestamp]:
    """Start date for any calendar to be passed to a mock prices class."""
    yield daily_limit - pd.Timedelta(14, "D")


@pytest.fixture(scope="class")
def xnys(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XNYS calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "XNYS" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("XNYS", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xlon(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XLON calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "XLON" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("XLON", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xasx(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XASX calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "XASX" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("XASX", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xhkg(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XHKG calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "XHKG" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("XHKG", start=cal_start, side=side)


@pytest.fixture(scope="class")
def x247(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """24/7 calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "24/7" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("24/7", start=cal_start, side=side)


@pytest.fixture(scope="class")
def cmes(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """CMES calendar that can be passed to a mock prices class.

    Due to xcals caching, if calendar passed to a mock prices class as
    "CMES" then calendar 'created' by that class will be same object as
    returned by this fixture.
    """
    yield xcals.get_calendar("CMES", start=cal_start, side=side)


@pytest.fixture
def PricesMockEmpty() -> abc.Iterator[type[m.PricesBase]]:
    class PricesMockEmpty_(m.PricesBase):
        """Mock PricesBase class with no base intervals defined."""

        def _request_data(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

        def prices_for_symbols(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

    yield PricesMockEmpty_


@pytest.fixture
def right_limits() -> abc.Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Use for an intrday / daily interval with a fixed right limit."""
    yield pd.Timestamp("2021-10-20 18:22", tz=UTC), pd.Timestamp("2021-10-20")


@pytest.fixture
def left_limits() -> abc.Iterator[dict[intervals.TDInterval, pd.Timestamp]]:
    """Use to define an intrday interval left limits as a timestamp."""
    yield {
        intervals.TDInterval.T1: pd.Timestamp("2021-09-20 18:23:00+0000", tz="UTC"),
        intervals.TDInterval.T5: pd.Timestamp("2021-08-21 18:23:00+0000", tz="UTC"),
        intervals.TDInterval.H1: pd.Timestamp("2020-10-20 18:23:00+0000", tz="UTC"),
    }


@pytest.fixture
def prices_mock_base_intervals(
    daily_limit, right_limits, left_limits
) -> abc.Iterator[
    tuple[
        type[intervals._BaseInterval],
        dict[intervals._BaseInterval, pd.Timedelta | pd.Timestamp],
        dict[intervals._BaseInterval, pd.Timestamp],
        dict[intervals._BaseInterval, pd.Timestamp],
    ]
]:
    """BaseInterval and corresponding limits for a PricesMock class.

    Defines left and right limits for both intraday and daily intervals.

    Left intervals defined as pd.Timedelta and pd.Timestamp, use that
    which required.
    """
    BaseInterval = intervals._BaseInterval(
        "BaseInterval",
        dict(
            T1=intervals.TIMEDELTA_ARGS["T1"],
            T5=intervals.TIMEDELTA_ARGS["T5"],
            H1=intervals.TIMEDELTA_ARGS["H1"],
            D1=intervals.TIMEDELTA_ARGS["D1"],
        ),
    )

    LIMITS = {
        BaseInterval.T1: pd.Timedelta(30, "D"),
        BaseInterval.T5: pd.Timedelta(60, "D"),
        BaseInterval.H1: pd.Timedelta(365, "D"),
        BaseInterval.D1: daily_limit,
    }

    LIMITS_FIXED = {
        BaseInterval.T1: left_limits[intervals.TDInterval.T1],
        BaseInterval.T5: left_limits[intervals.TDInterval.T5],
        BaseInterval.H1: left_limits[intervals.TDInterval.H1],
        BaseInterval.D1: daily_limit,
    }

    RIGHT_LIMITS = {
        BaseInterval.T1: right_limits[0],
        BaseInterval.T5: right_limits[0],
        BaseInterval.H1: right_limits[0],
        BaseInterval.D1: right_limits[1],
    }

    yield BaseInterval, LIMITS, LIMITS_FIXED, RIGHT_LIMITS


@pytest.fixture
def prices_mock_base_intervals_intraday_only(
    left_limits, right_limits
) -> abc.Iterator[
    tuple[
        type[intervals._BaseInterval],
        dict[intervals._BaseInterval, pd.Timedelta],
        dict[intervals._BaseInterval, pd.Timestamp],
        dict[intervals._BaseInterval, pd.Timestamp],
    ]
]:
    """BaseInterval and corresponding left and right limits for a PricesMock class.

    Defines only intraday intervals. Defines left limits as both timedelta
    and fixed timestamp.
    """
    BaseInterval = intervals._BaseInterval(
        "BaseInterval",
        dict(
            T1=intervals.TIMEDELTA_ARGS["T1"],
            T5=intervals.TIMEDELTA_ARGS["T5"],
            H1=intervals.TIMEDELTA_ARGS["H1"],
        ),
    )

    LIMITS = {
        BaseInterval.T1: pd.Timedelta(30, "D"),
        BaseInterval.T5: pd.Timedelta(60, "D"),
        BaseInterval.H1: pd.Timedelta(365, "D"),
    }

    LIMITS_FIXED = {
        BaseInterval.T1: left_limits[intervals.TDInterval.T1],
        BaseInterval.T5: left_limits[intervals.TDInterval.T5],
        BaseInterval.H1: left_limits[intervals.TDInterval.H1],
    }

    RIGHT_LIMITS = {
        BaseInterval.T1: right_limits[0],
        BaseInterval.T5: right_limits[0],
        BaseInterval.H1: right_limits[0],
    }

    yield BaseInterval, LIMITS, LIMITS_FIXED, RIGHT_LIMITS


@pytest.fixture
def prices_mock_base_intervals_daily_only(
    daily_limit, right_limits
) -> abc.Iterator[
    tuple[
        type[intervals._BaseInterval],
        dict[intervals._BaseInterval, pd.Timestamp],
        dict[intervals._BaseInterval, pd.Timestamp],
    ]
]:
    """BaseInterval and corresponding left and right limits for a PricesMock class.

    Defines only daily interval.
    """
    BaseInterval = intervals._BaseInterval(
        "BaseInterval", dict(D1=intervals.TIMEDELTA_ARGS["D1"])
    )
    LIMITS = {BaseInterval.D1: daily_limit}

    RIGHT_LIMITS = {
        BaseInterval.D1: right_limits[1],
    }

    yield BaseInterval, LIMITS, RIGHT_LIMITS


@pytest.fixture
def PricesMock(
    PricesMockEmpty, prices_mock_base_intervals
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase class with both intraday and daily intervals."""
    base_interval, limits, _, _ = prices_mock_base_intervals

    class PricesMock_(PricesMockEmpty):
        """Mock PricesBase class with both intraday and daily intervals."""

        BaseInterval = base_interval
        BASE_LIMITS = limits

    yield PricesMock_


@pytest.fixture
def PricesMockIntradayOnly(
    PricesMockEmpty, prices_mock_base_intervals_intraday_only
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase class with only intraday intervals."""
    base_interval, limits, _, _ = prices_mock_base_intervals_intraday_only

    class PricesMockIntradayOnly_(PricesMockEmpty):  # type: ignore[valid-type, misc]
        """Mock PricesBase class with only intraday intervals."""

        BaseInterval = base_interval
        BASE_LIMITS = limits

    yield PricesMockIntradayOnly_


@pytest.fixture
def PricesMockDailyOnly(
    PricesMockEmpty, prices_mock_base_intervals_daily_only
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase with only a daily interval."""
    base_interval, limits, _ = prices_mock_base_intervals_daily_only

    class PricesMockDailyOnly_(PricesMockEmpty):  # type: ignore[valid-type, misc]
        """Mock PricesBase with only a daily interval."""

        BaseInterval = base_interval
        BASE_LIMITS = limits

    yield PricesMockDailyOnly_


@pytest.fixture
def PricesMockBreakendPmOrigin(PricesMock) -> abc.Iterator[type[m.PricesBase]]:
    class PricesMockBreakendPmOrigin_(PricesMock):  # type: ignore[valid-type, misc]
        """Mock PricesBase class with PM_SUBSESSION_ORIGIN as 'break end'."""

        PM_SUBSESSION_ORIGIN = "break_end"

    yield PricesMockBreakendPmOrigin_


@pytest.fixture
def PricesMockFixedLimits(
    PricesMockEmpty, prices_mock_base_intervals
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase class with fixed limits and both intraday and daily intervals."""
    base_interval, _, limits, limits_right = prices_mock_base_intervals

    class PricesMockFixedLimits_(PricesMockEmpty):
        """Mock PricesBase class with both intraday and daily intervals."""

        BaseInterval = base_interval
        BASE_LIMITS = limits
        BASE_LIMITS_RIGHT = limits_right

    yield PricesMockFixedLimits_


@pytest.fixture
def PricesMockIntradayOnlyFixedLimits(
    PricesMockEmpty, prices_mock_base_intervals_intraday_only
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase class with fixed limits and only intraday intervals."""
    base_interval, _, limits, limits_right = prices_mock_base_intervals_intraday_only

    class PricesMockIntradayOnlyFixedLimits_(PricesMockEmpty):  # type: ignore[valid-type, misc]
        """Mock PricesBase class with only intraday intervals."""

        BaseInterval = base_interval
        BASE_LIMITS = limits
        BASE_LIMITS_RIGHT = limits_right

    yield PricesMockIntradayOnlyFixedLimits_


@pytest.fixture
def PricesMockDailyOnlyFixedLimits(
    PricesMockEmpty, prices_mock_base_intervals_daily_only
) -> abc.Iterator[type[m.PricesBase]]:
    """Mock PricesBase with fixed limits and only a daily interval."""
    base_interval, limits, limits_right = prices_mock_base_intervals_daily_only

    class PricesMockDailyOnlyFixedLimits_(PricesMockEmpty):  # type: ignore[valid-type, misc]
        """Mock PricesBase with only a daily interval."""

        BaseInterval = base_interval
        BASE_LIMITS = limits
        BASE_LIMITS_RIGHT = limits_right

    yield PricesMockDailyOnlyFixedLimits_


@pytest.fixture(scope="class")
def symbols() -> abc.Iterator[list[str]]:
    """Fictitious symbols for a mock prices class."""
    yield ["ONE.1", "TWO.22", "THREE3.3"]


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
    """Return dictionary representing 'pp' parameters."""
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


@pytest.fixture
def GetterMock(xnys, xlon) -> abc.Iterator[type[daterange.GetterIntraday]]:
    class GetterMock_(daterange.GetterIntraday):
        """Mock GetterIntraday class."""

        def __init__(
            self,
            daterange: (
                tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp] | None
            ) = None,
            daterange_sessions: tuple[pd.Timestamp, pd.Timestamp] | None = None,
            interval: intervals.BI | None = None,
            calendar: xcals.ExchangeCalendar | None = None,
            composite_calendar: calutils.CompositeCalendar | None = None,
            delay: pd.Timedelta = pd.Timedelta(0),  # noqa: B008
            limit: pd.Timestamp | None = None,
            ignore_breaks: bool | dict[intervals.BI, bool] = False,
            limit_right: pd.Timestamp | None = None,
        ):
            """Constructor.

            Parameters
            ----------
            daterange
                Value to be returned by `daterange`

            daterange_sessions
                Value to be returned by `daterange_sessions`

            interval
                Passed to base class constructor `interval` parameter.

            calendar : default: xnys
                Passed to base class constructor `calendar` parameter.

            composite_calendar : default: calutils.CompositeCalendar[xnys, xlon]
                Passed to base class constructor `composite_calendar` parameter.

            delay
                Passed to base class constructor `delay` parameter.

            limit : default: xnys.first_minute
                Passed to base class constructor `limit` parameter.

            limit_right: default: None
                Passed to base class constructor `limit_right` parameter.

            ignore_breaks : default: False for all intervals
                Passed to base class constructor `ignore_breaks` parameter.
            """
            self._daterange = daterange
            self._daterange_sessions = daterange_sessions

            calendar = calendar if calendar is not None else xnys
            if composite_calendar is None:
                composite_calendar = calutils.CompositeCalendar([xnys, xlon])
            limit = limit if limit is not None else xnys.first_minute

            super().__init__(
                calendar=calendar,
                composite_calendar=composite_calendar,
                delay=delay,
                limit=xnys.first_minute,
                interval=interval,
                ignore_breaks=ignore_breaks,
                limit_right=limit_right,
            )

        @property
        def daterange_sessions(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
            return self._daterange_sessions

        @property
        def daterange(
            self,
        ) -> tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp] | None:
            return self._daterange

    yield GetterMock_


class TestPricesBaseConstructor:
    """Verify expected errors raised during PricesBase instantiation."""

    def test_base_intervals_and_limits_defined(
        self, PricesMockEmpty, prices_mock_base_intervals, symbols, xnys
    ):
        """Verify raises error if base intervals or limits not defined."""
        match = re.escape(
            "Base intervals are not defined. Subclasses of `PricesBase` must define"
            " base intervals via the BaseInterval class attribute or the"
            " `_define_base_intervals` method."
        )
        with pytest.raises(AttributeError, match=match):
            PricesMockEmpty(symbols, xnys)

        base_intervals, limits, _, _ = prices_mock_base_intervals

        class PricesMockNoLimits(PricesMockEmpty):
            BaseInterval = base_intervals

        match = re.escape(
            "Base limits are not defined. Subclasses of `PricesBase` must define"
            " base limits via the BASE_LIMITS class attribute or the"
            " `_update_base_limits` method."
        )
        with pytest.raises(AttributeError, match=match):
            PricesMockNoLimits(symbols, xnys)

        # verify minimum requirements satisfied to instantiate subclass
        class PricesMockAsRequired(PricesMockNoLimits):
            BASE_LIMITS = limits

        assert PricesMockAsRequired(symbols, xnys)


class TestPricesBaseSetup:
    """Verify properties of PricesBase post-instantiation."""

    @pytest.fixture
    def PricesMockDailyNoLimit(
        self, PricesMock: type[m.PricesBase]
    ) -> abc.Iterator[type[m.PricesBase]]:
        """As PricesMock with daily interval with limit as None."""
        BASE_LIMITS_DAILY_NO_LIMIT = PricesMock.BASE_LIMITS.copy()
        BASE_LIMITS_DAILY_NO_LIMIT[PricesMock.BaseInterval.D1] = None

        class PricesMockDailyNoLimit_(PricesMock):  # type: ignore[valid-type, misc]
            """Mock PricesBase class with daily bi with no limit."""

            BASE_LIMITS = BASE_LIMITS_DAILY_NO_LIMIT

        yield PricesMockDailyNoLimit_

    def test_single_calendar(self, PricesMock, symbols, xnys, xasx, zero_td):
        """Verify post-setup properties when passing single calendar.

        Verifies passing calendar as calendar name and instance of ExchangeCalendar.
        """
        calendars = "XNYS"
        symbols_ = "ONE.1 TWO.22 THREE3.3"
        prices = PricesMock(symbols_, calendars)
        assert prices.symbols == symbols

        cal = xnys
        # if same parameters then cal will be the cached version
        assert prices.calendars == dict.fromkeys(symbols, cal)
        assert prices.calendars_symbols == {cal: symbols}
        assert prices.calendar_default == cal
        assert prices.calendars_unique == [cal]
        assert prices.calendars_names == ["XNYS"]
        assert prices.has_single_calendar

        assert prices.lead_symbol_default == symbols[0]

        assert prices.delays == dict.fromkeys(symbols, zero_td)
        assert prices.min_delay == zero_td
        assert prices.max_delay == zero_td
        assert prices.calendars_min_delay == {cal: zero_td}
        assert prices.calendars_max_delay == {cal: zero_td}

        assert prices.timezones == dict.fromkeys(symbols, cal.tz)
        assert prices.tz_default == cal.tz

        expected_cc = calutils.CompositeCalendar([cal])
        assert_frame_equal(prices.composite_calendar.schedule, expected_cc.schedule)
        assert prices.composite_calendar is prices.cc

        # Verify passing through single calendar as actual instance
        prices = PricesMock(symbols, xasx)
        assert prices.calendars == dict.fromkeys(symbols, xasx)

    def test_multiple_calendars(self, PricesMock, xlon, xasx, xnys, zero_td):
        """Verify post-setup properties when passing multiple calendars.

        Verifies:
            Passing multiple calendars as dict and list
                In both cases mixing defining calendars as instances of
                ExchangeCalendar and calendar names.
            Passing delays as dict and list
            Effect of passing `lead_symbol`
        """
        symbols = ["NY", "LON", "OZ", "LON2"]
        calendars = {  # mix them up
            symbols[0]: "XNYS",
            symbols[1]: xlon,
            symbols[3]: xlon,
            symbols[2]: "XASX",
        }
        prices = PricesMock(symbols, calendars)
        calendars_expected = {
            symbols[0]: xnys,
            symbols[1]: xlon,
            symbols[2]: xasx,
            symbols[3]: xlon,
        }
        assert prices.calendars == calendars_expected
        assert prices.symbols == symbols
        calendars_symbols_expected = {xlon: ["LON", "LON2"], xnys: ["NY"], xasx: ["OZ"]}
        assert prices.calendars_symbols == calendars_symbols_expected
        assert prices.calendar_default == xlon
        assert len(prices.calendars_unique) == 3
        assert all(cal in prices.calendars_unique for cal in (xlon, xnys, xasx))
        assert len(prices.calendars_names) == 3
        assert all(name in prices.calendars_names for name in ["XLON", "XASX", "XNYS"])
        assert not prices.has_single_calendar

        assert prices.lead_symbol_default == "LON"

        assert prices.delays == dict.fromkeys(symbols, zero_td)
        assert prices.min_delay == zero_td
        assert prices.max_delay == zero_td
        expected_calendar_delays = {xlon: zero_td, xnys: zero_td, xasx: zero_td}
        assert prices.calendars_min_delay == expected_calendar_delays
        assert prices.calendars_max_delay == expected_calendar_delays

        assert prices.timezones == {
            "NY": xnys.tz,
            "LON": xlon.tz,
            "LON2": xlon.tz,
            "OZ": xasx.tz,
        }
        assert prices.tz_default == xlon.tz

        expected_cc = calutils.CompositeCalendar([xlon, xnys, xasx])
        assert_frame_equal(prices.composite_calendar.schedule, expected_cc.schedule)
        assert prices.composite_calendar is prices.cc

        # verify can pass through calendars as a list, verify delays and
        # effect of lead_symbol
        calendars = [xnys, "XLON", xasx, "XLON"]
        delays = {"NY": 5, "LON": 10, "LON2": 15, "OZ": 0}
        prices = PricesMock(symbols, calendars, delays=delays, lead_symbol="NY")
        assert prices.calendars == calendars_expected
        assert prices.calendars_symbols == calendars_symbols_expected

        expected_delays = {
            "NY": pd.Timedelta(5, "min"),
            "LON": pd.Timedelta(10, "min"),
            "LON2": pd.Timedelta(15, "min"),
            "OZ": zero_td,
        }
        assert prices.delays == expected_delays
        assert prices.min_delay == zero_td
        assert prices.max_delay == pd.Timedelta(15, "min")
        assert prices.calendars_min_delay == {
            xlon: pd.Timedelta(10, "min"),
            xnys: pd.Timedelta(5, "min"),
            xasx: zero_td,
        }
        assert prices.calendars_max_delay == {
            xlon: pd.Timedelta(15, "min"),
            xnys: pd.Timedelta(5, "min"),
            xasx: zero_td,
        }

        # verifying effect of having changed `lead_symbol`
        assert prices.symbols == symbols
        assert prices.calendar_default == xnys
        assert prices.lead_symbol_default == "NY"
        assert prices.tz_default == xnys.tz

        # verify can pass delays through as a list
        delays = [5, 10, 0, 15]  # recalling symbols=["NY", "LON", "OZ", "LON2"]
        prices = PricesMock(symbols, calendars, delays=delays)
        assert prices.delays == expected_delays

    def test_base_limits(self, PricesMock, daily_limit):
        """Test base limits and that can be changed with `_update_base_limits`."""
        inst_args = (["ONE"], ["XLON"])
        BASE_LIMITS_COPY = PricesMock.BASE_LIMITS.copy()
        prices = PricesMock(*inst_args)
        assert prices.base_limits == PricesMock.BASE_LIMITS
        update = {prices.bi_daily: pd.Timestamp("2000-01-01")}
        prices._update_base_limits(update)
        new_limits = {**PricesMock.BASE_LIMITS, **update}
        assert prices.base_limits == new_limits
        # verify class attribute unchanged
        assert PricesMock.BASE_LIMITS == BASE_LIMITS_COPY

        # verify cannot set limit of an intraday interval to None, either via updating
        prev_limits = prices.base_limits.copy()
        bi = prices.bis.T5
        match = (
            "Intraday base interval limits must be of type pd.Timedelta or pd.Timestamp"
            f", although limit for {bi} would be defined as None."
        )
        with pytest.raises(ValueError, match=match):
            prices._update_base_limits({bi: None})
        assert prices.base_limits == prev_limits  # verify dict unchanged

        # or initial defintion
        class PricesMockInvalidBaseLimit1(PricesMock):
            """Mock PricesBase class with an invalid bi limit."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T5=intervals.TIMEDELTA_ARGS["T5"],
                    H1=intervals.TIMEDELTA_ARGS["H1"],
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T5: None,
                BaseInterval.H1: pd.Timedelta(365, "D"),
            }

        with pytest.raises(ValueError, match=match):
            PricesMockInvalidBaseLimit1(*inst_args)

        # verify cannot omit a base interval
        class PricesMockInvalidBaseLimit2(PricesMock):
            """Mock PricesBase class that fails to define limit of a bi."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T5=intervals.TIMEDELTA_ARGS["T5"],
                    H1=intervals.TIMEDELTA_ARGS["H1"],
                    D1=intervals.TIMEDELTA_ARGS["D1"],
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T5: pd.Timedelta(days=60),
                BaseInterval.D1: daily_limit,
            }

        def match(bis, limit_keys) -> str:
            return re.escape(
                "Base limits do not accurately represent base intervals. Base intervals"
                f" are {bis.__members__} although base limit keys would be"
                f" {limit_keys}."
            )

        bis = PricesMockInvalidBaseLimit2.BaseInterval
        limit_keys = PricesMockInvalidBaseLimit2.BASE_LIMITS.keys()
        with pytest.raises(ValueError, match=match(bis, limit_keys)):
            PricesMockInvalidBaseLimit2(*inst_args)

        # verify cannot include an interval that is not a base interval.
        base_limits_copy = PricesMock.BASE_LIMITS.copy()
        not_bi = intervals.TDInterval.T30
        base_limits_copy[not_bi] = pd.Timedelta(1, "s")
        limit_keys = base_limits_copy.keys()
        with pytest.raises(ValueError, match=match(prices.bis, limit_keys)):
            prices._update_base_limits({not_bi: pd.Timedelta(100, "D")})
        assert prices.base_limits == prev_limits  # verify dict unchanged

        # verify cannot define limit of daily base interval as non-date
        def match_daily_limit(limit) -> str:
            return re.escape(
                "If limit of daily base interval is defined as a pd.Timestamp"
                " then timestamp must represent a date, although being defined"
                f" as {limit}."
            )

        limits = [
            pd.Timestamp("2000-01-01 15:00"),
            pd.Timestamp("2000-01-01", tz=UTC),
        ]
        for limit in limits:
            with pytest.raises(ValueError, match=match_daily_limit(limit)):
                prices._update_base_limits({prices.bi_daily: limit})

        limit = limits[0]

        class PricesMockInvalidBaseLimit3A(PricesMock):
            """Mock PricesBase class with invalid daily base limit."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(D1=intervals.TIMEDELTA_ARGS["D1"]),
            )

            BASE_LIMITS = {BaseInterval.D1: limit}

        with pytest.raises(ValueError, match=match_daily_limit(limit)):
            PricesMockInvalidBaseLimit3A(*inst_args)

        limit = limits[1]

        class PricesMockInvalidBaseLimit3B(PricesMock):
            """Mock PricesBase class with invalid daily base limit."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(D1=intervals.TIMEDELTA_ARGS["D1"]),
            )

            BASE_LIMITS = {BaseInterval.D1: limit}

        with pytest.raises(ValueError, match=match_daily_limit(limit)):
            PricesMockInvalidBaseLimit3B(*inst_args)

    def test_calendar_too_short_warning(
        self, PricesMock, symbols, daily_limit, side, xnys
    ):
        """Verify raises `CalendarTooShortWarning`.

        Verify raises `CalendarTooShortWarning` when a calendar starts
        later than the earliest date for which prices are available for any
        symbol.
        """
        good_cal = xnys
        delta = pd.Timedelta(7, "D")
        start = daily_limit + delta
        cal = xcals.get_calendar("XLON", start=start, side=side)
        cal2 = xcals.get_calendar("XASX", start=start, side=side)

        def match(cal: xcals.ExchangeCalendar) -> str:
            return (
                f"Calendar '{cal.name}' is too short to support all available price"
                f" history. Calendar starts '{cal.first_session}' whilst earliest date"
                f" for which price history is available is '{daily_limit}'. Prices"
                f" will not be available for any date prior to {cal.first_session}."
            )

        with pytest.warns(errors.CalendarTooShortWarning, match=re.escape(match(cal))):
            PricesMock(symbols, cal)

        with pytest.warns(errors.CalendarTooShortWarning) as ws:
            PricesMock(symbols, [good_cal, cal, cal2])
        assert len(ws.list) == 2
        for match in (match(cal), match(cal2)):  # noqa: B020
            assert match in str(ws[0].message) or match in str(ws[1].message)

    def test_calendar_too_short_error(
        self, PricesMock, symbols, side, xnys, one_day, monkeypatch
    ):
        """Verify raises `CalendarTooShortError`.

        Verify raises `CalendarTooShortError` when a calendar's first
        minute is later than the earliest minute for which intraday prices
        are available for any symbol.
        """
        now = helpers.now()

        def mock_now(*_, tz=None, **__) -> pd.Timestamp:
            return pd.Timestamp(now.tz_convert(None), tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)

        intraday_limit = now - PricesMock.BASE_LIMITS[intervals.TDInterval.H1]
        start = intraday_limit.normalize().astimezone(None) + one_day
        cal = xcals.get_calendar("XLON", start=start, side=side)
        good_cal = xnys

        msg = re.escape(
            f"Calendar '{cal.name}' is too short to support all available intraday"
            f" price history. Calendar starts '{cal.first_minute}' whilst earliest"
            f" minute for which intraday price history is available is"
            f" '{intraday_limit}' (all calendars must be valid over at least the"
            " period for which intraday price data is available)."
        )

        with pytest.raises(errors.CalendarTooShortError, match=msg):
            _ = PricesMock(symbols, cal)

        with pytest.raises(errors.CalendarTooShortError, match=msg):
            _ = PricesMock(symbols, [good_cal, cal, cal])

        # just to verify is a good cal...
        _ = PricesMock(symbols, good_cal)

    def test_calendar_expired_error(self, PricesMock, symbols, cal_start, side):
        """Verify raises `CalendarExpiredError`.

        Verify raises `CalendarExpiredError` when a calendar ends earlier
        than 'tomorrow'.
        """
        end = pd.Timestamp.now().floor("D")
        cal = xcals.get_calendar("XLON", start=cal_start, end=end, side=side)
        match = re.escape(
            f"Calendar '{cal.name}' has expired. The calendar's right bound is"
            f" {cal.last_minute} although operation requires calendar to be valid"
            " through to at least"
        )
        with pytest.raises(errors.CalendarExpiredError, match=match):
            PricesMock(symbols, cal)

        good_cal = xcals.get_calendar("XNYS", start=cal_start, side=side)
        with pytest.raises(errors.CalendarExpiredError, match=match):
            PricesMock(symbols, [good_cal, cal, good_cal])

    def test_calendar_side_error(self, PricesMock, symbols, cal_start, side):
        """Verify raises ValueError when a calendar has an invalid side."""
        cal = xcals.get_calendar("XLON", start=cal_start, side="right")
        match = (
            "All calendars must have side 'left', although received calendar"
            f" '{cal.name}' with side 'right'."
        )
        with pytest.raises(ValueError, match=match):
            PricesMock(symbols, cal)

        good_cal = xcals.get_calendar("XNYS", start=cal_start, side=side)
        with pytest.raises(ValueError, match=match):
            PricesMock(symbols, [good_cal, cal, good_cal])

    def test_calendar_delays_definition_errors(self, PricesMock, symbols, xlon):
        """Verify raises errors when define calendars or delays badly.

        Verfies raises ValueError if pass `calendars` or `delays` as a list
        with fewer indices than `symbols` or a dict with a key not included to
        `symbols`.
        """

        def match(
            param: typing.Literal["calendars", "delays"],
            arg: dict | list,
        ) -> str:
            if isinstance(arg, list):
                type_, attr = ("list", "length")
            else:
                type_, attr = ("dict", "keys")
            return re.escape(
                f"If passing {param} as a {type_} then {type_} must have same {attr} as"
                f" symbols, although receieved {param} as {arg} for symbols {symbols}."
            )

        cal = xlon
        cals = [cal, cal]
        with pytest.raises(ValueError, match=match("calendars", cals)):
            PricesMock(symbols, [cal, cal])

        cals = dict.fromkeys(symbols[:-1], cal)
        cals["not_a_symbol"] = cal
        with pytest.raises(ValueError, match=match("calendars", cals)):
            PricesMock(symbols, cals)

        cals = dict.fromkeys(symbols, cal)
        cals["extra_symbol"] = cal
        with pytest.raises(ValueError, match=match("calendars", cals)):
            PricesMock(symbols, cals)

        delays = [5, 5]
        with pytest.raises(ValueError, match=match("delays", delays)):
            PricesMock(symbols, cal, delays=delays)

        delays = dict.fromkeys(symbols[:-1], 5)
        delays["not_a_symbol"] = 5
        with pytest.raises(ValueError, match=match("delays", delays)):
            PricesMock(symbols, cal, delays=delays)

        delays = dict.fromkeys(symbols, 5)
        delays["extra_symbol"] = 10
        with pytest.raises(ValueError, match=match("delays", delays)):
            PricesMock(symbols, cal, delays=delays)

    def test_lead_symbol_error(self, PricesMock, symbols):
        lead = "NOTASYMB"
        match = re.escape(
            f"`lead_symbol` received as '{lead}' although must be None or in {symbols}."
        )
        with pytest.raises(ValueError, match=match):
            PricesMock(symbols, "XNYS", lead_symbol=lead)

    def test_limits(
        self,
        PricesMock,
        PricesMockIntradayOnly,
        PricesMockDailyNoLimit,
        daily_limit,
        symbols,
        xnys,
        xlon,
        xhkg,
        one_min,
        monkeypatch,
    ):
        """Test limits properties.

        Test default right intrday / daily limit is 'now' / 'today'.
        """

        def mock_now(tz=None) -> pd.Timestamp:
            return pd.Timestamp("2022-02-14 21:21:05", tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)
        now = mock_now(tz=UTC)
        today = now.floor("D").tz_convert(None)

        calendars = [xnys, xhkg, xlon]
        prices = PricesMock(symbols, calendars)

        # verify `limits`.
        assert set(prices.limits.keys()) == set(PricesMock.BaseInterval)
        assert len(prices.limits) == len(PricesMock.BaseInterval)

        for bi in PricesMock.BaseInterval:
            if bi.is_daily:
                assert prices.limits[bi] == (daily_limit, today)
            else:
                ll = (now - PricesMock.BASE_LIMITS[bi]).ceil("min") + one_min
                rl = now.floor("min") + bi
                assert prices.limits[bi] == (ll, rl)
                assert prices.limit_intraday_bi(bi) == ll

        # verify `limit_daily`
        assert prices.limit_daily == daily_limit
        # verify `limit_right_daily` defaults to today
        assert prices.limit_right_daily == today

        limits_raw = {}
        # verify `limit_intraday`
        for bi in prices.bis_intraday:
            delta = PricesMock.BASE_LIMITS[bi]
            limit_raw = (now - delta).ceil("min") + one_min
            limits_raw[bi] = limit_raw
            limits_intraday = []
            for cal in calendars:
                expected = cal.minute_to_trading_minute(limit_raw, "next")
                limits_intraday.append(expected)
                assert prices.limit_intraday_bi_calendar(bi, cal) == expected
            if bi is prices.bis.T5:  # T5 is bi with longest history as H1 is unaligned
                assert prices.limit_intraday(cal) == expected
                expected_latest_intraday_limit = max(limits_intraday)
                assert prices.limit_intraday() == expected_latest_intraday_limit
                assert prices.limit_intraday(None) == expected_latest_intraday_limit

        # verify `limit_right_intraday`
        assert prices.limit_right_intraday == now.floor("min")

        # verify 'limit_sessions'
        assert len(prices.limits_sessions) == len(PricesMock.BaseInterval)

        # from manual inspection:
        lefts = {
            PricesMock.BaseInterval.T1: pd.Timestamp("2022-01-17"),
            PricesMock.BaseInterval.T5: pd.Timestamp("2021-12-17"),
            PricesMock.BaseInterval.H1: pd.Timestamp("2021-02-15"),
            PricesMock.BaseInterval.D1: daily_limit,
        }

        for bi in PricesMock.BaseInterval:
            assert prices.limits_sessions[bi] == (lefts[bi], today)

        limit_raw_T5 = limits_raw[prices.bis.T5]

        prices = PricesMockDailyNoLimit(symbols, calendars)
        # only test for differences to PricesMock

        assert set(prices.limits.keys()) == set(PricesMockDailyNoLimit.BaseInterval)
        assert len(prices.limits) == len(PricesMockDailyNoLimit.BaseInterval)
        bi_daily = PricesMockDailyNoLimit.BaseInterval.D1
        limit_daily = max(cal.first_session for cal in calendars)
        assert prices.limits[bi_daily] == (limit_daily, today)

        assert prices.limit_daily == limit_daily
        assert prices.limit_right_daily == today
        for bi in prices.bis_intraday:
            for cal in calendars:
                expected = cal.minute_to_trading_minute(limits_raw[bi], "next")
                assert prices.limit_intraday_bi_calendar(bi, cal) == expected
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw_T5, "next")
            assert prices.limit_intraday(cal) == expected_limit_intraday
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit
        assert len(prices.limits_sessions) == len(PricesMockDailyNoLimit.BaseInterval)
        assert prices.limits_sessions[bi_daily] == (limit_daily, today)

        prices = PricesMockIntradayOnly(symbols, calendars)
        # only test for differences to PricesMock

        assert set(prices.limits.keys()) == set(PricesMockIntradayOnly.BaseInterval)
        assert len(prices.limits) == len(PricesMockIntradayOnly.BaseInterval)
        assert pd.Timedelta(1, "min") in prices.bis
        assert pd.Timedelta(1, "D") not in prices.bis

        assert prices.limit_daily is None
        assert prices.limit_right_daily is None  # verify None when no daily interval
        for bi in prices.bis_intraday:
            for cal in calendars:
                expected = cal.minute_to_trading_minute(limits_raw[bi], "next")
                assert prices.limit_intraday_bi_calendar(bi, cal) == expected
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw_T5, "next")
            assert prices.limit_intraday(cal) == expected_limit_intraday
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit
        assert len(prices.limits_sessions) == len(PricesMockIntradayOnly.BaseInterval)

    def test_limits_fixed(
        self,
        PricesMock,
        PricesMockFixedLimits,
        PricesMockIntradayOnlyFixedLimits,
        PricesMockDailyOnlyFixedLimits,
        daily_limit,
        right_limits,
        left_limits,
        symbols,
        xnys,
        xlon,
        xhkg,
    ):
        """Test limit properties when class has fixed left and right limits."""
        right_limit, right_limit_daily = right_limits

        calendars = [xnys, xhkg, xlon]
        prices = PricesMockFixedLimits(symbols, calendars)

        # verify `limits`.
        assert set(prices.limits.keys()) == set(PricesMockFixedLimits.BaseInterval)
        assert len(prices.limits) == len(PricesMockFixedLimits.BaseInterval)

        for bi in PricesMockFixedLimits.BaseInterval:
            if bi.is_daily:
                assert prices.limits[bi] == (daily_limit, right_limit_daily)
            else:
                assert prices.limits[bi] == (left_limits[bi], right_limit)
                assert prices.limit_intraday_bi(bi) == left_limits[bi]

        # verify `limit_daily` and `limit_right_daily`
        assert prices.limit_daily == daily_limit
        assert prices.limit_right_daily == right_limit_daily
        # verify `limit_intraday_bi_calendar`
        for bi in prices.bis_intraday:
            for cal in calendars:
                expected = cal.minute_to_trading_minute(left_limits[bi], "next")
                assert prices.limit_intraday_bi_calendar(bi, cal) == expected
        # verify `limit_intraday`
        limit_raw_5T = left_limits[intervals.TDInterval.T5]  # unaligned at H1
        limits_intraday = []
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw_5T, "next")
            limits_intraday.append(expected_limit_intraday)
            assert prices.limit_intraday(cal) == expected_limit_intraday

        expected_latest_intraday_limit = max(limits_intraday)
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit

        # verify `limit_right_intraday`
        assert prices.limit_right_intraday == right_limits[0]

        # verify 'limit_sessions'
        assert len(prices.limits_sessions) == len(PricesMock.BaseInterval)

        # from manual inspection:
        lefts = {
            PricesMock.BaseInterval.T1: pd.Timestamp("2021-09-20"),
            PricesMock.BaseInterval.T5: pd.Timestamp("2021-08-23"),  # 21 is a Saturday
            PricesMock.BaseInterval.H1: pd.Timestamp("2020-10-20"),
            PricesMock.BaseInterval.D1: daily_limit,
        }

        for bi in PricesMock.BaseInterval:
            assert prices.limits_sessions[bi] == (lefts[bi], right_limits[1])

        prices = PricesMockDailyOnlyFixedLimits(symbols, calendars)
        assert set(prices.limits.keys()) == set(
            PricesMockDailyOnlyFixedLimits.BaseInterval
        )
        assert len(prices.limits) == len(PricesMockDailyOnlyFixedLimits.BaseInterval)
        bi_daily = PricesMockDailyOnlyFixedLimits.BaseInterval.D1
        assert prices.limits[bi_daily] == (daily_limit, right_limits[1])

        assert prices.limit_daily == daily_limit
        assert prices.limit_right_daily == right_limits[1]

        match = re.escape(
            "`limit_intraday` is not implemented when no intraday interval is defined."
        )
        with pytest.raises(NotImplementedError, match=match):
            prices.limit_intraday()

        # verify `limit_right_intraday`
        match = re.escape(
            "`limit_right_intraday` is not implemented when no intraday interval"
            " is defined."
        )
        with pytest.raises(NotImplementedError, match=match):
            prices.limit_right_intraday()

        assert len(prices.limits_sessions) == len(
            PricesMockDailyOnlyFixedLimits.BaseInterval
        )
        assert prices.limits_sessions[bi_daily] == (daily_limit, right_limits[1])

        prices = PricesMockIntradayOnlyFixedLimits(symbols, calendars)

        assert set(prices.limits.keys()) == set(
            PricesMockIntradayOnlyFixedLimits.BaseInterval
        )
        assert len(prices.limits) == len(PricesMockIntradayOnlyFixedLimits.BaseInterval)
        assert pd.Timedelta(1, "min") in prices.bis
        assert pd.Timedelta(1, "D") not in prices.bis

        assert prices.limit_daily is None
        assert prices.limit_right_daily is None  # verify None when no daily interval
        for bi in prices.bis_intraday:
            for cal in calendars:
                expected = cal.minute_to_trading_minute(left_limits[bi], "next")
                assert prices.limit_intraday_bi_calendar(bi, cal) == expected
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw_5T, "next")
            assert prices.limit_intraday(cal) == expected_limit_intraday
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit
        assert len(prices.limits_sessions) == len(
            PricesMockIntradayOnlyFixedLimits.BaseInterval
        )

    def test_live_prices(
        self,
        PricesMock,
        PricesMockDailyOnly,
        PricesMockIntradayOnly,
        PricesMockFixedLimits,
        PricesMockIntradayOnlyFixedLimits,
        PricesMockDailyOnlyFixedLimits,
        symbols,
        xnys,
        xlon,
        xhkg,
    ):
        """Test `live_prices` property."""
        calendars = [xnys, xhkg, xlon]

        # verifications against manual inspection of calendars' schedules.

        prices = PricesMock(symbols, calendars)
        assert prices.live_prices
        prices = PricesMockDailyOnly(symbols, calendars)
        assert prices.live_prices
        prices = PricesMockIntradayOnly(symbols, calendars)
        assert prices.live_prices

        prices = PricesMockFixedLimits(symbols, calendars)
        assert not prices.live_prices
        prices = PricesMockDailyOnlyFixedLimits(symbols, calendars)
        assert not prices.live_prices
        prices = PricesMockIntradayOnlyFixedLimits(symbols, calendars)
        assert not prices.live_prices

    def test_earliest(
        self,
        PricesMock,
        PricesMockIntradayOnly,
        PricesMockDailyNoLimit,
        daily_limit,
        symbols,
        xnys,
        xlon,
        xhkg,
        one_day,
        monkeypatch,
    ):
        """Test `earliest` properties.

        Tests:
            `_earliest_requestable_calendar_session`
            `_earliest_requestable_calendar_minute`
            `earliest_requestable_session`
            `earliest_requestable_minute`
        """

        def mock_now(tz=None) -> pd.Timestamp:
            return pd.Timestamp("2022-02-14 21:21:05", tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)
        calendars = [xnys, xhkg, xlon]

        # verifications against manual inspection of calendars' schedules.
        # NB only xnys is open on daily limit 1990-08-27, all are open on 1990-08-28

        prices = PricesMock(symbols, calendars)
        assert prices._earliest_requestable_calendar_session(xnys) == daily_limit
        xnys_open = xnys.session_open(daily_limit)
        assert prices._earliest_requestable_calendar_minute(xnys) == xnys_open
        for cal in (xlon, xhkg):
            session = daily_limit + one_day
            assert prices._earliest_requestable_calendar_session(cal) == session
            minute = cal.session_open(session)
            assert prices._earliest_requestable_calendar_minute(cal) == minute

        assert prices.earliest_requestable_session == daily_limit
        xlon_open = xlon.session_open(session)
        assert prices.earliest_requestable_minute == xlon_open

        prices = PricesMockDailyNoLimit(symbols, calendars)
        for cal in calendars:
            assert (
                prices._earliest_requestable_calendar_session(cal) == cal.first_session
            )
            assert prices._earliest_requestable_calendar_minute(cal) == cal.first_minute

        session = max(cal.first_session for cal in calendars)
        assert prices.earliest_requestable_session == session
        minute = max(cal.first_minute for cal in calendars)
        assert prices.earliest_requestable_minute == minute

        prices = PricesMockIntradayOnly(symbols, calendars)
        expected = pd.Timestamp("2021-12-17")  # based on 5T data, delta 60 days
        # all calendars opwn
        for cal in calendars:
            assert prices._earliest_requestable_calendar_session(cal) == expected
            expected_minute = cal.session_open(expected)
            assert prices._earliest_requestable_calendar_minute(cal) == expected_minute

        # verify general properties as xnys (latest calendar)
        assert prices.earliest_requestable_session == expected
        expected_minute = xnys.session_open("2021-12-17")
        assert prices.earliest_requestable_minute == expected_minute

    def test_last_requestable_session_(
        self, PricesMock, symbols, xnys, xlon, xhkg, one_min, monkeypatch
    ):
        """Test `last_requestable_session*` methods.

        Tests `last_requestable_session_all` and
        `last_requestable_session_any`.
        """
        prices = PricesMock(symbols, [xnys, xlon, xhkg])
        session = pd.Timestamp("2021-01-19")
        prev_session = pd.Timestamp("2021-01-18")
        xnys_prev_session = helpers.to_tz_naive(xnys.previous_session(session))
        # assert assumptions
        assert xnys_prev_session == pd.Timestamp("2021-01-15")

        def patch_now(ts: pd.Timestamp):
            monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: ts)

        xnys_open = xnys.session_open(session)
        patch_now(xnys_open - one_min)
        assert prices.last_requestable_session_all == xnys_prev_session

        patch_now(xnys_open)
        assert prices.last_requestable_session_all == session

        xhkg_open = xhkg.session_open("2021-01-18")
        patch_now(xhkg_open - one_min)
        assert prices.last_requestable_session_any == xnys_prev_session

        patch_now(xhkg_open)
        assert prices.last_requestable_session_any == prev_session

    def test_latest_requestable_minute(
        self,
        PricesMock,
        PricesMockFixedLimits,
        PricesMockIntradayOnlyFixedLimits,
        PricesMockDailyOnlyFixedLimits,
        symbols,
        right_limits,
        xnys,
        xlon,
        xhkg,
        monkeypatch,
    ):
        """Test `latest_requestable_minute` property."""

        def mock_now(tz=None) -> pd.Timestamp:
            return pd.Timestamp("2022-02-14 21:21:05", tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)
        calendars = [xnys, xhkg, xlon]

        now = mock_now(tz="UTC")

        # verifications against manual inspection of calendars' schedules.

        prices = PricesMock(symbols, calendars)
        assert prices.latest_requestable_minute == now.floor("min")

        prices = PricesMockDailyOnlyFixedLimits(symbols, calendars)
        assert prices.latest_requestable_minute == xnys.closes[right_limits[1]]

        prices = PricesMockIntradayOnlyFixedLimits(symbols, calendars)
        assert prices.latest_requestable_minute == right_limits[0]

        prices = PricesMockFixedLimits(symbols, calendars)
        assert prices.latest_requestable_minute == xnys.closes[right_limits[1]]

    def test__indices_aligned(
        self,
        PricesMock,
        PricesMockBreakendPmOrigin,
        GetterMock,
        symbols,
        xnys,
        xlon,
        xhkg,
        x247,
        monkeypatch,
        one_day,
    ):
        """Test `_indices_aligned` and `_indices_aligned_for_drg`."""
        now = pd.Timestamp("2021-12-31 23:59", tz=UTC)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)
        calendars = [xnys, xlon, xhkg]

        def assertions(prices: m.PricesBase, H1_aligned: pd.DatetimeIndex):
            bi = prices.bis.T1
            rtrn = prices._indices_aligned[bi]
            index = prices.cc.schedule.loc["2021-12-02":"2021-12-31"].index
            assert_index_equal(rtrn.index, index)
            assert rtrn.all()
            drg = GetterMock(daterange_sessions=(index[0], index[-1]), interval=bi)
            assert prices._indices_aligned_for_drg(drg)

            bi = prices.bis.T5
            rtrn = prices._indices_aligned[bi]
            index = prices.cc.schedule.loc["2021-11-02":"2021-12-31"].index
            assert_index_equal(rtrn.index, index)
            assert rtrn.all()
            drg = GetterMock(daterange_sessions=(index[0], index[-1]), interval=bi)
            assert prices._indices_aligned_for_drg(drg)

            rtrn = prices._indices_aligned[prices.bis.H1]
            index = prices.cc.schedule.loc["2021"].index
            assert_index_equal(rtrn.index, index)
            assert_index_equal(index[rtrn], H1_aligned)

        prices = PricesMock(symbols, calendars)
        # from manual inspection of schedules...
        H1_expected = pd.DatetimeIndex(
            [
                "2021-02-15",  # only xlon open
                "2021-04-05",  # only xnys open
                "2021-05-03",  # xlon not open, xhkg and xnys do not overlap
                "2021-05-31",  # only xhkg open
                "2021-08-30",  # xlon not open, xhkg and xnys do not overlap
                "2021-12-24",  # only xhkg and xlon open, no conflict as xhkg closes early  # noqa: E501
                "2021-12-27",  # only xnys open
                "2021-12-28",  # xlon not open, xhkg and xnys do not overlap
                "2021-12-31",  # all open but don't conflict due to early closes
            ]
        )
        assertions(prices, H1_expected)
        bi = prices.bis.H1
        # limit of range over which indices aligned at 1H
        start, end = start_end = pd.Timestamp("2021-12-24"), pd.Timestamp("2021-12-28")
        drg = GetterMock(daterange_sessions=start_end, interval=bi)
        assert prices._indices_aligned_for_drg(drg)
        drg = GetterMock(daterange_sessions=(start - one_day, end), interval=bi)
        assert not prices._indices_aligned_for_drg(drg)
        drg = GetterMock(daterange_sessions=(start, end + one_day), interval=bi)
        assert not prices._indices_aligned_for_drg(drg)

        prices = PricesMockBreakendPmOrigin(symbols, calendars)
        # additional alignments when PM_SUBSESSION_ORIGIN is "break_end"
        # xnys closed, xhkg and xlon don't conflict when pm has break_end origin
        additional_expected = pd.DatetimeIndex(
            [
                "2021-01-18",  # xhkg pm session close contiguous with start xlon session  # noqa: E501
                "2021-07-05",  # xhkg pm session and xlon session overlap but do not conflict  # noqa: E501
                "2021-09-06",  # xhkg pm session and xlon session overlap but do not conflict  # noqa: E501
                "2021-11-25",  # xhkg pm session close contiguous with start xlon session  # noqa: E501
            ]
        )
        H1_expected = H1_expected.union(additional_expected)
        assertions(prices, H1_expected)

        # calendars are not aligned
        calendars = [xnys, xnys, x247]
        prices = PricesMock(symbols, calendars)
        xnys_sessions = xnys.sessions_in_range("2021", "2021-12-31")
        all_dates = pd.date_range("2021", "2021-12-31")
        H1_expected = all_dates.difference(xnys_sessions)
        assertions(prices, H1_expected)

        # calendars are aligned
        calendars = [xlon, xlon, x247]
        prices = PricesMock(symbols, calendars)
        assertions(prices, all_dates)

    def test__indexes_status(
        self,
        PricesMock: type[m.PricesBase],
        GetterMock,
        symbols,
        xnys,
        xlon,
        xhkg,
        x247,
        xasx,
        cmes,
        monkeypatch,
        one_day,
    ):
        """Test `_indexes_status` and `_has_valid_fully_trading_indices`."""
        now = pd.Timestamp("2022", tz=UTC)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)
        symbols = ["ONE", "TWO"]

        def get_sessions(prices: m.PricesBase, bi: intervals.BI) -> pd.DatetimeIndex:
            return prices.cc.sessions_in_range(*prices.limits_sessions[bi])

        def get_calendars_sessions(
            prices: m.PricesBase,
            bi: intervals.BI,
            calendars: list[xcals.ExchangeCalendar],
        ) -> list[pd.DatetimeIndex]:
            range_ = prices.limits_sessions[bi]
            calendars_sessions = []
            for cal in calendars:
                sessions = cal.sessions_in_range(*range_)
                calendars_sessions.append(sessions)
            return calendars_sessions

        def assert_all_same(
            prices: m.PricesBase, bi: intervals.BI, value: bool | float
        ):
            sessions = get_sessions(prices, bi)
            expected = pd.Series(value, index=sessions, dtype="object")
            assert_series_equal(prices._indexes_status[bi], expected)

            drg = GetterMock(
                daterange_sessions=(sessions[0], sessions[-1]), interval=bi
            )
            rtrn = prices._has_valid_fully_trading_indices(drg)
            assert not rtrn if (np.isnan(value) or not value) else rtrn

        # xlon solo.
        prices = PricesMock(symbols, xlon)
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        # ...1H all partial indices
        assert_all_same(prices, prices.bis.H1, False)

        # xnys solo.
        prices = PricesMock(symbols, xnys)
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        # ...1H all partial indices
        assert_all_same(prices, prices.bis.H1, False)

        # 247 solo.
        prices = PricesMock(symbols, x247)
        for bi in prices.bis_intraday:
            assert_all_same(prices, bi, True)

        # 247 with xlon.
        prices = PricesMock(symbols, [x247, xlon])
        # all indices align and xlon partial indices are compensated by open 247
        for bi in prices.bis_intraday:
            assert_all_same(prices, bi, True)

        # 247 with xnys.
        prices = PricesMock(symbols, [x247, xnys])
        for bi in prices.bis_intraday[:-2]:
            assert_all_same(prices, bi, True)
        # ...1H conflict every day
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        expected = pd.Series(np.nan, index=sessions, dtype="object")
        # ...other than those sessions when xnys closed
        x247_sessions, xnys_sessions = get_calendars_sessions(prices, bi, [x247, xnys])
        expected[x247_sessions.difference(xnys_sessions)] = True
        assert_series_equal(prices._indexes_status[bi], expected)

        # XASX solo
        prices = PricesMock(symbols, xasx)
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        # on a normal day, no partial indices
        expected = pd.Series(True, index=sessions, dtype="object")
        # although there are a couple of early closes that are not aligned with 1H
        dates = ["2021-12-24", "2021-12-31"]
        expected[dates] = False
        assert_series_equal(prices._indexes_status[bi], expected)

        # XASX combined with xlon.
        prices = PricesMock(symbols, [xlon, xasx])
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        xasx_sessions, xlon_sessions = get_calendars_sessions(prices, bi, [xasx, xlon])
        # ...IH partial indices every session
        expected = pd.Series(False, index=sessions, dtype="object")
        # ...save when xlon closed
        expected[xasx_sessions.difference(xlon_sessions)] = True
        assert_series_equal(prices._indexes_status[bi], expected)

        # XASX combined with cmes.
        prices = PricesMock(symbols, [xlon, cmes])
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        # ...on a normal day, True (xasx enveloped by cmes and indices align)
        expected = pd.Series(True, index=sessions, dtype="object")
        # ...except when axsx early close (unaligned with !H) coincides with CMES hol.
        expected["2021-12-24"] = False
        assert_series_equal(prices._indexes_status[bi], expected)

        # xhkg and xasx
        prices = PricesMock(symbols, [xhkg, xasx])
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        xasx_sessions, xhkg_sessions = get_calendars_sessions(prices, bi, [xasx, xhkg])
        # ...on a normal day sessions will conflict
        expected = pd.Series(np.nan, index=sessions, dtype="object")
        # ...but if xasx open and xhkg closed, no partial indices
        expected[xasx_sessions.difference(xhkg_sessions)] = True
        # ...whilst if xhkg open and xasx closed, always partial indices
        expected[xhkg_sessions.difference(xasx_sessions)] = False
        assert_series_equal(prices._indexes_status[bi], expected)

        # from knowledge that 2021-02-12 through 2021-02-15 inclusive is period
        # when xasx is open whilst xhkg is closed, hence no partial indices
        start, end = pd.Timestamp("2021-02-12"), pd.Timestamp("2021-02-15")
        for date in pd.date_range(start, end):
            assert not xhkg.is_session(date)
        drg = GetterMock(daterange_sessions=(start, end), interval=bi)
        assert prices._has_valid_fully_trading_indices(drg)
        # but indices conflict on sessions either side
        drg = GetterMock(daterange_sessions=(start - one_day, end), interval=bi)
        assert not prices._has_valid_fully_trading_indices(drg)
        drg = GetterMock(daterange_sessions=(start, end + one_day), interval=bi)
        assert not prices._has_valid_fully_trading_indices(drg)

        # from knowledge that xhkg open on 2021-01-26 although xasx closed, and
        # hence no conflict but indices include a partial trading indice
        session = pd.Timestamp("2021-01-26")
        assert xhkg.is_session(session)
        assert not xasx.is_session(session)
        dr_sessions = [
            (session, session),
            (session - one_day, session),
            (session - one_day, session + one_day),
        ]
        for sessions_ in dr_sessions:
            drg = GetterMock(daterange_sessions=sessions_, interval=bi)
            assert not prices._has_valid_fully_trading_indices(drg)

    def test__subsessions_synced(self, PricesMock, xhkg, xlon, xnys):
        """Test `_subsessions_synced`."""
        prices = PricesMock("ONE", xhkg)

        calendar = prices.calendars[prices.symbols[0]]
        assert calendar.name == "XHKG"
        # knowing that, as of Jan 2022, XHKG subsessions are not synchronised at H1.
        assert prices._subsessions_synced(calendar, prices.BaseInterval.T1)
        assert prices._subsessions_synced(calendar, prices.BaseInterval.T5)
        assert not prices._subsessions_synced(calendar, prices.BaseInterval.H1)

        # verify consider synchronised with calendars that do not have a break:
        prices = PricesMock("ONE, TWO", [xlon, xnys])
        for cal in list(prices.calendars_unique):
            assert not cal.has_break
            for bi in prices.BaseInterval:
                if bi.is_daily:
                    continue
                assert prices._subsessions_synced(cal, bi)

    def test__ignore_breaks(
        self, PricesMock, PricesMockBreakendPmOrigin, xhkg, xlon, xnys
    ):
        """Tests methods and properties related to ignoring breaks.

        Tests methods:
            `_ignore_breaks_cal`
            `_ignore_breaks`
            `_ignore_breaks_any`
            `_get_bis_not_ignoring_breaks`
        """
        prices = PricesMock("ONE, TWO", [xhkg, xlon])
        # knowing that, as of Jan 2022, XHKG subsessions not synchronised at only H1.
        assert prices._ignore_breaks_any
        for bi in prices.bis_intraday:
            if bi is prices.bis.H1:
                assert prices._ignore_breaks_cal(xhkg, bi)
            else:
                assert not prices._ignore_breaks_cal(xhkg, bi)
        rtrn = prices._get_bis_not_ignoring_breaks(prices.bis_intraday, xhkg)
        assert rtrn == prices.bis_intraday[:2]
        rtrn = prices._get_bis_not_ignoring_breaks(prices.bis_intraday[1:], xhkg)
        assert rtrn == [prices.bis.T5]

        # verify breaks not ignored for calendar without breaks
        for bi in prices.bis_intraday:
            assert not prices._ignore_breaks_cal(xlon, bi)

        for bi in prices.bis_intraday:
            d = prices._ignore_breaks(bi)
            assert len(d) == 2
            if bi is prices.bis.H1:
                assert d[xhkg]
                assert not d[xlon]
            else:
                assert not d[xhkg] and not d[xlon]
        rtrn = prices._get_bis_not_ignoring_breaks(prices.bis_intraday, xlon)
        assert rtrn == prices.bis_intraday

        prices = PricesMockBreakendPmOrigin("ONE", xhkg)
        # verify breaks not ignored for any bi when PM_SUBSESSION_ORIGIN is "break_end"
        assert not prices._ignore_breaks_any
        for bi in prices.bis_intraday:
            assert not prices._ignore_breaks_cal(xhkg, bi)
            d = prices._ignore_breaks(bi)
            assert len(d) == 1
            assert not d[xhkg]
        rtrn = prices._get_bis_not_ignoring_breaks(prices.bis_intraday, xhkg)
        assert rtrn == prices.bis_intraday

        prices = PricesMock("ONE, TWO", [xnys, xlon])
        # verify no breaks ignored where no calendar has a break
        assert not prices._ignore_breaks_any

    def test_gpp_error(self, PricesMock, symbols, xlon):
        """Test `gpp` property raises error if called before `get`."""
        prices = PricesMock(symbols, xlon)
        match = (
            "An instance of `PricesBase.GetPricesParams` is not available as"
            " the `get` method has not been called."
        )
        with pytest.raises(errors.NoGetPricesParams, match=match):
            _ = prices.gpp

    def test_method_unavailable_no_daily_data(
        self, PricesMockIntradayOnly, symbols, xnys
    ):
        """Verify methods requiring daily data raise.

        Verifies methods requiring daily data raise error when daily
        interval is not included to base intervals.
        """
        prices = PricesMockIntradayOnly(symbols, xnys)
        for name in ("session_prices", "close_at"):
            method = getattr(prices, name)
            match = re.escape(
                f"`{name}` is not available as this method requires daily data although"
                " a daily base interval is not available to this prices class."
            )
            with pytest.raises(errors.MethodUnavailableNoDailyInterval, match=match):
                method("2000-01-06")


def test__calendars_latest_first_(PricesMock, cal_start, side):
    """Test `_calendars_latest_first_session` and `_calendars_latest_first_minute`."""
    xnys = xcals.get_calendar("XNYS", start=cal_start, side=side)
    xlon = xcals.get_calendar("XLON", start=cal_start, side=side)
    xhkg = xcals.get_calendar("XHKG", start=cal_start, side=side)

    symbols = "LON, NY, HK"
    prices = PricesMock(symbols, [xlon, xnys, xhkg])

    expected = max(xnys.first_session, xlon.first_session, xhkg.first_session)
    assert prices._calendars_latest_first_session == expected
    expected = max(xnys.first_minute, xlon.first_minute, xhkg.first_minute)
    assert prices._calendars_latest_first_minute == expected


def test__minute_to_session(PricesMock, cal_start, side, one_min, monkeypatch):
    """Test `_minute_to_session`.

    Tests for input unaffected by 'now' and tests for effect of 'now' and
    delay.
    """
    xnys = xcals.get_calendar("XNYS", start=cal_start, side=side)
    xlon = xcals.get_calendar("XLON", start=cal_start, side=side)

    symbols = "LON, NY"
    prices = PricesMock(symbols, [xlon, xnys])
    f = prices._minute_to_session

    # assert assumed sessions
    session = pd.Timestamp("2021-01-19")
    for cal in prices.calendars_unique:
        assert cal.is_session(session)
    next_session = pd.Timestamp("2021-01-20")
    xlon_prev_session = pd.Timestamp("2021-01-18")
    assert helpers.to_tz_naive(xlon.previous_session(session)) == xlon_prev_session
    xnys_prev_session = pd.Timestamp("2021-01-15")
    assert helpers.to_tz_naive(xnys.previous_session(session)) == xnys_prev_session

    # assert assumption that sessions overlap
    xlon_session_close = xlon.session_close(session)
    assert xlon_session_close == pd.Timestamp("2021-01-19 16:30", tz=UTC)
    xnys_session_open = xnys.session_open(session)
    assert xnys_session_open == pd.Timestamp("2021-01-19 14:30", tz=UTC)

    xnys_session_close = xnys.session_close(session)

    # minutes of nither calendar, all falling after session close
    minutes = [
        xnys_session_close + one_min,
        xnys_session_close,
    ]
    for minute in minutes:
        assert f(minute, "earliest", "previous") == session
        assert f(minute, "latest", "previous") == session
        assert f(minute, "earliest", "next") == next_session
        assert f(minute, "latest", "next") == next_session

    # minutes of xnys session, not of xlon session
    minutes = [
        xnys_session_close - one_min,
        xlon_session_close + one_min,
        xlon_session_close,
    ]
    for minute in minutes:
        assert f(minute, "earliest", "previous") == session
        assert f(minute, "latest", "previous") == session
        assert f(minute, "earliest", "next") == session
        assert f(minute, "latest", "next") == next_session

    # minutes of both calendars
    minutes = [
        xlon_session_close - one_min,
        xnys_session_open + one_min,
        xnys_session_open,
    ]
    for minute in minutes:
        assert f(minute, "earliest", "previous") == session
        assert f(minute, "latest", "previous") == session
        assert f(minute, "earliest", "next") == session
        assert f(minute, "latest", "next") == session

    # minutes of xlon session, not xnys session
    xlon_session_open = xlon.session_open(session)
    minutes = [
        xnys_session_open - one_min,
        xlon_session_open + one_min,
        xlon_session_open,
    ]
    for minute in minutes:
        assert f(minute, "earliest", "previous") == xnys_prev_session
        assert f(minute, "latest", "previous") == session
        assert f(minute, "earliest", "next") == session
        assert f(minute, "latest", "next") == session

    # minute of neither session, falling before session
    minute = xlon_session_open - one_min
    assert f(minute, "earliest", "previous") == xnys_prev_session
    assert f(minute, "latest", "previous") == xlon_prev_session
    assert f(minute, "earliest", "next") == session
    assert f(minute, "latest", "next") == session

    # Verify effect of 'now' and delay.

    symbols = "LON, LON2, NY, NY2"
    delays = [10, 20, 5, 15]
    prices = PricesMock(symbols, [xlon, xlon, xnys, xnys], delays=delays)
    f = prices._minute_to_session

    def patch_now(monkey, ts):
        monkey.setattr("pandas.Timestamp.now", lambda *_, **__: ts)

    minute = xlon_session_open
    args = ("latest", "previous")
    assert f(minute, *args) == session

    # now at limit of minute + min xlon delay (10)
    with monkeypatch.context() as m:
        now = minute + pd.Timedelta(10, "min")
        patch_now(m, now)
        assert f(minute, *args) == session
        # verify makes no difference if pass minute ahead of now
        assert f(minute + pd.Timedelta(5, "D"), *args) == session

        # now less than minute + min xlon delay (10)
        now = minute + pd.Timedelta(9, "min")
        patch_now(m, now)
        assert f(minute, *args) != session
        assert f(minute, *args) == xlon_prev_session

    minute = xnys_session_close
    args = ("earliest", "next")
    assert f(minute, *args) == next_session

    with monkeypatch.context() as m:
        # now at limit of minute + min xnys delay (5)
        now = minute + pd.Timedelta(5, "min")
        patch_now(m, now)
        assert f(minute, *args) == next_session

        # verify makes no difference if pass minute ahead of now
        assert f(minute + pd.Timedelta(5, "D"), *args) == next_session

        # now less than minute + min xlon delay (5)
        now = minute + pd.Timedelta(4, "min")
        patch_now(m, now)
        assert f(minute, *args) != next_session
        assert f(minute, *args) == session


def test__minute_to_latest_next_trading_minute(PricesMock, cal_start, side, one_min):
    """Test `_minute_to_latest_next_trading_minute`."""
    xnys = xcals.get_calendar("XNYS", start=cal_start, side=side)
    xlon = xcals.get_calendar("XLON", start=cal_start, side=side)
    xhkg = xcals.get_calendar("XHKG", start=cal_start, side=side)

    symbols = "LON, NY, HK"
    prices = PricesMock(symbols, [xlon, xnys, xhkg])
    f = prices._minute_to_latest_next_trading_minute

    # two consecutive sessions for all calendars (from knowledge of schedule)
    session = pd.Timestamp("2021-12-22")
    next_session = pd.Timestamp("2021-12-23")

    # verify prior to xhkg close
    xnys_session_open = xnys.session_open(session)
    for minute in (
        xhkg.session_open(session) - one_min,
        xhkg.session_open(session),
        xhkg.session_close(session) - one_min,
    ):
        assert f(minute) == xnys_session_open

    # verify xhkg close to prior to xlon close
    xhkg_next_session_open = xhkg.session_open(next_session)
    for minute in (
        xhkg.session_close(session),
        xlon.session_close(session) - one_min,
    ):
        assert f(minute) == xhkg_next_session_open

    # verify xlon close to prior to xnys close
    xlon_next_session_open = xlon.session_open(next_session)
    for minute in (
        xlon.session_close(session),
        xnys.session_close(session) - one_min,
    ):
        assert f(minute) == xlon_next_session_open

    # xnys close to prior to next xhkg close
    xnys_next_session_open = xnys.session_open(next_session)
    for minute in (
        xnys.session_close(session),
        xhkg.session_close(next_session) - one_min,
    ):
        assert f(minute) == xnys_next_session_open


def test__minute_to_earliest_previous_trading_minute(
    PricesMock, cal_start, side, one_min
):
    """Test `_minute_to_latest_next_trading_minute`."""
    xnys = xcals.get_calendar("XNYS", start=cal_start, side=side)
    xlon = xcals.get_calendar("XLON", start=cal_start, side=side)
    xhkg = xcals.get_calendar("XHKG", start=cal_start, side=side)

    symbols = "LON, NY, HK"
    prices = PricesMock(symbols, [xlon, xnys, xhkg])
    f = prices._minute_to_earliest_previous_trading_minute

    # two consecutive sessions for all calendars (from knowledge of schedule)
    session = pd.Timestamp("2021-12-22")
    next_session = pd.Timestamp("2021-12-23")

    # verify from xnys close to xnys open of next session
    xhkg_next_session_last_min = xhkg.last_minutes[next_session]
    for minute in (
        xnys.closes[next_session],
        xlon.closes[next_session],
        xnys.opens[next_session],
    ):
        assert f(minute) == xhkg_next_session_last_min
    assert f(xnys.opens[next_session] - one_min) != xhkg_next_session_last_min

    # verify from xnys open to xlon open of next session
    xnys_session_last_min = xnys.last_minutes[session]
    for minute in (
        xnys.opens[next_session] - one_min,
        xlon.opens[next_session],
    ):
        assert f(minute) == xnys_session_last_min
    assert f(xlon.opens[next_session] - one_min) != xhkg_next_session_last_min

    # verify from xlon open to xhkg open of next session
    xlon_session_last_min = xlon.last_minutes[session]
    for minute in (
        xlon.opens[next_session] - one_min,
        xhkg.opens[next_session],
    ):
        assert f(minute) == xlon_session_last_min
    assert f(xhkg.opens[next_session] - one_min) != xlon_session_last_min

    # verify from xhkg open of next session to xnys open of session
    xhkg_session_last_min = xhkg.last_minutes[session]
    for minute in (
        xhkg.opens[next_session] - one_min,
        xnys.opens[session],
    ):
        assert f(minute) == xhkg_session_last_min
    assert f(xnys.opens[session] - one_min) != xhkg_session_last_min


def test__get_trading_index(
    PricesMock,
    PricesMockBreakendPmOrigin,
    xhkg,
    monkeypatch,
):
    """Test `_get_trading_index`.

    Test covers only verifying that arguments are passed through and
    ignore_breaks argument provided by method.
    """
    now = pd.Timestamp("2021-12-31 23:59", tz=UTC)
    monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)

    cal = xhkg
    prices = PricesMock("ONE", cal)

    end_s = pd.Timestamp("2021-12-20")
    assert cal.is_session(end_s)
    start_s = cal.session_offset(end_s, -10)
    if start_s.tz is not None:
        start_s = start_s.tz_convert(None)

    def expected_index(
        bi: intervals.BI,
        ignore_breaks: bool,
        force: bool = False,
    ) -> pd.IntervalIndex:
        expected_args = (start_s, end_s, bi, True, "left")
        expected_kwargs = dict(force=force, ignore_breaks=ignore_breaks)
        return cal.trading_index(*expected_args, **expected_kwargs)

    interval = prices.bis.T5
    # ensure force being passed through
    for force in (True, False):
        ignore_breaks = False  # ignore_breaks should be False
        expected = expected_index(interval, ignore_breaks, force)
        rtrn = prices._get_trading_index(cal, interval, start_s, end_s, force)
        assert_index_equal(expected, rtrn)

    interval = prices.bis.H1
    # ensure force being passed through
    for force in (True, False):
        # ignore_breaks should be True
        ignore_breaks = True
        expected = expected_index(interval, ignore_breaks, force)
        rtrn = prices._get_trading_index(cal, interval, start_s, end_s, force)
        assert_index_equal(expected, rtrn)

    # verify effect of subsession pm origin as break_end
    prices_pm_origin = PricesMockBreakendPmOrigin("ONE", cal)
    ignore_breaks = False
    expected = expected_index(interval, ignore_breaks, force)
    rtrn = prices_pm_origin._get_trading_index(cal, interval, start_s, end_s, force)
    assert_index_equal(expected, rtrn)


class TestBis:
    """Test methods and properties that return base interval/s."""

    _now = pd.Timestamp("2022", tz=UTC)

    @pytest.fixture
    def now(self) -> abc.Iterator[pd.Timestamp]:
        yield self._now

    @pytest.fixture(autouse=True)
    def mock_now(self, now, monkeypatch):
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)

    @pytest.fixture
    def symbols(self) -> abc.Iterator[list[str]]:
        yield ["ONE", "TWO"]

    @pytest.fixture
    def PricesMockBis(
        self, PricesMock: type[m.PricesBase], daily_limit
    ) -> abc.Iterator[type[m.PricesBase]]:
        class PricesMockBis_(PricesMock):  # type: ignore[valid-type, misc]
            """Mock PricesBase class."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T1=intervals.TIMEDELTA_ARGS["T1"],
                    T2=intervals.TIMEDELTA_ARGS["T2"],
                    T5=intervals.TIMEDELTA_ARGS["T5"],
                    T10=intervals.TIMEDELTA_ARGS["T10"],
                    T15=intervals.TIMEDELTA_ARGS["T15"],
                    T30=intervals.TIMEDELTA_ARGS["T30"],
                    H1=intervals.TIMEDELTA_ARGS["H1"],
                    D1=intervals.TIMEDELTA_ARGS["D1"],
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T1: pd.Timestamp("2021-12-01", tz=UTC),
                BaseInterval.T2: pd.Timestamp("2021-11-01", tz=UTC),
                BaseInterval.T5: pd.Timestamp("2021-10-01", tz=UTC),
                BaseInterval.T10: pd.Timestamp("2021-09-01", tz=UTC),
                BaseInterval.T15: pd.Timestamp("2021-06-01", tz=UTC),
                BaseInterval.T30: pd.Timestamp("2021-03-01", tz=UTC),
                BaseInterval.H1: pd.Timestamp("2021-01-01", tz=UTC),
                BaseInterval.D1: daily_limit,
            }

            def __init__(
                self,
                *args,
                drg: daterange.GetterIntraday | None = None,
                ds_interval: intervals._TDIntervalBase | None | str = "none",
                anchor: mptypes.Anchor = None,
                **kwargs,
            ):
                """Mock constructor.

                Parameters
                ----------
                drg
                    Value will be returned by `.gpp.drg_intraday`.

                ds_interval
                    Value will be returned by `.gpp.ds_interval`.

                anchor
                    Value will be returned by `.gpp.anchor`.

                *args
                    passed to PricesBase.

                **kwargs
                    passed to PricesBase.
                """
                super().__init__(*args, **kwargs)

                @dataclasses.dataclass
                class GetPricesParamsMock:
                    """Mock GetPricesParams class."""

                    drg_intraday: daterange.GetterIntraday | None
                    drg_intraday_no_limit: daterange.GetterIntraday | None
                    ds_interval: intervals._TDIntervalBase | intervals.BI | None | str
                    anchor: mptypes.Anchor | None = None

                    @property
                    def pp_raw(self) -> str:
                        return "<mock pp>"

                    @property
                    def request_all_available_data(self) -> bool:
                        return False

                self._gpp = GetPricesParamsMock(drg, drg, ds_interval, anchor)

        yield PricesMockBis_

    @pytest.fixture
    def right_limits(self) -> abc.Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        """Use for an intrday / daily interval with a fixed right limit."""
        yield pd.Timestamp("2021-12-20 18:22", tz=UTC), pd.Timestamp("2021-12-20")

    @pytest.fixture
    def PricesRightLimitMockBis(
        self, PricesMockBis, right_limits
    ) -> abc.Iterator[type[m.PricesBase]]:
        right_limit_intraday, right_limit_daily = right_limits

        class PricesRightLimitMockBis_(PricesMockBis):  # type: ignore[valid-type, misc]
            """As PricesMockBis with right limit defined earlier than now."""

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T1=intervals.TIMEDELTA_ARGS["T1"],
                    T2=intervals.TIMEDELTA_ARGS["T2"],
                    T5=intervals.TIMEDELTA_ARGS["T5"],
                    T10=intervals.TIMEDELTA_ARGS["T10"],
                    T15=intervals.TIMEDELTA_ARGS["T15"],
                    T30=intervals.TIMEDELTA_ARGS["T30"],
                    H1=intervals.TIMEDELTA_ARGS["H1"],
                    D1=intervals.TIMEDELTA_ARGS["D1"],
                ),
            )

            BASE_LIMITS_RIGHT = {
                BaseInterval.T1: right_limit_intraday,
                BaseInterval.T2: right_limit_intraday,
                BaseInterval.T5: right_limit_intraday,
                BaseInterval.T10: right_limit_intraday,
                BaseInterval.T15: right_limit_intraday,
                BaseInterval.T30: right_limit_intraday,
                BaseInterval.H1: right_limit_intraday,
                BaseInterval.D1: right_limit_daily,
            }

        yield PricesRightLimitMockBis_

    @staticmethod
    def get_start_end_sessions(
        cc: calutils.CompositeCalendar, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return sessions for `prices` corresponding with `start` and `end`."""
        session_start = cc.minute_to_sessions(start, "next")[0]
        session_end = cc.minute_to_sessions(end, "previous")[-1]
        return (session_start, session_end)

    def get_mock_drg(
        self,
        GetterMock: type[daterange.GetterIntraday],
        cc: calutils.CompositeCalendar,
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> daterange.GetterIntraday:
        """Return instance of `GetterMock` representing period `start` through `end`.

        **kwargs passed on to `GetterMock`.
        """
        dr = (start, end), end
        daterange_sessions = self.get_start_end_sessions(cc, start, end)
        return GetterMock(
            daterange_sessions=daterange_sessions,
            daterange=dr,
            **kwargs,
        )

    def get_mock_drg_limit_available(
        self,
        prices: m.PricesBase,
        GetterMock: type[daterange.GetterIntraday],
        bi: intervals.BI,
        delta: int = 0,
        limit_end: bool = False,
        delta_end: int = 0,
    ) -> daterange.GetterIntraday:
        """Return drg representing limit of availability at `bi`.

        Parameters
        ----------
        bi
            drg will be set to reflect a daterange that covers period that
            data is availabile at `bi`.

        delta
            start of daterange will be displaced by `delta` mintues.

        limit_end
            True: set end of daterange as start (unadjusted for `delta`)
            Frame: set end of datearange to now.

        delta_end
            end of daterange will be displaced by `delta_end` mintues.
        """
        start = prices.BASE_LIMITS[bi]
        assert isinstance(start, pd.Timestamp)
        end = (start if limit_end else self._now) + pd.Timedelta(delta_end, "min")
        start += pd.Timedelta(delta, "min")
        return self.get_mock_drg(GetterMock, prices.cc, start, end)

    def get_mock_drg_limit_right_available(
        self,
        prices: m.PricesBase,
        GetterMock: type[daterange.GetterIntraday],
        bi: intervals.BI,
        delta: int = 0,
        limit_start: bool = False,
        delta_start: int = 0,
    ) -> daterange.GetterIntraday:
        """Return drg representing right limit of availability at `bi`.

        Parameters
        ----------
        bi
            drg will be set to reflect a daterange that covers period that
            data is availabile at `bi`.

        delta
            start of daterange will be displaced by `delta` mintues.

        limit_start
            True: set start of daterange as end (unadjusted for `delta`)
            Frame: set start of datearange to left limit for `bi`.

        delta_end
            end of daterange will be displaced by `delta_end` mintues.
        """
        end = prices.BASE_LIMITS_RIGHT[bi]
        start = end if limit_start else prices.BASE_LIMITS[bi]
        assert isinstance(start, pd.Timestamp)
        start += pd.Timedelta(delta_start, "min")
        end += pd.Timedelta(delta, "min")
        return self.get_mock_drg(GetterMock, prices.cc, start, end)

    def get_drg(
        self,
        calendar: xcals.ExchangeCalendar,
        composite_calendar: calutils.CompositeCalendar | None = None,
        delay: pd.Timedelta = pd.Timedelta(0),  # noqa: B008
        limit: pd.Timestamp | None = None,
        pp: mptypes.PP | None = None,
        interval: intervals.BI | None = None,
        ds_interval: intervals.TDInterval | None = None,
        end_alignment: mptypes.Alignment = mptypes.Alignment.BI,
        strict=True,
        ignore_breaks=False,
    ) -> daterange.GetterIntraday:
        """Return dictionary for getter 'pp' parameter."""
        if composite_calendar is None:
            composite_calendar = calutils.CompositeCalendar([calendar])
        if limit is None:
            limit = calendar.first_minute
        return daterange.GetterIntraday(
            calendar=calendar,
            composite_calendar=composite_calendar,
            delay=delay,
            limit=limit,
            pp=pp,
            interval=interval,
            ds_interval=ds_interval,
            end_alignment=end_alignment,
            strict=strict,
            ignore_breaks=ignore_breaks,
        )

    def test_bis_all_properties(self, PricesMockBis, symbols, xnys):
        prices = PricesMockBis(symbols, xnys)
        assert prices.bis is PricesMockBis.BaseInterval
        assert prices.bi_daily is PricesMockBis.BaseInterval.daily_bi()
        assert prices.bis_intraday == PricesMockBis.BaseInterval.intraday_bis()

    @staticmethod
    def set_prices_gpp_drg_properties(
        prices: m.PricesBase, drg: daterange.GetterIntraday
    ):
        """Set all drg properties of `prices`.gpp to `drg`."""
        prices.gpp.drg_intraday = drg
        prices.gpp.drg_intraday_no_limit = drg

    def test__bis_valid(self, PricesMockBis, GetterMock, symbols, xlon, xnys):
        """Test `_bis_valid`."""
        prices = PricesMockBis(symbols, [xlon, xnys])

        # sessions through which all intervals are aligned, from manual inspection.
        start_all_aligned = pd.Timestamp("2021-12-24")
        end_all_aligned = pd.Timestamp("2021-12-28")
        sessions_aligned = (start_all_aligned, end_all_aligned)

        def f(interval: int, drg: daterange.GetterIntraday) -> list[intervals.BI]:
            self.set_prices_gpp_drg_properties(prices, drg)
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            return prices._bis_valid

        drg_aligned = GetterMock(daterange_sessions=sessions_aligned)

        sessions_1h_unaligned = (pd.Timestamp("2020"), pd.Timestamp("2022"))
        drg_1h_unaligned = GetterMock(daterange_sessions=sessions_1h_unaligned)

        for drg in [drg_aligned, drg_1h_unaligned]:
            for interval in [1, 3, 7, 9, 11, 13, 17, 19, 23, 43]:
                assert f(interval, drg) == prices.bis[:1]

            for interval in [2, 4, 6, 8, 12, 14, 16, 18]:
                assert f(interval, drg) == prices.bis[:2]

            for interval in [5, 25, 35, 55]:
                assert f(interval, drg) == [prices.bis.T1, prices.bis.T5]

            for interval in [10, 20, 40, 50]:
                assert f(interval, drg) == prices.bis[:4]

            for interval in [15, 45, 75]:
                assert f(interval, drg) == [
                    prices.bis.T1,
                    prices.bis.T5,
                    prices.bis.T15,
                ]

            for interval in [30, 90]:
                assert f(interval, drg) == prices.bis_intraday[:-1]

        # Verify differs for 1h when drg unaligned
        for interval in [60, 120]:
            assert f(interval, drg_aligned) == prices.bis_intraday
            assert f(interval, drg_1h_unaligned) == prices.bis_intraday[:-1]

        # Verify when ds_interval is None that bis longer than period are excluded
        prices.gpp.ds_interval = None

        def get_drg(pp: dict) -> daterange.GetterIntraday:
            return daterange.GetterIntraday(
                calendar=xnys,
                composite_calendar=prices.cc,
                delay=pd.Timedelta(0),
                limit=xnys.first_minute,
                ignore_breaks=False,
                ds_interval=None,
                pp=pp,
            )

        # a trading hour, from inspection of schedules
        start = pd.Timestamp("2021-12-23 15:00", tz=UTC)
        end = pd.Timestamp("2021-12-23 16:00", tz=UTC)
        pp = dict(
            minutes=0,
            hours=0,
            days=0,
            weeks=0,
            months=0,
            years=0,
            start=start,
            end=end,
            add_a_row=False,
        )
        # all intraday bis should be valid save unaligned 1h
        prices.gpp.drg_intraday = get_drg(pp)
        assert prices._bis_valid == prices.bis_intraday[:-1]

        pp["end"] = pd.Timestamp("2021-12-23 15:12", tz=UTC)
        drg = get_drg(pp)
        self.set_prices_gpp_drg_properties(prices, drg)
        # only those bis <= 12 min duration should be valid
        assert prices._bis_valid == prices.bis_intraday[:-3]

        # verify same effect when pp defined with a duration (as opposed to start/end)
        pp["end"] = None
        pp["minutes"] = 12
        drg = get_drg(pp)
        prices.gpp.drg_intraday = drg
        assert prices._bis_valid == prices.bis_intraday[:-3]

    def test__bis_available(
        self,
        PricesMockBis,
        GetterMock,
        symbols,
        xlon,
        xnys,
    ):
        """Test `_bis_available_all`, `_bis_available_end` and `_bis_available_any`."""
        prices = PricesMockBis(symbols, [xlon, xnys])
        get_drg_args = (prices, GetterMock)

        def bis_available_all(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_all

        def bis_available_end(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_end

        def bis_available_any(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_any

        for i, bi in enumerate(prices.bis_intraday[:-1]):
            # start at limit for bi, end now
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi)
            assert bis_available_all(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            assert bis_available_all(bi.as_minutes, drg) == [bi]

            # start before limit for bi, end now
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi, -1)
            assert bis_available_all(30, drg) == prices.bis_intraday[i + 1 : -1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            assert bis_available_all(bi.as_minutes, drg) == []

            # start and end at limit for bi
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi, limit_end=True)
            assert bis_available_all(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[i:-1]

            assert bis_available_all(bi.as_minutes, drg) == [bi]
            assert bis_available_end(bi.as_minutes, drg) == [bi]
            assert bis_available_any(bi.as_minutes, drg) == [bi]

            # start and end beyond limit for bi
            match = re.escape(  # start of message only
                "The end of the requested period is earlier than the earliest"
                " timestamp at which intraday data is available for any base interval."
            )
            drg = self.get_mock_drg_limit_available(
                *get_drg_args, bi, -1, limit_end=True, delta_end=-1
            )

            if bi.as_minutes == 30:
                # Not even T30 data available to meet
                with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                    bis_available_all(30, drg)
                with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                    bis_available_end(30, drg)
                with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                    bis_available_any(30, drg)
            else:
                assert bis_available_end(30, drg) == prices.bis_intraday[i + 1 : -1]
                assert bis_available_any(30, drg) == prices.bis_intraday[i + 1 : -1]
                assert bis_available_all(30, drg) == prices.bis_intraday[i + 1 : -1]

            # Can only be met by this interval or a lower interval, i.e. can only be
            # met by intervals for which data not available
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                bis_available_end(bi.as_minutes, drg)
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                bis_available_any(bi.as_minutes, drg)

    def test__bis_available_fixed_right(
        self,
        PricesRightLimitMockBis,
        GetterMock,
        symbols,
        xlon,
        xnys,
    ):
        """Test `_bis_available_all`, `_bis_available_end` and `_bis_available_any`.

        Tests with Prices class that has a fixed right limit.
        """
        prices = PricesRightLimitMockBis(symbols, [xlon, xnys])
        get_drg_args = (prices, GetterMock)

        def bis_available_all(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_all

        def bis_available_end(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_end

        def bis_available_any(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "min")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_any

        for i, bi in enumerate(prices.bis_intraday[:-1]):
            # start at left limit, end at right limit for bi
            drg = self.get_mock_drg_limit_right_available(*get_drg_args, bi)
            assert bis_available_all(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            assert bis_available_all(bi.as_minutes, drg) == [bi]

            # start and end at right limit for bi
            drg = self.get_mock_drg_limit_right_available(
                *get_drg_args, bi, limit_start=True
            )
            assert bis_available_all(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            assert bis_available_all(bi.as_minutes, drg)
            assert bis_available_end(bi.as_minutes, drg)
            assert bis_available_any(bi.as_minutes, drg)

            # start before left limit for bi, end at right_limit
            drg = self.get_mock_drg_limit_right_available(
                *get_drg_args, bi, delta_start=-1
            )
            assert bis_available_all(30, drg) == prices.bis_intraday[i + 1 : -1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            assert bis_available_all(bi.as_minutes, drg) == []

            # start at left limit for bi, end after right_limit
            drg = self.get_mock_drg_limit_right_available(*get_drg_args, bi, 1)
            assert bis_available_all(30, drg) == []
            assert bis_available_end(30, drg) == []
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            # start before left limit for bi, end after right_limit
            drg = self.get_mock_drg_limit_right_available(
                *get_drg_args, bi, 1, delta_start=-1
            )
            assert bis_available_all(30, drg) == []
            assert bis_available_end(30, drg) == []
            assert bis_available_any(30, drg) == prices.bis_intraday[:-1]

            # start and end beyond limit for bi
            match = re.escape(  # start of message only
                "The start of the requested period is later than the latest"
                " timestamp at which intraday data is available for any base interval."
            )
            drg = self.get_mock_drg_limit_right_available(
                *get_drg_args, bi, 1, limit_start=True, delta_start=1
            )
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                bis_available_all(30, drg)
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                bis_available_end(30, drg)
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                bis_available_any(30, drg)

    def test_bis_stored_methods(self, PricesMockBis, GetterMock, symbols, xlon, xnys):
        """Tests `_bis_stored` and `_get_stored_bi_from_bis`."""
        prices = PricesMockBis(symbols, [xlon, xnys])
        get_drg_args = (prices, GetterMock)

        prices.gpp.ds_interval = intervals.to_ptinterval("30min")
        for bi in prices.bis_intraday:
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi)
            self.set_prices_gpp_drg_properties(prices, drg)
            assert prices._bis_stored == []
            assert prices._get_stored_bi_from_bis(prices.bis_intraday) is None

        bis_stored_available = [prices.bis.T2, prices.bis.T15, prices.bis.H1]
        for bi in bis_stored_available:
            prices._pdata[bi].requested_range = lambda *_, **__: True

        for bi in prices.bis_intraday:
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi)
            self.set_prices_gpp_drg_properties(prices, drg)
            # NB H1 not valid for ds_interval
            assert prices._bis_stored == [prices.bis.T2, prices.bis.T15]
            assert prices._get_stored_bi_from_bis(prices.bis_intraday) is prices.bis.T2
            bis = list(reversed(prices.bis_intraday))
            assert prices._get_stored_bi_from_bis(bis) is prices.bis.T15

    @pytest.fixture
    def prices_partial_indices(
        self, PricesMockBis: type[m.PricesBase], symbols, xasx
    ) -> tuple[m.PricesBase, pd.Timestamp]:
        """Instance of revised `PricesMockBis` to interrogate bis properties.

        Instance of `PricesMockBis` with sole calendar as xasx which has
        partial indices on "2021-12-24" for base intervals > T10 as a
        result of an early close at 03.10 (asserted by fixture).

        T1 base limit revised to "2021-12-29", i.e. after the early close
        such that T1 will not be an available bi for any period requested
        to "2021-12-24 03.10".

        Yields
        ------
        tuple[m.PricesBase, pd.Timestamp]
            [0] Instance of PricesMockBisAlt (revised `PricesMockBis`)
            [1] Early close
              pd.Timestamp("2021-12-24 03:10", tz=zoneinfo.ZoneInfo("UTC"))
        """
        early_close_session = pd.Timestamp("2021-12-24")
        early_close = xasx.session_close(early_close_session)
        # assert assumption that early close
        assert early_close == pd.Timestamp("2021-12-24 03:10", tz=UTC)

        revised_limits = PricesMockBis.BASE_LIMITS.copy()
        t1_limit = pd.Timestamp("2021-12-29", tz=UTC)
        revised_limits[PricesMockBis.BaseInterval.T1] = t1_limit

        class PricesMockBisAlt(PricesMockBis):  # type: ignore[valid-type, misc]
            """Mock PricesBase class with alternative limits."""

            BASE_LIMITS = revised_limits

        return PricesMockBisAlt(symbols, xasx), early_close

    def test__bis_no_partial_indices(self, prices_partial_indices, GetterMock, one_min):
        """Tests methods that return base intervals with no partial indices.

        Tests:
            `_bis_no_partial_indices`
            `_bis_available_no_partial_indices`
            `_bis_available_end_no_partial_indices`
        """
        prices, early_close = prices_partial_indices
        cc = prices.cc
        no_partial = prices._bis_no_partial_indices

        start = prices.BASE_LIMITS[prices.bis.T2]
        end = pd.Timestamp("2021-12-23 05:00", tz=UTC)
        # period with no partial indices
        drg = self.get_mock_drg(GetterMock, cc, start, end)
        self.set_prices_gpp_drg_properties(prices, drg)

        # T1 included as method doesn't check availability
        assert no_partial(prices.bis_intraday) == prices.bis_intraday
        # verify can pass through only select bis
        select_bis = prices.bis_intraday[2:4]
        assert no_partial(select_bis) == select_bis

        prices.gpp.ds_interval = intervals.TDInterval.H1
        # T1 excluded as not available (limit after drg)
        assert prices._bis_available_no_partial_indices == prices.bis_intraday[1:]
        assert prices._bis_available_end_no_partial_indices == prices.bis_intraday[1:]

        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)
        # all higher bis result in partial indices
        expected = [prices.bis.T1, prices.bis.T2, prices.bis.T5, prices.bis.T10]
        assert no_partial(prices.bis_intraday) == expected
        # verify can pass through only select bis
        select_bis = prices.bis_intraday[2:]
        assert no_partial(select_bis) == expected[-2:]

        assert prices._bis_available_no_partial_indices == expected[1:]
        assert prices._bis_available_end_no_partial_indices == expected[1:]

        start -= one_min
        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)
        assert no_partial(prices.bis_intraday) == expected
        # T2 can't cover start, T1 can't cover any, others partial, leaving T5 & T10
        assert prices._bis_available_no_partial_indices == prices.bis[2:4]
        # T2 can cover the end
        assert prices._bis_available_end_no_partial_indices == expected[1:]

    def test_bis_accuracy_methods(self, PricesMockBis, symbols, xnys, monkeypatch):
        """Test methods concerned with accuracy with which bis can represent period end.

        Tests following methods:
            `_bis_period_end_now`
            `_bis_most_accurate`

        Tests following properties:
            `_bis_end_most_accurate`
        """
        now = pd.Timestamp("2021-12-31 15:14", tz=UTC)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)

        cal = xnys
        ds_interval = intervals.TDInterval.H1  # all bis valid
        prices = PricesMockBis(symbols, cal, ds_interval=ds_interval)
        start = pd.Timestamp("2021", tz=UTC)  # start for all drg
        period_end_now = prices._bis_period_end_now
        bis_most_accurate = prices._bis_most_accurate

        # Verify methods for when end is earlier than now.

        # verify period end does not evaluate as now for any bis.
        for end in [
            pd.Timestamp("2021-12-31 15:13", tz=UTC),
            pd.Timestamp("2021-12-31 15:00", tz=UTC),
            pd.Timestamp("2021-12-31 12:00", tz=UTC),
            pd.Timestamp("2021-12-30 15:14", tz=UTC),
            pd.Timestamp("2021-12-30 15:13", tz=UTC),
            pd.Timestamp("2021-12-29 12:00", tz=UTC),
            pd.Timestamp("2021-06-29 12:00", tz=UTC),
            pd.Timestamp("2021-01-29 12:00", tz=UTC),
        ]:
            pp = get_pp(start=start, end=end)
            drg = self.get_drg(cal, pp=pp)
            self.set_prices_gpp_drg_properties(prices, drg)

            prices.gpp.anchor = mptypes.Anchor.OPEN
            assert period_end_now(prices.bis_intraday) == []
            prices.gpp.anchor = mptypes.Anchor.WORKBACK
            assert period_end_now(prices.bis_intraday) == []

        end = pd.Timestamp("2021-12-23 18:15", tz=UTC)
        pp = get_pp(start=start, end=end)
        drg = self.get_drg(cal, pp=pp)
        self.set_prices_gpp_drg_properties(prices, drg)
        # verify base intervals that can most accurately represent period end
        expected = [prices.bis.T1, prices.bis.T5, prices.bis.T15]
        prices.gpp.anchor = mptypes.Anchor.OPEN
        assert bis_most_accurate(prices.bis_intraday) == expected
        prices.gpp.anchor = mptypes.Anchor.WORKBACK  # anchor should make no difference.
        assert bis_most_accurate(prices.bis_intraday) == expected

        # with end late dec, all bis could serve prices at period end
        assert prices._bis_end_most_accurate == expected
        # although with end mid June, only base intervals, >T15 can
        end = pd.Timestamp("2021-06-16 18:15", tz=UTC)
        pp = get_pp(start=start, end=end)
        drg = self.get_drg(cal, pp=pp)
        self.set_prices_gpp_drg_properties(prices, drg)
        # of those able to represent end with greatest accuracy, leaves only T15
        assert prices._bis_end_most_accurate == [prices.bis.T15]

        # Verify methods for when end is now.

        pp = get_pp(start=start)  # no end date, i.e. to now
        drg = self.get_drg(cal, pp=pp)
        self.set_prices_gpp_drg_properties(prices, drg)

        # verify that when period end is 'now', what goes in comes back out...
        assert period_end_now(prices.bis_intraday) == prices.bis_intraday
        assert period_end_now(prices.bis_intraday[:3]) == prices.bis_intraday[:3]

        prices.gpp.anchor = mptypes.Anchor.OPEN
        # verify all bis are equally accurate when anchor is open and end 'now'
        assert bis_most_accurate(prices.bis_intraday) == prices.bis_intraday
        # when end is now, all bis can serve prices at period end
        assert prices._bis_end_most_accurate == prices.bis_intraday

        # although when anchor is workback, only those base intervals that are as
        # accurate as T1 (15:15) should be included.
        prices.gpp.anchor = mptypes.Anchor.WORKBACK
        expected = [prices.bis.T1, prices.bis.T5, prices.bis.T15]
        assert bis_most_accurate(prices.bis_intraday) == expected
        # when end is now, all bis can serve prices at period end
        assert prices._bis_end_most_accurate == expected

    def test__get_bi_from_bis(self, PricesMockBis, symbols, xhkg):
        cal = xhkg
        ds_interval = intervals.TDInterval.H1  # all bis valid
        prices = PricesMockBis(symbols, cal, ds_interval=ds_interval)
        prices.gpp.calendar = cal
        start = pd.Timestamp("2021", tz=UTC)  # start for all drg
        end = pd.Timestamp("2021-12-23 05:45", tz=UTC)
        pp = get_pp(start=start, end=end)
        drg = self.get_drg(cal, pp=pp)
        self.set_prices_gpp_drg_properties(prices, drg)
        prices.gpp.request_earliest_available_data = False
        f = prices._get_bi_from_bis

        bis_all = prices.bis_intraday
        # confirm most accurate bis for `end`
        expected = [prices.bis.T1, prices.bis.T5, prices.bis.T15]
        assert prices._bis_most_accurate(bis_all) == expected

        # no stored bis, verify returns highest most accurate bi
        assert f(bis_all, mptypes.Priority.END) == prices.bis.T15
        # although if priority PERIOD, verify simply returns highest bis
        # excluding H1 as H1 ignores breaks and there are other options
        assert f(bis_all, mptypes.Priority.PERIOD) == prices.bis.T30

        # if stored data available for T5
        for bi in [prices.bis.T1, prices.bis.T5]:
            prices._pdata[bi].requested_range = lambda *_, **__: True
        # verify when priority END that returns highest stored most accurate (T5)
        # rather simply highest most accurate (T15).
        assert f(bis_all, mptypes.Priority.END) == prices.bis.T5
        # if stored data also available for T15
        prices._pdata[prices.bis.T15].requested_range = lambda *_, **__: True
        # verify when priority PERIOD that returns highest stored (T15) rather than
        # highest prefered (T30)
        assert f(bis_all, mptypes.Priority.PERIOD) == prices.bis.T15

        # lose T15 form the store
        prices._pdata[prices.bis.T15].requested_range = lambda *_, **__: False
        # verify that when require earliest available data, returns highest interval
        # (that doesn't ignore breaks) regardless of what might be in the store.
        prices.gpp.request_earliest_available_data = True
        assert f(bis_all, mptypes.Priority.END) == prices.bis.T15
        assert f(bis_all, mptypes.Priority.PERIOD) == prices.bis.T30

        # verify same effect if ds_interval is None
        prices.gpp.request_earliest_available_data = False
        prices.gpp.ds_interval = None
        assert f(bis_all, mptypes.Priority.END) == prices.bis.T15
        assert f(bis_all, mptypes.Priority.PERIOD) == prices.bis.T30

        # verify returns H1 if breaks are no longer ignored for it...
        class PricesMockBisPMOrign(PricesMockBis):
            """Mock PricesBase class with PM_SUBSESSION_ORIGIN as 'break end'."""

            PM_SUBSESSION_ORIGIN = "break_end"

        prices = PricesMockBisPMOrign(symbols, cal, ds_interval=ds_interval)
        prices.gpp.calendar = cal
        drg = self.get_drg(cal, pp=pp)
        self.set_prices_gpp_drg_properties(prices, drg)
        prices.gpp.request_earliest_available_data = False
        f = prices._get_bi_from_bis
        bis_all = prices.bis_intraday

        # no stored bis, verify returns highest most accurate bi
        assert f(bis_all, mptypes.Priority.END) == prices.bis.T15
        # although if priority PERIOD, verify highest interval is now H1
        assert f(bis_all, mptypes.Priority.PERIOD) == prices.bis.H1

    def test_bis_properties(self, prices_partial_indices, GetterMock, one_min):
        """Test `_bis` and `_bis_end`."""
        prices, early_close = prices_partial_indices
        cc = prices.cc

        def match(
            start: pd.Timestamp,
            cal: xcals.ExchangeCalendar,
            part_period_available: bool = False,
            end_before_ll: bool = False,
        ) -> str:
            interval = prices.gpp.ds_interval
            anchor = prices.gpp.anchor
            factors = [bi for bi in prices.bis_intraday if not interval % bi]
            limit_start, limit_end = prices.limits[factors[-1]]
            earliest_minute = cal.minute_to_trading_minute(limit_start, "next")
            latest_minute = (
                cal.minute_to_trading_minute(limit_end, "previous") + one_min
            )
            available_period = (earliest_minute, latest_minute)
            if end_before_ll:
                s = (
                    "The end of the requested period is earlier than the earliest"
                    " timestamp at which intraday data is available for any base"
                    " interval."
                )
            else:
                s = (
                    "Data is unavailable at a sufficiently low base interval to"
                    f" evaluate prices at interval {interval} anchored '{anchor}'."
                )
            anchor_insert = " and that have no partial trading indices"
            insert1 = "" if anchor is mptypes.Anchor.OPEN else anchor_insert
            s += (
                f"\nBase intervals that are a factor of {interval}{insert1}:"
                f"\n\t{factors}."
                f"\nThe period over which data is available at {factors[-1]} is"
                f" {available_period}, although at this base interval the requested"
                f" period evaluates to {(start, early_close)}."
                "\nPeriod evaluated from parameters: <mock pp>."
            )
            if part_period_available:
                s += (
                    f"\nData is available from {helpers.fts(earliest_minute)} through"
                    f" to the end of the requested period. Consider passing `strict` as"
                    f" False to return prices for this part of the period."
                )
            return re.escape(s)

        # given fixture setup, T1 can not serve data for all or end of period
        start = prices.BASE_LIMITS[prices.bis.T5]  # only bi > T5 can cover all period
        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)
        prices.gpp.ds_interval = intervals.TDInterval.H1  # all bi are factors
        prices.gpp.anchor = mptypes.Anchor.OPEN

        assert prices._bis == prices.bis_intraday[2:]  # from T5
        assert prices._bis_end == prices.bis_intraday[1:]  # from T2

        # partial indices (due ot early close) within period makes bi > T10 invalid
        prices.gpp.anchor = mptypes.Anchor.WORKBACK
        # leaving only...
        assert prices._bis == [prices.bis.T5, prices.bis.T10]
        assert prices._bis_end == [prices.bis.T2, prices.bis.T5, prices.bis.T10]

        prices.gpp.anchor = mptypes.Anchor.OPEN  # all > T1 valid
        prices.gpp.ds_interval = intervals.TDInterval.T10  # only T2 through T10 factors
        # and start to left of T2 so T2 cannot serve
        assert prices._bis == [prices.bis.T5, prices.bis.T10]
        # although T2 can when only require the period end
        assert prices._bis_end == [prices.bis.T2, prices.bis.T5, prices.bis.T10]

        # set start to left of T5 limit
        start = prices.BASE_LIMITS[prices.bis.T5] - one_min
        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)

        # start precludes T2 and T5, all > T10 precluded as not factor of ds_interval
        assert prices._bis == [prices.bis.T10]
        # when only require end, as before
        assert prices._bis_end == [prices.bis.T2, prices.bis.T5, prices.bis.T10]

        prices.gpp.ds_interval = intervals.TDInterval.T4
        # >T2 precluded as not factors of interval, T2 precluded as limit > start
        msg = match(start, drg.cal, part_period_available=True)
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis

        assert prices._bis_end == [prices.bis.T2]  # although T2 can still serve end

        prices.gpp.ds_interval = intervals.TDInterval.T3  # until it can't...
        # can only be served by T1 which has limit later than `early_close`
        msg = match(start, drg.cal, part_period_available=False, end_before_ll=True)
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis_end

        prices.gpp.anchor = mptypes.Anchor.WORKBACK
        prices.gpp.ds_interval = intervals.TDInterval.H1

        # start precludes T2 through T5, ds_interval no longer precludes any indice
        # although, given workback anchor, partial indices preclude > T10, leaving:
        assert prices._bis == [prices.bis.T10]
        assert prices._bis_end == [prices.bis.T2, prices.bis.T5, prices.bis.T10]

        prices.gpp.ds_interval = intervals.TDInterval.T8
        # ...leaves nothing when require full period...
        msg = match(start, drg.cal, part_period_available=True)
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis
        # ...and only T2 when require end only
        assert prices._bis_end == [prices.bis.T2]

        prices.gpp.ds_interval = intervals.TDInterval.T7  # and now not even T2...
        # can only be met by T1, the limit for which falls after `early_close`
        msg = match(start, drg.cal, part_period_available=False, end_before_ll=True)
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis_end

        # verify error message for inferred interval
        prices.gpp.ds_interval = None
        prices.gpp.anchor = anchor = mptypes.Anchor.OPEN
        # set start to before leftmost limit
        highest_bi = prices.bis_intraday[-1]
        limit_start, limit_end = prices.limits[highest_bi]
        # Use cal from previous drg...
        earliest_minute = drg.cal.minute_to_trading_minute(limit_start, "next")
        latest_minute = drg.cal.minute_to_trading_minute(limit_end, "previous")
        available_period = (earliest_minute, latest_minute + one_min)
        start = limit_start - one_min
        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)

        msg = re.escape(
            "Data is unavailable at a sufficiently low base interval to evaluate"
            f" prices at an inferred interval anchored '{anchor}'.\nBase intervals:"
            f"\n\t{prices.bis_intraday}.\nThe period over which data is available at"
            f" {highest_bi} is {available_period}, although at this base"
            f" interval the requested period evaluates to {(start, early_close)}."
            "\nPeriod evaluated from parameters: <mock pp>."
            f"\nData is available from {helpers.fts(earliest_minute)} through to the"
            " end of the requested period. Consider passing `strict` as False to return"
            " prices for this part of the period."
        )
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis


def test_get_prices_params_cls(PricesMock, xnys, xhkg):
    """Test properties and methods of inner `GetPricesParams` class."""
    prices = PricesMock("ONE, TWO", [xhkg, xnys], delays=[15, 0])
    f = m.PricesBase.GetPricesParams

    def assert_properties(
        gpp: m.PricesBase.GetPricesParams,
        ds_interval: intervals.TDInterval,
        lead_symbol: str,
        anchor: mptypes.Anchor,
        openend: mptypes.OpenEnd,
        strict: bool,
        priority: mptypes.Priority,
    ):
        assert gpp.prices is prices
        assert gpp.ds_interval is ds_interval
        assert gpp.lead_symbol is lead_symbol
        assert gpp.anchor is anchor
        assert gpp.openend is openend
        assert gpp.strict is strict
        assert gpp.priority is priority

        calendar = prices.calendars[lead_symbol]
        assert gpp.calendar is calendar
        assert gpp.delay is prices.delays[lead_symbol]
        limit_f = gpp.intraday_limit
        for bi in prices.bis_intraday:
            bound = prices.limits[bi][0]
            expected_limit = calendar.minute_to_trading_minute(bound, "next")
            assert limit_f(bi) == expected_limit
        assert gpp.daily_limit == prices._earliest_requestable_calendar_session(
            calendar
        )

    def assert_drg_intraday_properties(
        drg: daterange.GetterIntraday,
        gpp: m.PricesBase.GetPricesParams,
        strict: bool,
        ds_interval: intervals.TDInterval,
        no_limit: bool = False,
    ):
        """Assert gpp parameters reflected in drg properties as expected."""
        assert isinstance(drg, daterange.GetterIntraday)
        assert drg.ds_interval is ds_interval
        assert drg._delay == gpp.delay
        assert drg._cc is prices.cc
        if no_limit:
            assert drg._limit == xnys.first_minute  # later to open of the two calendars
        else:
            for bi in gpp.prices.bis_intraday:
                assert drg._limit(bi) == gpp.intraday_limit(bi)
        assert drg._strict is strict
        assert drg.pp == gpp.pp(intraday=True)
        assert drg.anchor == gpp.anchor

    def assert_drg_daily_properties(
        drg: daterange.GetterDaily,
        gpp: m.PricesBase.GetPricesParams,
        strict: bool,
        ds_interval: intervals.TDInterval,
    ):
        """Assert gpp parameters reflected in drg properties as expected."""
        assert isinstance(drg, daterange.GetterDaily)
        assert drg.ds_interval is ds_interval
        assert drg._limit == gpp.daily_limit
        assert drg._strict is strict
        assert drg.pp == gpp.pp(intraday=False)

    start = pd.Timestamp("2021-12-15 12:22", tz=UTC)
    pp = get_pp(start=start, days=2)
    ds_interval = intervals.TDInterval.H2
    lead_symbol = "TWO"
    anchor = mptypes.Anchor.OPEN
    openend = mptypes.OpenEnd.MAINTAIN
    strict = True
    priority = mptypes.Priority.END
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    assert_properties(gpp, ds_interval, lead_symbol, anchor, openend, strict, priority)

    # expected starts from knowledge of schedule
    start_expected = pd.Timestamp("2021-12-15 14:30", tz=UTC)
    pp_expected = get_pp(start=start_expected, days=2)
    assert gpp.pp(intraday=True) == pp_expected
    start_expected_daily = pd.Timestamp("2021-12-15")
    pp_expected = get_pp(start=start_expected_daily, days=2)
    assert gpp.pp(intraday=False) == pp_expected
    assert not gpp.intraday_duration
    assert gpp.duration
    assert not gpp.request_earliest_available_data
    assert not gpp.request_all_available_data

    # assert parameters being passed through to drg.
    drg = gpp.drg_intraday
    assert_drg_intraday_properties(drg, gpp, strict, ds_interval)
    with pytest.raises(errors.StartTooEarlyError):
        drg.interval = prices.bis.T5
        _ = drg.daterange

    drg = gpp.drg_intraday_no_limit
    assert_drg_intraday_properties(drg, gpp, False, ds_interval, no_limit=True)
    drg.interval = prices.bis.T5
    acc_expected = pd.Timestamp("2021-12-16 21:00", tz=UTC)
    end_expected = pd.Timestamp("2021-12-16 22:30", tz=UTC)
    assert drg.daterange == ((start_expected, end_expected), acc_expected)
    assert not drg.ignore_breaks

    # check effect of anchor and openend being passed through to getter
    anchor_openends = [
        (mptypes.Anchor.OPEN, mptypes.OpenEnd.SHORTEN),
        (mptypes.Anchor.WORKBACK, mptypes.OpenEnd.MAINTAIN),
    ]
    for anchor, openend in anchor_openends:
        gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
        drg = gpp.drg_intraday_no_limit
        drg.interval = prices.bis.T5
        assert drg.daterange == ((start_expected, acc_expected), acc_expected)

    # Assert daily drg
    ds_interval = intervals.TDInterval.D2
    anchor, openend = mptypes.Anchor.OPEN, mptypes.OpenEnd.MAINTAIN
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    drg = gpp.drg_daily
    assert_drg_daily_properties(drg, gpp, strict, ds_interval)

    end_expected = pd.Timestamp("2021-12-16")
    assert drg.daterange == ((start_expected_daily, end_expected), end_expected)

    drg = gpp.drg_daily_raw
    assert_drg_daily_properties(drg, gpp, strict, intervals.ONE_DAY)
    assert drg.daterange == ((start_expected_daily, end_expected), end_expected)

    # alternative parameters
    end = pd.Timestamp("2021-12-15 03:10", tz=UTC)
    pp = get_pp(end=end, hours=3, minutes=30)
    ds_interval = intervals.TDInterval.H1
    lead_symbol = "ONE"
    anchor = mptypes.Anchor.OPEN
    openend = mptypes.OpenEnd.SHORTEN
    strict = False
    priority = mptypes.Priority.PERIOD
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    assert_properties(gpp, ds_interval, lead_symbol, anchor, openend, strict, priority)

    assert gpp.pp(intraday=True) == pp
    assert gpp.intraday_duration
    assert gpp.duration
    assert not gpp.request_earliest_available_data
    assert not gpp.request_all_available_data

    drg = gpp.drg_intraday
    assert_drg_intraday_properties(drg, gpp, strict, ds_interval)

    drg = gpp.drg_intraday_no_limit
    # assert ignore breaks and effect of
    drg.interval = prices.bis.H1
    assert drg.ignore_breaks
    # from knowledge of schedule...
    end_expected = pd.Timestamp("2021-12-15 02:30", tz=UTC)
    start_expected = pd.Timestamp("2021-12-14 05:30", tz=UTC)
    assert drg.daterange == ((start_expected, end_expected), end_expected)

    drg.interval = prices.bis.T5
    assert not drg.ignore_breaks
    start_expected = pd.Timestamp("2021-12-14 06:00", tz=UTC)
    assert drg.daterange == ((start_expected, end_expected), end_expected)

    # alternative parameters just to verify request_earliest_available_data True
    pp = get_pp(end=end)
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    assert not gpp.intraday_duration
    assert not gpp.duration
    assert gpp.request_earliest_available_data
    assert not gpp.request_all_available_data

    # alternative parameters just to verify request_all_available_data True
    pp = get_pp()
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    assert not gpp.intraday_duration
    assert not gpp.duration
    assert gpp.request_all_available_data
