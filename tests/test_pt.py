"""Tests for market_prices.pt module."""

from __future__ import annotations

import math
import re
import typing
from collections import abc
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest

import market_prices.pt as m
from market_prices import errors, helpers
from market_prices.helpers import UTC
from market_prices.intervals import TDInterval
from market_prices.utils import calendar_utils as calutils

from .utils import get_resource, multiple_sessions_freq

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


@pytest.fixture
def daily_pt() -> abc.Iterator[pd.DataFrame]:
    """Daily price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "ES=F"]
    > df = prices.get("1d", start="2021", end="2021-12-31")
    """
    yield get_resource("daily_pt")


@pytest.fixture
def intraday_pt() -> abc.Iterator[pd.DataFrame]:
    """Intraday price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "ES=F"]
    > df = prices.get("5min", start="2021-12-16", end="2022-01-06")
    """
    yield get_resource("intraday_pt")


@pytest.fixture
def intraday_1h_pt() -> abc.Iterator[pd.DataFrame]:
    """Intraday price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "ES=F"]
    > df = prices.get("1h", start="2021-12-16", end="2022-01-06", anchor="open")
    """
    yield get_resource("intraday_1h_pt")


@pytest.fixture
def intraday_asia_pt() -> abc.Iterator[pd.DataFrame]:
    """Intraday price table for symbols in asian timezones.

    Recreate table with:
    > symbols = ["BHP.AX", "9988.HK"]
    > df = prices.get("6h", start="2021-12-16", end="2022-01-06", anchor="open")
    """
    yield get_resource("intraday_asia_pt")


@pytest.fixture
def intraday_cc_overlapping_pt() -> abc.Iterator[pd.DataFrame]:
    """Intraday price table for symbols with overlapping calendars.

    Recreate table with:
    > symbols = ["BTC-USD", "ES=F"]
    > prices.get("15min", start="2021-12-21", end="2021-12-23", anchor="open")
    """
    yield get_resource("intraday_cc_overlapping_pt")


@pytest.fixture
def multiple_sessions_pt() -> abc.Iterator[pd.DataFrame]:
    """Multiple sessions price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "ES=F"]
    > df = prices.get("3d", start="2021", end="2021-12-31")
    """
    yield get_resource("multiple_sessions_pt")


@pytest.fixture
def multiple_sessions_alldays_pt() -> abc.Iterator[pd.DataFrame]:
    """Multiple sessions price table covering every day over period.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "BTC-USD"]
    > df = prices.get("3d", start="2021", end="2021-12-31", lead_symbol="BTC-USD")
    """
    yield get_resource("multiple_sessions_alldays_pt")


@pytest.fixture
def composite_intraday_pt() -> abc.Iterator[pd.DataFrame]:
    """Composite price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "BTC-USD"]
    > df = prices.get(start="2021-12-01", end="2022-01-06 15:11", composite=True)

    Covers period from 2021-12-01 09:30 (session open) through 2022-01-06 15:05 at
    5T interval. Then covers period from 2022-01-06 15:05 through 2022-01-06 15:11 at
    1T interval.
    """
    yield get_resource("composite_intraday_pt")


@pytest.fixture
def composite_daily_intraday_pt() -> abc.Iterator[pd.DataFrame]:
    """Composite price table for test symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L", "BTC-USD"]
    > df = prices.get(start="2020", end="2022-01-06 15:10", composite=True)

    Covers period from session 2020-01-02 through 2022-01-05, inclusive,
    as daily data. Then covers 2022-01-05 23:00 through 2022-01-06 20:10
    as intraday data with 5T interval. All UTC.
    """
    yield get_resource("composite_daily_intraday_pt")


@pytest.fixture
def intraday_detached_pt() -> abc.Iterator[pd.DataFrame]:
    """Intraday price table for symbols with non-overlapping sessions.

    Recreate table with:
    > symbols = ["MSFT", "9988.HK"]
    > df = prices.get("30min", start="2022-04-01", end="2022-04-07")
    """
    yield get_resource("intraday_detached_pt")


@pytest.fixture
def daily_pt_ss() -> abc.Iterator[pd.DataFrame]:
    """As daily_pt for single symbol "MSFT". NB. Frequency is None."""
    yield get_resource("daily_pt_ss")


@pytest.fixture
def intraday_pt_ss() -> abc.Iterator[pd.DataFrame]:
    """As intraday_pt for single symbol "MSFT"."""
    yield get_resource("intraday_pt_ss")


@pytest.fixture
def multiple_sessions_pt_ss() -> abc.Iterator[pd.DataFrame]:
    """As multiple_sessions_pt for single symbol "MSFT"."""
    yield get_resource("multiple_sessions_pt_ss")


@pytest.fixture
def composite_intraday_pt_ss() -> abc.Iterator[pd.DataFrame]:
    """As composite_intraday_pt for single symbol "MSFT"."""
    yield get_resource("composite_intraday_pt_ss")


@pytest.fixture
def composite_daily_intraday_pt_ss() -> abc.Iterator[pd.DataFrame]:
    """Composite daily intraday price table for single symbol "MSFT".

    Covers period from session 2020-01-02 through 2022-01-05, inclusive,
    as daily data. Then covers 2022-01-06 14:30 through 2022-01-06 20:10
    as intraday data with 5T interval. All UTC.
    """
    yield get_resource("composite_daily_intraday_pt_ss")


@pytest.fixture(
    params=[
        get_resource("daily_pt"),
        get_resource("intraday_pt"),
        get_resource("multiple_sessions_pt"),
        get_resource("composite_daily_intraday_pt"),
        get_resource("composite_intraday_pt"),
    ]
)
def price_tables(request) -> abc.Iterator[pd.DataFrame]:
    """Parameterized fixture of price tables."""
    yield request.param


@pytest.fixture(
    params=[
        get_resource("daily_pt_ss"),
        get_resource("intraday_pt_ss"),
        get_resource("multiple_sessions_pt_ss"),
        get_resource("composite_daily_intraday_pt_ss"),
        get_resource("composite_intraday_pt_ss"),
    ]
)
def price_tables_ss(request) -> abc.Iterator[pd.DataFrame]:
    """Parameterized fixture of price tables for single symbols."""
    yield request.param


@pytest.fixture(scope="session")
def symbols() -> abc.Iterator[list[str]]:
    """Symbols of regular resources' price tables."""
    yield ["MSFT", "AZN.L", "ES=F"]


@pytest.fixture(scope="session")
def symbol() -> abc.Iterator[str]:
    """Symbol of data of resources' single symbols tables."""
    yield "MSFT"


@pytest.fixture(scope="session")
def symbols_alldays() -> abc.Iterator[list[str]]:
    """Symbols of multiple_sessions_alldays_pt."""
    yield ["MSFT", "AZN.L", "BTC-USD"]


@pytest.fixture(scope="session")
def calendar_dates() -> abc.Iterator[tuple[str, str]]:
    yield "2020", "2022-01-08"


@pytest.fixture(scope="session")
def side() -> abc.Iterator[str]:
    yield "left"


@pytest.fixture(scope="session")
def xnys(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XNYS", *calendar_dates, side)


@pytest.fixture(scope="session")
def xlon(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XLON", *calendar_dates, side)


@pytest.fixture(scope="session")
def cmes(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("CMES", *calendar_dates, side)


@pytest.fixture(scope="session")
def calendars(xnys, xlon, cmes) -> abc.Iterator[list[xcals.ExchangeCalendar]]:
    yield [xnys, xlon, cmes]


@pytest.fixture(scope="session")
def xasx(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XASX", *calendar_dates, side)


@pytest.fixture(scope="session")
def xhkg(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XHKG", *calendar_dates, side)


@pytest.fixture(scope="session")
def calendars_asian(xasx, xhkg) -> abc.Iterator[list[xcals.ExchangeCalendar]]:
    yield [xasx, xhkg]


@pytest.fixture(scope="session")
def x247(calendar_dates, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("24/7", *calendar_dates, side)


@pytest.fixture(scope="session")
def calendars_alldays(xnys, xlon, x247) -> abc.Iterator[list[xcals.ExchangeCalendar]]:
    yield [xnys, xlon, x247]


@pytest.fixture(scope="session")
def calendars_overlapping(cmes, x247) -> abc.Iterator[list[xcals.ExchangeCalendar]]:
    yield [cmes, x247]


@pytest.fixture(scope="session")
def tz_default(xnys) -> abc.Iterator[ZoneInfo]:
    yield xnys.tz


@pytest.fixture(scope="session")
def tz_moscow() -> abc.Iterator[ZoneInfo]:
    yield ZoneInfo("Europe/Moscow")


class TestConstructorErrors:
    """Verify expected errors raised when attempting to call .pt accessor."""

    def test_new_constructor_errors(
        self, daily_pt, intraday_pt, multiple_sessions_pt, xnys
    ):
        """Verify raises errors defined in PT.__new__."""
        match = (
            "To use PT accessor index must be of type pd.DatetimeIndex or"
            " pd.IntervalIndex with left and right sides as pd.DatetimeIndex,"
            " although index is of type <class 'pandas.core.indexes."
        )
        daily_df = daily_pt.reset_index(drop=True)
        with pytest.raises(TypeError, match=match + "range.RangeIndex'>."):
            _ = daily_df.pt

        intraday_df = intraday_pt.copy()
        intraday_df.index = pd.interval_range(start=0, periods=len(intraday_pt), freq=1)

        with pytest.raises(TypeError, match=match + "base.Index'>."):
            _ = intraday_df.pt

        daily_df = daily_pt.copy()
        daily_df.index = daily_df.index + pd.Timedelta(1, "min")
        match = re.escape(
            "PT accessor not available where index is a pd.DatatimeIndex with"
            " one or more indices that have a time component (Index must be"
            " pd.IntervalIndex if any indice includes a time component)."
        )
        with pytest.raises(ValueError, match=match):
            _ = daily_df.pt

        multiple_sessions_pt.index = multiple_sessions_pt.index.left
        multiple_sessions_pt.index.freq = xnys.day * 3
        match = (
            "PT accessor not available where index DatatimeIndex and frequency"
            " is greater than one day."
        )
        with pytest.raises(ValueError, match=match):
            _ = multiple_sessions_pt.pt

    def test_init_constructor_errors(self, daily_pt, daily_pt_ss, symbol):
        """Verify raises errors defined in PT.__init__."""
        match = (
            "To use PricesTable accessor columns must by in dict_keys(['open',"
            " 'high', 'low', 'close', 'volume']) although columns evaluated as "
        )
        daily_pt_ss["extra_col"] = daily_pt_ss["open"] + 1
        with pytest.raises(
            KeyError, match=re.escape(match + f"{daily_pt_ss.columns}.")
        ):
            _ = daily_pt_ss.pt

        daily_pt[(symbol, "extra_col")] = daily_pt[(symbol, "open")] + 1
        with pytest.raises(
            KeyError, match=re.escape(match + f"{daily_pt.columns.levels[1]}.")
        ):
            _ = daily_pt.pt


class TestCommonProperties:
    """Verify properties common to all PT classes.

    Covers those properties that can be tested with a common test for all
    PT classes.
    """

    def test_common_properties(self, price_tables, symbols):
        """Test common properties of price tables with symbols."""
        df = price_tables
        assert_frame_equal(df.pt.prices, df)
        assert_index_equal(df.pt.index, df.index)
        assert_index_equal(df.pt.columns, df.columns)
        assert df.pt.symbols == symbols
        assert df.pt.has_symbols

    def test_common_properties_ss(self, price_tables_ss):
        """Test common properties of price tables for a single symbol."""
        df = price_tables_ss
        assert_frame_equal(df.pt.prices, df)
        assert_index_equal(df.pt.index, df.index)
        assert_index_equal(df.pt.columns, df.columns)
        assert df.pt.symbols is None
        assert not df.pt.has_symbols

        assert_frame_equal(df, df.pt.data_for_all_start)
        assert_frame_equal(df, df.pt.data_for_all_end)
        assert_frame_equal(df, df.pt.data_for_all)


class TestPriceTables:
    """Verify properties and methods common to all PT classes.

    Verifies those properties and methods that are common to all PT classes
    although where tests are necessarily class specific.
    """

    @pytest.fixture(scope="class")
    def session_utc(self) -> abc.Iterator[pd.Timestamp]:
        yield pd.Timestamp("2021-11-02", tz=UTC)

    @pytest.fixture(scope="class")
    def session_naive(self, session_utc) -> abc.Iterator[pd.Timestamp]:
        yield session_utc.tz_convert(None)

    @pytest.fixture(scope="class", autouse=True)
    def minute_utc(self) -> abc.Iterator[pd.Timestamp]:
        yield pd.Timestamp("2021-12-21 15:31", tz=UTC)

    @pytest.fixture(scope="class", autouse=True)
    def minute_naive(self, minute_utc) -> abc.Iterator[pd.Timestamp]:
        yield minute_utc.tz_convert(None)

    @pytest.fixture(scope="class", autouse=True)
    def minute_default_tz(self, minute_utc, tz_default) -> abc.Iterator[pd.Timestamp]:
        yield minute_utc.tz_convert(tz_default)

    def assert_interval_index_tz(self, df: pd.DataFrame, tz: ZoneInfo | None):
        """Assert tz of interval index."""
        assert df.index.left.tz is tz
        assert df.index.right.tz is tz

    def assert_frames_equal(self, df0: pd.DataFrame, df1: pd.DataFrame):
        """Assert data frame values are equivalent. Does not check index."""
        assert_frame_equal(df0.reset_index(drop=True), df1.reset_index(drop=True))

    def assert_interval_index_tz_properties(self, df, default_tz: ZoneInfo | None):
        self.assert_interval_index_tz(df, default_tz)
        assert df.pt.tz is default_tz
        self.assert_interval_index_tz(df.pt.naive, None)
        assert df.pt.naive.pt.tz is None
        self.assert_frames_equal(df.pt.naive, df)
        self.assert_interval_index_tz(df.pt.utc, UTC)
        assert df.pt.utc.pt.tz is UTC
        self.assert_frames_equal(df.pt.utc, df)

    def assert_set_tz_not_implemented(self, df):
        match = (
            f"set_tz is not implemented for {type(df.pt)}. Index of this class"
            " can be only timezone naive or have timezone as 'UTC'. Use"
            " .pt.utc and .pt.naive methods."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.set_tz(ZoneInfo("Europe/Moscow"))

    def assert_not_implemented(self, method, *args, **kwargs):
        match = (
            method.__name__ + f" is not implemented for {method.__self__.__class__}."
        )
        with pytest.raises(NotImplementedError, match=match):
            method(*args, **kwargs)

    def assert_reindexing(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        calendars: list[xcals.ExchangeCalendar],
    ):
        # NB ES=F / CMES not included to testing as the CMES calendar does not
        # accurately follow ES=F trading days / hours.
        for symbol, cal in zip(symbols, calendars):
            # values will be missing from rows as exchange times differ.
            assert df[symbol].isna().any(axis=None)
            reindexed = df.pt.reindex_to_calendar(cal)
            # values should not be missing in data reindexed to exchange calendar.
            assert reindexed[symbol].notna().all(axis=None)

    def assert_data_for_all(self, df: pd.DataFrame):
        interval_index = isinstance(df.index, pd.IntervalIndex)

        dfna = df[df.isna().any(axis=1)]

        start, end = dfna.pt.first_ts, dfna.pt.last_ts
        if interval_index:
            # index indice with right as end, not any following indice with left as end
            end -= helpers.ONE_SEC
        if interval_index and start == dfna.index[0].right:
            # for intervals of zero length (i.e. daily composite) require start to
            # be earlier than that interval's left side in order for that interval
            # to be included in the subset.
            start -= helpers.ONE_SEC

        test_df = df[start:end]
        bv_notna = test_df.notna().all(axis=1)
        df_expected_bounds = test_df[bv_notna]

        left_bound = df_expected_bounds.pt.first_ts
        if interval_index and left_bound == df_expected_bounds.index[0].right:
            left_bound -= helpers.ONE_SEC
        right_bound = df_expected_bounds.pt.last_ts
        if interval_index:
            right_bound -= helpers.ONE_SEC

        assert_frame_equal(test_df.pt.data_for_all, df[left_bound:right_bound])
        assert_frame_equal(test_df.pt.data_for_all_start, df[left_bound:end])
        assert_frame_equal(test_df.pt.data_for_all_end, df[start:right_bound])

    def test_daily_pt(
        self, daily_pt, daily_pt_ss, symbols, calendars, session_utc, session_naive
    ):
        df = daily_pt
        assert isinstance(df.pt, m.PTDaily)
        assert not df.pt.is_intraday
        assert df.pt.is_daily

        assert df.pt.first_ts == df.index[0]
        assert df.pt.last_ts == df.index[-1]

        assert df.index.tz is None
        assert df.pt.tz is None
        assert df.pt.naive.index.tz is None
        assert df.pt.naive.pt.tz is None
        assert_frame_equal(df.pt.naive, df)
        assert df.pt.convert_to_table_tz(session_utc) == session_naive
        assert df.pt.convert_to_table_tz(session_naive) == session_naive
        assert df.pt.utc.index.tz is UTC
        assert df.pt.utc.pt.tz is UTC
        assert_frame_equal(df.pt.utc, df, check_index_type=False)
        assert df.pt.utc.pt.convert_to_table_tz(session_utc) == session_utc
        assert df.pt.utc.pt.convert_to_table_tz(session_naive) == session_utc
        self.assert_set_tz_not_implemented(df)

        assert df.pt.freq is None
        assert df.pt.interval is TDInterval.D1
        assert df.pt.has_regular_interval

        xnys = calendars[0]
        assert_index_equal(daily_pt_ss.pt.get_trading_index(xnys), daily_pt_ss.index)

        for cal in calendars:
            ti = df.pt.get_trading_index(cal)
            assert isinstance(ti, pd.DatetimeIndex)
            assert ti.freq == cal.day
            assert df.index[0] <= ti[0]
            assert df.index[-1] >= ti[-1]

        self.assert_reindexing(df, symbols[:-1], calendars[:-1])

        self.assert_data_for_all(df)

    def test_intraday_pt(
        self,
        intraday_pt,
        intraday_pt_ss,
        intraday_1h_pt,
        tz_default,
        tz_moscow,
        symbols,
        calendars,
        minute_utc,
        minute_naive,
        minute_default_tz,
    ):
        df = intraday_pt
        df_1h = intraday_1h_pt
        assert isinstance(df.pt, m.PTIntraday)
        assert df.pt.is_intraday
        assert not df.pt.is_daily

        assert df.pt.first_ts == df.index[0].left
        assert df.pt.last_ts == df.index[-1].right

        self.assert_interval_index_tz_properties(df, tz_default)

        f = df.pt.convert_to_table_tz
        assert f(minute_utc) == minute_default_tz
        assert f(minute_utc).tz == tz_default
        assert f(minute_naive) == minute_naive.tz_localize(tz_default)
        assert f(minute_naive).tz == tz_default
        assert f(minute_default_tz) == minute_default_tz
        assert f(minute_default_tz).tz == tz_default

        df_tz_moscow = df.pt.set_tz(tz_moscow)
        self.assert_interval_index_tz(df_tz_moscow, tz_moscow)
        self.assert_frames_equal(df_tz_moscow, df)

        assert df.pt.freq is None
        assert df.pt.interval is TDInterval.T5
        assert df.pt.has_regular_interval

        xnys = calendars[0]
        assert_index_equal(
            intraday_pt_ss.pt.get_trading_index(xnys), intraday_pt_ss.pt.utc.index
        )
        for cal in calendars:
            ti = df.pt.get_trading_index(cal)
            assert isinstance(ti, pd.IntervalIndex)
            assert df.index[0].left <= ti[0].left
            assert df.index[-1].right >= ti[-1].right

        self.assert_reindexing(df, symbols[:-1], calendars[:-1])
        xlon = calendars[1]
        # verify raises error when opens are not synchronised for interval.
        sessions_xlon = pd.DatetimeIndex(df_1h.pt.sessions(xlon).unique())
        sessions_xnys = pd.DatetimeIndex(df_1h.pt.sessions(xnys).unique())
        non_compat_sessions = sessions_xlon.intersection(sessions_xnys)

        match = re.escape(
            f"Table index not compatible with calendar {xlon.name}. At least one"
            " table indice would conflict with a calendar indice for each of the"
            f" following sessions: \n{non_compat_sessions}."
        )
        with pytest.raises(errors.IndexConflictError, match=match):
            df_1h.pt.reindex_to_calendar(xlon)

        self.assert_data_for_all(df)

    def test_multiple_sessions_pt(
        self,
        multiple_sessions_pt,
        multiple_sessions_alldays_pt,
        xnys,
        minute_utc,
        minute_naive,
        minute_default_tz,
    ):
        df = multiple_sessions_pt
        assert isinstance(df.pt, m.PTMultipleSessions)
        assert not df.pt.is_intraday
        assert not df.pt.is_daily

        assert df.pt.first_ts == df.index[0].left
        assert df.pt.last_ts == df.index[-1].right

        self.assert_interval_index_tz_properties(df, None)
        self.assert_set_tz_not_implemented(df)

        f = df.pt.convert_to_table_tz
        assert f(minute_utc) == minute_naive
        assert f(minute_naive) == minute_naive
        assert f(minute_default_tz) == minute_default_tz.tz_convert(None)

        assert df.pt.freq == multiple_sessions_freq
        assert df.pt.interval is TDInterval.D3
        assert df.pt.has_regular_interval

        self.assert_not_implemented(df.pt.get_trading_index, xnys)
        self.assert_not_implemented(df.pt.reindex_to_calendar, xnys)

        # No na rows for regular table
        assert_frame_equal(df, df.pt.data_for_all_start)
        assert_frame_equal(df, df.pt.data_for_all_end)
        assert_frame_equal(df, df.pt.data_for_all)
        self.assert_data_for_all(multiple_sessions_alldays_pt)

    def test_composite_intraday_pt(
        self,
        composite_intraday_pt,
        tz_default,
        tz_moscow,
        xnys,
        minute_utc,
        minute_naive,
        minute_default_tz,
    ):
        df = composite_intraday_pt
        assert isinstance(df.pt, m.PTIntraday)
        assert df.pt.is_intraday
        assert not df.pt.is_daily

        assert df.pt.first_ts == df.index[0].left
        assert df.pt.last_ts == df.index[-1].right

        self.assert_interval_index_tz_properties(df, tz_default)

        f = df.pt.convert_to_table_tz
        assert f(minute_utc) == minute_default_tz
        assert f(minute_utc).tz == tz_default
        assert f(minute_naive) == minute_naive.tz_localize(tz_default)
        assert f(minute_naive).tz == tz_default
        assert f(minute_default_tz) == minute_default_tz
        assert f(minute_default_tz).tz == tz_default

        df_tz_moscow = df.pt.set_tz(tz_moscow)
        self.assert_interval_index_tz(df_tz_moscow, tz_moscow)
        self.assert_frames_equal(df_tz_moscow, df)

        assert df.pt.freq is None
        assert df.pt.interval is None
        assert not df.pt.has_regular_interval

        match = " requires price table to have a regular interval."
        with pytest.raises(ValueError, match="`get_trading_index`" + match):
            df.pt.get_trading_index(xnys)
        with pytest.raises(ValueError, match="`reindex_to_calendar`" + match):
            df.pt.reindex_to_calendar(xnys)

        self.assert_data_for_all(df)

    def test_composite_daily_intraday_pt(
        self,
        composite_daily_intraday_pt,
        xnys,
        minute_utc,
        minute_naive,
        minute_default_tz,
    ):
        df = composite_daily_intraday_pt
        assert isinstance(df.pt, m.PTDailyIntradayComposite)
        assert not df.pt.is_intraday
        assert not df.pt.is_daily

        assert df.pt.first_ts == df.index[0].left
        assert df.pt.last_ts == df.index[-1].right

        self.assert_interval_index_tz_properties(df, UTC)
        self.assert_set_tz_not_implemented(df)

        for minute in (minute_utc, minute_naive, minute_default_tz):
            rtrn = df.pt.convert_to_table_tz(minute)
            assert (rtrn, rtrn.tz) == (minute_utc, UTC)

        assert df.pt.freq is None
        assert df.pt.interval is None
        assert not df.pt.has_regular_interval

        self.assert_not_implemented(df.pt.get_trading_index, xnys)
        self.assert_not_implemented(df.pt.reindex_to_calendar, xnys)

        self.assert_data_for_all(df)

    def test_data_for_all(self, intraday_detached_pt):
        """Test data_for_all* props return empty when all rows have missing data."""
        df = intraday_detached_pt
        assert df.pt.data_for_all_start.empty
        assert df.pt.data_for_all_end.empty
        assert df.pt.data_for_all.empty


class TestIndicesTradingStatus:
    """Tests `indices_trading_status` method and dependent methods."""

    def assertions_no_partial_indices(self, df, symbols, calendars, reindex=True):
        # indices for which prices available for a symbol are considered trading indices
        # indices for which prices are not availbale are considered non-trading
        for symbol, cal in zip(symbols, calendars):
            bv = df[symbol].notna().all(axis=1)
            assert_series_equal(df.pt.indices_trading_status(cal), bv)
            sessions = df[bv].index
            non_sessions = df[~bv].index
            assert_index_equal(df.pt.indices_trading(cal), sessions)
            assert_index_equal(df.pt.indices_non_trading(cal), non_sessions)
            assert df.pt.indices_partial_trading(cal).empty
            assert not df.pt.indices_partial_trading_info(cal)

            assert not df.pt.indices_all_trading(cal)
            if reindex:
                df_ = df.pt.reindex_to_calendar(cal)
                assert df_.pt.indices_all_trading(cal)

    def test_daily_pt(self, daily_pt, symbols, calendars):
        # do not include ES=F as trading does not fully align with calendar.
        self.assertions_no_partial_indices(daily_pt, symbols[:-1], calendars[:-1])

    def test_intraday_pt(self, intraday_pt, symbols, calendars):
        self.assertions_no_partial_indices(intraday_pt, symbols, calendars)

    def test_composite_intraday_pt(self, composite_intraday_pt, symbols, calendars):
        self.assertions_no_partial_indices(
            composite_intraday_pt, symbols, calendars, reindex=False
        )

    def test_composite_daily_intraday_pt(
        self, composite_daily_intraday_pt, symbols, calendars
    ):
        # do not include ES=F as trading does not fully align with calendar.
        self.assertions_no_partial_indices(
            composite_daily_intraday_pt, symbols[:-1], calendars[:-1], reindex=False
        )

    def test_intraday_1h_pt(self, intraday_1h_pt, symbols, calendars, side):
        df = intraday_1h_pt
        # MSFT
        symbol, cal = symbols[0], calendars[0]
        bv = df[symbol].notna().all(axis=1)  # rows for which have prices
        # partial indices are the last indice of each session.
        bv_partial_trading = (bv + bv.shift(-1) == 1) & bv
        bv_trading_status = pd.Series(bv, dtype="object")
        bv_trading_status.loc[bv_partial_trading] = np.nan
        assert_series_equal(df.pt.indices_trading_status(cal), bv_trading_status)
        assert_index_equal(
            df.pt.indices_trading(cal), df.index[bv & ~bv_partial_trading]
        )
        assert_index_equal(df.pt.indices_non_trading(cal), df.index[~bv])

        partial_indices = df.index[bv_partial_trading]
        assert_index_equal(df.pt.indices_partial_trading(cal), partial_indices)

        # create indices_partial_trading_info
        delta = pd.Timedelta(30, "min")
        d = {}
        for indice in partial_indices:
            non_trading_part = pd.Interval(indice.right - delta, indice.right, side)
            d[indice] = pd.IntervalIndex([non_trading_part])
        assert d == df.pt.indices_partial_trading_info(cal)

        assert not df.pt.indices_all_trading(cal)

        # AZN.L
        symbol, cal = symbols[1], calendars[1]

        bv = df[symbol].notna().all(axis=1)
        # normally, partial indices are first indice of each session...
        bv_partial_trading = (bv + bv.shift(1) == 1) & bv
        bv_trading_status = pd.Series(bv, dtype="object")
        bv_trading_status.loc[bv_partial_trading] = np.nan

        # ...but 2021-12-24 is irregular due to different exchange hours.
        date = "2021-12-24"
        # First indice is a trading indice
        left = cal.session_open(date)
        right = left + pd.Timedelta(1, "h")
        indice1 = pd.Interval(left, right, side)
        bv_trading_status.loc[indice1] = True
        bv_partial_trading.loc[indice1] = False

        # last indice is a parital trading indice.
        left = cal.session_close(date) - pd.Timedelta(30, "min")
        right = left + pd.Timedelta(1, "h")
        indice2 = pd.Interval(left, right, side)
        bv_trading_status.loc[indice2] = np.nan
        bv_partial_trading.loc[indice2] = True

        assert_series_equal(df.pt.indices_trading_status(cal), bv_trading_status)
        assert_index_equal(
            df.pt.indices_trading(cal), df.index[bv & ~bv_partial_trading]
        )
        assert_index_equal(df.pt.indices_non_trading(cal), df.index[~bv])
        partial_indices = df.index[bv_partial_trading]
        assert_index_equal(df.pt.indices_partial_trading(cal), partial_indices)

        delta = pd.Timedelta(30, "min")
        d = {}
        for indice in partial_indices:
            if indice == indice2:
                # manual adjustment for 2021-12-24
                non_trading_part = pd.Interval(indice.right - delta, indice.right, side)
            else:
                non_trading_part = pd.Interval(indice.left, indice.left + delta, side)
            d[indice] = pd.IntervalIndex([non_trading_part])

        assert d == df.pt.indices_partial_trading_info(cal)
        assert not df.pt.indices_all_trading(cal)

    def test_multiple_sessions_pt(
        self, multiple_sessions_alldays_pt, symbols_alldays, calendars_alldays, one_day
    ):
        df = multiple_sessions_alldays_pt
        symbols, calendars = symbols_alldays, calendars_alldays
        for symbol, cal in zip(symbols, calendars):
            indices_trading_status = pd.Series(np.nan, index=df.index, dtype="object")
            bv = df[symbol].notna().all(axis=1)
            indices_non_trading = df.index[~bv]
            indices_trading_status.loc[indices_non_trading] = False
            assert_index_equal(df.pt.indices_non_trading(cal), indices_non_trading)

            all_sessions = cal.sessions
            left, right = df.pt.first_ts, df.pt.last_ts
            bv_sessions = (df.pt.first_ts <= all_sessions) & (
                all_sessions <= df.pt.last_ts
            )
            sessions = all_sessions[bv_sessions]
            non_sessions = pd.date_range(left, right).difference(sessions)

            info = df.pt.indices_partial_trading_info(cal)
            for indice in df.pt.indices_partial_trading(cal):
                dates = pd.date_range(indice.left, indice.right - one_day)
                non_trading_sessions = dates.intersection(non_sessions)
                assert not non_trading_sessions.empty
                assert not dates.intersection(sessions).empty
                assert_index_equal(info[indice], non_trading_sessions)

            for indice in df.pt.indices_trading(cal):
                dates = pd.date_range(indice.left, indice.right - one_day)
                assert dates.difference(sessions).empty

            indices_trading_status.loc[df.pt.indices_trading(cal)] = True
            assert_series_equal(
                df.pt.indices_trading_status(cal),
                indices_trading_status,
                check_dtype=False,
            )

        xnys, xlon, x247 = calendars_alldays
        assert not df.pt.indices_all_trading(xnys)
        assert not df.pt.indices_all_trading(xlon)
        assert df.pt.indices_all_trading(x247)

    def test_indices_partial_trading_info(
        self, intraday_asia_pt, calendars_asian, side
    ):
        """Test `indices_partial_trading_info` with multiple non trading periods."""
        df = intraday_asia_pt
        xasx, xhkg = calendars_asian

        session = "2021-12-16"
        side = "left"
        xasx_open = xasx.session_open(session)
        right = xasx_open + pd.Timedelta(6, "h")
        indice = pd.Interval(xasx_open, right, side)

        xhkg_non_trading_periods = [
            pd.Interval(xasx_open, xhkg.session_open(session), side),
            pd.Interval(
                xhkg.session_break_start(session), xhkg.session_break_end(session), side
            ),
        ]

        info = df.pt.utc.pt.indices_partial_trading_info(xhkg)
        assert_index_equal(info[indice], pd.IntervalIndex(xhkg_non_trading_periods))


class TestGetSubsetFromIndices:
    """Dedicated class-specific tests for `get_subset_from_indices` method."""

    def test_daily_pt(self, daily_pt, one_day):
        df = daily_pt
        f = df.pt.get_subset_from_indices
        assert_frame_equal(f(), df)

        start = df.index[-5]
        assert_frame_equal(f(start), df[start:])
        end = df.index[5]
        assert_frame_equal(f(end=end), df[:end])

        end = df.index[-2]
        assert_frame_equal(f(start, end), df[start:end])
        assert_frame_equal(
            f(start.tz_localize(UTC), end.tz_localize(UTC)), df[start:end]
        )

        dates = pd.date_range(df.index[0], df.index[-1])
        non_sessions = dates.difference(df.index)

        def match(parameter_name, date) -> str:
            return re.escape(f"`{parameter_name}` ({date}) is not an indice.")

        start_oob = df.index[0] - one_day
        for args in [
            (start_oob,),
            (start_oob, end),
            (non_sessions[0],),
            (non_sessions[0], end),
        ]:
            with pytest.raises(ValueError, match=match("start", args[0])):
                f(*args)

        end_oob = df.index[-1] + one_day
        for args in [
            (None, end_oob),
            (start, end_oob),
            (None, non_sessions[-1]),
            (start, non_sessions[-1]),
        ]:
            with pytest.raises(ValueError, match=match("end", args[1])):
                f(*args)

    def test_intraday_pt(self, intraday_pt, one_sec, one_min):
        # pylint: disable=too-complex
        df = intraday_pt
        f = df.pt.get_subset_from_indices
        assert_frame_equal(f(), df)

        i = -5
        start = df.index[i].left
        expected = df.iloc[i:]
        assert_frame_equal(f(start), expected)

        i = 5
        end_indice = df.index[i]
        expected = df.iloc[: i + 1]

        # test limits to return df with final indice as end_indice
        valid_ends = [
            end_indice.right,
            end_indice.right - one_sec,
            end_indice.right - one_min,
            end_indice.left + one_sec,
            end_indice.left + one_min,
        ]
        for end in valid_ends:
            assert_frame_equal(f(end=end), expected)

        # ...test beyond limits
        for end in [end_indice.left - one_sec, end_indice.left - one_min]:
            assert_frame_equal(f(end=end), df.iloc[:i])
        for end in [end_indice.right + one_sec, end_indice.right + one_min]:
            assert_frame_equal(f(end=end), df.iloc[: i + 2])

        # test passing start and end
        start, expected = end_indice.left, df.iloc[[i]]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)
        # check can pass as tz_naive
        assert_frame_equal(f(start.tz_localize(None), end.tz_localize(None)), expected)

        start, expected = df.index[i - 2].left, df.iloc[i - 2 : i + 1]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)

        # test invalid values

        # out-of-bounds
        invalid_starts = [df.pt.first_ts - one_min, df.pt.first_ts - one_sec]
        invalid_ends = [df.pt.last_ts + one_min, df.pt.last_ts + one_sec]

        # values within table bounds but outside of an interval
        bv_discountinuous = df.index.right[:-1] != df.index.left[1:]
        df_discontinuous = df[:-1][bv_discountinuous]

        if not df_discontinuous.empty:
            bound = df_discontinuous.index[0].right
            invalid_ends.append(bound + one_sec)
            invalid_starts.append(bound + one_sec)

            if bound + one_min != df_discontinuous.index[1].left:
                invalid_ends.append(bound + one_min)
                invalid_starts.append(bound + one_min)

        for end in invalid_ends:
            match = re.escape(
                f"`end` ({end}) is not the right side of or contained within an indice."
            )
            with pytest.raises(ValueError, match=match):
                f(None, end)

        # for start only, values within indice interval
        start = df.index[-i].left
        for ts in (start - one_min, start + one_min):
            if ts not in df.index.left:
                invalid_starts.append(ts)

        for start in invalid_starts:
            match = re.escape(f"`start` ({start}) is not the left side of an indice.")
            with pytest.raises(ValueError, match=match):
                f(start)

    def test_multiple_sessions_pt(self, multiple_sessions_pt, one_day):
        df = multiple_sessions_pt
        f = df.pt.get_subset_from_indices
        assert_frame_equal(f(), df)

        i = 5
        start = df.index[-i].left
        expected = df.iloc[-i:]

        assert_frame_equal(f(start), expected)

        end_indice = df.index[i]
        expected = df.iloc[: i + 1]

        # test limits to return df with final indice as end_indice
        valid_ends = [
            end_indice.right,
            end_indice.left + one_day,
            end_indice.right - one_day,
        ]
        for end in valid_ends:
            assert_frame_equal(f(end=end), expected)

        # ...test beyond limits
        assert_frame_equal(f(end=end_indice.left - one_day), df.iloc[:i])
        assert_frame_equal(f(end=end_indice.right + one_day), df.iloc[: i + 2])

        # test passing start and end
        start, expected = end_indice.left, df.iloc[[i]]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)
        # check can pass as tz_naive
        assert_frame_equal(f(start.tz_localize(None), end.tz_localize(None)), expected)

        start, expected = df.index[i - 2].left, df.iloc[i - 2 : i + 1]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)

        # test invalid values

        # out-of-bounds
        invalid_starts = [df.pt.first_ts - one_day]
        invalid_ends = [df.pt.last_ts + one_day]

        for end in invalid_ends:
            match = re.escape(
                f"`end` ({end}) is not the right side of or contained within an indice."
            )
            with pytest.raises(ValueError, match=match):
                f(None, end)

        # for start only, values within indice interval
        start = df.index[-i].left
        for ts in (start - one_day, start + one_day):
            if ts not in df.index.left:
                invalid_starts.append(ts)

        for start in invalid_starts:
            match = re.escape(f"`start` ({start}) is not the left side of an indice.")
            with pytest.raises(ValueError, match=match):
                f(start)

    def test_composite_intraday_pt(self, composite_intraday_pt, one_min, one_sec):
        df = composite_intraday_pt
        f = df.pt.get_subset_from_indices
        assert_frame_equal(f(), df)

        # Test basing tests on two intervals on either side of split between composite parts.
        last_5T_indice = df[df.index.length == TDInterval.T5].index[-1]
        i = df.index.get_loc(last_5T_indice)
        i_ = i + 1  # first 1T index
        first_1T_indice = df.index[i_]

        start = last_5T_indice.left
        expected = df.iloc[i:]
        assert_frame_equal(f(start), expected)

        # test for default start
        assert_frame_equal(f(end=first_1T_indice.right), df.iloc[: i_ + 1])

        # narrow further tests to just the indices immediately adjacent to the
        # change in interval
        start = last_5T_indice.left
        expected = df.iloc[i : i_ + 1]

        # test limits to return df with final indice as first_i_interval
        valid_ends = [
            first_1T_indice.right,
            first_1T_indice.right - one_sec,
            first_1T_indice.right - one_min + one_sec,
            first_1T_indice.left + one_sec,
        ]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)
        # check can pass as tz_naive
        assert_frame_equal(f(start.tz_localize(None), end.tz_localize(None)), expected)

        # test beyond limit (to left)
        expected = df.iloc[[i]]
        assert_frame_equal(f(start, first_1T_indice.left), expected)

        # out-of-bounds
        invalid_starts = [df.pt.first_ts - one_min, df.pt.first_ts - one_sec]
        invalid_ends = [df.pt.last_ts + one_min, df.pt.last_ts + one_sec]

        for end in invalid_ends:
            match = re.escape(
                f"`end` ({end}) is not the right side of or contained within an indice."
            )
            with pytest.raises(ValueError, match=match):
                f(None, end)

        for start in invalid_starts:
            match = re.escape(f"`start` ({start}) is not the left side of an indice.")
            with pytest.raises(ValueError, match=match):
                f(start)

    def test_composite_daily_intraday_pt(
        self, composite_daily_intraday_pt, one_day, one_min, one_sec
    ):
        df = composite_daily_intraday_pt
        f = df.pt.get_subset_from_indices
        assert_frame_equal(f(), df)

        # Test basing tests on two intervals on either side of split between composite parts.
        last_d_indice = df[df.index.length == pd.Timedelta(0)].index[-1]
        i = df.index.get_loc(last_d_indice)
        i_ = i + 1  # first intraday index
        first_i_indice = df.index[i_]

        start = last_d_indice.left
        expected = df.iloc[i:]
        assert_frame_equal(f(start), expected)

        # test for default start
        assert_frame_equal(f(end=first_i_indice.right), df.iloc[: i_ + 1])

        # narrow further tests to just the indices immediately adjacent to the
        # change in interval
        start = last_d_indice.left
        expected = df.iloc[i : i_ + 1]

        # test limits to return df with final indice as first_i_interval
        valid_ends = [
            first_i_indice.right,
            first_i_indice.right - one_sec,
            first_i_indice.right - one_min,
            first_i_indice.left + one_sec,
            first_i_indice.left + one_min,
        ]
        for end in valid_ends:
            assert_frame_equal(f(start, end), expected)
        # check can pass as tz_naive
        assert_frame_equal(f(start.tz_localize(None), end.tz_localize(None)), expected)

        # ... testing beyond limit (to left) covered within invalid values that fall
        # 'in_between'. Values between change of interval:
        in_between_ts = [
            first_i_indice.left - one_sec,
            first_i_indice.left - one_min,
            last_d_indice.right + one_sec,
            last_d_indice.right + one_min,
        ]

        # out-of-bounds
        invalid_starts = [df.pt.first_ts - one_day]
        invalid_ends = [df.pt.last_ts + one_min, df.pt.last_ts + one_sec]

        invalid_starts.extend(in_between_ts)
        invalid_ends.extend(in_between_ts)

        for end in invalid_ends:
            match = re.escape(
                f"`end` ({end}) is not the right side of or contained within an indice."
            )
            with pytest.raises(ValueError, match=match):
                f(None, end)

        for start in invalid_starts:
            match = re.escape(f"`start` ({start}) is not the left side of an indice.")
            with pytest.raises(ValueError, match=match):
                f(start)


class TestPriceAt:
    """Dedicated class-specific tests for `price_at` method."""

    @staticmethod
    def get_expected(
        df: pd.DataFrame, i: int, key: typing.Literal["open", "close"]
    ) -> pd.DataFrame:
        row = df.iloc[[i]]
        cols = [col for col in df.columns if col[1] == key]
        expected = row[cols].droplevel(1, axis=1)
        index = expected.index.left if key == "open" else expected.index.right
        expected.index = index
        return expected

    def test_intraday_pt(self, intraday_pt, tz_default, one_sec, one_min):
        # pylint: disable=too-complex
        # NB composite_intraday_pt not tested as implementation independent of
        # composite behavious.

        df = intraday_pt
        f = df.pt.price_at

        # test timestamps that lie within an indice with no gaps either side
        left = pd.Timestamp("2021-12-21 14:40:00", tz=UTC)
        right = pd.Timestamp("2021-12-21 14:45:00", tz=UTC)
        i = df.index.left.get_loc(left)

        expected = self.get_expected(df, i, "open")
        # test bounds of expected rtrn
        for ts in (
            left,
            left + one_sec,
            left + one_min,
            right - one_sec,
            right - one_min,
        ):
            pd.testing.assert_frame_equal(f(ts), expected)

        # test beyond bounds
        expected = self.get_expected(df, i - 1, "open")
        for ts in (left - one_sec, left - one_min):
            assert_frame_equal(f(ts), expected)

        expected = self.get_expected(df, i + 1, "open")
        for ts in (right + one_sec, right + one_min):
            assert_frame_equal(f(ts), expected)

        # assert that there is a gap here in the index
        gap_left = pd.Timestamp("2021-12-23 23:00:00", tz=UTC)
        assert gap_left in df.index.right
        assert not df.index.contains(gap_left).any()
        gap_right = pd.Timestamp("2021-12-24 08:00:00", tz=UTC)
        assert gap_right in df.index.left
        assert not df.index.contains(gap_right - one_sec).any()

        # test bounds within gap
        i = df.index.right.get_loc(gap_left)
        expected = self.get_expected(df.ffill(), i, "close")
        for ts in (
            gap_left,
            gap_left + one_sec,
            gap_left + one_min,
            gap_right - one_sec,
            gap_right - one_min,
        ):
            assert_frame_equal(f(ts), expected)

        # test immediately to left of gap left
        expected = self.get_expected(
            df, i, "open"
        )  # if prices were available for all sumbols ...
        # but prices not available for MSFT and AZN.L, so...
        expected_ = self.get_expected(df.ffill(), i - 1, "close")
        for s in ["MSFT", "AZN.L"]:
            expected[s] = expected_[s].iloc[0]
        for ts in (gap_left - one_sec, gap_left - one_min):
            assert_frame_equal(f(ts), expected)

        # test immediately to right of gap right
        expected = self.get_expected(
            df, i + 1, "open"
        )  # if prices were available for all symbols...
        # but prices not available for MSFT and ES=F, so...
        expected_ = self.get_expected(df.ffill(), i, "close")
        for s in ["MSFT", "ES=F"]:
            expected[s] = expected_[s].iloc[0]
        for ts in (gap_right, gap_right + one_sec, gap_right + one_min):
            assert_frame_equal(f(ts), expected)

        # test tz parameter
        i = df.index.left.get_loc(left)
        expected = self.get_expected(df, i, "open")

        # no tz parameter
        left_default_tz = left.tz_convert(tz_default)
        left_tz_naive = left_default_tz.tz_localize(None)
        for ts in (left, left_default_tz, left_tz_naive):
            rtrn = f(ts)
            assert_frame_equal(rtrn, expected)
            assert rtrn.index.tz is tz_default

        # tz to define return index only as ts is tz aware
        assert_frame_equal(f(left_default_tz, tz=UTC), expected.tz_convert(UTC))

        # tz to define ts (tz-naive) and return index
        assert_frame_equal(f(left.tz_localize(None), tz=UTC), expected.tz_convert(UTC))

        # Test a session label raises expected error
        session = pd.Timestamp("2021-12-21 00:00")

        msg = re.escape(
            "`ts` must have a time component or be tz-aware, although receieved"
            f" as {session}. To define ts as midnight pass as a tz-aware"
            " pd.Timestamp. For prices as at a session's close use .close_at()."
        )
        with pytest.raises(ValueError, match=msg):
            f(session)

        # Test bounds
        left_bound = df.index[0].left
        expected = self.get_expected(df, 0, "open")
        assert_frame_equal(f(left_bound), expected)

        match = (
            "`time` cannot be earlier than the first time for which prices are"
            " available. First time for which prices are available is"
            f" {left_bound.tz_convert(UTC)} although `time` received as "
        )
        for ts in (left_bound - one_sec, left_bound - one_min):
            with pytest.raises(
                errors.DatetimeTooEarlyError,
                match=re.escape(match + f"{ts.tz_convert(UTC)}."),
            ):
                f(ts)

        right_bound = df.index[-1].right
        expected = self.get_expected(df.ffill(), -1, "close")
        assert_frame_equal(f(right_bound), expected)

        match = (
            "`time` cannot be later than the most recent time for which prices are"
            f" available. Most recent time for which prices are available is"
            f" {right_bound.tz_convert(UTC)} although `time` received as "
        )
        for ts in (right_bound + one_sec, right_bound + one_min):
            with pytest.raises(
                errors.DatetimeTooLateError,
                match=re.escape(match + f"{ts.tz_convert(UTC)}."),
            ):
                f(ts)

    def test_composite_daily_intraday_pt(
        self, composite_daily_intraday_pt, one_sec, one_min
    ):
        df = composite_daily_intraday_pt
        # verify left bound of intraday part of table
        df_ip = df.pt.intraday_part
        left_bound = df_ip.index[0].left
        rtrn = df.pt.price_at(left_bound)
        expected = self.get_expected(df_ip, 0, "open")
        assert_frame_equal(rtrn, expected)

        # verify left of left bound raises error
        match = (
            "`time` cannot be earlier than the first time for which prices are"
            " available. First time for which prices are available is"
            f" {left_bound.tz_convert(UTC)} although `time` received as "
        )
        for ts in (left_bound - one_sec, left_bound - one_min):
            with pytest.raises(
                errors.DatetimeTooEarlyError,
                match=re.escape(match + f"{ts.tz_convert(UTC)}."),
            ):
                df.pt.price_at(ts)

    def test_daily_pt(self, daily_pt):
        df = daily_pt
        match = (
            "price_at is not implemented for daily price interval. Use"
            " `close_at` or `session_prices`."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.price_at(df.index[0])

    def test_multiple_sessions_pt(self, multiple_sessions_pt):
        df = multiple_sessions_pt
        match = (
            "`price_at` is not implemented for <class"
            " 'market_prices.pt.PTMultipleSessions'> as intervals are not intraday."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.price_at(df.index[0].right)


class TestCloseAt:
    """Tests for `close_at` and `session_prices`."""

    @pytest.fixture(scope="class")
    def session(self) -> abc.Iterator[pd.Timestamp]:
        yield pd.Timestamp("2021-12-21 00:00")

    @pytest.fixture
    def non_session(self, daily_pt) -> abc.Iterator[pd.Timestamp]:
        """First non-session date after first session of `daily_pt`."""
        df = daily_pt
        dates = pd.date_range(df.index[0], df.index[-1])
        non_sessions = dates.difference(df.index)
        yield non_sessions[0]

    @staticmethod
    def get_expected(df: pd.DataFrame, i: int) -> pd.DataFrame:
        row = df.iloc[[i]]
        cols = [col for col in df.columns if col[1] == "close"]
        expected = row[cols].droplevel(1, axis=1)
        return expected

    def test_daily_pt_close_at(
        self, daily_pt, session, non_session, tz_default, one_day, one_min
    ):
        df = daily_pt
        f = df.pt.close_at

        # test session
        i = df.index.get_loc(session)
        assert_frame_equal(f(session), self.get_expected(df, i))

        # test non-session
        session_with_gap_after = non_session - one_day
        i = df.index.get_loc(session_with_gap_after)
        assert_frame_equal(f(non_session), self.get_expected(df, i))

        # test bounds and out of bounds
        left_bound = df.index[0]
        assert_frame_equal(f(left_bound), self.get_expected(df, 0))

        oob_left = left_bound - one_day
        match = re.escape(
            "`date` cannot be earlier than the first date for which prices are"
            f" available. First date for which prices are available is {left_bound}"
            f" although `date` received as {oob_left}."
        )
        with pytest.raises(errors.DatetimeTooEarlyError, match=match):
            f(oob_left)

        right_bound = df.index[-1]
        assert_frame_equal(f(right_bound), self.get_expected(df, -1))

        oob_right = right_bound + one_day
        match = re.escape(
            "`date` cannot be later than the most recent date for which prices are"
            f" available. Most recent date for which prices are available is"
            f" {right_bound} although `date` received as {oob_right}."
        )
        with pytest.raises(errors.DatetimeTooLateError, match=match):
            f(oob_right)

        # test non-valid input
        ts = session + one_min
        match = (
            f"`date` can not have a time component, although receieved as {ts}."
            " For an intraday price use .price_at()."
        )
        with pytest.raises(ValueError, match=match):
            f(ts)

        # test non-valid input
        match = "`date` must be tz-naive, although receieved as "
        for ts in (
            session.tz_localize(UTC),
            session.tz_localize(tz_default),
        ):
            with pytest.raises(ValueError, match=re.escape(match + f"{ts}.")):
                f(ts)

    def test_daily_pt_session_prices(self, daily_pt, session, non_session, one_day):
        df = daily_pt
        f = df.pt.session_prices
        assert_frame_equal(f(session), df.loc[[session]])

        match = f"`session` {non_session} is not present in the table."
        with pytest.raises(ValueError, match=match):
            f(non_session)

        # test bounds
        left_bound = df.index[0]
        assert_frame_equal(f(left_bound), df.loc[[left_bound]])
        oob_left = left_bound - one_day
        with pytest.raises(errors.DatetimeTooEarlyError):
            f(oob_left)

        right_bound = df.index[-1]
        assert_frame_equal(f(right_bound), df.loc[[right_bound]])
        oob_right = right_bound + one_day
        with pytest.raises(errors.DatetimeTooLateError):
            f(oob_right)

    def test_composite_daily_intraday_pt(self, composite_daily_intraday_pt, one_day):
        df = composite_daily_intraday_pt
        df_dp = df.pt.daily_part
        right_bound = df_dp.index[-1]
        assert_frame_equal(df.pt.close_at(right_bound), self.get_expected(df_dp, -1))

        right_bound_utc = right_bound.tz_localize(UTC)
        indice = pd.Interval(right_bound_utc, right_bound_utc, "left")
        expected = df.loc[[indice]]
        expected.index = expected.index.left.tz_convert(None)
        assert_frame_equal(df.pt.session_prices(right_bound), expected)

        oob_right = right_bound + one_day

        def match(param) -> str:
            return re.escape(
                f"`{param}` cannot be later than the most recent {param} for which"
                f" prices are available. Most recent {param} for which prices are"
                f" available is {right_bound} although `{param}` received as"
                f" {oob_right}."
            )

        with pytest.raises(errors.DatetimeTooLateError, match=match("date")):
            df.pt.close_at(oob_right)

        with pytest.raises(errors.DatetimeTooLateError, match=match("session")):
            df.pt.session_prices(oob_right)

    def test_intraday_pt(self, intraday_pt):
        df = intraday_pt
        match = (
            "`close_at` is not implemented for intraday price intervals."
            " Use `price_at`."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.close_at(df.index[0].left)

        match = (
            "`session_prices` is not implemented for intraday price intervals."
            " Use `price_at`."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.session_prices(df.index[0].left)

    def test_multiple_sessions_pt(self, multiple_sessions_pt):
        df = multiple_sessions_pt
        match = (
            "`close_at` not implemented for <class"
            " 'market_prices.pt.PTMultipleSessions'> as table interval too high to"
            " offer close prices for a specific date."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.close_at(df.index[0].left)

        match = (
            "`session_prices` not implemented for"
            " <class 'market_prices.pt.PTMultipleSessions'> as table interval"
            " too high to offer prices for a specific session."
        )
        with pytest.raises(NotImplementedError, match=match):
            df.pt.session_prices(df.index[0].left)


class TestFillNa:
    """Dedicated tests for `fillna` method."""

    @staticmethod
    def assertions(df: pd.DataFrame, symbols: list[str]):
        # for initial tests ensure first and last rows do not have any missing values.
        df_notna = df[df.notna().all(axis=1)]
        start_label = df_notna.index[0]
        end_label = df_notna.index[-1]
        i_start = df.index.get_loc(start_label)
        i_end = df.index.get_loc(end_label)

        df_test = df.iloc[i_start : i_end + 1].copy()
        assert df_test.isna().any(axis=1).any()  # assert some rows in beween are na.

        rtrn_ff = df_test.pt.fillna("ffill")
        for s in symbols:
            bv = df_test[s].isna().any(axis=1)
            subset = rtrn_ff[bv][s]
            for col in subset:
                if col == "volume":
                    assert (subset[col] == 0).all()  # pylint: disable=compare-to-zero
                else:
                    assert_series_equal(subset[col], subset["close"], check_names=False)
                i_nas = rtrn_ff.index.get_indexer(subset.index)
                filled_values = rtrn_ff[s].iloc[i_nas]["close"].values
                prior_values = rtrn_ff[s].iloc[i_nas - 1]["close"].values
                # ensure being filled forward. Assertion will ensure start of each na
                # block being filled will prior close value.
                assert (filled_values == prior_values).all()

        rtrn_bf = df_test.pt.fillna("bfill")
        for s in symbols:
            bv = df_test[s].isna().any(axis=1)
            subset = rtrn_bf[bv][s]
            for col in subset:
                if col == "volume":
                    assert (subset[col] == 0).all()  # pylint: disable=compare-to-zero
                else:
                    assert_series_equal(subset[col], subset["close"], check_names=False)
                i_nas = rtrn_bf.index.get_indexer(subset.index)
                filled_values = rtrn_bf[s].iloc[i_nas]["close"].values
                following_values = rtrn_bf[s].iloc[i_nas + 1]["open"].values
                # ensure being filled backward. Assertion will ensure end of each na
                # block being filled will following open value.
                assert (filled_values == following_values).all()

        assert_frame_equal(df_test.pt.fillna("both"), rtrn_ff)

        # test ffill and bfill are filling in correct direction.

        # set up df_test with first and last rows that have some na values.
        df_isna = df[df.isna().any(axis=1)]
        start_label = df_isna.index[0]
        end_label = df_isna.index[-1]
        i_start = df.index.get_loc(start_label)
        i_end = df.index.get_loc(end_label)

        df_test = df.iloc[i_start : i_end + 1].copy()

        rtrn_ff = df_test.pt.fillna("ffill")
        assert rtrn_ff.isna().any(axis=1).iloc[0]
        rtrn_bf = df_test.pt.fillna("bfill")
        assert rtrn_bf.isna().any(axis=1).iloc[-1]
        rtrn_both = df_test.pt.fillna("both")
        assert rtrn_both.notna().all(axis=None)

        # for those symbols that have missing values in first row of df_test, make sure
        # that "both" is filling initial na rows backwards and everything else forwards.
        for s in symbols:
            if df_test[s].notna().all(axis=1).iloc[0]:
                continue
            df_notna = df_test[s][df_test.notna().all(axis=1)]
            start_label = df_notna.index[0]
            end_label = df_notna.index[-1]
            i_start = df_test.index.get_loc(start_label)
            assert_frame_equal(rtrn_both[s].iloc[:i_start], rtrn_bf[s].iloc[:i_start])
            assert_frame_equal(rtrn_both[s].iloc[i_start:], rtrn_ff[s].iloc[i_start:])

    def assertions_single_symbol(self, df: pd.DataFrame, symbol: str):
        """Verify method for table with single symbol.

        Verifies that for single symbol DataFrame method returns the same as
        the subset (for symbol) returned for multiple symbol DataFrame. Relies
        on separate verification of return for multiple symbol DataFrame.
        """
        df_test = df[symbol]
        assert not df_test.pt.has_symbols
        for method in ("ffill", "bfill", None):
            expected = df.pt.fillna(method)[symbol]
            rtrn = df_test.pt.fillna(method)
            assert_frame_equal(rtrn, expected)

    def test_daily_pt(self, daily_pt, symbols):
        self.assertions(daily_pt, symbols)
        self.assertions_single_symbol(daily_pt, symbols[0])

    def test_intraday_pt(self, intraday_pt, symbols):
        # NB no separate test for composite_intraday_pt as method implementation not
        # affected by composite behaviour.
        self.assertions(intraday_pt, symbols)
        self.assertions_single_symbol(intraday_pt, symbols[0])

    def test_composite_daily_intraday_pt(self, composite_daily_intraday_pt, symbols):
        self.assertions(composite_daily_intraday_pt, symbols)
        self.assertions_single_symbol(composite_daily_intraday_pt, symbols[0])

    def test_multiple_sessions_alldays_pt(
        self, multiple_sessions_alldays_pt, symbols_alldays
    ):
        self.assertions(multiple_sessions_alldays_pt, symbols_alldays)
        self.assertions_single_symbol(multiple_sessions_alldays_pt, symbols_alldays[0])


class TestOperate:
    """Tests `operate` method.

    Tests effect of each option in isolation.

    Test effect of passing a compatible set of all options although does
    not test any other combination of options.

    Where, for an option, the operator method serves as a wrapper over
    another methods, test simply expects return of that other method (which
    is assumed to be tested in its own right).
    """

    @staticmethod
    def df_na_start_end_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Return subset of df with na values in initial and end rows."""
        df_isna = df[df.isna().any(axis=1)]
        start_label = df_isna.index[0]
        end_label = df_isna.index[-1]
        i_start = df.index.get_loc(start_label)
        i_end = df.index.get_loc(end_label)
        return df.iloc[i_start : i_end + 1].copy()

    def no_args_assertion(self, df: pd.DataFrame):
        assert_frame_equal(df.pt.operate(), df)

    def assertions_all_options_combo(self, df: pd.DataFrame, symbols: list[str]):
        df_test = self.df_na_start_end_rows(df)

        rtrn = df_test.pt.operate(
            tz=UTC,
            data_for_all_start=True,
            fill="ffill",
            include=symbols[0],
            side="right",
            close_only=True,
        )

        # build expected
        df_ = df_test
        df_ = df_test.pt.utc
        df_ = df_[[symbols[0]]]
        df_ = df_.pt.data_for_all_start
        df_ = df_.pt.fillna("ffill")
        df_ = df_[[(symbols[0], "close")]]
        df_ = df_.droplevel(1, axis=1)  # type: ignore[assignment]  # does return frame
        if isinstance(df_.index, pd.IntervalIndex):
            df_.index = df_.index.right

        assert_frame_equal(rtrn, df_)

    def assertions_tz_utc_naive(self, df: pd.DataFrame, other_tz: ZoneInfo):
        """Assert tz as naive and utc and that other tz raises error."""
        f = df.pt.operate
        assert_frame_equal(f(tz=UTC), df.pt.utc)
        assert_frame_equal(f(tz=None), df.pt.naive)

        match = re.escape(
            f"`tz` for class {type(df.pt)} can only be UTC or timezone naive (None),"
            f" not {other_tz}."
        )
        with pytest.raises(ValueError, match=match):
            f(tz=other_tz)

    def assertions_tz(self, df: pd.DataFrame, other_tz: ZoneInfo):
        """Assert tz as naive, utc and `other_tz`."""
        f = df.pt.operate
        assert_frame_equal(f(tz=UTC), df.pt.utc)
        assert_frame_equal(f(tz=None), df.pt.naive)
        assert_frame_equal(f(tz=other_tz), df.pt.set_tz(other_tz))

    def assertions_include_exclude(self, df: pd.DataFrame, symbols: list[str]):
        f = df.pt.operate
        assert_frame_equal(f(include=symbols[0]), df[[symbols[0]]])
        assert_frame_equal(f(include=symbols[:2]), df[symbols[:2]])
        assert_frame_equal(f(include=symbols), df)

        assert_frame_equal(f(exclude=symbols[0]), df[symbols[1:]])
        assert_frame_equal(f(exclude=symbols[:2]), df[[symbols[2]]])

        match = "Cannot exclude all symbols."
        with pytest.raises(ValueError, match=match):
            f(exclude=list(reversed(symbols)))

        include, exclude = symbols[0], symbols[1]

        match = (
            "Pass only `exclude` or `include`, not both."
            f"\n`exclude` received as {exclude}.\n`include` received as {include}."
        )
        with pytest.raises(ValueError, match=match):
            f(include=include, exclude=exclude)

    def assertions_data_for_all(self, df: pd.DataFrame):
        df_test = self.df_na_start_end_rows(df)
        for method in ("data_for_all_start", "data_for_all_end", "data_for_all"):
            assert_frame_equal(
                df_test.pt.operate(**{method: True}), getattr(df_test.pt, method)
            )

    def assertions_fillna(self, df: pd.DataFrame):
        df_test = self.df_na_start_end_rows(df)
        for fill in ("ffill", "bfill", "both"):
            assert_frame_equal(df_test.pt.operate(fill=fill), df_test.pt.fillna(fill))

    def assertions_close_only(self, df: pd.DataFrame, symbols):
        rtrn = df.pt.operate(close_only=True)
        assert len(rtrn.columns) == len(symbols)
        for s in symbols:
            assert_series_equal(rtrn[s], df[(s, "close")], check_names=False)

    def assertions_lose_single_symbol(self, df: pd.DataFrame, symbols):
        f = df.pt.operate

        # test no effect if price table has more than one symbol
        assert_frame_equal(f(lose_single_symbol=True), df)
        assert_frame_equal(
            f(include=symbols[:2], lose_single_symbol=True), df[symbols[:2]]
        )

        # single symbol
        assert_frame_equal(
            f(include=symbols[0], lose_single_symbol=True),
            df[[symbols[0]]].droplevel(0, axis=1),
        )

    def assertions_side(self, df: pd.DataFrame):
        f = df.pt.operate
        df_ = df.copy()
        df_.index = df.index.left
        assert_frame_equal(f(side="left"), df_)
        df_.index = df.index.right
        assert_frame_equal(f(side="right"), df_)

    def test_daily_pt(self, daily_pt, tz_default, symbols):
        df = daily_pt
        self.no_args_assertion(df)
        self.assertions_tz_utc_naive(df, tz_default)
        self.assertions_include_exclude(df, symbols)
        self.assertions_data_for_all(df)
        self.assertions_fillna(df)
        self.assertions_close_only(df, symbols)
        self.assertions_lose_single_symbol(df, symbols)

        # confirm side option has no effect
        for side in ("left", "right"):
            assert_frame_equal(df.pt.operate(side=side), df)

        self.assertions_all_options_combo(df, symbols)

    def test_intraday_pt(self, intraday_pt, tz_moscow, symbols):
        # NB composite_intraday_pt NOT tested as operations are not
        # affected by composite nature of intraday table.

        df = intraday_pt

        # create subset to limit time for assert_frame_equals
        subset_start = pd.Timestamp("2021-12-21 00:00")
        subset_end = pd.Timestamp("2021-12-27 00:00")
        df_subset = df.pt.get_subset_from_indices(subset_start, subset_end)

        self.no_args_assertion(df)
        self.assertions_tz(df, tz_moscow)
        self.assertions_include_exclude(df_subset, symbols)
        self.assertions_data_for_all(df_subset)
        self.assertions_fillna(df_subset)
        self.assertions_close_only(df_subset, symbols)
        self.assertions_lose_single_symbol(df_subset, symbols)
        self.assertions_side(df_subset)
        self.assertions_all_options_combo(df, symbols)

    def test_multiple_sessions_pt(
        self,
        multiple_sessions_pt,
        multiple_sessions_alldays_pt,
        tz_default,
        symbols,
        symbols_alldays,
    ):
        df = multiple_sessions_pt
        df_alldays = multiple_sessions_alldays_pt
        self.no_args_assertion(df)
        self.assertions_tz_utc_naive(df, tz_default)
        self.assertions_include_exclude(df, symbols)
        self.assertions_data_for_all(df_alldays)
        self.assertions_fillna(df_alldays)
        self.assertions_close_only(df, symbols)
        self.assertions_lose_single_symbol(df, symbols)
        self.assertions_side(df)
        self.assertions_all_options_combo(df_alldays, symbols_alldays)

    def test_composite_daily_intraday_pt(
        self, composite_daily_intraday_pt, tz_default, symbols
    ):
        df = composite_daily_intraday_pt
        self.no_args_assertion(df)
        self.assertions_tz_utc_naive(df, tz_default)
        self.assertions_include_exclude(df, symbols)
        self.assertions_data_for_all(df)
        self.assertions_fillna(df)
        self.assertions_close_only(df, symbols)
        self.assertions_lose_single_symbol(df, symbols)
        self.assertions_side(df)
        self.assertions_all_options_combo(df, symbols)


def test_stacked(price_tables, symbols):
    table = price_tables

    # verify raises error if price table has more than one row
    match = re.escape(
        "Only price tables with a single row can be stacked (price table has"
        " 2 rows)."
    )
    with pytest.raises(ValueError, match=match):
        _ = table.iloc[0:2].pt.stacked

    # verify return as expected
    df = table.iloc[[0]]
    rtrn = df.pt.stacked
    # verify index and columns as expected
    assert isinstance(rtrn.index, pd.MultiIndex)
    assert_index_equal(rtrn.index.levels[0], df.index)
    assert_index_equal(rtrn.index.levels[1], df.columns.levels[0])
    columns = pd.Index(["open", "high", "low", "close", "volume"], name="")

    # verify `stack` had to order columns, and did order them.
    assert not (columns.values == df.columns.levels[1]).all()
    assert_index_equal(rtrn.columns, columns)

    # check values unchanged
    indice = df.index[0]
    for s in symbols:
        assert_series_equal(rtrn.loc[(indice, s)], df[s].iloc[0], check_names=False)


def assert_aggregations(symbols, subset: pd.DataFrame, row: pd.Series):
    """Assert aggregations for a row of downsampled data.

    Asserts `row` of return reflects aggregation of data from `subset`
    of table being downsampled
    """
    for s in symbols:
        subset_s, row_s = subset[s], row[s]
        if subset_s.isna().all(axis=None):
            assert row_s.isna().all(axis=None)
            continue
        assert subset_s.volume.sum() == row_s.volume
        assert subset_s.high.max() == row_s.high
        assert subset_s.low.min() == row_s.low
        assert subset_s.bfill().open.iloc[0] == row_s.open
        assert subset_s.ffill().close.iloc[-1] == row_s.close


class TestDownsampleDaily:
    """Dedicated tests for `PTDaily.downsample`."""

    def test_errors(self, daily_pt, daily_pt_ss, calendars, xnys, xlon):
        """Verify raising expected errors for daily price table."""
        df = daily_pt
        f = df.pt.downsample

        # test raises expected errors
        match = "Cannot downsample to a `pdfreq` with a unit more precise than 'd'."
        invalid_freqs = ("5min", "1ms", "3h", "120s", "1000ns", "26h")
        for freq in invalid_freqs:
            with pytest.raises(ValueError, match=match):
                f(freq, calendars[0])

        def match_f(freq) -> str:
            return re.escape(
                f"Received `pdfreq` as {freq} although must be either of type"
                " pd.offsets.CustomBusinessDay or acceptable input to"
                " pd.tseries.frequencies.to_offset that describes a frequency greater"
                ' than one day. For example "2d", "5d" "QS" etc.'
            )

        invalid_freqs = ("D2", "astring", "3E", "3")
        for freq in invalid_freqs:
            with pytest.raises(ValueError, match=match_f(freq)):
                f(freq, calendars[0])

        # Verify raises errors if `calendar` and `pdfreq` incompatible when
        # downsampling to CustomBusinessDay.
        advices = (
            "\nNB. Downsampling will downsample to a frequency defined in"
            " CustomBusinessDay when either `pdfreq` is passed as a CustomBusinessDay"
            " (or multiple of) or when the table has a CustomBusinessDay frequency and"
            ' `pdfreq` is passed with unit "d".'
        )

        # Verify raises error when calendar not passed although required
        match = re.escape(
            "To downsample to a frequency defined in terms of CustomBusinessDay"
            " `calendar` must be passed as an instance of"
            f" `exchange_calendars.ExchangeCalendar` although received {None}."
            + advices
        )
        with pytest.raises(TypeError, match=match):
            daily_pt.pt.downsample(xnys.day * 3, None)

        # Define a table with freq defined in CustomBusinessDay
        df_cbday_freq = daily_pt_ss.pt.reindex_to_calendar(xnys)
        assert isinstance(df_cbday_freq.pt.freq, pd.offsets.CustomBusinessDay)

        with pytest.raises(TypeError, match=match):
            df_cbday_freq.pt.downsample("3d", None)

        # Verify raises error when calendar.day does not match frequency (passed or
        # imferred) base.
        match = re.escape(
            "To downsample to a frequency defined in terms of CustomBusinessDay"
            " `calendar` must be passed as an instance of"
            " `exchange_calendars.ExchangeCalendar` which has a `calendar.day`"
            " attribute equal to the base CustomBusinessDay being downsampled to."
            f" Received calendar as {xlon}." + advices
        )
        with pytest.raises(ValueError, match=match):
            daily_pt.pt.downsample(xnys.day * 3, xlon)

        with pytest.raises(ValueError, match=match):
            df_cbday_freq.pt.downsample("3d", xlon)

    def test_cbdays_freq(self, daily_pt, calendars, symbols, one_day):
        """Verify daily price table with frequency as multiple of CustomBusinessDay."""
        df = daily_pt
        f = df.pt.downsample

        # freq as multiple of CustomBusinessDays
        for cal in calendars:
            # expected start if no initial rows are curtailed to maintain interval
            expected_start_origin = cal.date_to_session(df.pt.first_ts, "next")
            expected_start_origin = helpers.to_tz_naive(expected_start_origin)

            # expected right side of last indice will be calendar session after last
            # calendar session in table being downsampled (as right side of indice, this
            # data for this session is not aggregated within the indice (indice is
            # closed left)).
            expected_end = cal.date_to_session(df.pt.last_ts, "next")
            expected_end = helpers.to_tz_naive(expected_end)
            if expected_end == df.pt.last_ts:
                expected_end = cal.next_session(expected_end)
                expected_end = helpers.to_tz_naive(expected_end)

            sessions = cal.sessions_in_range(df.pt.first_ts, df.pt.last_ts)

            # check downsample for frequencies from 1 through 31 CustomBusinessDays
            for cbdays in range(1, 31):
                freq = cal.day * cbdays
                rtrn = f(freq, cal)
                num_sessions = len(sessions)
                excess = num_sessions % cbdays
                expected_num_indices = (num_sessions - excess) // cbdays
                assert len(rtrn) == expected_num_indices
                expected_start = cal.session_offset(expected_start_origin, excess)
                expected_start = helpers.to_tz_naive(expected_start)
                assert rtrn.pt.first_ts == expected_start
                assert rtrn.pt.last_ts == expected_end

                # Assert every indice of rtrn contains exactly i sessions.
                num_cbdays = []
                for intrvl in rtrn.index:
                    indice_sessions = cal.sessions_in_range(
                        intrvl.left, intrvl.right - one_day
                    )
                    num_cbdays.append(len(indice_sessions))
                cbdays_per_intrvl = set(num_cbdays)
                assert len(cbdays_per_intrvl) == 1
                assert cbdays_per_intrvl.pop() == cbdays

                # assert aggregations correct
                ds_sessions = cal.sessions_in_range(expected_start, df.pt.last_ts)
                if ds_sessions.tz is not None:
                    ds_sessions = ds_sessions.tz_convert(None)

                for i, start in enumerate(ds_sessions[::cbdays]):
                    end_i = (i * cbdays) + cbdays
                    try:
                        end = ds_sessions[end_i]
                    except IndexError:
                        end = expected_end
                    end -= one_day

                    subset = df[start:end]
                    row = rtrn.iloc[i]
                    assert_aggregations(symbols, subset, row)

    def test_d_freq(self, daily_pt, xnys, symbols, one_day):
        """Verify daily price table with frequency as multiple of days."""
        df = daily_pt
        f = df.pt.downsample
        expected_end = df.pt.last_ts + one_day
        table_start = df.pt.first_ts

        for days in range(1, 31):
            freq = str(days) + "d"
            offset = pd.tseries.frequencies.to_offset(freq)
            rtrn = f(freq, xnys)
            assert rtrn.pt.last_ts == expected_end

            # Assert every indice of rtrn has length i.
            values = rtrn.index.length.value_counts().index
            assert len(values) == 1
            assert values[0].days == days

            # verify indices, working back from last
            indice_left = expected_end - offset
            for i in reversed(range(len(rtrn))):
                indice = rtrn.iloc[[i]].index
                assert indice.left == indice_left
                indice_right = indice_left + offset
                assert indice.right == indice_right

                # verify aggregations
                subset = df[indice_left : indice_right - one_day]
                row = rtrn.iloc[i]
                assert_aggregations(symbols, subset, row)

                indice_left -= offset

            # verify all required table data is included in downsampled data
            assert indice_left < table_start

    def test_cbday_freq(self, daily_pt_ss, xnys):
        """Verify daily freq assumed as representing CustomBusinessDay.

        Verifies daily freq assumed as representing CustomBusinessDay if the price
        table has a frequency defined in terms of CustomBusinessDay.
        """
        # Define a table with freq defined in CustomBusinessDay
        df = daily_pt_ss.pt.reindex_to_calendar(xnys)
        assert isinstance(df.pt.freq, pd.offsets.CustomBusinessDay)

        assert_frame_equal(
            df.pt.downsample(xnys.day * 3, xnys),
            df.pt.downsample("3d", xnys),
        )

    def test_monthly_freq(self, daily_pt, xnys, x247, one_day, symbols):
        """Verify "MS" and "QS" frequencies."""
        df = daily_pt
        f = df.pt.downsample

        def assertions(
            unit: typing.Literal["MS", "QS"],
            cal: xcals.ExchangeCalendar,
            max_num_indices: int,
            expected_start: pd.Timestamp,
        ):
            for period in range(1, max_num_indices + 1):
                freq = str(period) + unit
                offset = pd.tseries.frequencies.to_offset(freq)
                rtrn = f(freq, cal)
                assert rtrn.pt.first_ts == expected_start
                assert len(rtrn) == max_num_indices // period

                for i in range(max_num_indices // period):
                    indice = rtrn.index[i]
                    # typing - operator * is supported with MS and QS offsets
                    left = expected_start + (i * offset)  # type: ignore[operator]
                    right = left + offset
                    assert indice.left == left
                    assert indice.right == right
                    subset = df[left : right - one_day]
                    row = rtrn.iloc[i]
                    assert_aggregations(symbols, subset, row)

        # first indice should be rolled back from "2021-01-04" to "2021-01-01" as this
        # period does not include calendar sessions for xnys
        cal, expected_start = xnys, pd.Timestamp("2021-01-01")
        assertions("MS", cal, 12, expected_start)
        assertions("QS", cal, 4, expected_start)
        # first indice should be rolled forwards from "2021-01-04" to "2021-02-01".
        for cal in (None, x247):
            assertions("MS", cal, 11, pd.Timestamp("2021-02-01"))
            assertions("QS", cal, 3, pd.Timestamp("2021-04-01"))

        # Verify effect of drop_incomplete_last_indice
        def assert_drop_incomplete_last_indice_unchanged(pdfreq: str):
            """Assert `drop_incomplete_last_indice` makes no difference."""
            expected = f(pdfreq, xnys)
            for drop_incomplete_last_indice in (True, False):
                # no incomplete last indice so..
                rtrn = f(pdfreq, xnys, drop_incomplete_last_indice)
                assert_frame_equal(rtrn, expected)

        # verify makes no difference if last indice is complete
        for pdfreq in ("MS", "QS"):
            assert_drop_incomplete_last_indice_unchanged(pdfreq)

        # Change table such that last indice is no longer complete
        df = df[:-5]
        f = df.pt.downsample

        def assert_drop_incomplete_last_indice_effect(pdfreq: str):
            """Verify effect of `drop_incomplete_last_indice`."""
            expected = daily_pt.pt.downsample(
                pdfreq, xnys
            )  # as if all indices complete
            # assert drops last incomplete indice
            rtrn = f(pdfreq, xnys, True)
            assert_frame_equal(rtrn, expected[:-1])

            # verify includes incomplete last indice
            rtrn = f(pdfreq, xnys, False)
            # assert all the same up to the last indice
            assert_frame_equal(rtrn[:-1], expected[:-1])
            # and that index the same throughout
            assert_index_equal(rtrn.index, expected.index)
            # assert aggregations for last row of return with incomplete last indice
            subset = df["2021-12-01":] if pdfreq == "MS" else df["2021-10-01":]
            assert_aggregations(symbols, subset, rtrn.iloc[-1])

        for pdfreq in ("MS", "QS"):
            assert_drop_incomplete_last_indice_effect(pdfreq)

    def test_reindex_equivalence(self, daily_pt, symbols, calendars):
        # NB this equivalence check works at a symbol level although not at a table
        # level - downsampling ta a CustomBusinessDay freq has the effect that the data
        # for those days that are present in the table but are not sessions of the
        # calendar (against which the downsampling is being evaluated) will be (rightly)
        # aggregated into the date for the prior calendar session.
        df = daily_pt

        for s, cal in zip(symbols, calendars):
            rtrn = df[s].pt.downsample(cal.day, cal)
            rtrn.index = rtrn.index.left
            df_reindex = df[s].pt.reindex_to_calendar(cal)
            assert_frame_equal(rtrn, df_reindex)

    def test_multiple_sessions_pt(self, multiple_sessions_pt):
        df = multiple_sessions_pt
        match = f"downsample is not implemented for {type(df.pt)}."
        with pytest.raises(NotImplementedError, match=match):
            df.pt.downsample("6D")


class TestDownsampleIntraday:
    """Dedicated tests for `PTIntraday.downsample`."""

    @pytest.fixture(scope="class")
    def cc(self, xlon, xnys) -> abc.Iterator[calutils.CompositeCalendar]:
        yield calutils.CompositeCalendar([xlon, xnys])

    @pytest.fixture(scope="class")
    def session(self, calendars) -> abc.Iterator[pd.Timestamp]:
        session_ = pd.Timestamp("2021-12-21")
        for cal in calendars:
            assert cal.is_session(session_)
        yield session_

    @pytest.fixture(scope="class")
    def prev_session(self, calendars) -> abc.Iterator[pd.Timestamp]:
        session = pd.Timestamp("2021-12-20")
        for cal in calendars:
            assert cal.is_session(session)
        yield session

    @pytest.fixture(scope="class")
    def next_session(self, calendars) -> abc.Iterator[pd.Timestamp]:
        session = pd.Timestamp("2021-12-22")
        for cal in calendars:
            assert cal.is_session(session)
        yield session

    @pytest.fixture(scope="class")
    def df_test_base_symbols(self) -> abc.Iterator[list[str]]:
        yield ["MSFT", "AZN.L"]

    @pytest.fixture
    def df_test_base(self, intraday_pt, df_test_base_symbols) -> abc.Iterator[str]:
        yield intraday_pt[df_test_base_symbols].dropna(how="all")

    @pytest.fixture(scope="class")
    def xlon_open(self, xlon, session) -> abc.Iterator[pd.Timestamp]:
        yield xlon.session_open(session)

    @pytest.fixture(scope="class")
    def xnys_open(self, xnys, session) -> abc.Iterator[pd.Timestamp]:
        yield xnys.session_open(session)

    @pytest.fixture(scope="class")
    def xnys_close(self, xnys, session) -> abc.Iterator[pd.Timestamp]:
        yield xnys.session_close(session)

    def test_errors(self, intraday_pt, composite_intraday_pt, one_min):
        """Verify raising expected errors for intraday price table."""
        df = intraday_pt
        f = df.pt.downsample

        minutes = (df.pt.interval - one_min).seconds // 60
        expected_interval = pd.Timedelta(minutes, "min")
        match = (
            "Downsampled interval must be higher than table interval, although"
            f" downsample interval evaluated as {expected_interval} whilst table"
            f" interval is {df.pt.interval}."
        )
        with pytest.raises(ValueError, match=match):
            f(str(minutes) + "min")

        minutes = (df.pt.interval + one_min).seconds // 60
        expected_interval = pd.Timedelta(minutes, "min")
        match = (
            "Table interval must be a factor of downsample interval, although"
            f" downsampled interval evaluated as {expected_interval} whilst table"
            f" interval is {df.pt.interval}."
        )
        with pytest.raises(ValueError, match=match):
            f(str(minutes) + "min")

        table_freq = df.pt.interval.as_pdfreq
        match = (
            'If anchor "open" then `calendar` must be passed as an instance of'
            " xcals.ExchangeCalendar."
        )
        with pytest.raises(TypeError, match=match):
            f(table_freq, anchor="open")

        def match_f(pdfreq) -> str:
            return re.escape(
                f"The unit of `pdfreq` must be in ['min', 'h'] although"
                f" received `pdfreq` as {pdfreq}."
            )

        invalid_pdfreqs = ["1d", "1s", "1ns", "1ms", "1ME", "1YE"]
        for pdfreq in invalid_pdfreqs:
            with pytest.raises(ValueError, match=match_f(pdfreq)):
                f(pdfreq)

        # verify no error raised for valid units
        valid_pdfreqs = [table_freq, table_freq[:-3] + "min", "1h"]
        for pdfreq in valid_pdfreqs:
            f(pdfreq)

        match = (
            "Cannot downsample a table for which a regular interval cannot be"
            " ascertained."
        )
        with pytest.raises(ValueError, match=match):
            composite_intraday_pt.pt.downsample(table_freq)

    @staticmethod
    def interval_resample_error(ds_freq: str, df: pd.DataFrame) -> str:
        offset = pd.tseries.frequencies.to_offset(ds_freq)
        return re.escape(
            f"Insufficient data to resample to {offset}.\nData from"
            f" {df.pt.first_ts} to {df.pt.last_ts} is insufficient"
            f" to create a single indice at {offset} when data is anchored"
            " to 'open'. Try anchoring on 'workback'."
        )

    def test_interval_error(self, intraday_pt, xlon, xnys):
        """Verify raises errors.PricesUnavailableIntervalResampleError.

        Verifies raises errors.PricesUnavailableIntervalResampleError when
        no data available over period when a calendar is open.
        """
        df = intraday_pt
        table_freq = df.pt.interval.as_pdfreq
        args = (table_freq, "open")
        msft, azn = "MSFT", "AZN.L"

        # create df where all rows are na for xlon but there is data for xnys
        bv_na = df[azn].isna().all(axis=1)
        start_ts = df[bv_na].pt.first_ts
        slice_start = df.index.left.get_loc(start_ts)
        bv_notna = df[azn][slice_start:].notna().all(axis=1)
        end_ts = df[slice_start:][bv_notna].pt.first_ts
        slice_end = df.index.left.get_loc(end_ts)
        df_test = df[slice_start:slice_end]

        match = self.interval_resample_error(df_test.pt.interval.as_pdfreq, df_test)
        with pytest.raises(errors.PricesUnavailableIntervalResampleError, match=match):
            df_test.pt.downsample(*args, xlon)

        # verify returns data for xnys calendar
        assert (
            df_test[msft].iloc[0].notna().all()
        )  # check xnys symbol has data for first row
        assert_frame_equal(df_test, df_test.pt.downsample(*args, xnys))

        # verify returns data when add the next row which covers the xlon open
        df_test_ = df[slice_start : slice_end + 1]
        assert_frame_equal(df_test_.iloc[-1:], df_test_.pt.downsample(*args, xlon))

    def test_interval_error2(
        self, intraday_pt, symbols, xlon, xlon_open, xnys_close, one_sec
    ):
        """Verify raises errors.PricesUnavailableIntervalResampleError.

        Verifies raises errors.PricesUnavailableIntervalResampleError when
        base data has fewer rows than required to provide data to cover a
        full downsampled interval.
        """
        df = intraday_pt[xlon_open : xnys_close - one_sec]
        factor = 3
        ds_interval = TDInterval(df.pt.interval * factor)
        ds_freq = ds_interval.as_pdfreq
        args = (ds_freq, "open", xlon)

        # Verify raises error when base data does not include the first row that
        # would be aggregated to a downsampled indice. In this case other rows
        # should be curtailed as surplus, leaving nothing to aggregate.
        for i in range(3):
            slice_start = (i * factor) + 1
            slice_end = slice_start + factor - 1
            # table will comprise (factor-1) rows, those rows excluding the first row
            # that would be required to cover a full downsampled interval.
            df_test = df[slice_start:slice_end]
            with pytest.raises(
                errors.PricesUnavailableIntervalResampleError,
                match=self.interval_resample_error(ds_freq, df_test),
            ):
                df_test.pt.downsample(*args)

        # Verify raises error when base data includes 'first row' although
        # `curtail_end=True`. In this case other rows should be curtailed to ensure
        # all downsampled indices comprise a full downsampled interval of data,
        # thereby leaving no data to aggregate.
        for i in range(3):
            slice_start = i * factor
            slice_end = slice_start + factor - 1
            df_test = df[slice_start:slice_end]
            # table will comprise (factor-1) rows, those rows excluding the last row
            # that would be required to cover a full downsampled interval.

            # verify default `curtail_end=False`` returns interval with incomplete data.
            rtrn = df_test.pt.downsample(*args)
            first_ts = xlon_open + (ds_interval * i)
            assert rtrn.pt.first_ts == first_ts
            assert rtrn.pt.last_ts == first_ts + ds_interval
            assert len(rtrn) == 1
            assert_aggregations(symbols, df_test, rtrn.iloc[0])

            with pytest.raises(
                errors.PricesUnavailableIntervalResampleError,
                match=self.interval_resample_error(ds_freq, df_test),
            ):
                df_test.pt.downsample(*args, curtail_end=True)

    def test_workback(self, intraday_pt, symbols):
        df = intraday_pt

        def f(freq) -> pd.DataFrame:
            return df.pt.downsample(freq, anchor="workback")

        base_minutes = (df.pt.interval).seconds // 60
        base_freq = str(base_minutes) + "min"

        # assert unchanged if downsample to table frequency
        assert_frame_equal(f(base_freq), df)

        test_freqs = [
            (pd.tseries.frequencies.to_offset(base_freq) * 2).freqstr,
            "30min",
            "1h",
            "4h",
            "8h",
        ]

        excess_rowss = []
        for freq in test_freqs:
            offset = pd.tseries.frequencies.to_offset(freq)
            freq_minutes = pd.Timedelta(offset).seconds // 60
            factor = freq_minutes // base_minutes
            excess_rows = len(df) % factor
            excess_rowss.append(excess_rows)

            rtrn = f(freq)
            expected_num_rows = len(df) // factor
            assert len(rtrn) == expected_num_rows
            # verify ends on same timestamp as base data
            assert rtrn.pt.last_ts == df.pt.last_ts

            # verify aggregations for each row of downsampled data
            num_base_rows = len(df)
            for i_, i in enumerate(reversed(range(expected_num_rows))):
                i_end = num_base_rows - (i_ * factor)
                i_start = i_end - factor
                subset = df.iloc[i_start:i_end]
                assert_aggregations(symbols, subset, rtrn.iloc[i])

            # verify no more data that should have been included and wasn't...
            assert 0 <= i_start < factor

        # verify that checked at least one frequency that resulted in excess rows
        assert sum(excess_rowss)

    def test_open_aggregations_xlon(
        self,
        df_test_base,
        xlon_open,
        xnys_close,
        xlon,
        df_test_base_symbols,
        prev_session,
    ):
        """Verify aggregations when basing on xlon.

        Also verifies curtailing excess rows and effect of `curtail_end`.
        """
        df = df_test_base
        table_interval = df.pt.interval

        slice_start = df.index.left.get_loc(xlon_open)
        slice_end = df.index.right.get_loc(xnys_close) + 1
        df_test = df[slice_start:slice_end]

        def f(ds_freq, curtail_end=False) -> pd.DataFrame:
            return df_test.pt.downsample(ds_freq, "open", xlon, curtail_end)

        excesses = []
        for factor in [1, 3, 5, 7]:
            ds_interval = TDInterval(table_interval * factor)
            ds_freq = ds_interval.as_pdfreq

            rtrn = f(ds_freq)

            assert rtrn.pt.first_ts == xlon_open
            excess = (xnys_close - xlon_open) % ds_interval
            excesses.append(excess)
            surplus = pd.Timedelta(0) if not excess else ds_interval - excess
            assert rtrn.pt.last_ts == xnys_close + surplus
            expected_num_rows = math.ceil((xnys_close - xlon_open) / ds_interval)
            assert len(rtrn) == expected_num_rows
            for i in range(expected_num_rows):
                i_start = i * factor
                subset = df_test[i_start : i_start + factor]
                row = rtrn.iloc[i]
                assert_aggregations(df_test_base_symbols, subset, row)

            rtrn_curtail = f(ds_freq, curtail_end=True)
            if not excess:
                assert_frame_equal(rtrn, rtrn_curtail)
            else:
                assert_frame_equal(rtrn[:-1], rtrn_curtail)

            rtrn_later_start = df[slice_start + 1 : slice_end].pt.downsample(
                ds_freq, anchor="open", calendar=xlon
            )
            assert_frame_equal(rtrn_later_start, rtrn[1:])

        # ensure at least one iteration was not a factor of period covered by table.
        assert pd.Series(excesses).sum()

        # verify rows prior to xlon start are excluded all the way back to the edge
        # case (immediately following previous xlon session). Also verify that other
        # side of edge included in return.
        xlon_prev_close = xlon.session_close(prev_session)
        xlon_prev_close_i = df.index.right.get_loc(xlon_prev_close)
        table_freq = df.pt.interval.as_pdfreq
        args = (table_freq, "open", xlon)
        rtrn = f(table_freq)
        rtrn_ = df[slice_start - 1 : slice_end].pt.downsample(*args)
        assert_frame_equal(rtrn, rtrn_)
        rtrn_ = df[xlon_prev_close_i + 1 : slice_end].pt.downsample(*args)
        assert_frame_equal(rtrn, rtrn_)

        rtrn_ = df[xlon_prev_close_i:slice_end].pt.downsample(*args)
        assert rtrn_.pt.first_ts == xlon_prev_close - df.pt.interval
        assert rtrn.pt.first_ts == xlon_open

    def test_open_aggregations_xnys(
        self,
        df_test_base,
        xlon_open,
        xnys_open,
        xnys_close,
        xnys,
        df_test_base_symbols,
        prev_session,
    ):
        """Verify aggregations when basing on xnys.

        Also verifies curtailing excess rows and effect of `curtail_end`.
        """
        df = df_test_base
        table_interval = df.pt.interval

        slice_start = df.index.left.get_loc(xlon_open)
        slice_end = df.index.right.get_loc(xnys_close) + 1
        df_test = df[slice_start:slice_end]

        def f(ds_freq, curtail_end=False) -> pd.DataFrame:
            return df_test.pt.downsample(ds_freq, "open", xnys, curtail_end)

        excesses = []
        for factor in [1, 3, 5, 7]:
            ds_interval = TDInterval(table_interval * factor)
            ds_freq = ds_interval.as_pdfreq

            rtrn = f(ds_freq)

            assert rtrn.pt.first_ts == xnys_open
            excess = (xnys_close - xnys_open) % ds_interval
            excesses.append(excess)
            surplus = pd.Timedelta(0) if not excess else ds_interval - excess
            assert rtrn.pt.last_ts == xnys_close + surplus
            expected_num_rows = math.ceil((xnys_close - xnys_open) / ds_interval)
            assert len(rtrn) == expected_num_rows
            i_base = df_test.index.left.get_loc(xnys_open)
            for i in range(expected_num_rows):
                i_start = i_base + (i * factor)
                subset = df_test[i_start : i_start + factor]
                row = rtrn.iloc[i]
                assert_aggregations(df_test_base_symbols, subset, row)

            rtrn_curtail = f(ds_freq, curtail_end=True)
            if not excess:
                assert_frame_equal(rtrn, rtrn_curtail)
            else:
                assert_frame_equal(rtrn[:-1], rtrn_curtail)

            rtrn_later_start = df_test[i_base + 1 : slice_end].pt.downsample(
                ds_freq, "open", xnys
            )
            assert_frame_equal(rtrn_later_start, rtrn[1:])

        # ensure at least one iteration was not a factor of period covered by table.
        assert pd.Series(excesses).sum()

        # verify rows prior to xnys start are excluded all the way back to the edge
        # case (immediately following previous xnys session). Also verify that other
        # side of edge included in return.
        xnys_prev_close = xnys.session_close(prev_session)
        xnys_prev_close_i = df.index.right.get_loc(xnys_prev_close)
        table_freq = df.pt.interval.as_pdfreq
        args = (table_freq, "open", xnys)
        rtrn = f(table_freq)
        rtrn_ = df[xnys_prev_close_i + 1 : slice_end].pt.downsample(*args)
        assert_frame_equal(rtrn, rtrn_)

        rtrn_ = df[xnys_prev_close_i:slice_end].pt.downsample(*args)
        assert rtrn_.pt.first_ts == xnys_prev_close - df.pt.interval
        assert rtrn.pt.first_ts == xnys_open

    def test_composite_calendar(
        self,
        df_test_base,
        xlon,
        xnys,
        next_session,
        xlon_open,
        xnys_open,
        xnys_close,
        cc,
        one_sec,
    ):
        # get df covering two sessions, starting on a xlon open and ending on an
        # xnys close.
        # NB. xlon opens before xnys, xnys closes after xlon
        df = df_test_base
        slice_start = df.index.left.get_loc(xlon_open)
        xnys_next_close = xnys.session_close(next_session)
        slice_end = df.index.right.get_loc(xnys_next_close) + 1
        xlon_next_open = xlon.session_open(next_session)
        df_test = df_test_base[slice_start:slice_end]

        factor = 3
        ds_minutes = df.pt.interval.as_minutes * factor
        ds_freq = str(ds_minutes) + "min"
        ds_interval = pd.Timedelta(ds_minutes, "min")

        def f(cal, comp_cal=None) -> pd.DataFrame:
            return df_test.pt.downsample(ds_freq, "open", cal, False, comp_cal)

        # When `calendar=xlon` the indices that fall after the xlon close should be
        # attributed to the xlon session of the close, with no na rows introduced.
        rtrn = f(xlon)
        assert not rtrn.isna().all(axis=1).any()
        assert rtrn.pt.first_ts == xlon_open
        assert rtrn.pt.last_ts == xnys_next_close
        num_i_session = (xnys_close - xlon_open) // ds_interval
        num_i_next_session = (xnys_next_close - xlon_next_open) // ds_interval
        # assuming freq is factor of each session duration - check this is the case.
        assert not (xnys_close - xlon_open) % ds_interval
        assert not (xnys_next_close - xlon_next_open) % ds_interval
        num_indices = num_i_session + num_i_next_session
        assert len(rtrn) == num_indices

        # As xlon opens before xnys, composite calendar should make no difference
        rtrn_cc = f(xlon, cc)
        assert_frame_equal(rtrn, rtrn_cc)

        # When `calendar=xnys` the indices that fall before the xnys open will be
        # attributed to the prior session, such that na rows are introduced between
        # sessions.
        rtrn = f(xnys)
        assert rtrn.isna().all(axis=1).any()
        na_rows = rtrn[xnys_close : xlon_next_open - one_sec]
        assert na_rows.isna().all(axis=None)

        assert rtrn.pt.first_ts == xnys_open
        assert rtrn.pt.last_ts == xnys_next_close

        rtrn_exc_na = rtrn.drop(na_rows.index)
        num_i_session = (xnys_close - xnys_open) // ds_interval
        num_i_next_session = (xnys_next_close - xlon_next_open) // ds_interval
        # assuming freq is factor of each session duration - check this is the case.
        assert not (xnys_close - xnys_open) % ds_interval
        assert not (xnys_next_close - xlon_next_open) % ds_interval
        num_indices = num_i_session + num_i_next_session
        assert len(rtrn_exc_na) == num_indices

        # As xlon opens before xnys, composite calendar should elimiate missing
        # rows between sessions
        rtrn_cc = f(xnys, cc)
        assert_frame_equal(rtrn_exc_na, rtrn_cc)

    def test_excess_row_handling_and_lengths(self, intraday_pt, cmes, xnys, one_sec):
        """Verify handling of excess rows at the start and/or end of sessions.

        Verifies handing where last downsampled indice of a session overlaps with
        the first downsampled indice of the following session.

        Also, takes advantage of knowledge of irregular indice lengths to
        verify methods `indices_length` and `by_indice_length`.
        """
        # pylint: disable=too-complex
        symbols = ["MSFT", "ES=F"]
        df = intraday_pt[symbols]
        cc = calutils.CompositeCalendar([cmes, xnys])

        def f(ds_interval: TDInterval, comp_cal=cc) -> pd.DataFrame:
            return df.pt.downsample(
                ds_interval.as_pdfreq, "open", xnys, composite_calendar=comp_cal
            )

        match = re.escape(
            "PriceTable interval is irregular. One or more indices were curtailed"
            " to prevent the last indice assigned to a (sub)session from overlapping"
            " with the first indice of the following (sub)session.\nUse"
            " .pt.indices_length and .pt.by_indice_length to interrogate."
        )

        def assert_indices_contiguous(index: pd.IntervalIndex):
            for i in range(len(index) - 1):
                assert index[i].right == index[i + 1].left

        def assert_sample_aggregations(sample: pd.DataFrame):
            for indice, row in sample.iterrows():
                left = indice.left  # type: ignore[union-attr]
                right = indice.right - one_sec  # type: ignore[union-attr]
                subset = df[left:right]
                assert_aggregations(symbols, subset, row)

        def assert_by_indice_length(
            df: pd.DataFrame, expected_contents: dict[pd.Timedelta, pd.DataFrame]
        ):
            gen = df.pt.by_indice_length
            for length, rtrn in gen:
                assert_frame_equal(expected_contents[length], rtrn)

        session = "2021-12-21"
        session_origin = xnys.session_open(session)
        session_cc_close = cc.session_close(session)
        next_session = xnys.next_session(session)
        next_session_origin = xnys.session_open(next_session)
        next_session_cc_open = cc.session_open(next_session)
        next_session_cc_close = cc.session_close(next_session)

        # verify any excess rows at end of session will be contiguous with excess rows
        # of at start of next session
        assert session_cc_close == next_session_cc_open

        # Sometimes the interval can be maintained by consolidating the overlapping
        # indices - undertaken by `PTIntraday._consolidate_resampled_duplicates`.
        # This will be the case where base rows attributed to last downsampled indice
        # of one session and the base rows attributed to the first downsampled indice
        # of the next session collectively represent a full downsampled indice.
        for ds_interval in (TDInterval.H1, TDInterval.H3):
            # confirm conditions to consolidate overlapping indices
            session_end_excess = (session_cc_close - session_origin) % ds_interval
            next_session_start_excess = (
                next_session_origin - next_session_cc_open
            ) % ds_interval
            assert session_end_excess + next_session_start_excess == ds_interval

            rtrn = f(ds_interval)

            # only verify assertions for session / next_session
            assert not ds_interval % df.pt.interval
            factor = ds_interval // df.pt.interval
            next_session_end_excess = (
                next_session_cc_close - next_session_origin
            ) % ds_interval
            end_slice = next_session_cc_close - next_session_end_excess - one_sec
            rtrn_selected = rtrn[session_origin:end_slice]
            df_selected = df[session_origin:end_slice]

            for i in range(len(rtrn_selected)):
                row = rtrn_selected.iloc[i]
                i_start = i * factor
                i_end = i_start + factor
                subset = df_selected[i_start:i_end]
                assert_aggregations(symbols, subset, row)

        # indices will overlap when the right side of the last indice of a session
        # is later than the left side of the first indice of the next session.
        # This is resovled in the first instance by curtailing the left side of the first
        # dowsampled indice of each session. These indices are curtailed to reflect only
        # the period for which the indice includes aggregated data (as opposed to the
        # full downsample interval which may have included a period at the left for
        # which no data was aggregated). If the overlapping persists then the right of
        # the last downsampled indice of each session is curtailed to the left of the
        # first downsampled indice of the next session.

        # Verify for case where curtails only left side of first session indice.
        # Set interval to ensure no excess rows at the end of session. Let there be
        # two intervals from session origin through session_cc_close.
        ds_interval = TDInterval((session_cc_close - session_origin) // 2)
        # check that there are excess rows at the start of the next session
        next_session_start_excess = (
            next_session_origin - next_session_cc_open
        ) % ds_interval
        assert next_session_start_excess

        with pytest.warns(errors.IntervalIrregularWarning, match=match):
            rtrn = f(ds_interval)
        assert not rtrn.index.is_overlapping

        # get the curtailed indice and the indices immediately either side
        slice_start = session_cc_close - ds_interval
        slice_end = session_cc_close + next_session_start_excess + ds_interval - one_sec
        rtrn_sample = rtrn[slice_start:slice_end]
        rtrn_index = rtrn_sample.index

        assert len(rtrn_sample) == 3
        assert_indices_contiguous(rtrn_index)

        # ensure indices lengths as expected
        assert rtrn_index[0].length == ds_interval == rtrn_index[2].length
        assert rtrn_index[1].length == next_session_start_excess

        assert_sample_aggregations(rtrn_sample)

        # verify `indices_length` for rtrn_sample
        lengths = rtrn_sample.pt.indices_length
        assert len(lengths) == 2
        assert lengths[ds_interval] == 2
        assert lengths[next_session_start_excess] == 1

        # verify `by_indice_length` for rtrn_sample
        expected_contents = {
            ds_interval: rtrn_sample.iloc[[0, -1]],
            next_session_start_excess: rtrn_sample.iloc[[1]],
        }
        assert_by_indice_length(rtrn_sample, expected_contents)

        # Verify for case where curtails left side of first indice of next session
        # and the right side of last indice of session.

        # verify there are excess rows at end of a session and start of the next
        # session, and that those excess rows do not comprise a full downsampled
        # interval.
        ds_interval = TDInterval.H5
        session_end_excess = (session_cc_close - session_origin) % ds_interval
        next_session_start_excess = (
            next_session_origin - next_session_cc_open
        ) % ds_interval
        assert session_end_excess
        assert next_session_start_excess
        assert session_end_excess + next_session_start_excess != ds_interval

        with pytest.warns(errors.IntervalIrregularWarning, match=match):
            rtrn = f(ds_interval)
        assert not rtrn.index.is_overlapping

        # get the curtailed indices and the indices immediately either side
        slice_start = session_cc_close - session_end_excess - ds_interval
        slice_end = session_cc_close + next_session_start_excess + ds_interval - one_sec
        rtrn_sample = rtrn[slice_start:slice_end]
        rtrn_index = rtrn_sample.index

        assert len(rtrn_sample) == 4
        assert_indices_contiguous(rtrn_index)

        # ensure indices lengths as expected
        assert rtrn_index[0].length == ds_interval == rtrn_index[-1].length
        assert rtrn_index[1].length == session_end_excess
        assert rtrn_index[2].length == next_session_start_excess

        # assert aggregations
        assert_sample_aggregations(rtrn_sample)

        # verify `indices_length` for rtrn_sample
        lengths = rtrn_sample.pt.indices_length
        assert len(lengths) == 3
        assert lengths[ds_interval] == 2
        assert lengths[session_end_excess] == 1
        assert lengths[next_session_start_excess] == 1

        # verify `by_indice_length` for rtrn_sample
        expected_contents = {
            ds_interval: rtrn_sample.iloc[[0, -1]],
            session_end_excess: rtrn_sample.iloc[[1]],
            next_session_start_excess: rtrn_sample.iloc[[2]],
        }
        assert_by_indice_length(rtrn_sample, expected_contents)

        # Verify case where there are excess rows at end of a session although not
        # at the start of the next session.
        session_open = xnys.session_open(session)
        next_session_open = xnys.session_open(next_session)
        # verify there are prices all the way through to the next session open
        assert next_session_open - df.pt.interval in df.index.left
        duration = next_session_open - session_open

        ds_intervals = (TDInterval.T25, TDInterval.T55, TDInterval.H5, TDInterval.H7)
        for ds_interval in ds_intervals:
            excess_rows = duration % ds_interval
            assert excess_rows

            with pytest.warns(errors.IntervalIrregularWarning, match=match):
                rtrn = f(ds_interval, comp_cal=None)
            assert not rtrn.index.is_overlapping

            # get the curtailed indice and the indices immediately either side
            slice_start = next_session_open - excess_rows - ds_interval
            slice_end = next_session_open + ds_interval - one_sec
            rtrn_sample = rtrn[slice_start:slice_end]
            rtrn_index = rtrn_sample.index

            assert len(rtrn_sample) == 3
            assert_indices_contiguous(rtrn_index)

            # ensure indices lengths as expected
            assert rtrn_index[0].length == ds_interval == rtrn_index[2].length
            assert rtrn_index[1].length == excess_rows

            assert_sample_aggregations(rtrn_sample)

            # verify `indices_length` for rtrn_sample
            lengths = rtrn_sample.pt.indices_length
            assert len(lengths) == 2
            assert lengths[ds_interval] == 2
            assert lengths[excess_rows] == 1

            # verify `by_indice_length` for rtrn_sample
            expected_contents = {
                ds_interval: rtrn_sample.iloc[[0, -1]],
                excess_rows: rtrn_sample.iloc[[1]],
            }
            assert_by_indice_length(rtrn_sample, expected_contents)

    def test_reindex_equivalence(self, intraday_pt, symbols, calendars):
        df = intraday_pt
        freq = df.pt.interval.as_pdfreq

        for s, cal in zip(symbols, calendars):
            rtrn = df.pt.downsample(freq, anchor="open", calendar=cal)
            bv = rtrn[s].notna().all(axis=1)
            rtrn_reindex = df.pt.reindex_to_calendar(cal)
            assert_frame_equal(rtrn[bv], rtrn_reindex)

        # Also serves to test single symbol tables
        for anchor in ["open", "workback"]:
            for s, cal in zip(symbols, calendars):
                rtrn = df[s].pt.downsample(freq, anchor=anchor, calendar=cal)
                rtrn_reindex = df[s].pt.reindex_to_calendar(cal)
                assert_frame_equal(rtrn.dropna(how="all"), rtrn_reindex)

    def test_composite_daily_intraday(self, composite_daily_intraday_pt):
        df = composite_daily_intraday_pt
        match = f"downsample is not implemented for {type(df.pt)}."
        with pytest.raises(NotImplementedError, match=match):
            df.pt.downsample("10min")


class TestPTIntraday:
    """Verifies methods and properties specific to `PTIntraday` class."""

    @pytest.fixture(scope="class")
    def cc(self, calendars) -> abc.Iterator[calutils.CompositeCalendar]:
        """Composite calendar for `calendars`.

        Composite calendar verified as not overlapping
        """
        cc = calutils.CompositeCalendar(calendars)
        assert not cc.sessions_overlap().any()
        yield cc

    @pytest.fixture(scope="class")
    def cc_overlapping(
        self, calendars_overlapping
    ) -> abc.Iterator[calutils.CompositeCalendar]:
        """Composite calendar for `calendars_overlapping`."""
        cc = calutils.CompositeCalendar(calendars_overlapping)
        assert cc.sessions_overlap().any()
        yield cc

    def test_indexed_left(self, intraday_pt):
        df = intraday_pt
        expected = df.copy()
        expected.index = expected.index.left.rename("left")
        assert_frame_equal(df.pt.indexed_left, expected)

    def test_indexed_right(self, intraday_pt):
        df = intraday_pt
        expected = df.copy()
        expected.index = expected.index.right.rename("right")
        assert_frame_equal(df.pt.indexed_right, expected)

    def test_sessions(self, intraday_pt, calendars, cc):
        """Verify `sessions` and `session_column` methods."""
        # NB. To get expected return test employs a different approach to method
        # under test. The approach here is NOT comprehensive. Whilst it works
        # within the confines of the test data, it would NOT work (at least not in
        # the below form) in all circumstances (due to reliance of ffill and bfill
        # without definitive knowledge of what session missing values at the edge
        # should be filled with - depending on the calendar, there might not even
        # be an open minute in the data for the required session). The below
        # implementation will (probably) be quicker than the actual implementation,
        # and could (probably) be adjusted to be comprehensive, although it is notably
        # more complex.
        df = intraday_pt
        f = df.pt.sessions
        calendars = list(calendars) + [cc]
        for cal in calendars:
            opens_ = pd.DatetimeIndex(cal.opens.values, tz=UTC)
            opens_arr = opens_.get_indexer(df.pt.utc.index.left, "ffill")
            closes_ = pd.DatetimeIndex(cal.closes.values, tz=UTC)
            closes_arr = closes_.get_indexer(df.pt.utc.index.left, "bfill")
            sessions = cal.sessions
            srs = pd.Series(pd.NaT, index=df.index, name="session")
            bv = srs.index.left.isin(opens_)
            bv = bv | ((opens_arr == closes_arr) & (~srs.index.left.isin(closes_)))
            srs.loc[bv] = sessions[opens_arr[bv]]

            # verify `direction` as default / "previous"
            rtrn = f(cal)
            srs_ = srs.copy()
            if pd.isna(srs_.iloc[0]):
                srs_.iloc[0] = sessions[opens_arr[0]]
            expected = srs_.ffill()
            assert_series_equal(rtrn, expected)

            rtrn_previous = f(cal, direction="previous")
            assert_series_equal(rtrn_previous, expected)

            # verify `direction` as "previous"
            rtrn_next = f(cal, direction="next")
            srs_ = srs.copy()
            if pd.isna(srs_.iloc[-1]):
                srs_.iloc[-1] = sessions[opens_arr[-1] + 1]
            assert_series_equal(rtrn_next, srs_.bfill())

            # verify `direction` as None
            rtrn_none = f(cal, direction=None)
            assert_series_equal(rtrn_none, srs)

        # verify `session_column` for last iteration of loop
        calendar = cal  # pylint: disable=undefined-loop-variable
        rtrn = df.pt.session_column(calendar, direction="previous")
        expected.name = (expected.name, expected.name)
        assert_frame_equal(rtrn, pd.concat([expected, df], axis=1))

    def test_indices_trading_minutes(self, intraday_1h_pt, calendars, one_sec):
        """Verify various methods related to trading minutes.

        Verifies:
            `indices_trading_minutes`
            `trading_minutes_interval`
            `indices_have_regular_trading_minutes`
            `indices_trading_minutes_values`
        """
        df = intraday_1h_pt
        possible_indice_mins = [0, 30, 60]

        # verify `indices_trading_minutes`, `trading_minutes_interval`
        # and `indices_have_regular_trading_minutes`
        for cal in calendars:
            mins = []
            for indice in df.index:
                indice_mins = cal.minutes_in_range(
                    indice.left, indice.right - one_sec, _parse=False
                )
                mins.append(len(indice_mins))
            rtrn = df.pt.indices_trading_minutes(cal)
            expected = pd.Series(mins, index=df.index, name="trading_mins")
            assert_series_equal(rtrn, expected)

            for indice_mins in set(mins):
                assert indice_mins in possible_indice_mins

            # create df_test where all indices have same number of trading minutes
            indice_mins_change = expected[expected != expected.iloc[0]].index[0]
            constant_trading_mins = expected[: indice_mins_change.left - one_sec]
            start = constant_trading_mins.index[0].left
            end = constant_trading_mins.index[-1].right - one_sec
            df_test = df[start:end]
            rtrn = df_test.pt.trading_minutes_interval(cal)
            expected_interval = TDInterval(pd.Timedelta(minutes=expected.iloc[0]))
            assert rtrn == expected_interval
            assert df_test.pt.indices_have_regular_trading_minutes(cal)

            # verify returns None on other side of edge where df includes indices
            # of a different number of trading minutes.
            df_test = df[start : indice_mins_change.right]
            assert df_test.pt.trading_minutes_interval(cal) is None
            assert not df_test.pt.indices_have_regular_trading_minutes(cal)

        # verify `indices_trading_minutes_values`
        for cal in calendars:
            rtrn = df.pt.indices_trading_minutes_values(cal)
            assert len(set(rtrn)) == len(rtrn)
            for indice_mins in rtrn:
                assert indice_mins in possible_indice_mins
            # extra check to verify that when indices minutes irreguar these methods
            # return None / False...
            assert len(rtrn) > 1  # given data
            assert df.pt.trading_minutes_interval(cal) is None
            assert not df.pt.indices_have_regular_trading_minutes(cal)


def test_pt_ss_class(
    daily_pt_ss,
    intraday_pt_ss,
    multiple_sessions_pt_ss,
    composite_intraday_pt_ss,
    composite_daily_intraday_pt_ss,
):
    """Verify .pt accessor creates instance of corresponding subclass."""
    assert isinstance(daily_pt_ss.pt, m.PTDaily)
    assert isinstance(intraday_pt_ss.pt, m.PTIntraday)
    assert isinstance(multiple_sessions_pt_ss.pt, m.PTMultipleSessions)
    assert isinstance(composite_intraday_pt_ss.pt, m.PTIntraday)
    assert isinstance(composite_daily_intraday_pt_ss.pt, m.PTDailyIntradayComposite)
