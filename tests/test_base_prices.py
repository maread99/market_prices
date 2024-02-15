"""Tests for market_prices.prices.base module.

Tests that require prices to be requested.

Notes
-----
Tests for the base module that do not require price data to be requested
are on `test_base`.

NB there's at least one test (test__get_bi_table) for a PricesBase method
defined on `test_yahoo.py` for the convenience of using fixtures and helpers
best defined there.
"""

from __future__ import annotations

from collections import abc
import datetime
import itertools
from pathlib import Path
import re
from typing import Literal
from zoneinfo import ZoneInfo

import attr
import exchange_calendars as xcals
import pandas as pd
from pandas.testing import assert_index_equal, assert_frame_equal
import pytest
import valimp

import market_prices.prices.base as m
from market_prices import errors, helpers, intervals, mptypes, pt, data
from market_prices.helpers import UTC
from market_prices.intervals import TDInterval, DOInterval
from market_prices.mptypes import Anchor, OpenEnd, Priority
from market_prices.prices import csv
from market_prices.support import tutorial_helpers as th

# from market_prices.utils import calendar_utils as calutils

from .utils import get_resource_pbt, RESOURCES_PATH, clean_temp_test_dir

# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, too-many-lines
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


@attr.s(auto_attribs=True)
class TstSymbol:
    """Data for a test symbol."""

    # pylint: disable=too-few-public-methods

    symbol: str
    calendar: str
    delay: int
    earliest_date: pd.Timestamp = attr.ib(converter=pd.Timestamp)


TST_SYMBOLS = {
    "MSFT": TstSymbol("MSFT", "XNYS", 0, "1986-03-13"),
    "AZN.L": TstSymbol("AZN.L", "XLON", 15, "1993-05-21"),
    "9988.HK": TstSymbol("9988.HK", "XHKG", 15, "2019-11-26"),
    "PETR3.SA": TstSymbol("PETR3.SA", "BVMF", 15, "2000-01-03"),
    "BTC-USD": TstSymbol("BTC-USD", "24/7", 0, "2014-09-17"),
    "ES=F": TstSymbol("ES=F", "CMES", 10, "2000-09-18"),
}


def get_tst_symbol_for_calendar(name: str) -> str:
    """Get a test symbol for a given calendar.

    Parameters
    ----------
    name
        Name of calendar for which require a test symbol.
    """
    for symb, data_ in TST_SYMBOLS.items():
        if data_.calendar == name:
            return symb
    raise ValueError(f"There is no test symbol for a calendar with name {name}.")


class PricesBaseTst(m.PricesBase):
    """PricesBase class for tests.

    Class concretes the abstract base with:
        BaseInterval and BASE_LIMITS as for PricesYahoo.
        _request_data to return prices from tables stored in resources.
        prices_for_symbols raises NotImplementedError.
        calendars and delays are ascertained from inspection of global
        `TST_SYMBOLS`.

    Parameters
    ----------
    symbols
        Symbols for which require prices.

    prices_tables
        Price tables covering all available price data, by base interval.
        (as return from `utils.get_resource_pbt`).

    lead_symbol
        As for `PricesBase`.

    recon_symbols
        Verify that `symbols` reconciles with symbols of `prices_tables`.
    """

    BaseInterval = intervals._BaseInterval(
        "BaseInterval",
        dict(
            T1=intervals.TIMEDELTA_ARGS["T1"],
            T2=intervals.TIMEDELTA_ARGS["T2"],
            T5=intervals.TIMEDELTA_ARGS["T5"],
            H1=intervals.TIMEDELTA_ARGS["H1"],
            D1=intervals.TIMEDELTA_ARGS["D1"],
        ),
    )

    BASE_LIMITS = {
        BaseInterval.T1: pd.Timedelta(30, "D"),
        BaseInterval.T2: pd.Timedelta(43, "D"),
        BaseInterval.T5: pd.Timedelta(60, "D"),
        BaseInterval.H1: pd.Timedelta(730, "D"),
        BaseInterval.D1: None,
    }

    def __init__(
        self,
        symbols: str | list[str],
        prices_tables: dict[str, pd.DataFrame],
        lead_symbol: str | None = None,
        recon_symbols: bool = True,
        calendars: (
            list[xcals.ExchangeCalendar] | dict[str, xcals.ExchangeCalendar] | None
        ) = None,
        delays: list[int] | dict[str, int] | None = None,
    ):
        self._prices_tables = prices_tables
        symbols = helpers.symbols_to_list(symbols)
        if recon_symbols:
            # verify that prices_tables are for symbols
            assert set(prices_tables["T1"].pt.symbols) == set(symbols)
        if getattr(self.BaseInterval, "D1", False):
            earliest = min(TST_SYMBOLS[symbol].earliest_date for symbol in symbols)
            self._update_base_limits({self.BaseInterval.D1: earliest})
        if calendars is None:
            calendars = [TST_SYMBOLS[symbol].calendar for symbol in symbols]
        if delays is None:
            delays = [TST_SYMBOLS[symbol].delay for symbol in symbols]
        super().__init__(symbols, calendars, lead_symbol, delays)

    def _request_data(
        self,
        interval: intervals.BI,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Request data.

        As PricesBase.
        """
        if start is None and interval.is_intraday:
            raise ValueError(
                "`start` cannot be None if `interval` is intraday. `interval`"
                f"receieved as 'f{interval}'."
            )
        df = self._prices_tables[interval.name]
        if not interval.is_daily:
            end -= helpers.ONE_SEC
        return df.loc[start:end].copy()

    def _get_class_instance(self, symbols: list[str], **kwargs) -> "PricesBaseTst":
        """Return an instance of the prices class with the same parameters as self."""
        diff = list(set(self.symbols) - set(symbols))
        tables = {
            bi: df.drop(columns=diff, level=0) for bi, df in self._prices_tables.items()
        }
        return super()._get_class_instance(symbols, prices_tables=tables, **kwargs)


class PricesBaseIntradayTst(PricesBaseTst):
    """PricesBase class for tests with only intraday base intervals.

    As `PricesBaseTst` except has no daily interval.
    """

    BaseInterval = intervals._BaseInterval(
        "BaseInterval",
        dict(
            T1=intervals.TIMEDELTA_ARGS["T1"],
            T2=intervals.TIMEDELTA_ARGS["T2"],
            T5=intervals.TIMEDELTA_ARGS["T5"],
            H1=intervals.TIMEDELTA_ARGS["H1"],
        ),
    )

    BASE_LIMITS = {
        BaseInterval.T1: pd.Timedelta(30, "D"),
        BaseInterval.T2: pd.Timedelta(43, "D"),
        BaseInterval.T5: pd.Timedelta(60, "D"),
        BaseInterval.H1: pd.Timedelta(730, "D"),
    }


class PricesBaseDailyTst(PricesBaseTst):
    """PricesBase class for tests with only intraday base intervals.

    As `PricesBaseTst` except only has daily interval.
    """

    BaseInterval = intervals._BaseInterval(
        "BaseInterval", {"D1": intervals.TIMEDELTA_ARGS["D1"]}
    )
    BASE_LIMITS = {BaseInterval.D1: None}


def mock_now(monkeypatch, now: pd.Timestamp):
    """Use `monkeypatch` to mock pd.Timestamp.now to return `now`."""

    def mock_now_(*_, tz=None, **__) -> pd.Timestamp:
        return pd.Timestamp(now.tz_convert(None), tz=tz)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now_)


# --- PricesBaseTst fixtures ---

ResourcePBT = tuple[dict[str, pd.DataFrame], pd.Timestamp]
# [0] keys as names of base intervals, values as corresponding prices tables.
# [1] timestamp when price tables created.


@pytest.fixture
def res_us_only() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single US equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("MSFT")
    at:
        Timestamp('2022-06-15 16:51:12', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("us_only")


@pytest.fixture
def prices_us(monkeypatch, res_us_only: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for single us equity.

    symbols: "MSFT"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_us_only`.
    """
    prices_tables, now = res_us_only
    mock_now(monkeypatch, now)
    yield PricesBaseTst("MSFT", prices_tables, "MSFT")


@pytest.fixture
def prices_us_intraday(
    monkeypatch, res_us_only: ResourcePBT
) -> abc.Iterator[PricesBaseIntradayTst]:
    """PricesBaseIntradayTst for single us equity.

    symbols: "MSFT"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_us_only`.
    """
    prices_tables, now = res_us_only
    mock_now(monkeypatch, now)
    yield PricesBaseIntradayTst("MSFT", prices_tables, "MSFT")


@pytest.fixture
def prices_us_daily(
    monkeypatch, res_us_only: ResourcePBT
) -> abc.Iterator[PricesBaseDailyTst]:
    """PricesBaseDailyTst for single us equity.

    symbols: "MSFT"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_us_only`.
    """
    prices_tables, now = res_us_only
    mock_now(monkeypatch, now)
    yield PricesBaseDailyTst("MSFT", prices_tables, "MSFT")


@pytest.fixture
def res_hk_only() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single Hong Kong equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("9988.HK")
    at:
        Timestamp('2022-06-16 15:27:12', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("hk_only")


@pytest.fixture
def prices_with_break(
    monkeypatch, res_hk_only: ResourcePBT
) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for single Hong Kong equity.

    symbols: "9988.HK"
    lead_symbol: "9988.HK"

    Price data provided by fixture `res_hk_only`.
    """
    prices_tables, now = res_hk_only
    mock_now(monkeypatch, now)
    yield PricesBaseTst("9988.HK", prices_tables, "9988.HK")


@pytest.fixture
def res_247_only() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single instrument that trades 24/7.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("BTC-USD")
    at:
        Timestamp('2022-06-17 13:26:44', tz=ZoneInfo("UTC"))

    NOTE: following warnings were raised on creating resource:
    PricesMissingWarning: Prices from Yahoo are missing for 'BTC-USD' at
    the base interval '0 days 00:02:00' for the following sessions:
    ['2022-05-05', '2022-05-06', '2022-05-07', '2022-05-08', '2022-05-09',
    '2022-05-10', '2022-05-11', '2022-05-12'].
    """
    yield get_resource_pbt("always_open_only")


@pytest.fixture
def prices_247(monkeypatch, res_247_only: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for for single instrument that trades 24/7.

    symbols: "BTC-USD"
    lead_symbol: "BTC-USD"

    Price data provided by fixture `res_247_only`.
    """
    prices_tables, now = res_247_only
    mock_now(monkeypatch, now)
    yield PricesBaseTst("BTC-USD", prices_tables, "BTC-USD")


@pytest.fixture
def res_us_lon() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single US equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("MSFT, AZN.L")
    at:
        Timestamp('2022-06-16 09:29:12', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("us_lon")


@pytest.fixture
def prices_us_lon(monkeypatch, res_us_lon: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for one us equity and one london equity.

    symbols: "MSFT, AZN.L"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_us_lon`.
    """
    prices_tables, now = res_us_lon
    mock_now(monkeypatch, now)
    yield PricesBaseTst("MSFT, AZN.L", prices_tables, "MSFT")


@pytest.fixture
def prices_lon_us(monkeypatch, res_us_lon: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for one london equity and one us equity.

    symbols: "MSFT, AZN.L"
    lead_symbol: "AZN.L"

    Price data provided by fixture `res_us_lon`.
    """
    prices_tables, now = res_us_lon
    mock_now(monkeypatch, now)
    yield PricesBaseTst("MSFT, AZN.L", prices_tables, "AZN.L")


@pytest.fixture
def res_hk_lon() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single Hong Kong equity and single London equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("9988.HK, AZN.L")
    at:
        Timestamp('2022-06-17 12:10:12', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("hk_lon")


@pytest.fixture
def prices_hk_lon(monkeypatch, res_hk_lon: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for single Hong Kong equity and single London equity.

    symbols: "9988.HK, AZN.L"
    lead_symbol: "9988.HK"

    Price data provided by fixture `res_hk_lon`.
    """
    prices_tables, now = res_hk_lon
    mock_now(monkeypatch, now)
    yield PricesBaseTst("9988.HK, AZN.L", prices_tables, "9988.HK")


@pytest.fixture
def res_us_hk() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single New York and single Hong Kong equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("MSFT, 9988.HK")
    at:
        Timestamp('2022-06-17 22:04:55', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("us_hk")


@pytest.fixture
def prices_us_hk(monkeypatch, res_us_hk: ResourcePBT) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for single New York and single Hong Kong equity.

    symbols: "MSFT, 9988.HK"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_us_hk`.
    """
    prices_tables, now = res_us_hk
    mock_now(monkeypatch, now)
    yield PricesBaseTst("MSFT, 9988.HK", prices_tables, "MSFT")


@pytest.fixture
def res_brz_hk() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for one Brazilian equity and one Hong Kong equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("PETR3.SA, 9988.HK")
    at:
        Timestamp('2022-06-16 15:46:12', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("brz_hk")


@pytest.fixture
def prices_detached(
    monkeypatch, res_brz_hk: ResourcePBT
) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for one Brazilian and one Hong Kong equity.

    The opening hours of the Brazilian and Hong Kong exchanges never
    overlap (considered 'detached').

    symbols: "PETR3.SA, 9988.HK"
    lead_symbol: "PETR3.SA"

    Price data provided by fixture `res_brz_hk`.
    """
    prices_tables, now = res_brz_hk
    mock_now(monkeypatch, now)
    yield PricesBaseTst("PETR3.SA, 9988.HK", prices_tables, "PETR3.SA")


@pytest.fixture
def res_lon_247() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for one London equity and one always trading equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("AZN.L, BTC-USD")
    at:
        Timestamp('2022-06-17 13:17:32', tz=ZoneInfo("UTC"))

    NOTE: following warnings were raised on creating resource:
    PricesMissingWarning: Prices from Yahoo are missing for 'BTC-USD' at
    the base interval '0 days 00:02:00' for the following sessions:
    ['2022-05-05', '2022-05-06', '2022-05-07', '2022-05-08', '2022-05-09',
    '2022-05-10', '2022-05-11', '2022-05-12'].
    PricesMissingWarning: Prices from Yahoo are missing for 'AZN.L' at the
    base interval '1 days 00:00:00' for the following sessions:
    ['2012-05-28'].
    """
    yield get_resource_pbt("lon_247")


@pytest.fixture
def prices_lon_247(
    monkeypatch, res_lon_247: ResourcePBT
) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for one London equity and one always trading equity.

    symbols: "AZN.L, BTC-USD"
    lead_symbol: "AZN.L"

    Price data provided by fixture `res_lon_247`.
    """
    prices_tables, now = res_lon_247
    mock_now(monkeypatch, now)
    yield PricesBaseTst("AZN.L, BTC-USD", prices_tables, "AZN.L")


@pytest.fixture
def res_247_245() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for one 247 equity and one 245 equity.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("BTC-USD, ES=F")
    at:
        Timestamp('2022-06-17 22:41:25', tz=ZoneInfo("UTC"))

    NOTE: following warnings were raised on creating resource:
    PricesMissingWarning: Prices from Yahoo are missing for 'BTC-USD' at
    the base interval '0 days 00:02:00' for the following sessions:
    ['2022-05-05', '2022-05-06', '2022-05-07', '2022-05-08', '2022-05-09',
    '2022-05-10', '2022-05-11', '2022-05-12'].
    PricesMissingWarning: Prices from Yahoo are missing for 'BTC-USD' at
    the base interval '0 days 01:00:00' for the following sessions:
    ['2020-06-17'].
    PricesMissingWarning: Prices from Yahoo are missing for 'ES=F' at the
    base interval '0 days 01:00:00' for the following sessions: ['2020-11-27'].
    PricesMissingWarning: Prices from Yahoo are missing for 'ES=F' at the
    base interval '1 days 00:00:00' for the following sessions: ['2000-11-23',
    '2001-01-15', '2001-02-19', '2001-05-28', '2001-07-04', '2001-09-03',
    '2001-09-12', '2001-09-13', '2001-09-14', '2001-11-22', '2002-01-21',
    '2003-07-04', '2005-01-17', '2005-02-21', '2005-05-30', '2005-07-04',
    '2005-09-05', '2007-01-15', '2007-02-19', '2007-05-28', '2007-07-04',
    '2007-09-03', '2007-11-22', '2008-01-21', '2008-02-18', '2008-05-26',
    '2008-07-04', '2008-09-01', '2008-11-27', '2009-01-19', '2009-02-16',
    '2009-05-25', '2009-07-03', '2009-09-07', '2009-11-26', '2010-01-18',
    '2010-02-15', '2010-05-31', '2010-07-05', '2010-09-06', '2010-11-25',
    '2011-01-17', '2011-02-21', '2011-05-30', '2011-07-04', '2011-09-05',
    '2011-11-24', '2012-01-16', '2012-02-20', '2012-05-28', '2012-07-04',
    '2012-09-03', '2012-10-29', '2012-10-30', '2012-11-22', '2013-01-21',
    '2013-02-18', '2013-05-27', '2013-07-04', '2013-09-02', '2013-11-28',
    '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01',
    '2014-11-27', '2015-01-19', '2015-02-16', '2015-05-25', '2015-07-03',
    '2015-09-07', '2015-11-26', '2016-01-18', '2016-02-15', '2016-05-30',
    '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24',
    '2017-01-16', '2017-02-20', '2017-05-29', '2017-07-04', '2017-09-04',
    '2017-11-23', '2018-01-15', '2018-02-19', '2018-05-28', '2018-07-04',
    '2018-09-03', '2018-11-22', '2019-01-21', '2019-02-18', '2019-05-27',
    '2019-07-04', '2019-09-02', '2019-11-28', '2020-01-20', '2020-02-17',
    '2020-05-25', '2020-07-03', '2020-09-07', '2020-11-26', '2021-01-18',
    '2021-02-15', '2021-05-31', '2021-07-05', '2021-09-06', '2022-01-17',
    '2022-02-21', '2022-05-30'].
    """
    yield get_resource_pbt("alwaysopen_weekdaysopen")


@pytest.fixture
def prices_247_245(
    monkeypatch, res_247_245: ResourcePBT
) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for one 247 equity and one 245 equity.

    symbols: "BTC-USD, ES=F"
    lead_symbol: "BTC-USD"

    Price data provided by fixture `res_lon_247`.
    """
    prices_tables, now = res_247_245
    mock_now(monkeypatch, now)
    yield PricesBaseTst("BTC-USD, ES=F", prices_tables, "BTC-USD")


@pytest.fixture
def res_us_lon_hk() -> abc.Iterator[ResourcePBT]:
    """PricesBaseTst resource for single equity of three different exchanges.

    Resource for one equity of each of New York, London and Hong Kong.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("MSFT, AZN.L, 9988.HK")
    at:
        Timestamp('2022-06-17 15:57:09', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("us_lon_hk")


@pytest.fixture
def prices_us_lon_hk(
    monkeypatch, res_us_lon_hk: ResourcePBT
) -> abc.Iterator[PricesBaseTst]:
    """PricesBaseTst for single Hong Kong equity and single London equity.

    symbols: "MSFT, AZN.L, 9988.HK"
    lead_symbol: "MSFT"

    Price data provided by fixture `res_hk_lon`.
    """
    prices_tables, now = res_us_lon_hk
    mock_now(monkeypatch, now)
    yield PricesBaseTst("MSFT, AZN.L, 9988.HK", prices_tables, "MSFT")


# --- Other fixtures ---


@pytest.fixture
def session_length_xnys() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "h")


@pytest.fixture
def session_length_xhkg() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "h")


@pytest.fixture
def session_length_xlon() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(8.5, "h")


@pytest.fixture
def session_length_bvmf() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(8, "h")


@pytest.fixture
def stricts() -> abc.Iterator[list[bool]]:
    """List of possible values for `strict` parameter."""
    yield [True, False]


@pytest.fixture
def priorities() -> abc.Iterator[list[Priority]]:
    """List of possible values for `priority` parameter."""
    yield [Priority.END, Priority.PERIOD]


# --- Helper functions ---


def get_symbols_for_calendar(prices: PricesBaseTst, name: str) -> str | list[str]:
    """Get symbol(s) of a `prices` instance associated with a given calendar `name`."""
    for cal, symbols in prices.calendars_symbols.items():
        if cal.name == name:
            return symbols[0] if len(symbols) == 1 else symbols
    raise ValueError(f"The PricesBaseTst instance has no calendar with name {name}.")


def get_calendar_from_name(prices: PricesBaseTst, name: str) -> xcals.ExchangeCalendar:
    """Get calendar of a given `name` from a `prices` instance."""
    for cal in prices.calendars_unique:
        if cal.name == name:
            return cal
    raise ValueError(f"The PricesBaseTst instance has no calendar with name {name}.")


def get_prices_limit_mock(
    prices: PricesBaseTst, bi: intervals.BI, limit: pd.Timestamp
) -> PricesBaseTst:
    """Return instance of PricesBaseTst with `limit` for `bi` set to limit.

    Otherwise instance will be unchanged from `prices`.
    """
    limits = prices.BASE_LIMITS.copy()
    limits[bi] = limit

    class PricesBaseTstMock(PricesBaseTst):
        """Mock PricesYahoo class."""

        BASE_LIMITS = limits

    lead = prices.lead_symbol_default
    return PricesBaseTstMock(prices.symbols, prices._prices_tables, lead)


def reset_prices(prices: PricesBaseTst) -> PricesBaseTst:
    """Return a new PricesBaseTst instance based on `prices`."""
    return PricesBaseTst(
        prices.symbols, prices._prices_tables, prices.lead_symbol_default
    )


def set_get_prices_params(
    prices: PricesBaseTst,
    pp_raw: mptypes.PP,
    ds_interval: intervals.PTInterval | None,
    lead_symbol: str | None = None,
    anchor: Anchor = Anchor.OPEN,
    openend: OpenEnd = OpenEnd.MAINTAIN,
    strict: bool = True,
    priority: mptypes.Priority = Priority.END,
) -> PricesBaseTst:
    """Return `prices` with gpp set.

    prices.gpp set to instance of prices.GetPricesParams with arguments
    as received.
    """
    if lead_symbol is None:
        lead_symbol = prices.lead_symbol_default
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


def _get_sessions_daterange_for_bi_cal(
    prices: PricesBaseTst,
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
    prices: PricesBaseTst,
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
    prices: PricesBaseTst,
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
    return start_session, end_session


def get_current_session(calendar: xcals.ExchangeCalendar) -> pd.Timestamp:
    today = helpers.now(intervals.BI_ONE_DAY)
    if calendar.is_session(today):
        if helpers.now() >= calendar.session_open(today):
            return today
        else:
            return calendar.previous_session(today)
    else:
        return calendar.date_to_session(today, "previous")


def get_consecutive_sessions(
    prices: m.PricesBase,
    bi: intervals.BI,
    calendar: xcals.ExchangeCalendar,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a given number of consecutive sessions.

    Sessions will be sessions of a given `calendar` for which intraday data
    is available at the given `bi`.
    """
    start, _ = th.get_sessions_range_for_bi(prices, bi, calendar)
    end = calendar.next_session(start)
    return start, end


def get_conforming_sessions(
    prices: PricesBaseTst,
    bi: intervals.BI,
    calendars: list[xcals.ExchangeCalendar],
    session_length: list[pd.Timedelta],
    num_sessions: int = 2,
) -> pd.DatetimeIndex:
    """Get sessions that conform with given requirements.

    Prices will be available at bi for at least one session (evaluated
    against CompositeCalendar) prior to first session.
    """
    # get sessions for which price data available at all base intervals
    session_start, session_end = th.get_sessions_range_for_bi(prices, bi)
    # margin allowing for prices at bi to be available over at least one prior session
    session_start = prices.cc.next_session(session_start)
    sessions = th.get_conforming_sessions(
        calendars, session_length, session_start, session_end, num_sessions
    )
    return sessions


def get_sessions_xnys_xhkg_xlon(bi: TDInterval, num: int = 2) -> pd.DatetimeIndex:
    """Get multiple consecutive regular sessions available at bi.

    Sessions are consecutive sessions of xnys, xhkg and xlon.

    Sessions will be consecutive regular sessions available at bi based on 'now' as is
    or as mocked.
    """
    calendar_names = ["XNYS", "XHKG", "XLON"]
    symbols = [get_tst_symbol_for_calendar(name) for name in calendar_names]
    # prices_tables not required for purposes...
    prices = PricesBaseTst(symbols, {}, recon_symbols=False)
    session_length = [
        pd.Timedelta(hours=6, minutes=30),
        pd.Timedelta(hours=6, minutes=30),
        pd.Timedelta(hours=8, minutes=30),
    ]
    calendars = [get_calendar_from_name(prices, name) for name in calendar_names]

    for bi_ in prices.bis_intraday:
        # convert `bi` from TDInteval to a BI
        if bi_ == bi:
            bi_req = bi_
            break

    return get_conforming_sessions(prices, bi_req, calendars, session_length, num)


# --- Common assertion functions ---


def assert_price_table(df: pd.DataFrame, prices: m.PricesBase):
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert not df.isna().all(axis=1).any()

    symbols = prices.symbols
    assert isinstance(df.columns, pd.MultiIndex)
    expected_columns = pd.MultiIndex.from_product(
        (symbols, ["open", "high", "low", "close", "volume"]), names=("symbol", "")
    )
    assert_index_equal(df.columns, expected_columns, check_order=False)


def assert_daily(df: pd.DataFrame, prices: m.PricesBase):
    """Assert `df` represents daily data."""
    assert_price_table(df, prices)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.pt.interval == intervals.ONE_DAY


def assertions_daily(
    df: pd.DataFrame,
    prices: PricesBaseTst,
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


def assert_prices_table_ii(df: pd.DataFrame, prices: m.PricesBase):
    """Assert `df` is a price table with interval index."""
    assert df.index.is_non_overlapping_monotonic
    assert isinstance(df.index, pd.IntervalIndex)
    assert_price_table(df, prices)


def assert_multiple_days(
    df: pd.DataFrame, prices: PricesBaseTst, interval: intervals.PTInterval
):
    """Assert `df` is price table with interval as a multiple of 1D."""
    assert df.pt.interval == interval
    assert_prices_table_ii(df, prices)


def assert_interval(df: pd.DataFrame, interval: TDInterval | intervals.BI):
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


def assertions_intraday_common(
    df: pd.DataFrame, prices: PricesBaseTst, interval: intervals.TDInterval
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
    interval: TDInterval,
    prices: PricesBaseTst,
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


def assertions_monthly(
    df: pd.DataFrame,
    prices: PricesBaseTst,
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


# --- Tests ---


def test__date_to_session(prices_us_lon_hk):
    prices = prices_us_lon_hk
    f = prices._date_to_session
    # from knowledge of schedule
    date = pd.Timestamp("2021-12-25")
    assert f(date, "earliest", "next") == pd.Timestamp("2021-12-27")
    assert f(date, "latest", "next") == pd.Timestamp("2021-12-29")
    assert f(date, "earliest", "previous") == pd.Timestamp("2021-12-23")
    assert f(date, "latest", "previous") == pd.Timestamp("2021-12-24")


def test_session_prices(prices_us_lon, one_day):
    prices = prices_us_lon
    f = prices.session_prices
    xnys = prices.calendar_default
    xlon = get_calendar_from_name(prices, "XLON")

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
    session = xnys_sessions.difference(xlon_session)[0]
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
    # NB placed at end as will fail if today flakylisted - test all can before skips.
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

    @staticmethod
    def match_liie(
        prices: PricesBaseTst,
        bis_period: list[intervals.BI],
        bis_accuracy: list[intervals.BI],
        anchor: Anchor = Anchor.OPEN,
    ) -> str:
        """Return error message for LastIndiceIntervalError."""
        bi_accurate = bis_accuracy[0]
        limit_start, limit_end = prices.limits[bi_accurate]
        cal = prices.gpp.calendar
        earliest_minute = cal.minute_to_trading_minute(limit_start, "next")
        latest_minute = cal.minute_to_trading_minute(limit_end, "previous")
        available_period = (earliest_minute, latest_minute)

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
            f"The period over which data is available at {bi_accurate}"
            f" is {available_period}, although at this base interval the"
            f" requested period evaluates to {would_be_period}.\n"
            f"Period evaluated from parameters: {prices.gpp.pp_raw}."
            "\nData that can express the period end with the greatest possible"
            f" accuracy is available from {helpers.fts(earliest_minute)} through to"
            " the end of the requested period. Consider passing `strict` as False to"
            " return prices for this part of the period.\nAlternatively, consider"
        )
        if anchor is mptypes.Anchor.OPEN:
            s += " creating a composite table (pass `composite` as True) or"
        s += " passing `priority` as 'period'."
        return re.escape(s)

    def test__get_bi_table_intraday_interval_bi(self, prices_us, stricts, priorities):
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
            assert_bounds(table_H1, (start, end_session_open + pd.Timedelta(1, "h")))

    def test__get_bi_table_intraday_interval_non_bi(
        self, prices_us, stricts, priorities, one_min
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
            start=start, periods=4, freq="min"
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
        assert_frame_equal(table, table_T4_T1)

        # verify `priority` being passed through. If  PERIOD then will get table
        # covering full period using T2 base data, such that end is a minute off.
        prices = set_get_prices_params(
            prices, pp_oob, ds_interval, anchor=anchor, priority=Priority.PERIOD
        )
        table = f()
        assert_most_common_interval(table, ds_interval)
        assert table.pt.last_ts == end - one_min
        assert table.pt.first_ts in pd.date_range(start=start, periods=4, freq="min")

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
            assert table.index[-1].length == pd.Timedelta(30, "min")
            assert_most_common_interval(table, ds_interval)

            # verify when openend is MAINTAIN
            prices = set_get_prices_params(
                prices, pp, ds_interval, openend=OpenEnd.MAINTAIN
            )
            delta = 30 if ds_interval is prices.bis.H1 else 90
            end_maintain = end_close + pd.Timedelta(delta, "min")
            table = f()
            assert_bounds(table, (start, end_maintain))
            assert_interval(table, ds_interval)

    def test__get_table_intraday_interval_H1(self, prices_lon_us):
        """Test `_get_table_intraday` for H1 ds_interval.

        Verifies that method has downsampled T5 data to H1 via
        table.`pt.downsample`.

        Verifies final indice is shortened by `openend` as "shorten".
        """
        prices = prices_lon_us
        # ds_interval H1 which can only be served by T5 data (cals not synced at H1)
        interval = prices.bis.T5
        range_start, _ = th.get_sessions_range_for_bi(prices, interval)
        _, range_end = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        length = pd.Timedelta(13, "h")
        start, end = th.get_conforming_cc_sessions(
            prices.cc, length, range_start, range_end, 2
        )
        pp = get_pp(start=start, end=end)

        # Verify for xlon lead
        lead = prices.lead_symbol_default
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
            end_maintain = end_close - pd.Timedelta(30, "min")
            assert_bounds(table, (start_open, end_maintain))
            assert_interval(table, ds_interval)

            # verify when openend is SHORTEN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.SHORTEN, **prices_kwargs
            )
            table = prices._get_table_intraday()
            assert_bounds(table, (start_open, end_close))
            assert table.index[-1].length == pd.Timedelta(30, "min")
            assert_most_common_interval(table, ds_interval)

        # Verify for xnys lead
        lead = get_symbols_for_calendar(prices, "XNYS")
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
            end_maintain = end_close + pd.Timedelta(delta, "min")
            assert_bounds(table, (start_open, end_maintain))
            assert_interval(table, ds_interval)

            # verify when openend is SHORTEN
            prices = set_get_prices_params(
                prices, openend=OpenEnd.SHORTEN, **prices_kwargs
            )
            table = prices._get_table_intraday()
            assert_bounds(table, (start_open, end_close))
            assert table.index[-1].length == pd.Timedelta(30, "min")
            assert_most_common_interval(table, ds_interval)

    def assertions_downsample_bi_table(
        self,
        prices: PricesBaseTst,
        factors: tuple[int],
        bi: intervals.BI,
        starts: list[pd.Timestamp] | pd.Timestamp,
        end: pd.Timestamp,
        expected_starts: list[pd.Timestamp] | pd.Timestamp | None = None,
        expected_end: pd.Timestamp | None = None,
        lead_symbol: str | None = None,
        ignore_breaks: bool = False,
    ):
        """Make assertions for `_downsample_bi_table` tests.

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
                    right = row.name.right - helpers.ONE_SEC
                    subset = df_bi[left:right]
                    for s in prices.symbols:
                        subset_s, row_s = subset[s], row[s]
                        if subset_s.isna().all(axis=None):
                            assert row_s.isna().all(axis=None)
                            continue
                        assert subset_s.volume.sum() == row_s.volume
                        assert subset_s.high.max() == row_s.high
                        assert subset_s.low.min() == row_s.low
                        assert subset_s.bfill().open.iloc[0] == row_s.open
                        assert subset_s.ffill().close.iloc[-1] == row_s.close

    def test__downsample_bi_table_lon_us(self, prices_lon_us, one_min):
        """Tests `_downsample_bi_table` for symbols on overlapping exchanges.

        Test inputs limited to circumstances in which method called by
        `_get_table_intraday`.
        """
        prices = prices_lon_us
        lead = prices.lead_symbol_default
        xlon = prices.calendar_default
        bi = prices.bis.T5

        start, end = get_consecutive_sessions(prices, bi, xlon)
        # test start as both session and time
        starts = (start, prices.cc.session_open(start) + one_min)
        xlon_close = xlon.session_close(end)
        self.assertions_downsample_bi_table(
            prices, (2, 6), bi, starts, end, None, xlon_close, lead
        )

    def test__downsample_bi_table_with_breaks(self, prices_with_break, one_min):
        """Tests `_downsample_bi_table` for symbol with a break.

        Test inputs limited to circumstances in which method called by
        `_get_table_intraday`.
        """
        prices = prices_with_break
        cal = prices.calendar_default
        bi = prices.bis.T5
        start, end = get_consecutive_sessions(prices, bi, cal)
        self.assertions_downsample_bi_table(prices, (2, 5, 7, 12), bi, start, end)

        # Test for interval where breaks are ignored
        bi = prices.bis.H1
        start, end = get_consecutive_sessions(prices, bi, cal)
        # test start as both session and time
        starts = (start, cal.session_open(start) + one_min)
        factors = (2, 5)
        self.assertions_downsample_bi_table(
            prices, factors, bi, starts, end, None, None, None, True
        )

    def test__downsample_bi_table_detached(
        self, prices_detached, one_min, session_length_bvmf, session_length_xhkg
    ):
        """Tests `_downsample_bi_table` for symbols of non-overlapping exchanges.

        Test inputs limited to circumstances in which method called by
        `_get_table_intraday`.
        """
        prices = prices_detached
        lead = prices.lead_symbol_default
        bi = prices.bis.T5
        bvmf = prices.calendar_default
        # get the other calendar
        xhkg = get_calendar_from_name(prices, "XHKG")
        calendars = [bvmf, xhkg]
        session_length = [session_length_bvmf, session_length_xhkg]

        start, end = get_conforming_sessions(prices, bi, calendars, session_length, 2)
        bvmf_open = bvmf.session_open(start)
        # NB 21 is limit before last indice of hkg am session overlaps with pm session
        self.assertions_downsample_bi_table(
            prices, (2, 5, 7, 12, 21), bi, start, end, bvmf_open, None, lead
        )
        with pytest.warns(errors.IntervalIrregularWarning):
            self.assertions_downsample_bi_table(
                prices, (22,), bi, start, end, bvmf_open, None, lead
            )

        # Test for interval where breaks are ignored
        bi = prices.bis.H1
        start, end = get_conforming_sessions(prices, bi, calendars, session_length, 2)
        bvmf_open = bvmf.session_open(start)
        # test start as both session and time
        starts = (start, bvmf.session_open(start) + one_min)
        exp_starts = (bvmf_open, bvmf_open + one_min)
        self.assertions_downsample_bi_table(
            prices, (2, 5), bi, starts, end, exp_starts, None, lead, True
        )


class TestForcePartialIndices:
    """Tests for `_force_partial_indices`."""

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
        closes = pd.DatetimeIndex(cal.closes[start:end], tz=UTC)

        bv = index.right != index_forced.right
        # assert all indices with right side changed now reflect a close
        assert index_forced.right[bv].isin(closes).all()
        # assert that all changed table indices had right > corresponding session close
        changed_indices_sessions = table.pt.sessions(cal)[bv]
        for indice, session in changed_indices_sessions.items():
            assert (indice.left < cal.session_close(session)) & (
                indice.right > cal.session_close(session)
            )
        # check all session closes are represented
        assert closes.isin(index_forced.right).all()

    def test_us_lon(self, prices_us_lon, session_length_xnys):
        """Test for two overlapping calendars."""
        prices = prices_us_lon
        xnys = prices.calendar_default
        # use T5 data so that can get H1 data out for cals that are unsynced at H1
        lead = prices.lead_symbol_default
        start, end = get_sessions_daterange_for_bi(
            prices, prices.bis.T5, xnys, session_length_xnys
        )
        pp = get_pp(start=start, end=end)
        prices = set_get_prices_params(
            prices, pp, prices.bis.H1, lead, openend=OpenEnd.SHORTEN
        )
        table = prices._get_table_intraday()

        index = table.index
        index_forced = prices._force_partial_indices(table)

        slc = slice(start, end)
        closes = pd.DatetimeIndex(prices.cc.closes[slc], tz=UTC)
        xlon = get_calendar_from_name(prices, "XLON")

        # Verify left side of forced index as expected
        bv = index.left != index_forced.left
        # assert all indices with left side changed now reflect an open
        opens = pd.DatetimeIndex(prices.cc.opens[slc], tz=UTC)
        assert index_forced.left[bv].isin(opens).all()

        # assert all changed table indices had left < corresponding lon session open
        changed_indices_sessions = table.pt.sessions(prices.cc, direction="next")[bv]
        for indice, session in changed_indices_sessions.items():
            assert (indice.left < xlon.session_open(session)) & (
                indice.right > xlon.session_open(session)
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
        for indice, session in changed_indices_sessions.items():
            if xnys.is_session(session):
                us_close = (indice.left < xnys.session_close(session)) & (
                    indice.right > xnys.session_close(session)
                )
            else:
                us_close = False
            if xlon.is_session(session):
                uk_close = (indice.left < xlon.session_close(session)) & (
                    indice.right > xlon.session_close(session)
                )
            else:
                uk_close = False
            assert us_close or uk_close

        # check all cc closes are represented
        last_session = prices.gpp.calendar.date_to_session(end, "previous")
        last_close = prices.gpp.calendar.session_close(last_session)
        assert closes[closes <= last_close].isin(index_forced.right).all()

        # Verify that if AZN.L lead then no left side of forced index is unchanged
        lead = get_symbols_for_calendar(prices, "XLON")
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
        slc = slice(start, end)
        am_closes = pd.DatetimeIndex(cal.break_starts[slc], tz=UTC)
        pm_opens = pd.DatetimeIndex(cal.break_ends[slc], tz=UTC)
        pm_closes = pd.DatetimeIndex(cal.closes[slc], tz=UTC)
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
        for indice, session in changed_indices_sessions.items():
            break_end = cal.session_break_end(session)
            assert (indice.left < break_end) & (indice.right > break_end)

        # assert all table indices with changed right side had right > corresponding
        # (sub)session close.
        changed_indices_sessions = table.pt.sessions(cal)[bv_right]
        for indice, session in changed_indices_sessions.items():
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

    def test_detached_calendars(self, prices_detached):
        """Test for two detached calendars, one of which has breaks."""
        prices = prices_detached
        lead = get_symbols_for_calendar(prices, "XHKG")
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
        bvmf = prices.calendar_default
        slc = slice(start, end)
        brz_closes = pd.DatetimeIndex(bvmf.closes[slc], tz=UTC)
        hk_am_closes = pd.DatetimeIndex(xhkg.break_starts[slc], tz=UTC)
        hk_pm_opens = pd.DatetimeIndex(xhkg.break_ends[slc], tz=UTC)
        hk_pm_closes = pd.DatetimeIndex(xhkg.closes[slc], tz=UTC)
        all_closes = brz_closes.union(hk_pm_closes).union(hk_am_closes)

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
        for indice, session in changed_indices_sessions.items():
            break_end = xhkg.session_break_end(session)
            assert (indice.left < break_end) & (indice.right > break_end)

        # assert all changed table indices had right > corresponding (sub)session close.
        changed_indices_sessions = table.pt.sessions(prices.cc)[bv_right]
        for indice, session in changed_indices_sessions.items():
            if bvmf.is_session(session):
                bvmf_close = bvmf.session_close(session)
                brz_close = (indice.left < bvmf_close) & (indice.right > bvmf_close)
            else:
                brz_close = False
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
            assert brz_close or hk_am_close or hk_pm_close

        # check all xhkg pm subsession opens are represented
        assert hk_pm_opens.dropna().isin(index_forced.left).all()

        # check all (sub)session closes are represented
        # only consider those brz_closes that fall before the last hk close
        # (as lead symbol is hk, table will end before the end session's brz close,
        # and this can be more than one session before if hk has holidays towards
        # the end session).
        bv = (brz_closes > hk_pm_closes[0]) & (brz_closes < hk_pm_closes[-1])
        brz_closes_ = brz_closes[bv]
        for closes in (brz_closes_, hk_pm_closes, hk_am_closes.dropna()):
            assert closes.isin(index_forced.right).all()


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
        yield get_tst_symbol_for_calendar("XLON")

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
        sessions = sessions.tz_localize(UTC)
        assert_index_equal(daily_part.index.left, sessions)
        assert_index_equal(daily_part.index.right, sessions)
        # verify not missing anything inbetween
        assert_frame_equal(table, pd.concat([daily_part, intraday_part]))

    def test__get_daily_intraday_composite_error(self, prices_247_245):
        """Test raises error when calendars overlap."""
        prices = prices_247_245
        lead = get_symbols_for_calendar(prices, "CMES")

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
        calendars = prices.calendar_default, get_calendar_from_name(prices, "XLON")

        session_length = [session_length_xnys, session_length_xlon]
        _, end_session = get_conforming_sessions(
            prices, prices.bis.T1, calendars, session_length, 2
        )
        end_session_open = prices.cc.session_open(end_session)
        end = end_session_open + pd.Timedelta(7, "min")
        _, start_session = get_conforming_sessions(
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
        limit_T1_mock = end_session_open - pd.Timedelta(1, "h")
        prices = get_prices_limit_mock(prices, prices.bis.T1, limit_T1_mock)
        end = end_session_open + pd.Timedelta(3, "min")
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table_edgecase1 = prices._get_table_composite()

        # verify T5 parts of composite tables are the same
        assert_frame_equal(table_edgecase1[:-3], table[:-7])

        # Test edge case 2.
        limit_T1_mock = end_session_open + pd.Timedelta(6, "min")
        prices = get_prices_limit_mock(prices, prices.bis.T1, limit_T1_mock)
        end = end_session_open + pd.Timedelta(9, "min")
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table_edgecase2 = prices._get_table_composite()
        # verify T5 parts of composite tables are the same
        assert_frame_equal(table_edgecase2[:-4], table[:-7])
        # verify other part is being met from T2 data, and table ends one minute short
        # of what would be max accuaracy
        T2_data = table_edgecase2[-4:]
        assert_interval(T2_data, prices.bis.T2)
        assert len(T2_data) == 4
        assert T2_data.pt.first_ts == end_session_open
        assert T2_data.pt.last_ts == end - one_min

        # Verify raises error under edge case 2 when 'next table' has same interval
        # as table1.
        limit_T1_mock = end_session_open + pd.Timedelta(20, "min")  # unable to serve
        # T2 unable to serve from start of day
        limit_T2_mock = end_session_open + pd.Timedelta(6, "min")
        limits = prices.BASE_LIMITS.copy()
        limits[prices.bis.T1] = limit_T1_mock
        limits[prices.bis.T2] = limit_T2_mock

        class PricesMock(PricesBaseTst):
            """Mock PricesBaseTst class."""

            BASE_LIMITS = limits

        prices = PricesMock(symbols, prices._prices_tables, prices.lead_symbol_default)

        end = end_session_open + pd.Timedelta(8, "min")
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
        length = pd.Timedelta(13, "h")
        _start_session, end_session = get_sessions_daterange_for_bi(
            prices, prices.bis.T2, length_end_session=length
        )
        while not (prices.cc.sessions_length(end_session, end_session) == length).iloc[
            0
        ]:
            end_session = prices.cc.previous_session(end_session)
            if end_session == _start_session:
                raise ValueError(f"Unable to get a 'T2' session of length {length}.")

        end_session_open = prices.cc.session_open(end_session)
        end = end_session_open + pd.Timedelta(6, "min")

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
        assert table.pt.first_ts == helpers.to_utc(start_session)
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
        limit_T2_mock = end_session_open + pd.Timedelta(1, "min")
        prices = get_prices_limit_mock(prices, prices.bis.T2, limit_T2_mock)
        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)
        table = prices._get_table_composite()

        # Verify that as T2 data not available from session open, fulfilled with T5 data
        # to the extent possible (i.e. to max accuracy available)
        assert (table[-1:].index.length == prices.bis.T5).all()
        # table 1 daily indices have no length
        assert (table[:-1].index.length == pd.Timedelta(0)).all()
        assert table.pt.first_ts == helpers.to_utc(start_session)
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
        end = end_session_open + pd.Timedelta(1, "h")
        start_session = calendar.session_offset(end_session, -20)

        pp = get_pp(start=start_session, end=end)
        prices = set_get_prices_params(prices, pp, ds_interval=None, lead_symbol=lead)

        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices._get_table_composite()


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

    def test_gpp(self, prices_us_lon):
        """Test `PricesBase.GetPricesParams` is being initialised.

        Tests parameters passed through as expected from arguments passed
        to `get`.
        """
        prices = prices_us_lon
        assert not prices.has_data

        # verify options default values
        prices.get("5min", minutes=30)
        assert prices.has_data
        assert isinstance(prices.gpp, m.PricesBase.GetPricesParams)
        assert prices.gpp.anchor is Anchor.OPEN
        assert prices.gpp.openend is OpenEnd.MAINTAIN
        assert prices.gpp.prices is prices
        assert prices.gpp.ds_interval is TDInterval.T5
        assert prices.gpp.lead_symbol == prices.lead_symbol_default
        assert prices.gpp.strict is True

        # passing as default
        lead = prices.lead_symbol_default
        strict = True
        prices.get(
            "10min",
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
        lead = get_symbols_for_calendar(prices, "XLON")
        strict = False
        prices.get(
            "3min",
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
        interval = intervals.TDInterval.T5

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

        msg = "Got unexpected keyword argument: 'minute'"
        with pytest.raises(valimp.InputsError, match=msg):
            prices.get(minute=3)

        # Verify that a parameter that takes a Literal raises exception if pass
        # non-valid value. Only test for one such parameter.
        msg = re.escape(
            "anchor\n\tTakes a value from <('workback', 'open')> although"
            " received 'wrkback'."
        )
        with pytest.raises(valimp.InputsError, match=msg):
            prices.get("30min", anchor="wrkback")

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
        """Test that parsing validates `interval`.

        Only tests that `interval` is being validated by `to_ptinterval`
        via `intevals.parse_interval`. `to_ptinterval` is
        comprehensively tested elsewhere.
        """
        # Verify invalid interval unit
        match = re.escape(
            "`interval` unit must by one of ['MIN', 'T', 'H', 'D', 'M']"
            " (or lower-case) although evaluated to 'G'."
        )
        with pytest.raises(ValueError, match=match):
            prices_us.get("3G")

    def test_intervals_inferred(self, prices_us):
        """Test intervals inferred as expected."""
        prices = prices_us
        cal = prices.calendar_default

        bi = prices.bis.T1
        rng_start, rng_end = th.get_sessions_range_for_bi(prices, bi)
        rng_start_open = cal.session_open(rng_start)
        rng_end_close = cal.session_close(rng_end)

        # Verify return at intraday intervals.

        # verify end as time. Given end T5 is highest intraday interval that can fulfil.
        end = cal.session_open(rng_end) + pd.Timedelta(5, "min")
        df = prices.get(start=rng_start, end=end)
        assertions_intraday_common(df, prices, prices.bis.T5)
        assert_bounds(df, (rng_start_open, end))

        # verify start as time.
        start = cal.session_open(rng_start) + pd.Timedelta(5, "min")
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
        assert_bounds(df, (rng_start_open, rng_start_open + pd.Timedelta(4, "min")))
        df = prices.get(minutes=4)
        assert df.pt.interval in (prices.bis.T1, prices.bis.T2)

        # verify hours
        df = prices.get(start=rng_start, hours=4)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert_bounds(df, (rng_start_open, rng_start_open + pd.Timedelta(4, "h")))
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
        end = cal.session_offset(rng_start, 4)
        df = prices.get(start=rng_start, end=end)
        assertions_intraday_common(df, prices, prices.bis.H1)
        assert df.pt.first_ts == rng_start_open
        assert expected_end in df.index[-1]

        # verify start only, 5 sessions prior to last or current session
        limit_left, limit_right = prices.limits[prices.bis.D1]
        last_session = cal.date_to_session(limit_right, "previous")
        if pd.Timestamp.now(tz=UTC) < cal.session_open(last_session):
            last_session = cal.previous_session(last_session)
        start = cal.session_offset(last_session, -4)
        df = prices.get(start=start)
        assert df.pt.interval in prices.bis_intraday
        assert df.pt.first_ts == cal.session_open(start)

        # Verify return at daily interval

        # verify days
        df = prices.get(days=6)
        assert_daily(df, prices)
        assert df.pt.last_ts == last_session
        expected_start = cal.session_offset(last_session, -5)
        assert df.pt.first_ts == expected_start

        # verify start and end describing > 5 sessions
        end = cal.session_offset(rng_start, 5)
        df = prices.get(start=rng_start, end=end)
        assertions_daily(df, prices, rng_start, end)

        # verify start only, 6 sessions prior to last/current_session
        # 6 rather than 5 as if a live session then won't be counted
        start = cal.session_offset(last_session, -6)
        df = prices.get(start=start)
        assertions_daily(df, prices, start, last_session)

        # verify no parameters
        df = prices.get()
        assertions_daily(df, prices, limit_left, last_session)

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
        f("2h", hours=2, anchor="workback")
        with pytest.raises(errors.PricesUnavailableIntervalDurationError):
            f("2h", hours=1, minutes=59)

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
        kwargs = dict(start=session, days=1, anchor="workback")
        # also verify can pass interval as pd.Timedelta
        f(session_length_xnys, **kwargs)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f(session_length_xnys + one_min, **kwargs)

        # Verify raises PricesUnavailableIntervalPeriodError
        start = cal.session_close(session) - pd.Timedelta(2, "min")
        end = cal.session_open(cal.next_session(session)) + pd.Timedelta(2, "min")
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f("5min", start=start, end=end)

        start = cal.session_open(session) + prices.bis.T5
        end = start + pd.Timedelta(4, "min")
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            f("5min", start=start, end=end)

    def test_interval_only_param(self, prices_us, one_day, one_min):
        """Test passing interval as only parameter.

        Tests base intervals and intervals that require downsampling.
        """
        prices = prices_us
        f = prices.get

        cal = prices.calendar_default
        for bi in prices.bis_intraday:
            limit_left, limit_right = prices.limits[bi]
            first_from = cal.minute_to_trading_minute(limit_left, "next")
            first_to = cal.minute_offset_by_sessions(first_from, 1)
            last_to = cal.minute_to_trading_minute(limit_right, "previous")
            last_from = cal.minute_offset_by_sessions(last_to, -1)
            if bi is prices.bis.H1:
                last_to += pd.Timedelta(30, "min")  # provide for possibly unaligned end
            df = f(bi)
            assert first_from <= df.pt.first_ts <= first_to
            # + one_min to cover processing between evaluating last_to and evaluating df
            assert last_from <= df.pt.last_ts <= last_to + one_min
            assertions_intraday_common(df, prices, bi)
            # check for interval that requires downsampling
            for factor in (3, 7):
                interval = bi * factor
                last_to_ = last_to + (factor * bi) + one_min
                df = f(interval)
                assert first_from <= df.pt.first_ts <= first_to
                assert last_from <= df.pt.last_ts <= last_to_
                assertions_intraday_common(df, prices, interval)

        df_daily = f(prices.bi_daily)
        assert_daily(df_daily, prices)
        limit_left, limit_right = prices.limits[prices.bis.D1]
        last_session = cal.date_to_session(limit_right, "previous")
        if pd.Timestamp.now(tz=UTC) < cal.session_open(last_session):
            last_session = cal.previous_session(last_session)
        assert df_daily.pt.first_ts == limit_left
        assert df_daily.pt.last_ts == last_session

        df_mult_days = f("3D")
        assert_prices_table_ii(df_mult_days, prices)
        first_to = cal.session_offset(limit_left, 3)
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

    def test_intervals_overlapping_247(self, prices_247):
        """Test raises warning when downsampled intervals would overlap.

        Verifies raises `errors.IntervalIrregularWarning`.
        """
        prices = prices_247
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices.get("7min", days=2)
        assert_most_common_interval(df, intervals.TDInterval.T7)

    def test_intervals_overlapping_with_break(self, prices_with_break, one_min):
        """Test raises warning when downsampled intervals would overlap.

        Verifies raises `errors.IntervalIrregularWarning`.
        """
        prices = prices_with_break
        rng_start, rng_end = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session_length = pd.Timedelta(hours=6, minutes=30)
        sessions = th.get_conforming_cc_sessions(
            prices.cc, session_length, rng_start, rng_end, 4
        )
        i = 0
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
        session = th.get_conforming_cc_sessions(
            prices.cc, regular_session_length, *sessions_range, 1
        )[0]

        start = cal.session_open(session)
        start_utc = start
        start_local = sl = start.astimezone(prices.tz_default)
        start_str = sl.strftime("%Y-%m-%d %H:%M")
        start_str2 = sl.strftime("%Y-%m-%d %H")
        start_int = pd.Timestamp(start_str).value
        start_dt = datetime.datetime(sl.year, sl.month, sl.day, sl.hour, sl.minute)

        end = cal.session_close(session)
        end -= pd.Timedelta(30, "min")
        end_utc = end
        end_local = el = end.astimezone(prices.tz_default)
        end_str = el.strftime("%Y-%m-%d %H:%M")
        end_str2 = el.strftime("%Y-%m-%d %H:%M")
        end_int = pd.Timestamp(end_str).value
        end_dt = datetime.datetime(el.year, el.month, el.day, el.hour, el.minute)

        starts = (start_utc, start_local, start_str, start_str2, start_int, start_dt)
        ends = (end_utc, end_local, end_str, end_str2, end_int, end_dt)

        df_base = prices.get("2h", starts[0], ends[0])
        for start, end in zip(starts[1:], ends[1:]):
            df = prices.get("2h", start, end)
            assert_frame_equal(df, df_base)

        # Verify `tzin`
        symb_xnys = get_symbols_for_calendar(prices, "XNYS")
        for tzin in ("America/New_York", ZoneInfo("America/New_York"), symb_xnys):
            df = prices.get("2h", start_utc, end_str, tzin=tzin)
            assert_frame_equal(df, df_base)

        # verify can pass as non-default symbol
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        start_lon_tz = start_utc.astimezone(prices.timezones[symb_xlon])
        start_lon_str = start_lon_tz.strftime("%Y-%m-%d %H:%M")
        end_lon_tz = end_utc.astimezone(prices.timezones[symb_xlon])
        end_lon_str = end_lon_tz.strftime("%Y-%m-%d %H:%M")
        df = prices.get(
            "2h", start_lon_str, end_lon_str, tzin=symb_xlon, tzout=symb_xnys
        )
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
        df = prices.get("5min", start_session, end_session)
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
        df = prices.get("10min", start, end)
        assertions_intraday(df, TDInterval.T10, prices, start, end, num_rows // 2)
        df = prices.get("5min", start, end)
        assertions_intraday(df, TDInterval.T5, prices, start, end, num_rows)

        # verify passing `start` and `end` as non-trading times
        assert_frame_equal(df, prices.get("5min", start - one_min, end + one_min))
        delta = pd.Timedelta(45, "min")
        assert_frame_equal(df, prices.get("5min", start - delta, end + delta))

        # verify passing `start` and `end` 6 min inside session bounds knocks 2 indices
        # off each end
        delta = pd.Timedelta(6, "min")
        assert_frame_equal(df[2:-2], prices.get("5min", start + delta, end - delta))

        # verify just one second inside the session bounds will knock off the
        # first/last indices
        assert_frame_equal(df[1:-1], prices.get("5min", start + one_sec, end - one_sec))

        # Verify passing `start` and `end` as mix of data and time
        assert_frame_equal(df_daily, prices.get("1D", start_session, end))
        assert_frame_equal(df_daily, prices.get("1D", start, end_session))
        assert_frame_equal(df_daily[1:], prices.get("1D", start + one_sec, end_session))

        assert_frame_equal(df, prices.get("5min", start_session, end))
        assert_frame_equal(df[:-1], prices.get("5min", start_session, end - one_sec))
        assert_frame_equal(df, prices.get("5min", start, end_session))

    def test_start_end_none_bi(self, prices_us):
        """Test `start` and `end` as None and intervals as base intervals."""
        prices = prices_us
        cal = prices.calendar_default
        for bi in prices.bis_intraday:
            limit_left = prices.limits[bi][0]
            first_from = cal.minute_to_trading_minute(limit_left, "next")
            for ds_interval in (bi, bi * 3):
                df = prices.get(ds_interval, openend="shorten")
                assert first_from <= df.pt.first_ts <= first_from + ds_interval

    def test_start_end_none_ds(
        self, prices_us, session_length_xnys, one_sec, monkeypatch
    ):
        """Test `start` and `end` as None and intervals as ds intervals."""
        prices = prices_us
        cal = prices.calendar_default
        # define start and end as standard sessions for which intraday data available
        sessions_range = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        session = th.get_conforming_cc_sessions(
            prices.cc, session_length_xnys, *sessions_range, 2
        )[-1]
        now_session_open = cal.session_open(sessions_range[-1])
        for bi in prices.bis_intraday:
            for ds_interval in (bi, bi * 3):
                mock_now(monkeypatch, now_session_open + ds_interval + one_sec)
                df = prices.get(ds_interval, start=session, openend="shorten")
                assert df.pt.last_ts == now_session_open + (ds_interval * 2)

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
        num_rows = int((session_length_xnys * num_sessions) / pd.Timedelta(30, "min"))
        df = prices.get("30min", start=start_session, days=num_sessions)
        assertions_intraday(df, TDInterval.T30, prices, open_, close, num_rows)
        assert_frame_equal(df, prices.get("30min", end=end_session, days=num_sessions))

        # verify to bound as time
        # getting intraday data
        start = open_ + pd.Timedelta(88, "min")
        df_intra = prices.get("30min", start, days=num_sessions - 1)
        exp_start = open_ + pd.Timedelta(90, "min")
        exp_end = cal.session_open(end_session) + pd.Timedelta(90, "min")
        num_rows -= 13
        if exp_start.time() != exp_end.time():
            # adjust for different DST observance
            if exp_end.time() > exp_start.time():
                exp_end -= pd.Timedelta(1, "h")
                num_rows -= 2
            else:
                exp_end += pd.Timedelta(1, "h")
                num_rows += 2
        assertions_intraday(
            df_intra, TDInterval.T30, prices, exp_start, exp_end, num_rows
        )
        end = exp_end + one_min  # verify aligning as expected
        assert_frame_equal(
            df_intra, prices.get("30min", end=end, days=num_sessions - 1)
        )
        # getting daily data
        df = prices.get("1D", start, days=num_sessions - 1)
        assertions_daily(df, prices, sessions[1], end_session)
        df = prices.get("1D", end=end, days=num_sessions - 1)
        assertions_daily(df, prices, start_session, sessions[-2])

        # verify to now as trading time
        # one sec short of 30 min post last session open
        mock_now(monkeypatch, exp_end - one_sec)
        df = prices.get("30min", days=num_sessions - 1)
        assert_frame_equal(df_intra, df)

        # verify to now as non-trading time
        mock_now(monkeypatch, close + pd.Timedelta(2, "h"))
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
        while prices.tz_default.dst(sessions[0]) != prices.tz_default.dst(sessions[-1]):
            start_range += one_day
            sessions = th.get_conforming_cc_sessions(
                prices.cc, session_length_xnys, start_range, end_range, num_sessions
            )
        start_session = sessions[0]

        num_sessions_in_week = int(
            cal.sessions_distance(start_session, start_session + pd.Timedelta(6, "D"))
        )

        # verify getting intraday data with bound as date
        end_session = cal.session_offset(start_session, num_sessions_in_week - 1)
        df = prices.get("30min", start_session, weeks=1)
        # Verifications piggy backs on separate testing of durations in terms of
        # trading sessions
        expected = prices.get("30min", start_session, days=num_sessions_in_week)
        assert_frame_equal(df, expected)
        expected = prices.get("30min", end=end_session, days=num_sessions_in_week)
        assert_frame_equal(df, expected)

        # verify getting intraday data with bound as time
        # end_session will now be one session later
        end_session2 = cal.session_offset(start_session, num_sessions_in_week)
        open_ = cal.session_open(start_session)
        start = open_ + pd.Timedelta(28, "min")
        df = prices.get("30min", start, weeks=1)
        # Verification piggy backs on separate testing of durations in terms of
        # trading sessions
        expected = prices.get("30min", start, days=num_sessions_in_week)
        assert_frame_equal(df, expected)
        end = cal.session_open(end_session2) + pd.Timedelta(32, "min")
        df_intra = prices.get("30min", end=end, weeks=1)
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
        now = cal.session_open(end_session2) + pd.Timedelta(30, "min") - one_sec
        # one sec short of 30 min post last session open
        mock_now(monkeypatch, now - one_sec)
        df = prices.get("30min", weeks=1)
        assert_frame_equal(df_intra, df)

        # verify to now as non-trading time
        mock_now(monkeypatch, cal.session_open(end_session) + pd.Timedelta(2, "h"))
        assert_frame_equal(df_daily, prices.get("D", weeks=1), check_freq=False)

    def test_trading_time_duration(self, prices_us, monkeypatch):
        """Test for periods defined as duration in terms of trading time."""
        prices = prices_us
        cal = prices.calendar_default

        # Verify durations in trading terms
        prev_session, session = get_consecutive_sessions(prices, prices.bis.T1, cal)

        # verify bounds with a session
        df = prices.get("5min", session, minutes=20)
        open_, close = cal.session_open_close(session)
        assertions_intraday(
            df, TDInterval.T5, prices, open_, open_ + pd.Timedelta(20, "min"), 4
        )

        df = prices.get("15min", end=session, hours=1)
        assertions_intraday(
            df, TDInterval.T15, prices, close - pd.Timedelta(1, "h"), close, 4
        )

        # verify bounds with a time
        bound = open_ + pd.Timedelta(30, "min")
        df = prices.get("15min", bound, hours=1, minutes=15)
        delta = pd.Timedelta(hours=1, minutes=15)
        assertions_intraday(df, TDInterval.T15, prices, bound, bound + delta, 5)

        # verify crossing sessions
        df = prices.get("15min", end=bound, hours=1, minutes=15)
        prev_close = cal.session_close(prev_session)
        exp_start = prev_close - pd.Timedelta(45, "min")
        assertions_intraday(df, TDInterval.T15, prices, exp_start, bound, 5)

        # verify Silver Rule (if bound unaligned then start from next aligned indice)
        df = prices.get("25min", start=bound, hours=1, minutes=40)
        exp_start = bound + pd.Timedelta(20, "min")
        exp_end = exp_start + pd.Timedelta(hours=1, minutes=40)
        assertions_intraday(df, TDInterval.T25, prices, exp_start, exp_end, 4)

        # verify limit before next indice would be included
        assert_frame_equal(df, prices.get("25min", start=bound, hours=2, minutes=4))
        df = prices.get("25min", start=bound, hours=2, minutes=5)
        exp_end += pd.Timedelta(25, "min")
        assertions_intraday(df, TDInterval.T25, prices, exp_start, exp_end, 5)

        # verify default end with now as trading time
        now = open_ + pd.Timedelta(32, "min")
        mock_now(monkeypatch, now)
        df = prices.get("5min", minutes=20)
        exp_end = open_ + pd.Timedelta(35, "min")
        exp_start = exp_end - pd.Timedelta(20, "min")
        assertions_intraday(df, TDInterval.T5, prices, exp_start, exp_end, 4)

        # verify default end with now as non-trading time
        now = close + pd.Timedelta(2, "h")
        mock_now(monkeypatch, now)
        df = prices.get("2min", minutes=41)
        assertions_intraday(
            df, TDInterval.T2, prices, close - pd.Timedelta(40, "min"), close, 20
        )

    def test_lead_symbol(self, prices_us_lon, session_length_xnys, session_length_xlon):
        """Test effect of `lead_symbol`."""
        prices = prices_us_lon
        xnys = prices.calendar_default
        symb_xnys = prices.lead_symbol_default
        xlon = get_calendar_from_name(prices, "XLON")
        symb_xlon = get_symbols_for_calendar(prices, "XLON")

        # verify effect of lead_symbol on known differing expected daily returns
        kwargs = dict(interval="D", end="2021-12-28", days=3)
        df = prices.get(**kwargs, lead_symbol=symb_xnys)
        exp_start, exp_end = pd.Timestamp("2021-12-23"), pd.Timestamp("2021-12-28")
        assertions_daily(df, prices, exp_start, exp_end)

        df = prices.get(**kwargs, lead_symbol=symb_xlon)
        exp_start, exp_end = pd.Timestamp("2021-12-22"), pd.Timestamp("2021-12-24")
        assertions_daily(df, prices, exp_start, exp_end)

        # verify effect of lead_symbol on intraday data
        calendars = [xnys, xlon]
        session_length = [session_length_xnys, session_length_xlon]
        session = get_conforming_sessions(
            prices, prices.bis.T1, calendars, session_length, 1
        )[0]

        # start as one hour prior to XNYS open
        xnys_open = xnys.session_open(session)
        start = xnys_open - pd.Timedelta(1, "h")
        args, kwargs = ("6min", start), {"minutes": 30}
        # verify start rolls forward to XNYS open
        df = prices.get(*args, **kwargs, lead_symbol=symb_xnys)
        half_hour = pd.Timedelta(30, "min")
        assertions_intraday(
            df, TDInterval.T6, prices, xnys_open, xnys_open + half_hour, 5
        )
        # verify start as start given XLON is open
        df = prices.get(*args, **kwargs, lead_symbol=symb_xlon)
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
        _, session = get_consecutive_sessions(prices, prices.bis.T1, cal)
        start = cal.session_open(session)
        args = ("10min", start)
        kwargs = {"minutes": 30}
        base_df = prices.get(*args, **kwargs)
        exp_end = start + pd.Timedelta(30, "min")
        assertions_intraday(base_df, TDInterval.T10, prices, start, exp_end, 3)

        df = prices.get(*args, **kwargs, add_a_row=True)
        assert_frame_equal(df[1:], base_df, check_freq=False)
        exp_start = cal.previous_close(start) - pd.Timedelta(10, "min")
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

    def test_open_indices_us(self, prices_us, session_length_xnys):
        """Test indices as expected when `anchor` is "open".

        Tests at an interval aligned with session closes and an interval that is
        unaligned.

        Tests over two consecutive regular sessions.
        """
        prices = prices_us
        xnys = prices.calendar_default
        start_session, end_session = get_conforming_sessions(
            prices, prices.bis.T1, [xnys], [session_length_xnys], 2
        )
        df = prices.get("30min", start_session, end_session, anchor="open", tzout=UTC)
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
        df = prices.get("90min", start_session, end_session, anchor="open", tzout=UTC)
        assert df.pt.has_regular_interval
        assert not df.pt.indices_have_regular_trading_minutes(xnys)

        misalignment = pd.Timedelta(1, "h")
        start = xnys.session_open(start_session)
        end_start_session = xnys.session_close(start_session) + misalignment
        start_end_session = xnys.session_open(end_session)
        end = xnys.session_close(end_session) + misalignment
        starts = [start, start_end_session]
        ends = [end_start_session, end]
        interval = TDInterval.T90
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

    def test_open_indices_with_break(
        self, prices_with_break, session_length_xhkg, monkeypatch
    ):
        """Test indices as expected when `anchor` is "open".

        Tests symbol with break at an interval aligned with (sub)session
        closes and an interval that is unaligned.

        Tests over two consecutive regular sessions and, separately, to a mock now.
        """
        # Verify for calendar with break
        prices = prices_with_break
        xhkg = prices.calendar_default
        start_session, end_session = get_conforming_sessions(
            prices, prices.bis.T1, [xhkg], [session_length_xhkg], 2
        )

        df = prices.get("30min", start_session, end_session, anchor="open", tzout=UTC)
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
        df = prices.get("40min", start_session, end_session, anchor="open", tzout=UTC)
        assert df.pt.has_regular_interval
        assert not df.pt.indices_have_regular_trading_minutes(xhkg)

        starts, ends = [], []
        for session in (start_session, end_session):
            starts.append(xhkg.session_open(session))
            ends.append(xhkg.session_break_start(session) + pd.Timedelta(10, "min"))
            starts.append(xhkg.session_break_end(session))
            ends.append(xhkg.session_close(session) + pd.Timedelta(20, "min"))
        interval = TDInterval.T40
        exp_index = self.create_index(starts, ends, interval)
        assert_index_equal(df.index, exp_index)

        # # verify to now includes live indice and nothing beyond
        now = ends[-1] - pd.Timedelta(50, "min")
        mock_now(monkeypatch, now)
        # should lose last indice
        df_ = prices.get("40min", start_session, end_session, anchor="open", tzout=UTC)
        # last indice be the same although not values as now is 10 minutes short
        # of the full indice
        assert_frame_equal(df_[:-1], df[:-2])
        assert df_.index[-1] == df.index[-2]

    def test_workback_indices_us(
        self, prices_us, session_length_xnys, monkeypatch, one_min
    ):
        """Test indices as expected when `anchor` is "workback".

        Tests to session end and to minute of a session.
        """
        prices = prices_us
        xnys = prices.calendar_default
        start_session, end_session = get_conforming_sessions(
            prices, prices.bis.T1, [xnys], [session_length_xnys], 2
        )

        df = prices.get(
            "90min", start_session, end_session, anchor="workback", tzout=UTC
        )
        assert len(df.pt.indices_length) == 2
        assert len(df.pt.indices_partial_trading(xnys)) == 1
        assert df.pt.indices_have_regular_trading_minutes(xnys)

        interval = TDInterval.T90
        end = xnys.session_close(end_session)
        start = end - (interval * 4)
        index_end = self.create_single_index(start, end, interval)

        end = xnys.session_close(start_session) - pd.Timedelta(1, "h")
        start = end - (interval * 3)
        index_start = self.create_single_index(start, end, interval)

        indice_cross_sessions = pd.Interval(
            index_start[-1].right, index_end[0].left, closed="left"
        )
        index_middle = pd.IntervalIndex([indice_cross_sessions])

        index = index_start.union(index_middle).union(index_end)
        assert_index_equal(df.index, index)

        # Verify to specific minute
        end = xnys.session_close(start_session) - pd.Timedelta(43, "min")
        df = prices.get("30min", end=end, hours=4, anchor="workback", tzout=UTC)
        interval = TDInterval.T30
        start = end - (8 * interval)
        index = self.create_single_index(start, end, interval)
        assert_index_equal(df.index, index)

        # Verify to now
        mock_now(monkeypatch, end - one_min)
        prices = reset_prices(prices)
        df_ = prices.get("30min", hours=4, anchor="workback", tzout=UTC)
        assert_frame_equal(df_, df)

    def test_workback_indices_with_break(self, prices_with_break, session_length_xhkg):
        """Test indices as expected when `anchor` is "workback".

        Tests symbol with break.
        Tests to session end and to minute of a session.
        """
        prices = prices_with_break
        xhkg = prices.calendar_default
        session = get_conforming_sessions(
            prices, prices.bis.T1, [xhkg], [session_length_xhkg], 1
        )[0]

        df = prices.get("40min", session, session, anchor="workback", tzout=UTC)
        assert len(df.pt.indices_length) == 2
        assert len(df.pt.indices_partial_trading(xhkg)) == 1
        assert df.pt.indices_have_regular_trading_minutes(xhkg)

        interval = TDInterval.T40
        end = xhkg.session_close(session)
        start = end - (interval * 4)
        index_end = self.create_single_index(start, end, interval)

        end = xhkg.session_break_start(session) - pd.Timedelta(20, "min")
        start = end - (interval * 3)
        index_start = self.create_single_index(start, end, interval)

        indice_cross_sessions = pd.Interval(
            index_start[-1].right, index_end[0].left, closed="left"
        )
        index_middle = pd.IntervalIndex([indice_cross_sessions])

        index = index_start.union(index_middle).union(index_end)
        assert_index_equal(df.index, index)

    def test_mult_cal_indices(self, prices_us_lon):
        """Test indices as expected when evaluted against multiple calendars.

        Tests expected output over two sessions for symbols trading on two
        different calendars.
        Tests for indices anchored both on "open" and "workback".
        """
        prices = prices_us_lon
        xnys = prices.calendar_default
        xnys_symb = prices.lead_symbol_default
        xlon = get_calendar_from_name(prices, "XLON")
        xlon_symb = get_symbols_for_calendar(prices, "XLON")
        start_session, end_session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 2)

        kwargs = {
            "interval": "1h",
            "start": start_session,
            "end": end_session,
            "tzout": UTC,
        }
        interval = TDInterval.H1
        half_hour = pd.Timedelta(30, "min")

        # Verify for indices anchored "open" with xlon as lead cal
        df = prices.get(**kwargs, anchor="open", lead_symbol=xlon_symb)
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
        df = prices.get(**kwargs, anchor="open", lead_symbol=xnys_symb)
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
        df = prices.get(**kwargs, anchor="workback", lead_symbol=xlon_symb)

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
        df = prices.get(**kwargs, anchor="workback", lead_symbol=xnys_symb)

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

    def test_force(self, prices_with_break, session_length_xhkg):
        """Test effect of `force`.

        Testing limited to ensuring option results in force being
        implemented. Does not test the actual implementation of force.
        """
        prices = prices_with_break
        xhkg = prices.calendar_default
        session = session = get_conforming_sessions(
            prices, prices.bis.T1, [xhkg], [session_length_xhkg], 1
        )[0]

        # verify not forced
        df = prices.get("40min", session, session, anchor="open", force=False)
        assert df.pt.has_regular_interval
        assert xhkg.session_break_start(session) not in df.index.right
        assert df.pt.last_ts != xhkg.session_close(session)
        assert not df.pt.indices_all_trading(xhkg)
        assert len(df.pt.indices_partial_trading(xhkg)) == 2

        # verify not forced by default
        df_ = prices.get("40min", session, session, anchor="open")
        assert_frame_equal(df_, df)

        # verify forced
        df_f = prices.get("40min", session, session, anchor="open", force=True)
        assert not df_f.pt.has_regular_interval
        assert xhkg.session_break_start(session) in df_f.index.right
        assert df_f.pt.last_ts == xhkg.session_close(session)
        assert df_f.pt.indices_all_trading(xhkg)
        assert df_f.pt.indices_partial_trading(xhkg).empty

    def test_overlapping_sessions(self, prices_247):
        """Test raises warning when intervals overlap.

        Tests raises `IntervalIrregularWarning` when last indice of session
        overlaps with first indice of next session.
        """
        prices = prices_247
        x247 = prices.calendar_default
        start_session, end_session = get_consecutive_sessions(
            prices, prices.bis.T1, x247
        )

        # verify does not overlap on limit
        start_session_close = x247.session_close(start_session)
        end_session_open = x247.session_open(end_session)
        assert start_session_close == end_session_open
        df = prices.get("8h", start_session, end_session, anchor="open")
        assert df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 1
        assert start_session_close in df.index.left
        assert start_session_close in df.index.right

        # verify warns
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices.get("7h", start_session, end_session, anchor="open")
        assert not df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 2
        assert start_session_close in df.index.left
        assert start_session_close in df.index.right

    def test_overlapping_subsessions(self, prices_with_break, session_length_xhkg):
        """Test raises warning when intervals overlap.

        Tests raises IntervalIrregularWarning when last indice of a morning
        subsession overlaps with first indice of afternoon subsession.
        """
        prices = prices_with_break
        xhkg = prices.calendar_default
        session = session = get_conforming_sessions(
            prices, prices.bis.T1, [xhkg], [session_length_xhkg], 1
        )[0]

        # verify am/pm indices do not overlap on limit
        df = prices.get("105min", session, session, anchor="open")
        assert df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 1
        assert xhkg.session_break_end(session) in df.index.left
        assert xhkg.session_break_end(session) in df.index.right

        # verify warns
        with pytest.warns(errors.IntervalIrregularWarning):
            df = prices.get("106min", session, session, anchor="open")
        assert not df.pt.has_regular_interval
        assert len(df.pt.indices_length) == 2
        assert xhkg.session_break_end(session) in df.index.left
        assert xhkg.session_break_end(session) in df.index.right

    def test_cal_break_ignored_with_origin_open(
        self, prices_with_break, session_length_xhkg
    ):
        """Test a circumstance when indices not aligned to pm open.

        Tests when source does not anchor prices on afternoon open (but
        rather the session open) and data not available at a lower interval
        from which indices respecting break can be evaluated.
        """
        prices = prices_with_break
        assert prices.PM_SUBSESSION_ORIGIN == "open"
        xhkg = prices.calendar_default
        interval = prices.bis.H1
        session = get_conforming_sessions(
            prices, interval, [xhkg], [session_length_xhkg], 1
        )[0]

        df = prices.get("1h", session, session, anchor="open", tzout=UTC)
        assert df.pt.has_regular_interval
        assert (df.index.left.minute == 30).all()
        assert (df.index.right.minute == 30).all()
        start = xhkg.session_open(session)
        end = xhkg.session_close(session) + pd.Timedelta(30, "min")
        index = self.create_single_index(start, end, interval)
        assert_index_equal(df.index, index)

    def test_cal_break_ignored_with_partial_overlap(self, prices_hk_lon):
        """Test  circumstance when indices not aligned to pm open.

        Tests when indice of a trading index evaluated against the calendar
        that observes the break *partially* overlaps an indice of a trading
        index evaluated against another calendar.
        """
        prices = prices_hk_lon
        xhkg = prices.calendar_default
        start_session, end_session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 2)
        interval = TDInterval.T50

        df = prices.get("50min", start_session, end_session, anchor="open", tzout=UTC)
        df = df[:9]  # only take what's necessary to prove the point
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

    def test_openend_no_trading_after(self, prices_us, session_length_xnys):
        """Test effect of "openend" option.

        Tests "maintain" and "shorten" where no symbol trades after an
        unaligned session close.
        """
        prices = prices_us
        xnys = prices.calendar_default
        session = get_conforming_sessions(
            prices, prices.bis.T5, [xnys], [session_length_xnys], 1
        )[0]
        start, end = xnys.session_open_close(session)

        # verify maintain when no symbol trades after unaligned close
        df = prices.get("1h", start, end, openend="maintain")
        assert df.pt.has_regular_interval
        half_hour = pd.Timedelta(30, "min")
        exp_end = end + half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 7)

        # verify maintain is default
        df_ = prices.get("1h", start, end)
        assert_frame_equal(df, df_)

        # verify shorten
        df = prices.get("1h", start, end, openend="shorten")
        assert not df.pt.has_regular_interval
        last_indice = df.index[-1]
        assert last_indice.right == end
        assert last_indice.left == end - half_hour
        assert last_indice.length == half_hour
        # verify rest of table as expected
        exp_end = end - half_hour
        assertions_intraday(df[:-1], prices.bis.H1, prices, start, exp_end, 6)

    def test_openend_trading_after(self, prices_lon_247, session_length_xlon):
        """Test effect of "openend" option.

        Tests "maintain" and "shorten" where a symbol trades after an
        unaligned session close.
        Tests "shorten" acts as "maintain" if data unavailable to evalute
        shorter final indice.
        """
        # verify maintain when no symbol trades after unaligned close
        prices = prices_lon_247
        xlon = prices.calendar_default
        session = get_conforming_sessions(
            prices, prices.bis.T5, [xlon], [session_length_xlon], 1
        )[0]
        start, end = xlon.session_open_close(session)

        # verify maintain when symbol trades after unaligned close
        df = prices.get("1h", start, end, openend="maintain")
        assert df.pt.has_regular_interval
        half_hour = pd.Timedelta(30, "min")
        exp_end = end - half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 8)

        # verify shorten
        df = prices.get("1h", start, end, openend="shorten")
        assert not df.pt.has_regular_interval
        last_indice = df.index[-1]
        assert last_indice.right == end
        assert last_indice.left == end - half_hour
        assert last_indice.length == half_hour
        # verify rest of table as expected
        exp_end = end - half_hour
        assertions_intraday(df[:-1], prices.bis.H1, prices, start, exp_end, 8)

        # verify shorten as maintain when data not available to crete short indice
        session = get_conforming_sessions(
            prices, prices.bis.H1, [xlon], [session_length_xlon], 1
        )[0]
        start, end = xlon.session_open_close(session)
        df = prices.get("1h", start, end, openend="shorten")
        assert df.pt.has_regular_interval
        exp_end = end - half_hour
        assertions_intraday(df, prices.bis.H1, prices, start, exp_end, 8)

    @staticmethod
    def assertions_test_daily(
        prices: PricesBaseTst,
        lead: str,
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        exp_end: pd.Timestamp,
    ):
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
                    end = cal.date_to_session(end, "next")
                assert start == helpers.to_tz_naive(cal.session_offset(end, -interval))

            # verify sessions missing to left of table start are fewer in number
            # than one interval
            excess = len(cal.sessions_in_range(period_start, start - helpers.ONE_DAY))
            assert excess < interval

            # verify indices contiguous
            assert_index_equal(df.index.left[1:], df.index.right[:-1])

    def test_daily_single_cal(self, prices_us, one_day):
        """Test indices where interval is a multiple of days.

        Tests for symbols associated with single and multiple calendars.
        """
        prices = prices_us
        xnys = prices.calendar_default
        symb = prices.lead_symbol_default
        period_start = pd.Timestamp("2021-12-01")
        period_end = pd.Timestamp("2021-12-31")
        assert xnys.is_session(period_end)
        exp_end = period_end + one_day
        self.assertions_test_daily(prices, symb, period_start, period_end, exp_end)

    def test_daily_mult_cal(self, prices_lon_247, one_day):
        """Test indices where interval is a multiple of days.

        Tests for symbols associated with multiple calendars.
        """
        prices = prices_lon_247
        xlon = prices.calendar_default
        symb = prices.lead_symbol_default
        period_start = pd.Timestamp("2021-12-01")
        period_end = pd.Timestamp("2021-12-31")
        assert xlon.is_session(period_end)
        exp_end = period_end + one_day
        self.assertions_test_daily(prices, symb, period_start, period_end, exp_end)
        symb_247 = get_symbols_for_calendar(prices, "24/7")
        self.assertions_test_daily(prices, symb_247, period_start, period_end, exp_end)

    def assertions_test_monthly(self, prices, monkeypatch):
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
            now = pd.Timestamp("2021-12-22 13:22", tz=UTC)
            mock_now(m, now)
            df_now = prices.get("2M", months=7)
            assert_index_equal(df_now.index, df.index)

    def test_monthly_single_cal(self, prices_us, monkeypatch):
        """Test indices where interval is a multiple of months.

        Tests for symbols associated with a single calendar.
        """
        self.assertions_test_monthly(prices_us, monkeypatch)

    def test_monthly_mult_cal(self, prices_us_lon, monkeypatch):
        """Test indices where interval is a multiple of months.

        Tests for symbols associated with multiple calendars.
        """
        self.assertions_test_monthly(prices_us_lon, monkeypatch)

    def test_raises_PricesUnavailableIntervalPeriodError(self, prices_us, one_sec):
        """Test raises `errors.PricesUnavailableIntervalPeriodError`.

        Tests raises when expected, with IntradayPricesUnavailable expected
        to take precendence.

        Tests when `start` and `end` represent period less than one interval.

        Tests 'open' and 'workback' anchors independently.
        """
        prices = prices_us
        cal = prices.calendar_default

        start_session_T5, _ = th.get_sessions_range_for_bi(prices, prices.bis.T5)
        anchors = ("open", "workback")

        prev_session = start_session_T5
        session = cal.next_session(prev_session)
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
        delta = pd.Timedelta(2, "min")
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
        start_session_T1, _ = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        prev_session = start_session_T1
        session = cal.next_session(prev_session)
        session_open = cal.session_open(session)
        prev_session_close = cal.session_close(prev_session)

        delta = pd.Timedelta(2, "min")
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

        session = get_conforming_sessions(
            prices, prices.bis.T5, [cal], [session_length_xnys], 1
        )[0]
        session_open = cal.session_open(session)

        prev_session_T1, session_T1 = get_conforming_sessions(
            prices, prices.bis.T1, [cal], [session_length_xnys], 2
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

        delta = pd.Timedelta(2, "min")
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
        assert_frame_equal(df_, df)

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
        end = l_limit + pd.Timedelta(2, "h")
        # error depends on strict
        get_kwargs = dict(interval="3h", end=end, hours=6, anchor="workback")
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(**get_kwargs, strict=True)
        with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
            prices.get(**get_kwargs, strict=False)

    # ------------------- Tests related to data availability ------------------

    def test_raises_PricesIntradayUnavailableError(self, prices_us, one_min):
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
                        "3min",
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
                prices.get("3min", session_T5, priority=priority)

        # although returns data from limit if strict False
        df = prices.get("3min", session_T5, strict=False)
        assert df.pt.first_ts >= prices.limits[prices.bis.T1][0] - one_min
        assert df.pt.interval == TDInterval.T3

    def test_raises_LastIndiceInaccurateError(self, prices_us):
        prices = prices_us
        xnys = prices.calendar_default

        start_T1, end_T1 = th.get_sessions_range_for_bi(prices, prices.bis.T1)
        start_T5, _ = th.get_sessions_range_for_bi(prices, prices.bis.T5)
        start_H1, _ = th.get_sessions_range_for_bi(prices, prices.bis.H1)

        start_T5_oob = xnys.session_offset(start_T5, -2)
        start_H1_oob = xnys.session_offset(start_H1, -2)

        limit_T1 = prices.limits[prices.bis.T1][0]
        limit_T5 = prices.limits[prices.bis.T5][0]
        limit_H1 = prices.limits[prices.bis.H1][0]

        # period end that can only be represented with T1 or T5 data
        end = xnys.session_close(end_T1) - pd.Timedelta(15, "min")

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
        end = xnys.session_close(start_T5) - pd.Timedelta(3, "min")
        df = prices.get(start=start_T5, end=end)
        assert df.pt.interval == prices.bis.T5
        assert df.pt.last_ts == end - pd.Timedelta(2, "min")

        # verify error not raised when interval passed and anchor "open",
        # regardless of final indice not representing period end
        df_ = prices.get("5min", start=start_T5, end=end)
        assert_frame_equal(df, df_)

        # verify that raises errors when anchor="workback"
        # set period such that T1 data only available over period end and
        # period end can only be served with T1 data.
        end = xnys.session_close(start_T1) - pd.Timedelta(3, "min")
        # whilst prices available when anchor "open"
        df = prices.get("10min", start=start_T5, end=end, anchor="open")
        assert df.pt.interval == TDInterval.T10
        # verify not when anchor is "workback"
        with pytest.raises(errors.LastIndiceInaccurateError):
            prices.get("10min", start=start_T5, end=end, anchor="workback")

        # verify will return later part of period if strict False
        df = prices.get(
            "10min", start=start_T5, end=end, anchor="workback", strict=False
        )
        assert df.pt.last_ts == end
        assert df.pt.first_ts >= limit_T1
        assert df.index[0].length == TDInterval.T10
        assert df.index[-1].length == TDInterval.T10

        # verify will return full period if priority "period", althrough with
        # lesser end accuracy
        df = prices.get(
            "10min", start=start_T5, end=end, anchor="workback", priority="period"
        )
        assert df.pt.last_ts < end
        assert df.pt.first_ts < limit_T1
        assert df.index[0].length == TDInterval.T10
        assert df.index[-1].length == TDInterval.T10

    def test_raises_direct_errors(self, prices_us):
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
        start_H1_oob = xnys.session_offset(start_H1, -2)
        start_T5 = th.get_sessions_range_for_bi(prices, prices.bis.T5)[0]

        end = xnys.session_close(start_T5) - pd.Timedelta(3, "min")
        end = end.astimezone(prices.tz_default)

        with pytest.raises(errors.LastIndiceInaccurateError):
            prices.get(start=start_H1_oob, end=end)

        # now prices.gpp available...match message
        bi = prices.bis.T5
        bi_limit_start, bi_limit_end = prices.limits[bi]
        earliest_minute = xnys.minute_to_trading_minute(bi_limit_start, "next")
        latest_minute = xnys.minute_to_trading_minute(bi_limit_end, "previous")
        available_period = (earliest_minute, latest_minute)

        drg = prices.gpp.drg_intraday_no_limit
        drg.interval = bi

        msg = re.escape(
            "Full period not available at any synchronised intraday base interval."
            " The following base intervals could represent the end indice with the"
            " greatest possible accuracy although have insufficient data available"
            f" to cover the full period:\n\t{[bi]}.\nThe period over which data is"
            f" available at {bi} is {available_period}, although"
            " at this base interval the requested period evaluates to"
            f" {drg.daterange[0]}."
            f"\nPeriod evaluated from parameters: {prices.gpp.pp_raw}."
            "\nData that can express the period end with the greatest possible accuracy"
            f" is available from {helpers.fts(earliest_minute)} through to the end of"
            " the requested period. Consider passing `strict` as False to return prices"
            " for this part of the period."
            f"\nAlternatively, consider creating a composite table (pass `composite`"
            " as True) or passing `priority` as 'period'."
        )
        with pytest.raises(errors.LastIndiceInaccurateError, match=msg):
            prices.get(start=start_H1_oob, end=end)

        # Verify raises `PricesUnavailableInferredIntervalError` when interval
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
        with pytest.raises(errors.PricesUnavailableInferredIntervalError, match=msg):
            prices.get(start=start, end=start_H1_oob)

    def test_raises_daily_only_direct_errors(self, prices_us_daily):
        """Test get() directly raises expected errors when no intraday interval.

        Tests expected errors that are raised directly by get() when the
        no intraday base interval is defined.
        """
        prices = prices_us_daily
        limit = prices.limit_daily
        match = re.escape(
            "Intraday prices unavailable as prices class does not have any"
            " intraday base intervals defined."
        )
        with pytest.raises(errors.PricesIntradayIntervalError, match=match):
            prices_us_daily.get("5min", limit, days=2)

    def test_raises_intraday_only_direct_errors(self, prices_us, prices_us_intraday):
        """Test get() directly raises expected errors when no daily interval.

        Tests expected errors that are raised directly by get() when the
        daily base interval is not defined.
        """
        prices = prices_us_intraday
        xnys = prices.calendar_default
        start_T5 = th.get_sessions_range_for_bi(prices, prices.bis.T5)[0]
        match = re.escape(
            "Daily and monthly prices unavailable as prices class does not"
            " have a daily base interval defined."
        )
        with pytest.raises(errors.PricesDailyIntervalError, match=match):
            prices.get("1D", start_T5, days=10)

        start = th.get_sessions_range_for_bi(prices, prices.bis.H1)[0]

        # test raises when cannot fulfill inferred intraday interval
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(end=start, days=4, strict=True, priority="period")
        # verify that if daily interval available then will advise of such
        with pytest.raises(errors.PricesUnavailableInferredIntervalError):
            prices_us.get(end=start, days=4, strict=True, priority="period")

        # test raises when cannot fulfill inferred daily interval as no daily data
        with pytest.raises(errors.PricesDailyIntervalError, match=match):
            prices.get(end=start, days=10, strict=True, priority="period")
        # verify that if daily interval available then will return
        df = prices_us.get(end=start, days=10, strict=True, priority="period")
        assert df.pt.interval == prices_us.bis.D1

        # test raises when cannot create composite as no daily data
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(end=start, days=4, composite=True)
        # verify that if daily interval available then will return composite table
        df = prices_us.get(end=start, days=4, composite=True)
        assert df.index[0].left == df.index[0].right
        assert xnys.is_session(df.index[0].left.tz_convert(None))
        assert df.index[-1].length == prices.bis.H1

    # -------------- Tests related to post-processing options -------------

    def test_tzout(self, prices_us_lon_hk):
        """Test `tzout` option.

        Tests for expected default behaviour and overriding default.
        """
        prices = prices_us_lon_hk
        session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 1)[0]
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        tzhkg = prices.timezones[symb_xhkg]
        tzny = prices.timezones[symb_xnys]
        tzlon = prices.timezones[symb_xlon]

        kwargs_daily = dict(interval="1D", end=session, days=10)
        # verify tzout for daily data returns as tz-naive
        assert prices.get(**kwargs_daily).index.tz is None
        assert prices.get(**kwargs_daily, tzout=tzhkg).index.tz is None
        # unless tz is UTC
        assert prices.get(**kwargs_daily, tzout=UTC).index.tz is UTC

        # verify `tzout` defaults to timezone that `tzin` evaluates to
        kwargs_intraday = dict(end=session, days=2)
        df = prices.get(**kwargs_intraday)
        assert df.index.left.tz == df.index.right.tz == df.pt.tz == tzny
        assert prices.get(**kwargs_intraday, lead_symbol=symb_xhkg).pt.tz == tzhkg
        # verify tzin overrides `lead_symbol`
        assert (
            prices.get(**kwargs_intraday, lead_symbol=symb_xhkg, tzin=symb_xlon).pt.tz
            == tzlon
        )

        # verify passing `tzout` overrides default
        kwargs_intraday.update({"lead_symbol": symb_xnys, "tzin": symb_xlon})
        assert prices.get(**kwargs_intraday, tzout=symb_xhkg).pt.tz == tzhkg
        assert prices.get(**kwargs_intraday, tzout="Asia/Hong_Kong").pt.tz == tzhkg
        assert prices.get(**kwargs_intraday, tzout=tzhkg).pt.tz == tzhkg

    def test_post_processing_params(self, prices_us_lon_hk):
        """Test post-processing params implemented via .pt.operate."""
        prices = prices_us_lon_hk
        session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 1)[0]
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        kwargs = dict(end=session, days=2)
        df = prices.get(**kwargs)

        def assertion(**options):
            rtrn = prices.get(**kwargs, **options)
            assert_frame_equal(rtrn, df.pt.operate(**options))

        # assert last row has a missing value (which should be the case as
        # lead is xnys and xnys and xhkg do not overlap).
        assert df.iloc[-1].isna().any()
        assertion(fill="ffill")
        assertion(include=symb_xnys)
        assertion(exclude=[symb_xnys, symb_xlon])
        assertion(side="right")
        assertion(close_only=True)
        # verify for passing multiple options
        assertion(close_only=True, fill="bfill", side="left", exclude=symb_xhkg)

    def test_post_processing_params_lose_single_symbol(self, prices_us):
        """Test lose_single_symbol param implemented via .pt.operate."""
        prices = prices_us
        session, _ = th.get_sessions_range_for_bi(prices, prices.bis.T1)

        kwargs = dict(end=session, days=2)
        df = prices.get(**kwargs)
        rtrn = prices.get(**kwargs, lose_single_symbol=True)
        assert_frame_equal(rtrn, df.pt.operate(lose_single_symbol=True))


def test_request_all_prices(prices_us_lon):
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
        assert earliest_minute <= first_ts <= earliest_minute + bi

        last_ts = table.pt.last_ts
        if xnys.is_trading_minute(now):
            latest_minute = now
        else:
            latest_minute = xnys.previous_close(now)
        assert latest_minute - bi <= last_ts <= latest_minute + bi

    bi = prices.bis.D1
    table = prices._pdata[bi]._table

    first_ts = table.pt.first_ts
    limit = limits[bi][0]
    assert limit == first_ts

    last_ts = table.pt.last_ts
    latest_session = get_current_session(xnys)
    assert last_ts == latest_session


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
        column: Literal["open", "close"],
    ) -> float:
        return table.loc[session, (s, column)]

    def assertions(
        self,
        table: pd.DataFrame,
        df: pd.DataFrame,
        indice: pd.Timestamp,
        values: dict[str, tuple[pd.Timestamp, Literal["open", "close"]]],
        tz: ZoneInfo = UTC,
    ):
        self.assert_price_at_rtrn_format(table, df)
        assert df.index[0] == indice
        assert df.index.tz is tz
        for s, (session, col) in values.items():
            assert df[s].iloc[0] == self.get_cell(table, s, session, col)

    def test_oob(self, prices_us_lon_hk, one_min):
        """Test raises errors when minute out-of-bounds.

        Also tests does return at limits.
        """
        prices = prices_us_lon_hk
        # verify raises error if `minute` oob
        limit_left_session = prices.limits[prices.bis.D1][0]
        limit_left = max(
            c.session_close(limit_left_session) for c in prices.calendars_unique
        )
        df_left_limit = prices.price_at(limit_left)  # at limit
        assert df_left_limit.index[0] == limit_left

        with pytest.raises(errors.PriceAtUnavailableLimitError):
            prices.price_at(limit_left - one_min)

        oob = prices.earliest_requestable_minute - one_min
        with pytest.raises(errors.DatetimeTooEarlyError):
            prices.price_at(oob)

        limit_right = now = helpers.now()
        df_limit_right = prices.price_at(limit_right, UTC)  # at limit
        current_minute = prices.cc.minute_to_trading_minute(now, "previous")
        if not prices.cc.is_open_on_minute(now):
            current_minute += one_min
        assert df_limit_right.index[0] == current_minute
        with pytest.raises(errors.DatetimeTooLateError):
            limit_right = now = helpers.now()  # reset
            prices.price_at(limit_right + one_min)

    def test__price_at_from_daily(self, prices_us_lon_hk, one_min, monkeypatch):
        prices = prices_us_lon_hk

        xnys = prices.calendar_default
        xlon = get_calendar_from_name(prices, "XLON")
        xhkg = get_calendar_from_name(prices, "XHKG")
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        table = prices.get("1D", "2021-12-22", "2021-12-29")

        prices = reset_prices(prices)
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
            symb_xnys: (prev_session, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg.session_open(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (prev_session, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (session, "open"),
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
            symb_xnys: (prev_session, "close"),
            symb_xlon: (session, "open"),
            symb_xhkg: (session, "close"),
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
        # unchanged as from daily data unable to get value for AZN.L open as at xnys open
        indice = indice
        values = values
        self.assertions(table, df, indice, values)

        # prior to xnys close
        minute = xnys.session_close(session) - one_min
        df = f(minute, UTC)
        indice = indice  # as reason for prior
        values = values  # as reason for prior
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys.session_close(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (session, "close"),
            symb_xlon: (session, "close"),
            symb_xhkg: (session, "close"),
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
            symb_xnys: (prev_session_msft, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys.session_open(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (session, "open"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys.session_close(session)
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (session, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # verify for a time on 25 when all calendars closed
        # also verify changing tz for other value
        minute = pd.Timestamp("2021-12-25 13:00", tz=UTC)
        tz = prices.tz_default
        df = f(minute, tz)
        indice = xlon.session_close(prev_session)
        values = {
            symb_xnys: (prev_session_msft, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values, tz)

        # verify for minute as None. Repeat example for 27
        prices = reset_prices(prices)
        f = prices._price_at_from_daily

        # now prior to xnys open
        now = xnys.session_open(session) - one_min
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = xlon.session_close(prev_session)
        values = {
            symb_xnys: (prev_session_msft, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # now as xnys open
        now = xnys.session_open(session)
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            symb_xnys: (session, "close"),  # close as latest price
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # now as after xnys open, verify indice reflects 'now' as live session
        now = xnys.session_open(session) + pd.Timedelta(22, "min")
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            symb_xnys: (session, "close"),  # close as latest price
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        now = xnys.session_close(session)
        mock_now(monkeypatch, now)
        df = f(None, UTC)
        indice = now
        values = {
            symb_xnys: (session, "close"),
            symb_xlon: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }
        self.assertions(table, df, indice, values)

    def test_timezone(self, prices_us_hk, one_min):
        """Test tz parameter processed as mptypes.PricesTimezone."""
        prices = prices_us_hk
        f = prices.price_at
        xnys = prices.calendar_default
        xhkg = get_calendar_from_name(prices, "XHKG")

        symb_xnys = prices.lead_symbol_default
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")
        table = prices.get("1D", "2019-12-11", "2019-12-12")

        # following assertions from knowledge of schedule
        # two consecutive sessions (of both calendars) that can only be
        # served with daily data
        session = pd.Timestamp("2019-12-12")
        prev_session = pd.Timestamp("2019-12-11")

        minute = xhkg.session_open(session) - one_min
        indice = xnys.session_close(prev_session)
        values = {
            symb_xnys: (prev_session, "close"),
            symb_xhkg: (prev_session, "close"),
        }

        # verify tz be default is tz associated with lead symbol
        df = f(minute)
        self.assertions(table, df, indice, values, prices.tz_default)

        df = f(minute, tz=UTC)
        self.assertions(table, df, indice, values, UTC)

        df = f(minute, tz=symb_xhkg)
        self.assertions(table, df, indice, values, xhkg.tz)

        df = f(minute, tz=symb_xnys)
        self.assertions(table, df, indice, values, xnys.tz)

        df = f(minute, tz="Europe/London")
        self.assertions(table, df, indice, values, ZoneInfo("Europe/London"))

        # verify tz also defines tz of a timezone naive minute
        minute = minute.astimezone(None) + xhkg.tz.utcoffset(session)
        # assert should return different values if minute treated as UTC
        assert minute.tz_localize(UTC) > xhkg.session_close(session)

        df = f(minute, tz=xhkg.tz)
        # although verify same return as above given that tz-naive minute
        # treated as having tz as `tz`
        self.assertions(table, df, indice, values, xhkg.tz)

    def test_daily(self, prices_us_lon_hk, monkeypatch):
        """Test returns prices from daily as expected.

        Tests returns from daily when now

        test_single_symbol_T1_and_now
        """
        prices = prices_us_lon_hk
        xnys = prices.calendar_default
        xlon = get_calendar_from_name(prices, "XLON")
        xhkg = get_calendar_from_name(prices, "XHKG")

        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        # verify minute after intraday limit returns via intraday data
        limit_id = prices.limit_intraday()
        minute = limit_id + pd.Timedelta(1, "h")
        df = prices.price_at(minute)
        assert df.notna().all(axis=None)
        assert prices._pdata[prices.bis.D1]._table is None
        # only required by assert method for symbols...
        table_ = prices._pdata[prices.bis.T5]._table
        self.assert_price_at_rtrn_format(table_, df)

        # verify minute prior to intraday limit returns via _price_at_from_daily
        minute = xnys.previous_close(limit_id) - pd.Timedelta(1, "h")
        session_xnys = helpers.to_tz_naive(xnys.minute_to_session(minute))
        session_xhkg = helpers.to_tz_naive(xhkg.minute_to_session(minute, "previous"))
        session_xlon = helpers.to_tz_naive(xlon.minute_to_session(minute, "previous"))
        table = prices.get("1D", end=session_xnys, days=10)

        prices = reset_prices(prices)
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
            symb_xnys: (session_xnys, "open"),
            symb_xlon: (session_xlon, "close"),
            symb_xhkg: (session_xhkg, "close"),
        }
        self.assertions(table, df, indice, values)

        # verify None served via `_prices_at_from_daily` (use data frm previous example)
        mock_now(monkeypatch, minute)
        df = f(None, UTC)
        self.assert_price_at_rtrn_format(table, df)
        indice = minute
        values = {
            symb_xnys: (session_xnys, "close"),
            symb_xlon: (session_xlon, "close"),
            symb_xhkg: (session_xhkg, "close"),
        }
        self.assertions(table, df, indice, values)
        assert df.index[0] == minute  # as cal open, minute will be 'now'

    def test_single_symbol_T1_and_now(
        self, prices_us, session_length_xnys, monkeypatch, one_min
    ):
        """Tests single symbol with for data available at all bi.

        Tests both sides of edges of notable changes in expected values.

        Also tests that `now` and within min_delay of now returns daily
        close, whilst immediately prior to min_delay returns intraday data.
        """
        prices = prices_us
        xnys = prices.calendar_default
        symb = prices.lead_symbol_default

        delta = pd.Timedelta(33, "min")

        bi = prices.bis.T1
        session_prev, session = get_conforming_sessions(
            prices, bi, [xnys], [session_length_xnys], 2
        )
        session_before = prices.cc.previous_session(session_prev)

        prev_close = xnys.session_close(session_prev)
        open_ = xnys.session_open(session)
        close = xnys.session_close(session)
        open_next = xnys.session_open(xnys.next_session(session))

        table = prices.get("1min", session_before, session, tzout=UTC)
        tableD1 = prices.get("1D", session_before, session)

        delay = 20
        prices = reset_prices(prices)
        prices._delays[symb] = pd.Timedelta(delay, "min")  # mock delay
        f = prices.price_at

        # prior to xlon open
        minute = open_ - one_min
        df = f(minute, UTC)
        indice = prev_close
        values = {symb: (prev_close - bi, "close")}
        self.assertions(table, df, indice, values)

        # xlon open to before close
        for minute in (open_, open_ + one_min, open_ + delta, close - one_min):
            df = f(minute, UTC)
            values = {symb: (minute, "open")}
            self.assertions(table, df, minute, values)

        # xlon close
        minute = close
        df = f(minute, UTC)
        indice = minute
        values = {symb: (close - bi, "close")}
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
            values = {symb: (close - bi, "close")}
            self.assertions(table, df, close, values)

        # verify that from (now - min_delay) through now returns day close
        now = close - delta
        mock_now(monkeypatch, now)

        for minute in (None, now):
            df = f(minute, UTC)
            indice = now
            values = {symb: (session, "close")}
            self.assertions(tableD1, df, indice, values)

        for i in range(delay):
            minute = now - pd.Timedelta(i, "min")
            df = f(minute, UTC)
            indice = minute  # indices as requested minute
            values = {symb: (session, "close")}
            self.assertions(tableD1, df, indice, values)

        # verify that on delay limit prices return for intraday data
        minute = now - pd.Timedelta(delay, "min")
        df = f(minute, UTC)
        indice = minute
        values = {symb: (minute, "open")}
        self.assertions(table, df, indice, values)

    def test_when_all_data_available(self, prices_us_lon_hk, one_min):
        """Test for session over which data available at all bi."""
        prices = prices_us_lon_hk
        xnys = prices.calendar_default
        xlon = get_calendar_from_name(prices, "XLON")
        xhkg = get_calendar_from_name(prices, "XHKG")
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        bi = prices.bis.T1
        delta = pd.Timedelta(43, "min")

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

        table = prices.get("1min", session_before, session_after, tzout=UTC)

        prices = reset_prices(prices)
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_prev_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg am subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg break
        minute += delta
        df = f(minute, UTC)
        indice = xhkg_break_start  # as during break not a trading minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break end
        minute = xhkg_break_end
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_end, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg pm subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # prior to xlon open
        minute = xlon_open - one_min
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (minute, "open"),
        }
        self.assertions(table, df, indice, values)

        # xlon open (possible xhkg close)
        minute = xlon_open
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (minute, "open"),
            symb_xhkg: ali_value,
        }
        self.assertions(table, df, indice, values)

        # after xlon open
        minute += delta
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (minute, "open"),
            symb_xhkg: ali_value,
        }
        self.assertions(table, df, indice, values)

        if not xhkg_xlon_touch:
            # at xhkg close
            minute = xhkg_close
            df = f(minute, UTC)
            indice = minute
            values = {
                symb_xnys: (xnys_prev_close - bi, "close"),
                symb_xlon: (minute, "open"),
                symb_xhkg: (xhkg_close - bi, "close"),
            }
            self.assertions(table, df, indice, values)

        # prior to xnys open
        minute = xnys_open - one_min
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (minute, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys_open
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_open, "open"),
            symb_xlon: (minute, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xlon close
        minute = xlon_close
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (minute, "open"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys_close
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_close - bi, "close"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xnys close
        minute += delta
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

    def test_data_available_from_T5(self, prices_us_lon_hk, one_min):
        """Test with minutes for which data only available at bi >= T5."""
        prices = prices_us_lon_hk
        xnys = prices.calendar_default
        xlon = get_calendar_from_name(prices, "XLON")
        xhkg = get_calendar_from_name(prices, "XHKG")
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        delta = pd.Timedelta(43, "min")
        offset = pd.Timedelta(3, "min")
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

        table = prices.get("5min", session_before, session_after, tzout=UTC)

        # reset prices
        prices = reset_prices(prices)
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_prev_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify every minute expected to return indice
        minute += one_min
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (indice, "open"),
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
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg break
        minute += delta
        df = f(minute, UTC)
        indice = xhkg_break_start  # as during break not a trading minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_start - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # on xhkg break end
        minute = xhkg_break_end
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (xhkg_break_end, "open"),
        }
        self.assertions(table, df, indice, values)

        # during xhkg pm subsession
        minute += delta
        df = f(minute, UTC)
        indice = minute - offset
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (indice, "open"),
        }
        self.assertions(table, df, indice, values)

        # prior to xlon open
        minute = xlon_open - one_min
        df = f(minute, UTC)
        indice = minute - bi_less_one_min
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (xlon_prev_close - bi, "close"),
            symb_xhkg: (indice, "open"),
        }
        self.assertions(table, df, indice, values)

        # xlon open (possible xhkg close)
        minute = xlon_open
        df = f(minute, UTC)
        indice = minute
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (minute, "open")
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (minute, "open"),
            symb_xhkg: ali_value,
        }
        self.assertions(table, df, indice, values)

        # after xlon open
        minute += delta
        df = f(minute, UTC)
        indice = minute - offset
        ali_value = (xhkg_close - bi, "close") if xhkg_xlon_touch else (indice, "open")
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (indice, "open"),
            symb_xhkg: ali_value,
        }
        self.assertions(table, df, indice, values)

        if not xhkg_xlon_touch:
            # at xhkg close
            minute = xhkg_close
            df = f(minute, UTC)
            indice = minute
            values = {
                symb_xnys: (xnys_prev_close - bi, "close"),
                symb_xlon: (minute, "open"),
                symb_xhkg: (xhkg_close - bi, "close"),
            }
            self.assertions(table, df, indice, values)

        # verifying all minutes that would return expected indice
        # prior to xnys open
        minute = xnys_open - bi
        indice = minute
        values = {
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (minute, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
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
            symb_xnys: (xnys_prev_close - bi, "close"),
            symb_xlon: (indice, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xnys open
        minute = xnys_open
        indice = minute
        values = {
            symb_xnys: (xnys_open, "open"),
            symb_xlon: (minute, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
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
            symb_xnys: (minute, "open"),
            symb_xlon: (minute, "open"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xlon close
        minute = xlon_close
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (minute, "open"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xlon close
        minute = xlon_close + one_min
        df = f(minute, UTC)
        indice = xlon_close
        values = {
            symb_xnys: (indice, "open"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # one indice after xlon close
        minute = xlon_close + bi
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (indice, "open"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # xnys close
        minute = xnys_close
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: (xnys_close - bi, "close"),
            symb_xlon: (xlon_close - bi, "close"),
            symb_xhkg: (xhkg_close - bi, "close"),
        }
        self.assertions(table, df, indice, values)

        # after xnys close
        minute += delta
        df = f(minute, UTC)
        indice = indice  # unchanged
        values = values  # unchanged
        self.assertions(table, df, indice, values)

    def test_data_available_from_H1(self, prices_us_hk, one_min):
        """Test with minutes for which intraday data only available at H1.

        Also tests indices unaligned with subsession am close, pm open and
        pm close with multiple calendars.
        """
        prices = prices_us_hk
        xnys = prices.calendar_default
        xhkg = get_calendar_from_name(prices, "XHKG")
        symb_xnys = prices.lead_symbol_default
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        delta = pd.Timedelta(63, "min")
        offset = pd.Timedelta(3, "min")
        delta_reg = delta - offset

        bi = prices.bis.H1
        bi_less_one_min = bi - one_min
        half_hour = pd.Timedelta(30, "min")

        # consecutive sessions
        sessions = get_sessions_xnys_xhkg_xlon(bi, 4)
        session_before, session_prev, session, session_after = sessions

        xnys_prev_close = xnys.session_close(session_prev)
        xhkg_prev_close = xhkg.session_close(session_prev)

        xhkg_open = xhkg.session_open(session)
        xnys_open = xnys.session_open(session)
        xhkg_close = xhkg.session_close(session)
        xnys_close = xnys.session_close(session)
        xhkg_break_start = xhkg.session_break_start(session)
        xhkg_break_end = xhkg.session_break_end(session)

        table = prices.get("1h", session_before, session_after, tzout=UTC)

        prices = reset_prices(prices)
        f = prices.price_at

        # prior to xhkg open
        minute = xhkg_open - one_min
        df = f(minute, UTC)
        indice = xnys_prev_close
        msft_prev_close_value = (xnys_prev_close - half_hour, "close")
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (xhkg_prev_close - half_hour, "close"),
        }
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = xhkg_open
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (xhkg_open, "open"),
        }
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice
        # and one minute other side of either bound.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify minutes either side of bounds
        minute += one_min
        indice = minute
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (indice, "open"),
        }
        for i in (0, 1, 2, bi.as_minutes - 2, bi.as_minutes - 1):
            minute_ = minute + pd.Timedelta(i, "min")
            df = f(minute_, UTC)
            self.assertions(table, df, indice, values)

        # verify one minute after right bound.
        minute = minute_ + one_min
        indice = minute
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on xhkg break start
        minute = xhkg_break_start
        df = f(minute, UTC)
        indice = minute
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (xhkg_break_start - half_hour, "close"),
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
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (xhkg_break_end - half_hour, "open"),
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
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (indice, "open"),
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
            symb_xnys: msft_prev_close_value,
            symb_xhkg: (indice, "open"),
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # xhkg close and through and post unaligned final interval
        indice = xhkg_close
        ali_close_value = (xhkg_close - half_hour, "close")
        values = {
            symb_xnys: msft_prev_close_value,
            symb_xhkg: ali_close_value,
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
            symb_xnys: (xnys_open, "open"),
            symb_xhkg: ali_close_value,
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
            symb_xnys: (indice, "open"),
            symb_xhkg: ali_close_value,
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
            symb_xnys: (indice, "open"),
            symb_xhkg: ali_close_value,
        }
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # minutes expected to return as open of final indice of xnys session
        indice = xnys_close - half_hour
        values = {
            symb_xnys: (xnys_close - half_hour, "open"),
            symb_xhkg: ali_close_value,
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
            symb_xnys: (xnys_close - half_hour, "close"),
            symb_xhkg: ali_close_value,
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
        xhkg = prices.calendar_default
        symb_xhkg = prices.lead_symbol_default

        delta = pd.Timedelta(63, "min")
        offset = pd.Timedelta(3, "min")
        delta_reg = delta - offset

        bi = prices.bis.H1
        bi_less_one_min = bi - one_min
        half_hour = pd.Timedelta(30, "min")
        # four consecutive sessions
        sessions = get_conforming_sessions(prices, bi, [xhkg], [session_length_xhkg], 4)
        session_before, session_prev, session, session_after = sessions

        prev_close = xhkg.session_close(session_prev)
        open_ = xhkg.session_open(session)
        close = xhkg.session_close(session)
        break_start = xhkg.session_break_start(session)
        break_end = xhkg.session_break_end(session)

        table = prices.get("1h", session_before, session_after, tzout=UTC)

        prices = reset_prices(prices)
        f = prices.price_at

        # prior to xhkg open
        minute = open_ - one_min
        df = f(minute, UTC)
        indice = prev_close
        values = {symb_xhkg: (prev_close - half_hour, "close")}
        self.assertions(table, df, indice, values)

        # xhkg open
        minute = open_
        indice = minute
        values = {symb_xhkg: (open_, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # During xhkg am subsession. Verify every minute of expected indice
        # and one minute other side of either bound.

        # verify one minute before left bound.
        minute += delta_reg - one_min
        indice = minute - bi_less_one_min
        values = {symb_xhkg: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify minutes either side of bounds
        minute += one_min
        indice = minute
        values = {symb_xhkg: (indice, "open")}
        for i in (0, 1, 2, bi.as_minutes - 2, bi.as_minutes - 1):
            minute_ = minute + pd.Timedelta(i, "min")
            df = f(minute_, UTC)
            self.assertions(table, df, indice, values)

        # verify one minute after right bound.
        minute = minute_ + one_min
        indice = minute
        values = {symb_xhkg: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # on break start and during break
        # select minutes represending start and end of break and minutes either
        # side of unaligned final indice of am session.
        indice = break_start
        values = {symb_xhkg: (break_start - half_hour, "close")}
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
        values = {symb_xhkg: (break_end - half_hour, "open")}
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
        values = {symb_xhkg: (indice, "open")}
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
        values = {symb_xhkg: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify returns open of second to last indice
        minute = close - half_hour - one_min
        indice = close - half_hour - bi
        values = {symb_xhkg: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)

        # verify prior to close returns open of last unaligned indice
        indice = close - half_hour
        values = {symb_xhkg: (indice, "open")}
        for minute in (
            close - half_hour,
            close - one_min,
        ):
            df = f(minute, UTC)
            self.assertions(table, df, indice, values)

        # xhkg close and through and post unaligned final interval
        indice = close
        values = {symb_xhkg: (close - half_hour, "close")}
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
        values = {symb_xhkg: (indice, "open")}
        df = f(minute, UTC)
        self.assertions(table, df, indice, values)


def test_close_at(prices_us_lon_hk, one_day, monkeypatch):
    prices = prices_us_lon_hk
    xhkg = get_calendar_from_name(prices, "XHKG")

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
        now = xhkg.session_close(mock_today) - pd.Timedelta(1, "h")
        mock_now(monkey, now)

        prices = reset_prices(prices)
        # now during XHKG open and MSFT lead symbol so daily table to 'now'
        # should not inlcude session for 23rd
        assert prices.get("1D", days=10).pt.last_ts == pd.Timestamp("2021-12-22")
        # verify that close_at DOES include the 23rd as HKG is open
        rtrn = prices.close_at()
        # given that, notwithstanding time before XNYS open, data is available for
        # all for the 23rd, return should be as if end of 23rd.
        assert_frame_equal(rtrn, df.iloc[[0]], check_freq=False)

    # verify not passing a session returns as at most recent date.
    # NB placed at end as will fail if today flakylisted - test all can before skips.
    current_sessions = [get_current_session(cal) for cal in prices.calendars_unique]
    current_session = max(current_sessions)
    assert prices.close_at().index[0] == current_session


def test_price_range(prices_us_lon_hk, one_day, monkeypatch):
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
    prices = prices_us_lon_hk
    f = prices.price_range

    xhkg = get_calendar_from_name(prices, "XHKG")
    xlon = get_calendar_from_name(prices, "XLON")

    symb_xnys = prices.lead_symbol_default
    symb_xlon = get_symbols_for_calendar(prices, "XLON")
    symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

    _, session = get_sessions_xnys_xhkg_xlon(prices.bis.T1, 2)
    minute = xlon.session_close(session) - pd.Timedelta(43, "min")
    session_T5 = get_sessions_xnys_xhkg_xlon(prices.bis.T5, 2)[-1]
    minute_T5 = xhkg.session_close(session_T5) - pd.Timedelta(78, "min")

    def get(kwargs, **others) -> pd.DataFrame:
        return prices.get(**kwargs, **others, composite=True, openend="shorten")

    def assertions(
        rng: pd.DataFrame,
        table: pd.DataFrame,
        tz: ZoneInfo = prices.tz_default,
        to_now: bool = False,
    ):
        assert_prices_table_ii(rng, prices)
        assert len(rng.index) == 1
        assert rng.pt.tz == tz

        start = table.pt.first_ts
        end = table.pt.last_ts
        if isinstance(table.pt, (pt.PTDailyIntradayComposite, pt.PTDaily)):
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
    tz = UTC
    rng = f(**kwargs, tzout=tz)
    assertions(rng, table, tz)

    tz = ZoneInfo("Australia/Perth")
    rng = f(**kwargs, tzout=tz)
    assertions(rng, table, tz)

    rng = f(**kwargs, tzout=symb_xhkg)
    assertions(rng, table, xhkg.tz)

    # verify output in terms of `tzin` if tzout not otherwise passed.
    tzin = ZoneInfo("Australia/Perth")
    rng = f(**kwargs, tzin=tzin)
    assertions(rng, table, tzin)

    lead = symb_xhkg
    table_ = get(kwargs, lead_symbol=lead)
    rng = f(**kwargs, lead_symbol=lead)
    assertions(rng, table_, xhkg.tz)

    # but not if `tzout` passed
    tzout = ZoneInfo("Europe/Rome")
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
    include = [symb_xhkg, symb_xlon]
    rng_include = f(**kwargs, include=include)
    assert_frame_equal(rng_include, rng[include])
    rng_exclude = f(**kwargs, exclude=symb_xnys)
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
    mock_now(monkeypatch, pd.Timestamp.now(tz=UTC) - pd.Timedelta(5, "D"))
    # verify for passing `start` and for requesting to now
    kwargs = dict(start=minute)
    test_it(kwargs, to_now=True)

    # verify for no arguments
    test_it({}, to_now=True)


def test_prices_for_symbols(prices_us_lon):
    """Verify `prices_for_symbols`.

    Notes
    -----
    H1 interval not tested as not synchronised for xnys/xlon calendars.
    """
    prices = prices_us_lon
    symbols = prices.symbols
    f = prices.prices_for_symbols

    for s, cal in prices.calendars.items():
        if cal.name == "XNYS":
            symb_us = s
        elif cal.name == "XLON":
            symb_lon = s

    _ = prices.get("1d", start="2021-12-31", end="2022-01-05")

    # set up inraday data as period within a single session during which
    # us and lon calendars overlap (from inspection of calendar schedules).
    cal_us = prices.calendars[symb_us]
    cal_lon = prices.calendars[symb_lon]

    session = pd.Timestamp("2022-06-08")
    us_open = cal_us.opens[session]
    lon_close = cal_lon.closes[session]
    assert us_open + pd.Timedelta(1, "h") < lon_close  # verify overlap > one hour
    start = us_open - pd.Timedelta(2, "h")
    end = lon_close + pd.Timedelta(2, "h")

    _ = prices.get("5min", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("2min", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("1min", start, us_open, lead_symbol="AZN.L")
    _ = prices.get("5min", us_open, end)
    _ = prices.get("2min", us_open, end)
    _ = prices.get("1min", us_open, end)

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


@pytest.fixture
def to_csv_dir() -> abc.Iterator[Path]:
    """resources/csv directory"""
    path = RESOURCES_PATH / "to_csv"
    assert path.is_dir()
    yield path


@pytest.fixture
def symbols() -> abc.Iterator[list[str]]:
    """Selection of symbols represented in resources files in to_csv dir."""
    yield ["MSFT", "AZN.L", "9988.HK"]


@pytest.fixture
def calendars(symbols) -> abc.Iterator[dict[str, str]]:
    """Mapping of `symbols` to calendar names."""
    calendars = {
        "MSFT": "XNYS",
        "AZN.L": "XLON",
        "9988.HK": "XHKG",
    }
    assert set(symbols) == set(calendars)
    yield calendars


def test_to_csv(to_csv_dir, temp_dir, symbols, calendars, one_day):
    match = re.escape(
        "Price data has been found for all symbols at a least one interval, however, you may find that not all the expected price data is available. See the `limits` property for available base intervals and the limits between which price data is available at each of these intervals. See the `csv_paths` property for paths to all csv files that were found for the requested symbols. See the 'path' parameter and 'Notes' sections of help(PricesCsv) for advices on how csv files should be named and formatted and for use of the `read_csv_kwargs` parameter.\n\nThe following errors and/or warnings occurred during parsing:\n\n0) For symbol 'AZN.L with interval '1:00:00' no indice aligned with index evaluated from calendar 'XLON'.\n\n1) Prices are not available at base interval 1:00:00 as (aligned) data was not found at this interval for symbols '['AZN.L']'."
    )
    # get prices from csv files. Verify warns that 1H interval not available (files
    # are there but data for XLON instrument does not align with XLON as file was
    # generated form downsampled T5 data).
    with pytest.warns(csv.PricesCsvParsingConsolidatedWarning, match=match):
        prices = csv.PricesCsv(to_csv_dir, symbols, calendars)

    def assert_output_as_original_files(paths: list[Path]):
        for path in paths:
            content = path.open().read()
            content_orig = (to_csv_dir / path.name).open().read()
            assert content == content_orig

    # export all data to .csv files in temp dir
    paths = prices.to_csv(temp_dir)

    # verify that, when intervals are default, data is not included for unaligned
    # base intervals (as if were to have been exported then data would have to
    # have been downsampled from a lower base interval).
    assert not any("_1H_" in path.name.upper() for path in paths)
    assert_output_as_original_files(paths)

    # verify can pass list of multiple intervals
    clean_temp_test_dir()
    intervals = ["5T", "2T"]
    paths = prices.to_csv(temp_dir, intervals)
    assert len(paths) == 6
    assert_output_as_original_files(paths)

    # verify effect of include
    clean_temp_test_dir()
    symbol = symbols[0]
    paths = prices.to_csv(temp_dir, intervals, include=symbol)
    assert len(paths) == 2
    assert all(symbol in path.name for path in paths)
    assert_output_as_original_files(paths)

    # verify effect of exclude
    clean_temp_test_dir()
    paths = prices.to_csv(temp_dir, intervals, exclude=symbol)
    assert len(paths) == 4
    assert not any(symbol in path.name for path in paths)
    assert_output_as_original_files(paths)

    # verify can pass single interval, verify can pass an unaligned base interval
    clean_temp_test_dir()
    paths = prices.to_csv(temp_dir, "1H")
    # verify output as original 1H files created from downsampled data...
    assert_output_as_original_files(paths)

    # verify can pass non-base interval
    clean_temp_test_dir()
    df = prices.get("10T")
    paths = prices.to_csv(temp_dir, "10T")
    prices_reloaded = csv.PricesCsv(temp_dir, symbols, calendars)
    df_reloaded = prices_reloaded.get("10T")
    assert_frame_equal(df, df_reloaded)

    # verify can pass get_params
    clean_temp_test_dir()
    get_params = {"start": df.index[7].right, "end": df.index[-7].left}
    paths = prices.to_csv(temp_dir, "5T", get_params=get_params)
    assert len(paths) == 3
    prices_reloaded = csv.PricesCsv(temp_dir, symbols, calendars)
    df_reloaded = prices_reloaded.get("5T")
    assert df_reloaded.pt.interval == prices.bis.T5
    assert df_reloaded.pt.first_ts == get_params["start"]
    assert df_reloaded.pt.last_ts == get_params["end"]

    # verify raises when no get_params
    clean_temp_test_dir()
    match = re.escape(
        "It was not possible to export prices as an error was raised when prices were"
        " requested for interval 0:01:00. The error is included at the top of the"
        " traceback.\nNB prices have not been exported for any interval."
    )
    with pytest.raises(errors.PricesUnavailableForExport, match=match):
        prices.to_csv(temp_dir, include=["NOT_A_SYMBOL"])
    assert not list(temp_dir.iterdir())

    # verify raises when pass get_params
    get_params = {"start": df.pt.first_ts - (one_day * 7)}
    match = re.escape(
        "It was not possible to export prices as an error was raised when"
        " prices were requested for interval 0:05:00. The error is included at the"
        " top of the traceback. Prices were requested with the following kwargs:"
        " {'start': Timestamp('2023-12-11 09:40:00-0500', tz='America/New_York')}"
        "\nNB prices have not been exported for any interval."
    )
    with pytest.raises(errors.PricesUnavailableForExport, match=match):
        prices.to_csv(temp_dir, "5T", get_params=get_params)
    assert not list(temp_dir.iterdir())
