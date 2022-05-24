"""Tests for market_prices.prices.base module.

Notes
-----
Some `PricesBase` methods are tested on `test_yahoo.py` under the default
`PricesYahoo` implementation. See dedicated section of `test_yahoo.py`
"""

from __future__ import annotations

import itertools
import dataclasses
import typing
from collections import abc
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytz
import pytest

import market_prices.prices.base as m
from market_prices import helpers, intervals, errors, daterange, mptypes
from market_prices.prices.yahoo import PricesYahoo
from market_prices.utils import calendar_utils as calutils

from .utils import get_resource


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
def t1_us_lon() -> abc.Iterator[pd.DataFrame]:
    """'T1' price table for us and lon symbols.

    Recreate table with:
    > symbols = ["MSFT", "AZN.L"]
    > df = prices.get(
        "1T",
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
        "5T",
        start=pd.Timestamp("2022-01"),
        end=pd.Timestamp("2022-02-07"),
        anchor="open",
    )
    """
    yield get_resource("t5_us_lon")


def test_create_composite(t1_us_lon, t5_us_lon, one_day):
    f = m.create_composite

    first_df = t5_us_lon
    start = pd.Timestamp("2022-02-03 14:00", tz=pytz.UTC)
    stop = pd.Timestamp("2022-02-09 15:32", tz=pytz.UTC)
    second_df = t1_us_lon[start:stop]

    start_indice = first_df.index[33]
    end_indice = second_df.index[-6]
    assert end_indice.right == pd.Timestamp("2022-02-09 15:28", tz=pytz.UTC)
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
        _ = f(second, first)  # pylint: disable=arguments-out-of-order

    # verify raises error when 'first' overlaps 'second'
    with pytest.raises(ValueError, match=match):
        subset = second_df[: first_df.index[-1].left - one_day]
        f(first, (subset, subset.index[-5]))


def test_inferred_intraday_interval(calendars_extended, one_min, monkeypatch):
    # pylint: disable=too-complex
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
    session = start_date = helpers.to_tz_naive(  # TODO xcals 4.0 lose wrapper
        cal.date_to_session(pd.Timestamp("2021-10-01"), "next")
    )
    session = cal.session_offset(session, 7)
    time = session.replace(minute=33, hour=11)
    session = cal.session_offset(session, 7)
    midnight = helpers.to_utc(session)  # TODO xcals 4.0 this STAYS STAYS STAYS
    # TODO xcals 4.0 lose wrapper
    end_date = helpers.to_tz_naive(cal.session_offset(session, 7))

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
    end = helpers.to_tz_naive(end)  # TODO xcals 4.0 lose line
    assert_intraday(start=start_date, end=end)  # 4 days diff

    end = cal.session_offset(end, 1)
    end = helpers.to_tz_naive(end)  # TODO xcals 4.0 lose line
    assert_intraday(start=start_date, end=end)  # 5 days diff, on limit

    end = cal.session_offset(end, 1)
    end = helpers.to_tz_naive(end)  # TODO xcals 4.0 lose line
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
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
    assert_intraday(start=start)  # 4 days diff to prior session

    start = cal.session_offset(start, -1)
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
    assert_intraday(start=start)  # 5 days diff, on limit

    start = cal.session_offset(start, -1)
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
    assert_daily(start=start)  # 6 days diff, over limit

    def mock_now_after_close(*_, **__) -> pd.Timestamp:
        return last_session_close + (5 * one_min)

    monkeypatch.setattr("pandas.Timestamp.now", mock_now_after_close)

    assert_intraday(start=last_session)  # same date

    # as now after close, 5 days will count to, and inclusive of last_session
    start = cal.session_offset(last_session, -3)
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
    assert_intraday(start=start)  # 4 days diff

    start = cal.session_offset(start, -1)
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
    assert_intraday(start=start)  # 5 days diff, on limit

    start = cal.session_offset(start, -1)
    start = helpers.to_tz_naive(start)  # TODO xcals 4.0 lose line
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
    """Limit of daily price availability for PricesMock."""
    yield pd.Timestamp("1990-08-27")


@pytest.fixture(scope="class")
def cal_start(daily_limit) -> abc.Iterator[pd.Timestamp]:
    """Start date for any calendar to be passed to PricesMock."""
    yield daily_limit - pd.Timedelta(14, "D")


@pytest.fixture(scope="class")
def xnys(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XNYS calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "XNYS" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("XNYS", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xlon(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XLON calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "XLON" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("XLON", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xasx(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XASX calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "XASX" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("XASX", start=cal_start, side=side)


@pytest.fixture(scope="class")
def xhkg(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XHKG calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "XHKG" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("XHKG", start=cal_start, side=side)


@pytest.fixture(scope="class")
def x247(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """24/7 calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "24/7" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("24/7", start=cal_start, side=side)


@pytest.fixture(scope="class")
def cmes(cal_start, side) -> abc.Iterator[xcals.ExchangeCalendar]:
    """CMES calendar that can be passed to PricesMock.

    Due to xcals caching, if calendar passed to PricesMock as "CMES" then
    calendar 'created' by PricesMock will be same object as returned by
    this fixture.
    """
    yield xcals.get_calendar("CMES", start=cal_start, side=side)


@pytest.fixture
def PricesMock(daily_limit) -> abc.Iterator[type[m.PricesBase]]:
    class PricesMock_(m.PricesBase):
        """Mock PricesBase class."""

        BaseInterval = intervals._BaseInterval(
            "BaseInterval",
            dict(
                T1=pd.Timedelta(1, "T"),
                T5=pd.Timedelta(5, "T"),
                H1=pd.Timedelta(1, "H"),
                D1=pd.Timedelta(1, "D"),
            ),
        )

        BASE_LIMITS = {
            BaseInterval.T1: pd.Timedelta(30, "D"),
            BaseInterval.T5: pd.Timedelta(60, "D"),
            BaseInterval.H1: pd.Timedelta(365, "D"),
            BaseInterval.D1: daily_limit,
        }

        def _request_data(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

        def prices_for_symbols(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

    yield PricesMock_


@pytest.fixture
def PricesMockBreakendPmOrigin(PricesMock) -> abc.Iterator[type[m.PricesBase]]:
    class PricesMockBreakendPmOrigin_(PricesMock):  # type: ignore[valid-type, misc]
        """Mock PricesBase class with PM_SUBSESSION_ORIGIN as 'break end'."""

        # pylint: disable=too-few-public-methods
        PM_SUBSESSION_ORIGIN = "break_end"

    yield PricesMockBreakendPmOrigin_


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
            daterange: tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
            | None = None,
            daterange_sessions: tuple[pd.Timestamp, pd.Timestamp] | None = None,
            interval: intervals.BI | None = None,
            calendar: xcals.ExchangeCalendar | None = None,
            composite_calendar: calutils.CompositeCalendar | None = None,
            delay: pd.Timedelta = pd.Timedelta(0),
            limit: pd.Timestamp | None = None,
            ignore_breaks: bool | dict[intervals.BI, bool] = False,
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


class TestPricesBaseSetup:
    """Verify properties of PricesBase post-instantiation."""

    @pytest.fixture(scope="class")
    def symbols(self) -> abc.Iterator[list[str]]:
        yield ["ONE.1", "TWO.22", "THREE3.3"]

    @pytest.fixture
    def PricesMockDailyNoLimit(
        self, PricesMock: type[m.PricesBase]
    ) -> abc.Iterator[type[m.PricesBase]]:
        """As PricesMock with daily interval with limit as None."""
        BASE_LIMITS_DAILY_NO_LIMIT = PricesMock.BASE_LIMITS.copy()
        BASE_LIMITS_DAILY_NO_LIMIT[PricesMock.BaseInterval.D1] = None

        class PricesMockDailyNoLimit_(PricesMock):  # type: ignore[valid-type, misc]
            """Mock PricesBase class with daily bi with no limit."""

            # pylint: disable=too-few-public-methods
            BASE_LIMITS = BASE_LIMITS_DAILY_NO_LIMIT

        yield PricesMockDailyNoLimit_

    @pytest.fixture
    def PricesMockNoDaily(
        self, PricesMock: type[m.PricesBase]
    ) -> abc.Iterator[type[m.PricesBase]]:
        """As PricesMock with no daily interval."""

        class PricesMockNoDaily_(PricesMock):  # type: ignore[valid-type, misc]
            """Mock PricesBase class with no daily bi."""

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T1=pd.Timedelta(1, "T"),
                    T5=pd.Timedelta(5, "T"),
                    H1=pd.Timedelta(1, "H"),
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T1: pd.Timedelta(30, "D"),
                BaseInterval.T5: pd.Timedelta(60, "D"),
                BaseInterval.H1: pd.Timedelta(365, "D"),
            }

        yield PricesMockNoDaily_

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
        assert prices.calendars == {symbol: cal for symbol in symbols}
        assert prices.calendars_symbols == {cal: symbols}
        assert prices.calendar_default == cal
        assert prices.calendars_unique == [cal]
        assert prices.calendars_names == ["XNYS"]
        assert prices.has_single_calendar

        assert prices._lead_symbol_default == symbols[0]

        assert prices.delays == {symbol: zero_td for symbol in symbols}
        assert prices.min_delay == zero_td
        assert prices.max_delay == zero_td
        assert prices.calendars_min_delay == {cal: zero_td}
        assert prices.calendars_max_delay == {cal: zero_td}

        assert prices.timezones == {s: cal.tz for s in symbols}
        assert prices.tz_default == cal.tz

        expected_cc = calutils.CompositeCalendar([cal])
        assert_frame_equal(prices.composite_calendar.schedule, expected_cc.schedule)
        assert prices.composite_calendar is prices.cc

        # Verify passing through single calendar as actual instance
        prices = PricesMock(symbols, xasx)
        assert prices.calendars == {s: xasx for s in symbols}

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

        assert prices._lead_symbol_default == "LON"

        assert prices.delays == {symbol: zero_td for symbol in symbols}
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
            "NY": pd.Timedelta(5, "T"),
            "LON": pd.Timedelta(10, "T"),
            "LON2": pd.Timedelta(15, "T"),
            "OZ": zero_td,
        }
        assert prices.delays == expected_delays
        assert prices.min_delay == zero_td
        assert prices.max_delay == pd.Timedelta(15, "T")
        assert prices.calendars_min_delay == {
            xlon: pd.Timedelta(10, "T"),
            xnys: pd.Timedelta(5, "T"),
            xasx: zero_td,
        }
        assert prices.calendars_max_delay == {
            xlon: pd.Timedelta(15, "T"),
            xnys: pd.Timedelta(5, "T"),
            xasx: zero_td,
        }

        # verifying effect of having changed `lead_symbol`
        assert prices.symbols == symbols
        assert prices.calendar_default == xnys
        assert prices._lead_symbol_default == "NY"
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

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T5=pd.Timedelta(5, "T"),
                    H1=pd.Timedelta(1, "H"),
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

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T5=pd.Timedelta(5, "T"),
                    H1=pd.Timedelta(1, "H"),
                    D1=pd.Timedelta(1, "D"),
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T5: pd.Timedelta(days=60),
                BaseInterval.D1: daily_limit,
            }

        def match(bis, limit_keys) -> str:  # pylint: disable=function-redefined
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
        base_limits_copy[not_bi] = pd.Timedelta(1, "S")
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
            pd.Timestamp("2000-01-01", tz=pytz.UTC),
        ]
        for limit in limits:
            with pytest.raises(ValueError, match=match_daily_limit(limit)):
                prices._update_base_limits({prices.bi_daily: limit})

        limit = limits[0]

        class PricesMockInvalidBaseLimit3A(PricesMock):
            """Mock PricesBase class with invalid daily base limit."""

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(D1=pd.Timedelta(1, "D")),
            )

            BASE_LIMITS = {BaseInterval.D1: limit}

        with pytest.raises(ValueError, match=match_daily_limit(limit)):
            PricesMockInvalidBaseLimit3A(*inst_args)

        limit = limits[1]

        class PricesMockInvalidBaseLimit3B(PricesMock):
            """Mock PricesBase class with invalid daily base limit."""

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(D1=pd.Timedelta(1, "D")),
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
            return re.escape(
                f"Calendar '{cal.name}' is too short to support all available price"
                f" history. Calendar starts '{cal.first_session}' whilst earliest date"
                f" for which price history is available is '{daily_limit}'. Prices"
                f" will not be available for any date prior to {cal.first_session}."
            )

        with pytest.warns(errors.CalendarTooShortWarning, match=match(cal)):
            PricesMock(symbols, cal)

        with pytest.warns(errors.CalendarTooShortWarning, match=match(cal)) as ws:
            PricesMock(symbols, [good_cal, cal, cal2])
        assert len(ws.list) == 2

        with pytest.warns(errors.CalendarTooShortWarning, match=match(cal2)):
            PricesMock(symbols, [good_cal, cal, cal2])

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
        start = intraday_limit.normalize() + one_day
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

        cals = {s: cal for s in symbols[:-1]}
        cals["not_a_symbol"] = cal
        with pytest.raises(ValueError, match=match("calendars", cals)):
            PricesMock(symbols, cals)

        cals = {s: cal for s in symbols}
        cals["extra_symbol"] = cal
        with pytest.raises(ValueError, match=match("calendars", cals)):
            PricesMock(symbols, cals)

        delays = [5, 5]
        with pytest.raises(ValueError, match=match("delays", delays)):
            PricesMock(symbols, cal, delays=delays)

        delays = {s: 5 for s in symbols[:-1]}
        delays["not_a_symbol"] = 5
        with pytest.raises(ValueError, match=match("delays", delays)):
            PricesMock(symbols, cal, delays=delays)

        delays = {s: 5 for s in symbols}
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
        PricesMockNoDaily,
        PricesMockDailyNoLimit,
        daily_limit,
        symbols,
        xnys,
        xlon,
        xhkg,
        one_min,
        monkeypatch,
    ):
        """Test limits properties."""

        def mock_now(tz=None) -> pd.Timestamp:
            return pd.Timestamp("2022-02-14 21:21:05", tz=tz)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now)
        now = mock_now(tz=pytz.UTC)
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
                ll = (now - PricesMock.BASE_LIMITS[bi]).ceil("T") + one_min
                rl = now.floor("T") + bi
                assert prices.limits[bi] == (ll, rl)

        # verify `limit_daily`
        assert prices.limit_daily == daily_limit

        # verify `limit_intraday`
        delta = PricesMock.BASE_LIMITS[PricesMock.BaseInterval.T5]  # unaligned at H1
        limit_raw = (now - delta).ceil("T") + one_min
        limits_intraday = []
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw, "next")
            limits_intraday.append(expected_limit_intraday)
            assert prices.limit_intraday(cal) == expected_limit_intraday
        expected_latest_intraday_limit = max(limits_intraday)
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit

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

        prices = PricesMockDailyNoLimit(symbols, calendars)
        # only test for differences to PricesMock

        assert set(prices.limits.keys()) == set(PricesMockDailyNoLimit.BaseInterval)
        assert len(prices.limits) == len(PricesMockDailyNoLimit.BaseInterval)
        bi_daily = PricesMockDailyNoLimit.BaseInterval.D1
        assert prices.limits[bi_daily] == (None, today)

        assert prices.limit_daily is None
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw, "next")
            assert prices.limit_intraday(cal) == expected_limit_intraday
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit
        assert len(prices.limits_sessions) == len(PricesMockDailyNoLimit.BaseInterval)
        assert prices.limits_sessions[bi_daily] == (None, today)

        prices = PricesMockNoDaily(symbols, calendars)
        # only test for differences to PricesMock

        assert set(prices.limits.keys()) == set(PricesMockNoDaily.BaseInterval)
        assert len(prices.limits) == len(PricesMockNoDaily.BaseInterval)
        assert pd.Timedelta(1, "T") in prices.bis
        assert not pd.Timedelta(1, "D") in prices.bis

        assert prices.limit_daily is None
        for cal in calendars:
            expected_limit_intraday = cal.minute_to_trading_minute(limit_raw, "next")
            assert prices.limit_intraday(cal) == expected_limit_intraday
        assert prices.limit_intraday() == expected_latest_intraday_limit
        assert prices.limit_intraday(None) == expected_latest_intraday_limit
        assert len(prices.limits_sessions) == len(PricesMockNoDaily.BaseInterval)

    def test_earliest(
        self,
        PricesMock,
        PricesMockNoDaily,
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
            session = helpers.to_tz_naive(
                cal.first_session
            )  # TODO xcals 4.0 lose wrapper
            assert prices._earliest_requestable_calendar_session(cal) == session
            assert prices._earliest_requestable_calendar_minute(cal) == cal.first_minute

        # TODO xcals 4.0 lose wrapper
        session = helpers.to_tz_naive(max([cal.first_session for cal in calendars]))
        assert prices.earliest_requestable_session == session
        minute = max([cal.first_minute for cal in calendars])
        assert prices.earliest_requestable_minute == minute

        prices = PricesMockNoDaily(symbols, calendars)
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

    def test_last_requestable(
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
        now = pd.Timestamp("2021-12-31 23:59", tz=pytz.UTC)
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
                "2021-12-24",  # only xhkg and xlon open, no conflict as xhkg closes early
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
                "2021-01-18",  # xhkg pm session close contiguous with start xlon session
                "2021-07-05",  # xhkg pm session and xlon session overlap but do not conflict
                "2021-09-06",  # xhkg pm session and xlon session overlap but do not conflict
                "2021-11-25",  # xhkg pm session close contiguous with start xlon session
            ]
        )
        H1_expected = H1_expected.union(additional_expected)
        assertions(prices, H1_expected)

        # calendars are not aligned
        calendars = [xnys, xnys, x247]
        prices = PricesMock(symbols, calendars)
        xnys_sessions = xnys.sessions_in_range("2021", "2021-12-31")
        # TODO xcals 4.0 lose if clause
        if xnys_sessions.tz is not None:
            xnys_sessions = xnys_sessions.tz_convert(None)
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
        # pylint: disable=too-complex, unbalanced-tuple-unpacking
        now = pd.Timestamp("2022", tz=pytz.UTC)
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
                # TODO xcals 4.0 lose if clause
                if sessions.tz is not None:
                    sessions = sessions.tz_convert(None)
                calendars_sessions.append(sessions)
            return calendars_sessions

        def assert_all_same(
            prices: m.PricesBase, bi: intervals.BI, value: bool | float
        ):
            sessions = get_sessions(prices, bi)
            expected = pd.Series(value, index=sessions)
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
        expected = pd.Series(np.nan, index=sessions)
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
        expected = pd.Series(True, index=sessions)
        # although there are a couple of early closes that are not aligned with 1H
        dates = ["2021-12-24", "2021-12-31"]
        expected[dates] = False  # type: ignore[call-overload]
        assert_series_equal(prices._indexes_status[bi], expected)

        # XASX combined with xlon.
        prices = PricesMock(symbols, [xlon, xasx])
        for bi in prices.bis_intraday[:2]:
            assert_all_same(prices, bi, True)
        bi = prices.bis.H1
        sessions = get_sessions(prices, bi)
        xasx_sessions, xlon_sessions = get_calendars_sessions(prices, bi, [xasx, xlon])
        # ...IH partial indices every session
        expected = pd.Series(False, index=sessions)
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
        expected = pd.Series(True, index=sessions)
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
        expected = pd.Series(np.NaN, index=sessions)
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
    assert xlon_session_close == pd.Timestamp("2021-01-19 16:30", tz=pytz.UTC)
    xnys_session_open = xnys.session_open(session)
    assert xnys_session_open == pd.Timestamp("2021-01-19 14:30", tz=pytz.UTC)

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
        monkey.setattr("pandas.Timestamp.now", lambda *a, **k: ts)

    minute = xlon_session_open
    args = ("latest", "previous")
    assert f(minute, *args) == session

    # now at limit of minute + min xlon delay (10)
    with monkeypatch.context() as m:
        now = minute + pd.Timedelta(10, "T")
        patch_now(m, now)
        assert f(minute, *args) == session
        # verify makes no difference if pass minute ahead of now
        assert f(minute + pd.Timedelta(5, "D"), *args) == session

        # now less than minute + min xlon delay (10)
        now = minute + pd.Timedelta(9, "T")
        patch_now(m, now)
        assert not f(minute, *args) == session
        assert f(minute, *args) == xlon_prev_session

    minute = xnys_session_close
    args = ("earliest", "next")
    assert f(minute, *args) == next_session

    with monkeypatch.context() as m:
        # now at limit of minute + min xnys delay (5)
        now = minute + pd.Timedelta(5, "T")
        patch_now(m, now)
        assert f(minute, *args) == next_session

        # verify makes no difference if pass minute ahead of now
        assert f(minute + pd.Timedelta(5, "D"), *args) == next_session

        # now less than minute + min xlon delay (5)
        now = minute + pd.Timedelta(4, "T")
        patch_now(m, now)
        assert not f(minute, *args) == next_session
        assert f(minute, *args) == session


def test__minute_to_latest_trading_minute(PricesMock, cal_start, side, one_min):
    """Test `_minute_to_latest_trading_minute`."""
    xnys = xcals.get_calendar("XNYS", start=cal_start, side=side)
    xlon = xcals.get_calendar("XLON", start=cal_start, side=side)
    xhkg = xcals.get_calendar("XHKG", start=cal_start, side=side)

    symbols = "LON, NY, HK"
    prices = PricesMock(symbols, [xlon, xnys, xhkg])
    f = prices._minute_to_latest_trading_minute

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
    now = pd.Timestamp("2021-12-31 23:59", tz=pytz.UTC)
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
    """Tests methods and properties that return base interval/s."""

    _now = pd.Timestamp("2022", tz=pytz.UTC)

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

            # pylint: disable=too-few-public-methods

            BaseInterval = intervals._BaseInterval(
                "BaseInterval",
                dict(
                    T1=pd.Timedelta(1, "T"),
                    T2=pd.Timedelta(2, "T"),
                    T5=pd.Timedelta(5, "T"),
                    T10=pd.Timedelta(10, "T"),
                    T15=pd.Timedelta(15, "T"),
                    T30=pd.Timedelta(30, "T"),
                    H1=pd.Timedelta(1, "H"),
                    D1=pd.Timedelta(1, "D"),
                ),
            )

            BASE_LIMITS = {
                BaseInterval.T1: pd.Timestamp("2021-12-01", tz=pytz.UTC),
                BaseInterval.T2: pd.Timestamp("2021-11-01", tz=pytz.UTC),
                BaseInterval.T5: pd.Timestamp("2021-10-01", tz=pytz.UTC),
                BaseInterval.T10: pd.Timestamp("2021-09-01", tz=pytz.UTC),
                BaseInterval.T15: pd.Timestamp("2021-06-01", tz=pytz.UTC),
                BaseInterval.T30: pd.Timestamp("2021-03-01", tz=pytz.UTC),
                BaseInterval.H1: pd.Timestamp("2021-01-01", tz=pytz.UTC),
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

                self._gpp = GetPricesParamsMock(drg, drg, ds_interval, anchor)

        yield PricesMockBis_

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
        start: pd.Timstamp,
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
            start of daterange will be displaced by `delta_end` mintues.
        """
        start = prices.BASE_LIMITS[bi]
        assert isinstance(start, pd.Timestamp)
        end = (start if limit_end else self._now) + pd.Timedelta(delta_end, "T")
        start += pd.Timedelta(delta, "T")
        return self.get_mock_drg(GetterMock, prices.cc, start, end)

    def get_drg(
        self,
        calendar: xcals.ExchangeCalendar,
        composite_calendar: calutils.CompositeCalendar | None = None,
        delay: pd.Timedelta = pd.Timedelta(0),
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
        # pylint: disable=too-complex
        prices = PricesMockBis(symbols, [xlon, xnys])

        # sessions through which all intervals are aligned, from manual inspection.
        start_all_aligned = pd.Timestamp("2021-12-24")
        end_all_aligned = pd.Timestamp("2021-12-28")
        sessions_aligned = (start_all_aligned, end_all_aligned)

        def f(interval: int, drg: daterange.GetterIntraday) -> list[intervals.BI]:
            self.set_prices_gpp_drg_properties(prices, drg)
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "T")
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
        start = pd.Timestamp("2021-12-23 15:00", tz=pytz.UTC)
        end = pd.Timestamp("2021-12-23 16:00", tz=pytz.UTC)
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

        pp["end"] = pd.Timestamp("2021-12-23 15:12", tz=pytz.UTC)
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

    def test__bis_available(self, PricesMockBis, GetterMock, symbols, xlon, xnys):
        """Test `_bis_available` and `_bis_available_end`."""
        prices = PricesMockBis(symbols, [xlon, xnys])
        get_drg_args = (prices, GetterMock)

        def bis_available(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "T")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available

        def bis_available_end(interval: int, drg) -> list[intervals.BI]:
            prices.gpp.ds_interval = intervals.to_ptinterval(str(interval) + "T")
            self.set_prices_gpp_drg_properties(prices, drg)
            return prices._bis_available_end

        for i, bi in enumerate(prices.bis_intraday[:-1]):
            # start at limit for bi, end now
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi)
            assert bis_available(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]

            assert bis_available(bi.as_minutes, drg) == [bi]

            # start before limit for bi, end now
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi, -1)
            assert bis_available(30, drg) == prices.bis_intraday[i + 1 : -1]
            assert bis_available_end(30, drg) == prices.bis_intraday[:-1]

            assert bis_available(bi.as_minutes, drg) == []

            # start and end at limit for bi
            drg = self.get_mock_drg_limit_available(*get_drg_args, bi, limit_end=True)
            assert bis_available(30, drg) == prices.bis_intraday[i:-1]
            assert bis_available_end(30, drg) == prices.bis_intraday[i:-1]

            assert bis_available_end(bi.as_minutes, drg) == [bi]

            # start and end beyond limit for bi
            drg = self.get_mock_drg_limit_available(
                *get_drg_args, bi, -1, limit_end=True, delta_end=-1
            )
            assert bis_available(30, drg) == prices.bis_intraday[i + 1 : -1]
            assert bis_available_end(30, drg) == prices.bis_intraday[i + 1 : -1]

            assert bis_available_end(bi.as_minutes, drg) == []

    def test_bis_stored_methods(self, PricesMockBis, GetterMock, symbols, xlon, xnys):
        """Tests `_bis_stored` and `_get_stored_bi_from_bis`."""
        prices = PricesMockBis(symbols, [xlon, xnys])
        get_drg_args = (prices, GetterMock)

        prices.gpp.ds_interval = intervals.to_ptinterval("30T")
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
            [1] Early close pd.Timestamp("2021-12-24 03:10", tz=pytz.UTC)
        """
        # pylint: disable=redundant-yields-doc
        early_close_session = pd.Timestamp("2021-12-24")
        early_close = xasx.session_close(early_close_session)
        # assert assumption that early close
        assert early_close == pd.Timestamp("2021-12-24 03:10", tz=pytz.UTC)

        revised_limits = PricesMockBis.BASE_LIMITS.copy()
        t1_limit = pd.Timestamp("2021-12-29", tz="UTC")
        revised_limits[PricesMockBis.BaseInterval.T1] = t1_limit

        class PricesMockBisAlt(PricesMockBis):  # type: ignore[valid-type, misc]
            """Mock PricesBase class with alternative limits."""

            # pylint: disable=too-few-public-methods
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
        end = pd.Timestamp("2021-12-23 05:00", tz=pytz.UTC)
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
        now = pd.Timestamp("2021-12-31 15:14", tz=pytz.UTC)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: now)

        cal = xnys
        ds_interval = intervals.TDInterval.H1  # all bis valid
        prices = PricesMockBis(symbols, cal, ds_interval=ds_interval)
        start = pd.Timestamp("2021", tz=pytz.UTC)  # start for all drg
        period_end_now = prices._bis_period_end_now
        bis_most_accurate = prices._bis_most_accurate

        # Verify methods for when end is earlier than now.

        # verify period end does not evaluate as now for any bis.
        for end in [
            pd.Timestamp("2021-12-31 15:13", tz=pytz.UTC),
            pd.Timestamp("2021-12-31 15:00", tz=pytz.UTC),
            pd.Timestamp("2021-12-31 12:00", tz=pytz.UTC),
            pd.Timestamp("2021-12-30 15:14", tz=pytz.UTC),
            pd.Timestamp("2021-12-30 15:13", tz=pytz.UTC),
            pd.Timestamp("2021-12-29 12:00", tz=pytz.UTC),
            pd.Timestamp("2021-06-29 12:00", tz=pytz.UTC),
            pd.Timestamp("2021-01-29 12:00", tz=pytz.UTC),
        ]:
            pp = get_pp(start=start, end=end)
            drg = self.get_drg(cal, pp=pp)
            self.set_prices_gpp_drg_properties(prices, drg)

            prices.gpp.anchor = mptypes.Anchor.OPEN
            assert period_end_now(prices.bis_intraday) == []
            prices.gpp.anchor = mptypes.Anchor.WORKBACK
            assert period_end_now(prices.bis_intraday) == []

        end = pd.Timestamp("2021-12-23 18:15", tz=pytz.UTC)
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
        end = pd.Timestamp("2021-06-16 18:15", tz=pytz.UTC)
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
        start = pd.Timestamp("2021", tz=pytz.UTC)  # start for all drg
        end = pd.Timestamp("2021-12-23 05:45", tz=pytz.UTC)
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

            # pylint: disable=too-few-public-methods
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
        ) -> str:
            interval = prices.gpp.ds_interval
            anchor = prices.gpp.anchor
            factors = [bi for bi in prices.bis_intraday if not interval % bi]
            highest_factor_limit = prices.BASE_LIMITS[factors[-1]]
            earliest_minute = cal.minute_to_trading_minute(highest_factor_limit, "next")
            anchor_insert = " and that have no partial trading indices"
            insert1 = "" if anchor is mptypes.Anchor.OPEN else anchor_insert
            s = (
                "Data is unavailable at a sufficiently low base interval to evaluate"
                f" prices at interval {interval} anchored '{anchor}'.\nBase intervals"
                f" that are a factor of {interval}{insert1}:\n\t{factors}.\nThe"
                f" earliest minute from which data is available at {factors[-1]} is"
                f" {earliest_minute}, although at this base interval the requested"
                f" period evaluates to {(start, early_close)}."
                "\nPeriod evaluated from parameters: <mock pp>."
            )
            if part_period_available:
                s += (
                    f"\nData is available from {earliest_minute} through to the end"
                    f" of the requested period. Consider passing `strict` as False"
                    f" to return prices for this part of the period."
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
        msg = match(start, drg.cal, part_period_available=False)
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
        msg = match(start, drg.cal, part_period_available=False)
        with pytest.raises(errors.PricesIntradayUnavailableError, match=msg):
            _ = prices._bis_end

        # verify error message for inferred interval
        prices.gpp.ds_interval = None
        prices.gpp.anchor = anchor = mptypes.Anchor.OPEN
        # set start to before leftmost limit
        highest_bi = prices.bis_intraday[-1]
        leftmost_limit = prices.BASE_LIMITS[highest_bi]
        # Use cal from previous drg...
        earliest_minute = drg.cal.minute_to_trading_minute(leftmost_limit, "next")
        start = leftmost_limit - one_min
        drg = self.get_mock_drg(GetterMock, cc, start, early_close)
        self.set_prices_gpp_drg_properties(prices, drg)

        msg = re.escape(
            "Data is unavailable at a sufficiently low base interval to evaluate"
            f" prices at an inferred interval anchored '{anchor}'.\nBase intervals:"
            f"\n\t{prices.bis_intraday}.\nThe earliest minute from which data is"
            f" available at {highest_bi} is {earliest_minute}, although at this base"
            f" interval the requested period evaluates to {(start, early_close)}."
            "\nPeriod evaluated from parameters: <mock pp>."
            f"\nData is available from {earliest_minute} through to the end of the"
            " requested period. Consider passing `strict` as False to return prices"
            " for this part of the period."
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
            assert drg._limit == prices._earliest_requestable_calendar_minute(drg.cal)
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

    start = pd.Timestamp("2021-12-15 12:22", tz=pytz.UTC)
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
    start_expected = pd.Timestamp("2021-12-15 14:30", tz=pytz.UTC)
    pp_expected = get_pp(start=start_expected, days=2)
    assert gpp.pp(intraday=True) == pp_expected
    start_expected_daily = pd.Timestamp("2021-12-15")
    pp_expected = get_pp(start=start_expected_daily, days=2)
    assert gpp.pp(intraday=False) == pp_expected
    assert not gpp.intraday_duration
    assert gpp.duration
    assert not gpp.request_earliest_available_data

    # assert parameters being passed through to drg.
    drg = gpp.drg_intraday
    assert_drg_intraday_properties(drg, gpp, strict, ds_interval)
    with pytest.raises(errors.StartTooEarlyError):
        drg.interval = prices.bis.T5
        _ = drg.daterange

    drg = gpp.drg_intraday_no_limit
    assert_drg_intraday_properties(drg, gpp, False, ds_interval, no_limit=True)
    drg.interval = prices.bis.T5
    acc_expected = pd.Timestamp("2021-12-16 21:00", tz=pytz.UTC)
    end_expected = pd.Timestamp("2021-12-16 22:30", tz=pytz.UTC)
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
    end = pd.Timestamp("2021-12-15 03:10", tz=pytz.UTC)
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

    drg = gpp.drg_intraday
    assert_drg_intraday_properties(drg, gpp, strict, ds_interval)

    drg = gpp.drg_intraday_no_limit
    # assert ignore breaks and effect of
    drg.interval = prices.bis.H1
    assert drg.ignore_breaks
    # from knowledge of schedule...
    end_expected = pd.Timestamp("2021-12-15 02:30", tz=pytz.UTC)
    start_expected = pd.Timestamp("2021-12-14 05:30", tz=pytz.UTC)
    assert drg.daterange == ((start_expected, end_expected), end_expected)

    drg.interval = prices.bis.T5
    assert not drg.ignore_breaks
    start_expected = pd.Timestamp("2021-12-14 06:00", tz=pytz.UTC)
    assert drg.daterange == ((start_expected, end_expected), end_expected)

    # alternative parameters just to verify request_earliest_available_data True
    pp = get_pp(end=end)
    gpp = f(prices, pp, ds_interval, lead_symbol, anchor, openend, strict, priority)
    assert not gpp.intraday_duration
    assert not gpp.duration
    assert gpp.request_earliest_available_data
