"""Tests added with `PricesCSV` to test returns around limits.

Tests added to verify effect of requesting periods where extremes lie on
and over either or both of left and right limits and with 'strict' as both
True and False. Also tests effect of 'priority'.
"""

from __future__ import annotations

import itertools
import re
import typing
import warnings
from collections import abc
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from market_prices import errors, intervals
from market_prices.helpers import UTC
from market_prices.intervals import TDInterval
from market_prices.prices.csv import PricesCsv
from market_prices.mptypes import Priority


from .utils import RESOURCES_PATH


# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=too-many-arguments, too-many-public-methods
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-statements, too-many-lines, line-too-long
# pylint: disable=protected-access, no-self-use, unused-argument, invalid-name
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   invalid-name: names in tests not expected to strictly conform with snake_case.

# Any flake8 disabled violations handled via per-file-ignores on .flake8


@pytest.fixture(scope="class")
def csv_dir() -> abc.Iterator[Path]:
    """resources/csv directory"""
    path = RESOURCES_PATH / "csv"
    assert path.is_dir()
    yield path


@pytest.fixture(scope="class")
def symbols() -> abc.Iterator[list[str]]:
    """Selection of symbols represented in resources csv files."""
    yield ["MSFT", "AZN.L", "9988.HK"]


@pytest.fixture(scope="class")
def calendars(symbols) -> abc.Iterator[dict[str, str]]:
    """Mapping of `symbols` to calendar names."""
    cals = {
        "MSFT": xcals.get_calendar("XNYS", start="1986"),
        "AZN.L": xcals.get_calendar("XLON", start="1986"),
        "9988.HK": xcals.get_calendar("XHKG", start="1986"),
    }
    assert set(symbols) == set(cals)
    yield cals


@pytest.fixture(scope="class")
def prices(csv_dir, symbols, calendars) -> abc.Iterator[PricesCsv]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prices = PricesCsv(csv_dir, symbols, calendars, verbose=False)

    assert prices.limits == {
        prices.bis.T1: (
            pd.Timestamp("2022-05-18 16:00", tz="UTC"),
            pd.Timestamp("2022-06-15 16:30", tz="UTC"),
        ),
        prices.bis.T2: (
            pd.Timestamp("2022-05-05 16:00", tz="UTC"),
            pd.Timestamp("2022-06-15 16:30", tz="UTC"),
        ),
        prices.bis.T5: (
            pd.Timestamp("2022-04-18 16:00", tz="UTC"),
            pd.Timestamp("2022-06-15 16:00", tz="UTC"),
        ),
        prices.bis.D1: (pd.Timestamp("1986-03-13"), pd.Timestamp("2022-06-15")),
    }

    yield prices


@pytest.fixture(scope="class")
def loll(prices) -> abc.Iterator[pd.Timestamp]:
    """Timestamp that lies significantly to Left of leftmost left intraday limit."""
    ts = pd.Timestamp("2022-03-15 09:52", tz=prices.tz_default)
    assert all(
        ts < lim[0] for intrvl, lim in prices.limits.items() if intrvl.is_intraday
    )
    yield ts


@pytest.fixture(scope="class")
def rorl(prices) -> abc.Iterator[pd.Timestamp]:
    """Timestamp that lies significantly to right of rightmost right intraday limit."""
    ts = pd.Timestamp("2022-07-13 09:53", tz=prices.tz_default)
    assert all(
        ts > lim[1] for intrvl, lim in prices.limits.items() if intrvl.is_intraday
    )
    yield ts


@pytest.fixture(scope="class")
def ts_T5(prices) -> abc.Iterator[tuple[pd.Timestamp, ...]]:
    """Timestamps available at an interval >= T5."""
    tss = (
        pd.Timestamp("2022-04-27 09:36", tz=prices.tz_default),
        pd.Timestamp("2022-04-27 09:37", tz=prices.tz_default),
    )
    assert all(
        prices.limits[prices.bis.T5][0] < ts < prices.limits[prices.bis.T2][0]
        for ts in tss
    )
    yield tss


@pytest.fixture(scope="class")
def ts_T2(prices) -> abc.Iterator[tuple[pd.Timestamp, ...]]:
    """Timestamps available at an interval >= T2."""
    tss = (
        pd.Timestamp("2022-05-11 09:35", tz=prices.tz_default),
        pd.Timestamp("2022-05-11 09:36", tz=prices.tz_default),
        pd.Timestamp("2022-05-11 09:37", tz=prices.tz_default),
    )
    assert all(
        prices.limits[prices.bis.T2][0] < ts < prices.limits[prices.bis.T1][0]
        for ts in tss
    )
    yield tss


@pytest.fixture(scope="class")
def ts_T1(prices) -> abc.Iterator[tuple[pd.Timestamp, ...]]:
    """Timestamps available at an interval >= T1."""
    tss = (
        pd.Timestamp("2022-05-25 09:35", tz=prices.tz_default),
        pd.Timestamp("2022-05-25 09:36", tz=prices.tz_default),
        pd.Timestamp("2022-05-25 09:37", tz=prices.tz_default),
    )
    assert all(
        prices.limits[prices.bis.T1][0] < ts < prices.limits[prices.bis.T1][1]
        for ts in tss
    )
    yield tss


@pytest.fixture(scope="class")
def days_to_T2(prices, loll) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of dflt calendar days from loll to a ts available at >= 2T."""
    days = 40
    target = loll + prices.calendar_default.day * days
    assert prices.limits[prices.bis.T2][0] < target < prices.limits[prices.bis.T1][0]
    yield days, target


@pytest.fixture(scope="class")
def days_back_to_T2(prices, rorl) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of dflt calendar days from rorl to a ts available at >= 2T."""
    days = 40
    target = rorl - prices.calendar_default.day * days
    assert prices.limits[prices.bis.T2][0] < target < prices.limits[prices.bis.T1][0]
    yield days, target


@pytest.fixture(scope="class")
def days_to_T5(prices, loll) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of dflt calendar days from loll to a ts available at >= 5T."""
    days = 30
    target = loll + prices.calendar_default.day * days
    assert prices.limits[prices.bis.T5][0] < target < prices.limits[prices.bis.T2][0]
    yield days, target


@pytest.fixture(scope="class")
def days_back_to_T5(prices, rorl) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of dflt calendar days from rorl to a ts available at >= 5T."""
    days = 50
    target = rorl - prices.calendar_default.day * days
    assert prices.limits[prices.bis.T5][0] < target < prices.limits[prices.bis.T2][0]
    yield days, target


@pytest.fixture(scope="class")
def loll_daily(prices) -> abc.Iterator[pd.Timestamp]:
    """Timestamp that lies significantly to left of leftmost left daily limit."""
    ts = pd.Timestamp("1986-02-28")
    assert prices.calendar_default.first_session < ts < prices.limit_daily
    yield ts


@pytest.fixture(scope="class")
def rorl_daily(prices) -> abc.Iterator[pd.Timestamp]:
    """Timestamp that lies significantly to right of rightmost right daily limit."""
    ts = pd.Timestamp("2022-07-13")
    assert prices.limits[prices.bis.D1][1] < ts < prices.calendar_default.last_session
    yield ts


@pytest.fixture(scope="class")
def days_to_D1(prices, loll_daily) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of dflt calendar days from loll to a ts available at 1D."""
    days = 70
    target = loll_daily + prices.calendar_default.day * (days - 1)
    assert prices.limit_daily < target < prices.limits[prices.bis.D1][1]
    yield days, target


@pytest.fixture(scope="class")
def days_back_to_D1(prices, rorl_daily) -> abc.Iterator[tuple[int, pd.Timestamp]]:
    """Number of calendar days from rorl to a ts available at 1D."""
    days = 70
    target = rorl_daily - prices.calendar_default.day * (days - 1)
    assert prices.limit_daily < target < prices.limits[prices.bis.D1][1]
    yield days, target


@pytest.fixture
def priorities() -> abc.Iterator[list[str]]:
    yield [p.value for p in Priority]


@pytest.fixture
def stricts() -> abc.Iterator[list[bool]]:
    yield [True, False]


@pytest.fixture
def intrvls() -> abc.Iterator[tuple[None, str]]:
    yield (None, "5min")


@pytest.fixture
def intrvls_dmo() -> abc.Iterator[tuple[None, str, str]]:
    yield (None, "1D", "1M")


def check(
    prices: PricesCsv,
    df: pd.DataFrame,
    interval: intervals.BI,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
):
    """Check a DataFrame as expected"""
    if start is None:
        start = prices.limits[interval][0]
    if end is None:
        end = prices.limits[interval][1]
    assert isinstance(df, pd.DataFrame)
    assert df.pt.interval == interval
    assert df.index[0].left == start
    assert df.index[-1].right == end


def check_daily(
    prices: PricesCsv,
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
):
    """Check a DataFrame as expected"""
    if start is None:
        start = prices.limits[prices.bis.D1][0]
    if end is None:
        end = prices.limits[prices.bis.D1][1]
    assert isinstance(df, pd.DataFrame)
    assert df.pt.interval == prices.bis.D1
    assert df.index[0] == start
    assert df.index[-1] == end


def check_monthly(
    prices, df, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
):
    if start is None:
        start = pd.tseries.frequencies.to_offset("1MS").rollforward(prices.limit_daily)
    if end is None:
        right_limit = prices.limits[prices.bis.D1][1]
        end = pd.tseries.frequencies.to_offset("1MS").rollback(right_limit)
    assert isinstance(df, pd.DataFrame)
    assert df.pt.interval is intervals.DOInterval.M1
    assert df.index[0].left == start
    assert df.index[-1].right == end


class TestStartEnd:
    """Tests when defining period with `start` and `end` parameters."""

    def test_start_loll_end_ok(
        self,
        prices,
        loll,
        loll_daily,
        ts_T5,
        ts_T2,
        ts_T1,
        priorities,
        stricts,
        one_min,
        intrvls_dmo,
    ):
        """Test with start left of left limit, end within limits.

        Also, initial verifcations check data is returned at expected
        intervals.
        """
        # end can be served by T5 only
        ends_T5 = ts_T5
        expected_end_T5 = ends_T5[0] - one_min

        for end, priority in itertools.product(ends_T5, priorities):
            # verify all same as data only available at 5T
            df = prices.get(start=loll, end=end, priority=priority, strict=False)
            check(prices, df, prices.bis.T5, end=expected_end_T5)

        # end can be served by T5 or T2
        ends_T2 = ts_T2
        for priority in priorities:
            df = prices.get(start=loll, end=ends_T2[0], priority=priority, strict=False)
            check(prices, df, prices.bis.T5, end=ends_T2[0])

        for end in ends_T2[1:]:
            df = prices.get(start=loll, end=end, priority="period", strict=False)
            check(prices, df, prices.bis.T5, end=ends_T2[0])

            df = prices.get(start=loll, end=end, priority="end", strict=False)
            check(prices, df, prices.bis.T2, end=ends_T2[1])

        # end can be served by T5, T2 or T1
        ends_T1 = ts_T1
        for priority in priorities:
            df = prices.get(start=loll, end=ends_T1[0], priority=priority, strict=False)
            check(prices, df, prices.bis.T5, end=ends_T1[0])

        for end in ends_T1[1:]:
            df = prices.get(start=loll, end=end, priority="period", strict=False)
            check(prices, df, prices.bis.T5, end=ends_T1[0])

        df = prices.get(start=loll, end=ends_T1[1], priority="end", strict=False)
        check(prices, df, prices.bis.T2, end=ends_T1[1])

        df = prices.get(start=loll, end=ends_T1[2], priority="end", strict=False)
        check(prices, df, prices.bis.T1, end=ends_T1[2])

        # verify as required at the edge

        end = ends_T5[0]
        one_min = pd.Timedelta(1, "min")
        limit_start_5T, _ = prices.limits[prices.bis.T5]
        start_ool = limit_start_5T - prices.bis.T5
        start_valid = start_ool + one_min

        # verify returns on latest start that will resolve to limit
        for strict, priority in itertools.product(stricts, priorities):
            df = prices.get(
                start=start_valid, end=end, priority=priority, strict=strict
            )
            check(prices, df, prices.bis.T5, end=expected_end_T5)
            # check when explicitly defining interval
            df = prices.get(
                "5min", start=start_valid, end=end, priority=priority, strict=strict
            )
            check(prices, df, prices.bis.T5, end=expected_end_T5)

        # verify raises error on latest start that resolves prior to limit when strict True
        match = re.escape(
            f"Full period not available at any synchronised intraday base interval. The following base intervals could represent the end indice with the greatest possible accuracy although have insufficient data available to cover the full period:\n\t[<BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), Timestamp('2022-04-27 13:35:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), 'end': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'add_a_row': False}}.\nData that can express the period end with the greatest possible accuracy is available from 2022-04-18 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period.\nAlternatively, consider creating a composite table (pass `composite` as True) or passing `priority` as 'period'."
        )
        with pytest.raises(errors.LastIndiceInaccurateError, match=match):
            prices.get(start=start_ool, end=end, priority="end", strict=True)

        match = re.escape(
            "Given the parameters receieved, the interval was inferred as intraday although the request can only be met with daily data. To return daily prices pass `interval` as a daily interval, for example '1D'.\nNB. The interval will only be inferred as daily if `end` and `start` are defined (if passed) as sessions (timezone naive and with time component as 00:00) and any duration is defined in terms of either `days` or `weeks`, `months` and `years`. Also, if both `start` and `end` are passed then the distance between them should be no less than 6 sessions.\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), 'end': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'add_a_row': False}."
        )
        with pytest.raises(errors.PricesUnavailableInferredIntervalError, match=match):
            # PricesUnavailableInferredIntervalError as could be met from daily
            prices.get(start=start_ool, end=end, priority="period", strict=True)

        # verify same for when explicitly defining interval
        match = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), Timestamp('2022-04-27 13:35:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), 'end': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority in priorities:
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    "5min", start=start_ool, end=end, priority=priority, strict=True
                )

        # verify doesn't raise when strict False
        for priority in priorities:
            df = prices.get(start=start_ool, end=end, priority=priority, strict=False)
            check(prices, df, prices.bis.T5, end=expected_end_T5)
            # check when explicitly defining interval
            df = prices.get(
                "5min", start=start_ool, end=end, priority=priority, strict=False
            )
            check(prices, df, prices.bis.T5, end=expected_end_T5)

        # test daily and monthly interval
        start_valid_D1 = prices.limit_daily
        end = start_valid_D1 + prices.calendar_default.day * 70
        start_ool_D1 = start_valid_D1 - pd.Timedelta(1, "D")

        end_exp_mo = pd.tseries.frequencies.to_offset("1MS").rollback(end)
        for intrvl, strict in itertools.product(intrvls_dmo, stricts):
            df = prices.get(intrvl, start_valid_D1, end, strict=strict)
            if intrvl == "1M":
                check_monthly(prices, df, end=end_exp_mo)
            else:
                check_daily(prices, df, end=end)

        matches = [
            re.escape(
                "Prices unavailable as start evaluates to 1986-03-12 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        ]
        matches.append(
            re.escape(
                "Prices unavailable as start evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        )
        matches.append(
            re.escape(
                "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        )
        for priority, start, intrvl in itertools.product(
            priorities, [start_ool_D1, loll_daily], intrvls_dmo
        ):
            df = prices.get(
                intrvl, start=start, end=end, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df, end=end_exp_mo)
                if start == start_ool_D1:
                    continue
                match = matches[2]
            else:
                check_daily(prices, df, end=end)
                match = matches[0] if start == start_ool_D1 else matches[1]
            with pytest.raises(errors.StartTooEarlyError, match=match):
                prices.get(intrvl, start=start, end=end, priority=priority, strict=True)

    def test_start_ok_end_rorl(
        self,
        prices,
        rorl,
        ts_T5,
        ts_T2,
        ts_T1,
        priorities,
        stricts,
        intrvls,
        intrvls_dmo,
        one_min,
        one_day,
    ):
        """Test with start within limits, end right of right limit."""
        starts_T5 = ts_T5
        starts_T2 = ts_T2
        starts_T1 = ts_T1
        all_starts = tuple(itertools.chain(starts_T1, starts_T2, starts_T5))

        for start, priority in itertools.product(all_starts, priorities):
            # as neither period nor end can be met return will be based on highest interval
            df = prices.get(start=start, end=rorl, priority=priority, strict=False)
            mod = pd.Timedelta((start.time().minute % 5), "min")
            expected_start = start if not mod else start + (prices.bis.T5 - mod)
            check(prices, df, prices.bis.T5, start=expected_start)
            # verify when explicitly defining interval
            df = prices.get(
                "5min", start=start, end=rorl, priority=priority, strict=False
            )
            check(prices, df, prices.bis.T5, start=expected_start)

        # check when define lower interval that returns at that interval
        for start, priority in itertools.product(starts_T2, priorities):
            df = prices.get(
                "2min", start=start, end=rorl, priority=priority, strict=False
            )
            mod = pd.Timedelta((start.time().minute % 2), "min")
            expected_start = start if not mod else start + (prices.bis.T2 - mod)
            check(prices, df, prices.bis.T2, start=expected_start)

        # verify as required at the edge
        start = starts_T5[0]
        expected_start_T5 = pd.Timestamp("2022-04-27 09:40", tz=prices.tz_default)
        _, limit_end_5T = prices.limits[prices.bis.T5]
        end_valid = limit_end_5T

        # verify returns on latest valid end
        for strict, priority in itertools.product(stricts, priorities):
            df = prices.get(
                start=start, end=end_valid, priority=priority, strict=strict
            )
            check(prices, df, prices.bis.T5, start=expected_start_T5)
            # check when explicitly defining interval
            df = prices.get(
                "5min", start=start, end=end_valid, priority=priority, strict=strict
            )
            check(prices, df, prices.bis.T5, start=expected_start_T5)

        end_ool_1 = end_valid + one_min
        end_ool_5 = end_valid + one_min * 5

        # verify raises as expected when strict and end earliest value later than limit

        # will raise when priority 'end' as can hit end with T1 data, but T1 data not available to
        # cover start of period.
        match = re.escape(
            f"Full period available at the following synchronised intraday base intervals although these do not allow for representing the end indice with the greatest possible accuracy:\n\t[<BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe following base intervals could represent the end indice with the greatest possible accuracy although have insufficient data available to cover the full period:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>].\nThe period over which data is available at {prices.bis.T1} is (Timestamp('2022-05-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:30:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), Timestamp('2022-06-15 16:01:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:01:00+0000', tz='UTC'), 'add_a_row': False}}.\nData that can express the period end with the greatest possible accuracy is available from 2022-05-18 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period.\nAlternatively, consider creating a composite table (pass `composite` as True) or passing `priority` as 'period'."
        )
        with pytest.raises(errors.LastIndiceInaccurateError, match=match):
            prices.get(start=start, end=end_ool_1, priority="end", strict=True)

        # will return as although parameters cannot be fulfilled by T1 it can by T5 as end will
        # not evaluate beyond T5 limit until passed as limit + 5 minutes (until which just
        # evaluates as the limit).
        df = prices.get(start=start, end=end_ool_1, priority="period", strict=True)
        check(prices, df, prices.bis.T5, start=expected_start_T5)

        # might as well verify a composite whilst here...
        # NB this works because there's an extra half hour of data available for T1 relative to T5
        df = prices.get(
            start=start, end=end_ool_1, priority="end", composite=True, strict=True
        )
        assert isinstance(df, pd.DataFrame)
        assert df.index[0].left == expected_start_T5
        assert df.index[-1].right == end_ool_1

        # but not going to pass at 5min beyond the limit...
        match = re.escape(
            "Given the parameters receieved, the interval was inferred as intraday although the request can only be met with daily data. To return daily prices pass `interval` as a daily interval, for example '1D'.\nNB. The interval will only be inferred as daily if `end` and `start` are defined (if passed) as sessions (timezone naive and with time component as 00:00) and any duration is defined in terms of either `days` or `weeks`, `months` and `years`. Also, if both `start` and `end` are passed then the distance between them should be no less than 6 sessions.\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:05:00+0000', tz='UTC'), 'add_a_row': False}."
        )
        with pytest.raises(errors.PricesUnavailableInferredIntervalError, match=match):
            prices.get(start=start, end=end_ool_5, priority="period", strict=True)

        # verify same for when explicitly define interval
        match = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-27 13:40:00+0000', tz='UTC'), Timestamp('2022-06-15 16:05:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-27 13:36:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:05:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from the start of the requested period through to 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority in priorities:
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    "5min", start=start, end=end_ool_5, priority=priority, strict=True
                )

        # verify doesn't raise when False

        df = prices.get(start=start, end=end_ool_1, priority="end", strict=False)
        assert isinstance(df, pd.DataFrame)
        assert df.pt.interval == prices.bis.T1
        assert df.index[0].left == prices.limits[prices.bis.T1][0]
        assert df.index[-1].right == end_ool_1

        # will be fulfilled by 1T data within confined of what's available
        df = prices.get(
            "5min", start=start, end=end_ool_5, priority="end", strict=False
        )
        assert isinstance(df, pd.DataFrame)
        assert df.pt.interval == prices.bis.T5
        assert df.index[0].left == prices.limits[prices.bis.T1][0]
        assert df.index[-1].right == end_ool_5

        for intrvl in intrvls:
            df = prices.get(intrvl, start, end_ool_5, priority="period", strict=False)
            check(prices, df, prices.bis.T5, start=expected_start_T5)

        # test for daily and monthly interval
        end_valid_D1 = prices.limits[prices.bis.D1][1]
        start = end_valid_D1 - prices.calendar_default.day * 70
        end_ool_D1 = end_valid_D1 + pd.Timedelta(1, "D")

        start_exp_mo = pd.tseries.frequencies.to_offset("1MS").rollforward(start)
        for intrvl, strict, priority in itertools.product(
            intrvls_dmo, stricts, priorities
        ):
            df = prices.get(
                intrvl, start, end_valid_D1, priority=priority, strict=strict
            )
            if intrvl == "1M":
                check_monthly(prices, df, start_exp_mo)
            else:
                check_daily(prices, df, start)

        match = re.escape(
            f"`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `end` evaluates to 2022-06-16."
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(intrvl, start, end_ool_D1, priority=priority, strict=False)
            if intrvl == "1M":
                check_monthly(prices, df, start_exp_mo)
                continue
            else:
                check_daily(prices, df, start)
            with pytest.raises(errors.EndTooLateError, match=match):
                prices.get(
                    intrvl, start=start, end=end_ool_D1, priority=priority, strict=True
                )

        # last day of month is not valid as would require the full month to be included which in
        # turn would include data later than the last session available
        end_ool_mo = (
            pd.tseries.frequencies.to_offset("1MS").rollforward(end_valid_D1) - one_day
        )
        # although one day earlier and the end will evaluate to the end of the prior month, i.e.
        # within the available data
        end_valid_mo = end_ool_mo - one_day
        for priority in priorities:
            df = prices.get("1M", start, end_valid_mo, priority=priority, strict=True)
            check_monthly(prices, df, start_exp_mo)

        match = re.escape(
            "`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval 'DOInterval.M1' is 2022-05-31, although `end` evaluates to 2022-06-30."
        )
        for priority in priorities:
            with pytest.raises(errors.EndTooLateError, match=match):
                prices.get("1M", start, end_ool_mo, priority=priority, strict=True)

    def test_start_loll_end_rorl(
        self,
        prices,
        priorities,
        stricts,
        one_min,
        one_day,
        intrvls,
        intrvls_dmo,
        loll_daily,
        rorl_daily,
    ):
        """Test with start within limits, end right of right limit."""
        _start_limit_T5, _end_limit_T5 = prices.limits[prices.bis.T5]
        start_limit_T5, end_limit_T5 = (
            _start_limit_T5 - 4 * one_min,
            _end_limit_T5 + 4 * one_min,
        )
        start_ool_T5, end_ool_T5 = start_limit_T5 - one_min, end_limit_T5 + one_min

        # will reutrn at 5T given priority
        for strict in stricts:
            df = prices.get(
                start=start_limit_T5, end=end_limit_T5, priority="period", strict=strict
            )
            check(prices, df, prices.bis.T5, _start_limit_T5, _end_limit_T5)

        # raises as 'end' priority can be achieved with T2 (for which a later data is available)
        match = re.escape(
            f"Full period available at the following synchronised intraday base intervals although these do not allow for representing the end indice with the greatest possible accuracy:\n\t[<BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe following base intervals could represent the end indice with the greatest possible accuracy although have insufficient data available to cover the full period:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>].\nThe period over which data is available at {prices.bis.T2} is (Timestamp('2022-05-05 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:30:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-18 15:56:00+0000', tz='UTC'), Timestamp('2022-06-15 16:04:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:56:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:04:00+0000', tz='UTC'), 'add_a_row': False}}.\nData that can express the period end with the greatest possible accuracy is available from 2022-05-05 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period.\nAlternatively, consider creating a composite table (pass `composite` as True) or passing `priority` as 'period'."
        )
        with pytest.raises(errors.LastIndiceInaccurateError, match=match):
            prices.get(
                start=start_limit_T5, end=end_limit_T5, priority="end", strict=True
            )

        # will return using T2 data which can meet the end, but not the start
        df = prices.get(
            start=start_limit_T5, end=end_limit_T5, priority="end", strict=False
        )
        check(prices, df, prices.bis.T2, end=end_limit_T5)

        # test other side of limits

        # strict True
        match = re.escape(
            "Given the parameters receieved, the interval was inferred as intraday although the request can only be met with daily data. To return daily prices pass `interval` as a daily interval, for example '1D'.\nNB. The interval will only be inferred as daily if `end` and `start` are defined (if passed) as sessions (timezone naive and with time component as 00:00) and any duration is defined in terms of either `days` or `weeks`, `months` and `years`. Also, if both `start` and `end` are passed then the distance between them should be no less than 6 sessions.\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:05:00+0000', tz='UTC'), 'add_a_row': False}."
        )
        with pytest.raises(errors.PricesUnavailableInferredIntervalError, match=match):
            prices.get(
                start=start_ool_T5, end=end_ool_T5, priority="period", strict=True
            )
        match = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), Timestamp('2022-06-15 16:05:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-04-18 15:55:00+0000', tz='UTC'), 'end': Timestamp('2022-06-15 16:05:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
            prices.get(
                "5min",
                start=start_ool_T5,
                end=end_ool_T5,
                priority="period",
                strict=True,
            )
        with pytest.raises(errors.LastIndiceInaccurateError):  # as prior reasoning
            prices.get(start=start_ool_T5, end=end_ool_T5, priority="end", strict=True)
        with pytest.raises(errors.PricesIntradayUnavailableError):
            prices.get(
                "5min", start=start_ool_T5, end=end_ool_T5, priority="end", strict=True
            )

        # strict False
        for intrvl in intrvls:
            # gets what it can
            df = prices.get(
                intrvl, start_ool_T5, end_ool_T5, priority="period", strict=False
            )
            check(prices, df, prices.bis.T5)

        # will meet with 1T data that can meet end
        df = prices.get(None, start_ool_T5, end_ool_T5, priority="end", strict=False)
        check(prices, df, prices.bis.T1, end=end_ool_T5)

        # will meet with downsampled 1T data that can meet end
        df = prices.get("5min", start_ool_T5, end_ool_T5, priority="end", strict=False)
        check(prices, df, prices.bis.T5, prices.limits[prices.bis.T1][0], end_ool_T5)

        # going way out
        for intrvl, priority in itertools.product(intrvls, priorities):
            # gets what it can
            df = prices.get(
                intrvl,
                start_ool_T5 - one_day * 5,
                end_ool_T5 + one_day * 5,
                priority=priority,
                strict=False,
            )
            check(prices, df, prices.bis.T5)

        # check daily and monthly intervals
        start_valid_D1, end_valid_D1 = prices.limits[prices.bis.D1]
        start_ool_D1, end_ool_D1 = start_valid_D1 - one_day, end_valid_D1 + one_day

        # on limit daily
        for strict, priority, intrvl in itertools.product(
            stricts, priorities, intrvls_dmo
        ):
            df = prices.get(
                intrvl, start_valid_D1, end_valid_D1, priority=priority, strict=strict
            )
            if intrvl == "1M":
                check_monthly(prices, df)
            else:
                check_daily(prices, df)

        # on limit monthly

        # first day of month is not valid as would require the full month to be included which in
        # turn would include data earlier than the first session available
        start_ool_mo = pd.tseries.frequencies.to_offset("1MS").rollback(start_valid_D1)
        # although one day later and the start will evaluate to the start of the next month, i.e.
        # within the available data
        start_valid_mo = start_ool_mo + one_day

        # same principle for end
        end_ool_mo = (
            pd.tseries.frequencies.to_offset("1MS").rollforward(end_valid_D1) - one_day
        )
        end_valid_mo = end_ool_mo - one_day

        for priority, strict in itertools.product(priorities, stricts):
            df = prices.get(
                "1M", start_valid_mo, end_valid_mo, priority=priority, strict=strict
            )
            check_monthly(prices, df)

        # outside of limits

        match_start = re.escape(
            """Prices unavailable as start evaluates to 1986-03-12 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."""
        )
        match_end = re.escape(
            f"`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `end` evaluates to 2022-06-16."
        )
        match_start_mo = re.escape(
            "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
        )
        match_end_mo = re.escape(
            "`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval 'DOInterval.M1' is 2022-05-31, although `end` evaluates to 2022-06-30."
        )

        for priority, intrvl in itertools.product(priorities, intrvls_dmo):
            # start outside, end inside
            df = prices.get(
                intrvl, start_ool_D1, end_valid_D1, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df)
                df = prices.get(
                    "1M", start_ool_mo, end_valid_mo, priority=priority, strict=False
                )
                check_monthly(prices, df)
                with pytest.raises(errors.StartTooEarlyError, match=match_start_mo):
                    prices.get(
                        "1M", start_ool_mo, end_valid_mo, priority=priority, strict=True
                    )
            else:
                check_daily(prices, df)
                with pytest.raises(errors.StartTooEarlyError, match=match_start):
                    prices.get(
                        intrvl,
                        start_ool_D1,
                        end_valid_D1,
                        priority=priority,
                        strict=True,
                    )

            # start inside, end outside
            df = prices.get(
                intrvl, start_valid_D1, end_ool_D1, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df)
                df = prices.get(
                    "1M", start_valid_mo, end_ool_mo, priority=priority, strict=False
                )
                check_monthly(prices, df)
                with pytest.raises(errors.EndTooLateError, match=match_end_mo):
                    prices.get(
                        intrvl,
                        start_valid_mo,
                        end_ool_mo,
                        priority=priority,
                        strict=True,
                    )
            else:
                check_daily(prices, df)
                with pytest.raises(errors.EndTooLateError, match=match_end):
                    prices.get(
                        intrvl,
                        start_valid_D1,
                        end_ool_D1,
                        priority=priority,
                        strict=True,
                    )

            # start outside, end outside
            df = prices.get(
                intrvl, start_ool_D1, end_ool_D1, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df)
                df = prices.get(
                    "1M", start_ool_mo, end_ool_mo, priority=priority, strict=False
                )
                check_monthly(prices, df)
                with pytest.raises(errors.StartTooEarlyError, match=match_start_mo):
                    prices.get(
                        "1M", start_ool_mo, end_ool_mo, priority=priority, strict=True
                    )

            else:
                check_daily(prices, df)
                with pytest.raises(errors.StartTooEarlyError, match=match_start):
                    prices.get(
                        intrvl, start_ool_D1, end_ool_D1, priority=priority, strict=True
                    )
                # further out
                df = prices.get(
                    intrvl, loll_daily, rorl_daily, priority=priority, strict=False
                )
                check_daily(prices, df)
                with pytest.raises(errors.StartTooEarlyError):
                    prices.get(
                        intrvl, loll_daily, rorl_daily, priority=priority, strict=True
                    )


class TestDuration:
    """Tests when defining period with `start` or `end` and duration parameters."""

    def test_defined_ool_evaluated_ok(
        self,
        prices,
        priorities,
        loll,
        rorl,
        loll_daily,
        rorl_daily,
        intrvls,
        intrvls_dmo,
        days_to_T2,
        days_to_T5,
        days_back_to_T2,
        days_back_to_T5,
        days_to_D1,
        days_back_to_D1,
    ):
        """Defined extreme outside limts, evaluted extreme in limits."""
        # start defined ool
        # strict False
        days, end = days_to_T2
        df = prices.get(start=loll, days=days, priority="end", strict=False)
        check(prices, df, prices.bis.T2, end=end)

        mod = pd.Timedelta((end.time().minute % 5), "min")
        expected_end = end + (prices.bis.T5 - mod)
        for intrvl, priority in itertools.product(intrvls, priorities):
            if intrvl is None and priority == "end":
                continue  # already covered above
            df = prices.get(
                intrvl, start=loll, days=days, priority=priority, strict=False
            )
            # end 09:55 as start will have been advanced from 09:52 to 09:55
            check(prices, df, prices.bis.T5, end=expected_end)

        # strict True
        match = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-15 13:55:00+0000', tz='UTC'), Timestamp('2022-04-27 13:55:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 30, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority in priorities:
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    "5min",
                    start=loll,
                    days=days_to_T5[0],
                    priority=priority,
                    strict=True,
                )

        # end defined ool
        # strict False

        # priority not important as cannot meet end, will be returned at highest
        # avaialble intraday interval, i.e. 5T. end will first be evaluated from passed
        # value such that aligns with 5T. start will then be evaluated off this aligned
        # value, such that falls earlier than what would be the 2T target
        days, start = days_back_to_T2
        mod = pd.Timedelta((end.time().minute % 5), "min")
        expected_start = start - (prices.bis.T5 - mod)
        for intrvl, priority in itertools.product(intrvls, priorities):
            df = prices.get(
                intrvl, end=rorl, days=days, priority=priority, strict=False
            )
            check(prices, df, prices.bis.T5, start=expected_start)

        # strict True
        matches = [
            re.escape(
                f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-29 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 50, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from the start of the requested period through to 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
            )
        ]
        matches.append(
            re.escape(
                f"Data is unavailable at a sufficiently low base interval to evaluate prices at an inferred interval anchored 'Anchor.OPEN'.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-04-29 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 50, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from the start of the requested period through to 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
            )
        )
        for intrvl, priority in itertools.product(intrvls, priorities):
            match = matches[1] if intrvl is None else matches[0]
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    intrvl,
                    end=rorl,
                    days=days_back_to_T5[0],
                    priority=priority,
                    strict=True,
                )

        # test daily and monthly intervals
        # start defined ool
        days, end = days_to_D1
        expected_end_mo = pd.tseries.frequencies.to_offset("1MS").rollback(end)
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(
                intrvl, start=loll_daily, days=days, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df, end=expected_end_mo)
            else:
                check_daily(prices, df, end=end)

        matches = [
            re.escape(
                "Prices unavailable as start evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        ]
        matches.append(
            re.escape(
                "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = matches[1] if intrvl == "1M" else matches[0]
            with pytest.raises(errors.StartTooEarlyError, match=match):
                prices.get(
                    intrvl, start=loll_daily, days=days, priority=priority, strict=True
                )

        # end defined ool
        days, start = days_back_to_D1
        expected_start_mo = pd.tseries.frequencies.to_offset("1MS").rollforward(start)
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(
                intrvl, end=rorl_daily, days=days, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df, start=expected_start_mo)
            else:
                check_daily(prices, df, start=start)

        match_daily = re.escape(
            f"`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `end` evaluates to 2022-07-13."
        )
        match_mo = re.escape(
            "`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval 'DOInterval.M1' is 2022-05-31, although `end` evaluates to 2022-06-30."
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = match_mo if intrvl == "1M" else match_daily
            with pytest.raises(errors.EndTooLateError, match=match):
                prices.get(
                    intrvl, end=rorl_daily, days=days, priority=priority, strict=True
                )

    def test_defined_ok_evaluated_ool(
        self,
        prices,
        priorities,
        intrvls,
        intrvls_dmo,
        days_to_T2,
        days_to_T5,
        days_back_to_T2,
        days_to_D1,
        days_back_to_D1,
    ):
        """Defined extreme within limts, evaluted extreme outside limits."""
        # end defined within limits
        # strict False
        days, end = days_to_T2
        df = prices.get(end=end, days=days, priority="end", strict=False)
        check(prices, df, prices.bis.T2, end=end)

        mod = pd.Timedelta((end.time().minute % 5), "min")
        expected_end = end - mod
        for intrvl, priority in itertools.product(intrvls, priorities):
            if intrvl is None and priority == "end":
                continue  # already covered above
            df = prices.get(intrvl, end=end, days=days, priority=priority, strict=False)
            check(prices, df, prices.bis.T5, end=expected_end)

        # strict True
        days, end = days_to_T5
        match = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-15 13:50:00+0000', tz='UTC'), Timestamp('2022-04-27 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 30, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-04-27 13:52:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through to the end of the requested period. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority in priorities:
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get("5min", end=end, days=days, priority=priority, strict=True)

        # start defined within limits
        # strict True

        # priority not important as cannot meet end, will be returned at highest
        # avaialble intraday interval, i.e. 5T
        days, start = days_back_to_T2
        mod = pd.Timedelta((end.time().minute % 5), "min")
        expected_start = start + mod

        for intrvl, priority in itertools.product(intrvls, priorities):
            df = prices.get(
                intrvl, start=start, days=days, priority=priority, strict=False
            )
            check(prices, df, prices.bis.T5, start=expected_start)

        # strict True
        matches = [
            re.escape(
                f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-05-13 13:55:00+0000', tz='UTC'), Timestamp('2022-07-13 13:55:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 40, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-05-13 13:53:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}}.\nData is available from the start of the requested period through to 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
            )
        ]
        matches.append(
            re.escape(
                f"Data is unavailable at a sufficiently low base interval to evaluate prices at an inferred interval anchored 'Anchor.OPEN'.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-05-13 13:55:00+0000', tz='UTC'), Timestamp('2022-07-13 13:55:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 40, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-05-13 13:53:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}}.\nData is available from the start of the requested period through to 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
            )
        )
        for intrvl, priority in itertools.product(intrvls, priorities):
            match = matches[1] if intrvl is None else matches[0]
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    intrvl,
                    start=start,
                    days=days,
                    priority=priority,
                    strict=True,
                )

        # test daily and monthly intervals
        # end defined within limits
        days, end = days_to_D1
        expected_end_mo = pd.tseries.frequencies.to_offset("1MS").rollback(end)
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(intrvl, end=end, days=days, priority=priority, strict=False)
            if intrvl == "1M":
                check_monthly(prices, df, end=expected_end_mo)
            else:
                check_daily(prices, df, end=end)

        matches = [
            re.escape(
                "Prices unavailable as start evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        ]
        matches.append(
            re.escape(
                "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = matches[1] if intrvl == "1M" else matches[0]
            with pytest.raises(errors.StartTooEarlyError, match=match):
                prices.get(intrvl, end=end, days=days, priority=priority, strict=True)

        # start defined within limits
        days, start = days_back_to_D1
        expected_start_mo = pd.tseries.frequencies.to_offset("1MS").rollforward(start)
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(
                intrvl, start=start, days=days, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df, start=expected_start_mo)
            else:
                check_daily(prices, df, start=start)

        match_daily = re.escape(
            f"`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `end` evaluates to 2022-07-13."
        )
        match_mo = re.escape(
            "`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval 'DOInterval.M1' is 2022-05-31, although `end` evaluates to 2022-06-30."
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = match_mo if intrvl == "1M" else match_daily
            with pytest.raises(errors.EndTooLateError, match=match):
                prices.get(
                    intrvl, start=start, days=days, priority=priority, strict=True
                )

    def test_defined_ool_evaluated_ool(
        self,
        prices,
        priorities,
        loll,
        rorl,
        loll_daily,
        rorl_daily,
        intrvls,
        intrvls_dmo,
    ):
        """Defined extreme outside limts, evaluted extreme outside limits."""
        # define start
        # strict False
        diff = rorl - loll
        weeks = int(diff.total_seconds() // (3600 * 24 * 7))
        # cannot meet end, so interval will always be highest intraday interval available
        for priority, intrvl in itertools.product(priorities, intrvls):
            df = prices.get(intrvl, loll, weeks=weeks, priority=priority, strict=False)
            check(prices, df, prices.bis.T5)

        # strict True
        match_inf = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at an inferred interval anchored 'Anchor.OPEN'.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-15 13:55:00+0000', tz='UTC'), Timestamp('2022-07-12 13:55:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 17, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        match_5T = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-15 13:55:00+0000', tz='UTC'), Timestamp('2022-07-12 13:55:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 17, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority, intrvl in itertools.product(priorities, intrvls):
            match = match_inf if intrvl is None else match_5T
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    df=prices.get(
                        intrvl, loll, weeks=weeks, priority=priority, strict=True
                    )
                )

        # define end
        # strict False
        # cannot meet end, so interval will always be highest intraday interval available
        for priority, intrvl in itertools.product(priorities, intrvls):
            df = prices.get(
                intrvl, end=rorl, weeks=weeks, priority=priority, strict=False
            )
            check(prices, df, prices.bis.T5)

        # strict True
        match_inf = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at an inferred interval anchored 'Anchor.OPEN'.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-16 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 17, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        match_5T = re.escape(
            f"Data is unavailable at a sufficiently low base interval to evaluate prices at interval {TDInterval.T5} anchored 'Anchor.OPEN'.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-16 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 17, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}.\nData is available from 2022-04-18 16:00 UTC through 2022-06-15 16:00 UTC. Consider passing `strict` as False to return prices for this part of the period."
        )
        for priority, intrvl in itertools.product(priorities, intrvls):
            match = match_inf if intrvl is None else match_5T
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    df=prices.get(
                        intrvl, end=rorl, weeks=weeks, priority=priority, strict=True
                    )
                )

        # define start, daily and monthly
        diff = rorl_daily - loll_daily
        weeks = int(diff.total_seconds() // (3600 * 24 * 7))
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(
                intrvl, start=loll_daily, weeks=weeks, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df)
            else:
                check_daily(prices, df)

        matches = [
            re.escape(
                "Prices unavailable as start evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        ]
        matches.append(
            re.escape(
                "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
            )
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = matches[1] if intrvl == "1M" else matches[0]
            with pytest.raises(errors.StartTooEarlyError, match=match):
                prices.get(
                    intrvl,
                    start=loll_daily,
                    weeks=weeks,
                    priority=priority,
                    strict=True,
                )

        # define end, daily and monthly
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            df = prices.get(
                intrvl, end=rorl_daily, weeks=weeks, priority=priority, strict=False
            )
            if intrvl == "1M":
                check_monthly(prices, df)
            else:
                check_daily(prices, df)

        match_daily = re.escape(
            f"`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `end` evaluates to 2022-07-13."
        )
        match_mo = re.escape(
            "`end` cannot evaluate to a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval 'DOInterval.M1' is 2022-05-31, although `end` evaluates to 2022-06-30."
        )
        for intrvl, priority in itertools.product(intrvls_dmo, priorities):
            match = match_mo if intrvl == "1M" else match_daily
            with pytest.raises(errors.EndTooLateError, match=match):
                prices.get(
                    intrvl, end=rorl_daily, weeks=weeks, priority=priority, strict=True
                )


class TestOutsideLimits:
    """Defining period with both start and end outside of limits, on same side."""

    def test_period_defined_to_left_of_available(
        self,
        prices,
        loll,
        loll_daily,
        one_day,
        intrvls,
        intrvls_dmo,
        priorities,
        stricts,
    ):
        # to left of intraday data, although available at daily interval

        # intraday interval
        match = re.escape(
            f"The end of the requested period is earlier than the earliest timestamp at which intraday data is available for any base interval.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-03-15 13:55:00+0000', tz='UTC'), Timestamp('2022-03-16 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': Timestamp('2022-03-16 13:52:00+0000', tz='UTC'), 'add_a_row': False}}."
        )
        for priority, strict in itertools.product(priorities, stricts):
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    "5min",
                    start=loll,
                    end=loll + one_day,
                    priority=priority,
                    strict=strict,
                )
            with pytest.raises(errors.PricesIntradayUnavailableError):
                prices.get("5min", start=loll, days=2, priority=priority, strict=strict)
            with pytest.raises(errors.PricesIntradayUnavailableError):
                prices.get("5min", end=loll, days=2, priority=priority, strict=strict)

        # interval inferred
        match_start = "Given the parameters receieved, the interval was inferred as intraday although the request can only be met with daily data. To return daily prices pass `interval` as a daily interval, for example '1D'.\nNB. The interval will only be inferred as daily if `end` and `start` are defined (if passed) as sessions (timezone naive and with time component as 00:00) and any duration is defined in terms of either `days` or `weeks`, `months` and `years`. Also, if both `start` and `end` are passed then the distance between them should be no less than 6 sessions."
        for priority, strict in itertools.product(priorities, stricts):
            match = re.escape(
                match_start
                + "\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': Timestamp('2022-03-17 13:52:00+0000', tz='UTC'), 'add_a_row': False}."
            )
            with pytest.raises(
                errors.PricesUnavailableInferredIntervalError, match=match
            ):
                prices.get(
                    None,
                    start=loll,
                    end=loll + one_day * 2,
                    priority=priority,
                    strict=strict,
                )
            match = re.escape(
                match_start
                + "\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 2, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'end': None, 'add_a_row': False}."
            )
            with pytest.raises(
                errors.PricesUnavailableInferredIntervalError, match=match
            ):
                prices.get(None, start=loll, days=2, priority=priority, strict=strict)
            match = re.escape(
                match_start
                + "\nPeriod parameters were evaluted as {'minutes': 0, 'hours': 0, 'days': 2, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-03-15 13:52:00+0000', tz='UTC'), 'add_a_row': False}."
            )
            with pytest.raises(
                errors.PricesUnavailableInferredIntervalError, match=match
            ):
                prices.get(None, end=loll, days=2, priority=priority, strict=strict)

        # to left of intraday data and daily data

        # interval intraday and inferred as intraday
        # match = re.escape(
        #     "Prices unavailable as end would resolve to an earlier minute than the earliest minute of calendar 'XNYS'. The calendar's earliest minute is 1986-01-02 14:30 UTC (this bound should coincide with the earliest minute or date for which price data is available)."
        # )
        for intrvl, priority, strict in itertools.product(intrvls, priorities, stricts):
            if intrvl is None:
                match = re.escape(
                    f"The end of the requested period is earlier than the earliest timestamp at which intraday data is available for any base interval.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('1986-02-28 14:30:00+0000', tz='UTC'), Timestamp('1986-02-28 21:00:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('1986-02-28 00:00:00'), 'end': Timestamp('1986-03-02 00:00:00'), 'add_a_row': False}}."
                )
            else:
                match = re.escape(
                    f"The end of the requested period is earlier than the earliest timestamp at which intraday data is available for any base interval.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('1986-02-28 14:30:00+0000', tz='UTC'), Timestamp('1986-02-28 21:00:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 0, 'weeks': 0, 'months': 0, 'years': 0, 'start': Timestamp('1986-02-28 00:00:00'), 'end': Timestamp('1986-03-02 00:00:00'), 'add_a_row': False}}."
                )
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(
                    intrvl,
                    start=loll_daily,
                    end=loll_daily + one_day * 2,
                    priority=priority,
                    strict=strict,
                )
            with pytest.raises(errors.PricesIntradayUnavailableError):
                prices.get(
                    intrvl, start=loll_daily, days=2, priority=priority, strict=strict
                )
            with pytest.raises(errors.PricesIntradayUnavailableError):
                prices.get(
                    intrvl, end=loll_daily, days=2, priority=priority, strict=strict
                )

        # intervals daily, monthly and inferred as daily
        # raises StartTooEarlyError if strict, otherwise EndTooEarlyError
        days = 7
        for intrvl, priority, strict in itertools.product(
            intrvls_dmo, priorities, stricts
        ):
            if not strict:
                if intrvl == "1M":
                    match = re.escape(
                        "Prices unavailable as end evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
                    )
                else:
                    match = re.escape(
                        "Prices unavailable as end evaluates to 1986-03-07 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
                    )
                with pytest.raises(errors.EndTooEarlyError, match=match):
                    prices.get(
                        intrvl,
                        loll_daily,
                        loll_daily + one_day * days,
                        priority=priority,
                        strict=strict,
                    )
                with pytest.raises(errors.EndTooEarlyError):
                    prices.get(
                        intrvl, loll_daily, days=days, priority=priority, strict=strict
                    )
                with pytest.raises(errors.EndTooEarlyError):
                    prices.get(
                        intrvl,
                        end=loll_daily,
                        days=days,
                        priority=priority,
                        strict=strict,
                    )

            else:
                if intrvl == "1M":
                    match = re.escape(
                        "Prices unavailable as start evaluates to 1986-03-01 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
                    )
                else:
                    match = re.escape(
                        "Prices unavailable as start evaluates to 1986-02-28 which is earlier than the earliest session for which price data is available. The earliest session for which prices are available is 1986-03-13."
                    )
                with pytest.raises(errors.StartTooEarlyError, match=match):
                    prices.get(
                        intrvl,
                        loll_daily,
                        loll_daily + one_day * days,
                        priority=priority,
                        strict=strict,
                    )
                with pytest.raises(errors.StartTooEarlyError):
                    prices.get(
                        intrvl, loll_daily, days=days, priority=priority, strict=strict
                    )

    def test_period_defined_to_right_of_available(
        self,
        prices,
        rorl,
        rorl_daily,
        one_day,
        intrvls_dmo,
        priorities,
        stricts,
    ):
        # to right of all data

        # intraday interval
        match_stl = re.escape(
            f"`start` cannot be a later time than the latest time for which prices are available.\nThe latest time for which prices are available for interval '{TDInterval.T1}' is 2022-06-15 20:00 UTC, although `start` received as 2022-07-13 13:53 UTC."
        )
        match = re.escape(
            f"The start of the requested period is later than the latest timestamp at which intraday data is available for any base interval.\nBase intervals that are a factor of {TDInterval.T5} and for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-07-11 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 2, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}."
        )
        for priority, strict in itertools.product(priorities, stricts):
            with pytest.raises(errors.StartTooLateError, match=match_stl):
                prices.get(
                    "5min",
                    start=rorl,
                    end=rorl + one_day * 2,
                    priority=priority,
                    strict=strict,
                )
            with pytest.raises(errors.StartTooLateError):
                prices.get("5min", start=rorl, days=2, priority=priority, strict=strict)
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get("5min", end=rorl, days=2, priority=priority, strict=strict)

        # interval inferred as intraday
        match = re.escape(
            f"The start of the requested period is later than the latest timestamp at which intraday data is available for any base interval.\nBase intervals for which timestamps of all calendars are synchronised:\n\t[<BaseInterval.T1: datetime.timedelta(seconds=60)>, <BaseInterval.T2: datetime.timedelta(seconds=120)>, <BaseInterval.T5: datetime.timedelta(seconds=300)>].\nThe period over which data is available at {prices.bis.T5} is (Timestamp('2022-04-18 16:00:00+0000', tz='UTC'), Timestamp('2022-06-15 16:00:00+0000', tz='UTC')), although at this base interval the requested period evaluates to (Timestamp('2022-07-11 13:50:00+0000', tz='UTC'), Timestamp('2022-07-13 13:50:00+0000', tz='UTC')).\nPeriod evaluated from parameters: {{'minutes': 0, 'hours': 0, 'days': 2, 'weeks': 0, 'months': 0, 'years': 0, 'start': None, 'end': Timestamp('2022-07-13 13:53:00+0000', tz='UTC'), 'add_a_row': False}}."
        )
        for priority, strict in itertools.product(priorities, stricts):
            with pytest.raises(errors.StartTooLateError, match=match_stl):
                prices.get(
                    None,
                    start=rorl,
                    end=rorl + one_day * 2,
                    priority=priority,
                    strict=strict,
                )
            with pytest.raises(errors.StartTooLateError):
                prices.get(None, start=rorl, days=2, priority=priority, strict=strict)
            with pytest.raises(errors.PricesIntradayUnavailableError, match=match):
                prices.get(None, end=rorl, days=2, priority=priority, strict=strict)

        # intervals daily, monthly and inferred as daily
        days = 10
        for intrvl, priority, strict in itertools.product(
            intrvls_dmo, priorities, stricts
        ):
            match = re.escape(
                f"`start` cannot be a later date than the latest date for which prices are available.\nThe latest date for which prices are available for interval '{TDInterval.D1}' is 2022-06-15, although `start` received as 2022-07-13."
            )
            with pytest.raises(errors.StartTooLateError, match=match):
                prices.get(
                    intrvl,
                    rorl_daily,
                    rorl_daily + one_day * days,
                    priority=priority,
                    strict=strict,
                )
            with pytest.raises(errors.StartTooLateError):
                prices.get(
                    intrvl, rorl_daily, days=days, priority=priority, strict=strict
                )
            # when duration from end, raises EndTooEarlyError if strict, otherwise StartTooEarlyError
            if not strict:
                with pytest.raises(errors.StartTooLateError):
                    prices.get(
                        intrvl,
                        end=rorl_daily,
                        days=days,
                        priority=priority,
                        strict=strict,
                    )
            else:
                with pytest.raises(errors.EndTooLateError):
                    prices.get(
                        intrvl,
                        end=rorl_daily,
                        days=days,
                        priority=priority,
                        strict=strict,
                    )


def get_symbols_for_calendar(prices: PricesCsv, name: str) -> str | list[str]:
    """Get symbol(s) of a `prices` instance associated with a given calendar `name`."""
    for cal, symbols in prices.calendars_symbols.items():
        if cal.name == name:
            return symbols[0] if len(symbols) == 1 else symbols
    raise ValueError(f"The PricesBaseTst instance has no calendar with name {name}.")


def get_calendar_from_name(prices: PricesCsv, name: str) -> xcals.ExchangeCalendar:
    """Get calendar of a given `name` from a `prices` instance."""
    for cal in prices.calendars_unique:
        if cal.name == name:
            return cal
    raise ValueError(f"The PricesBaseTst instance has no calendar with name {name}.")


class TestPriceAt:
    """Tests for `price_at`."""

    def assertions(
        self,
        prices: PricesCsv,
        df: pd.DataFrame,
        indice: pd.Timestamp,
        values: dict[str, tuple[pd.Timestamp, typing.Literal["open", "close"]]],
        tz: ZoneInfo = UTC,
    ):
        expected_columns = pd.Index(prices.symbols, name="symbol")
        assert_index_equal(expected_columns, df.columns, check_order=False)
        assert len(df) == 1
        row = df.iloc[0]
        assert row.name == indice
        assert df.index.tz is tz
        for s, v in values.items():
            assert df[s].iloc[0] == v

    def test_price_at(
        self,
        prices,
        one_day,
        one_min,
    ):
        """Tests for `price_at` at right limit."""
        xnys = prices.calendar_default
        xlon = get_calendar_from_name(prices, "XLON")
        xhkg = get_calendar_from_name(prices, "XHKG")
        symb_xnys = prices.lead_symbol_default
        symb_xlon = get_symbols_for_calendar(prices, "XLON")
        symb_xhkg = get_symbols_for_calendar(prices, "XHKG")

        session = prices.limit_right_daily
        prev_session = session - one_day
        for c in prices.calendars_unique:
            assert c.is_session(session)
            assert c.is_session(prev_session)

        prev_close_xnys = xnys.closes[prev_session]
        prev_close_xlon = xlon.closes[prev_session]
        open_xhkg = xhkg.opens[session]
        close_xhkg = xhkg.closes[session]
        open_xlon = xlon.opens[session]
        open_xnys = xnys.opens[session]
        close_xlon = xlon.closes[session]
        close_xnys = xnys.closes[session]

        T1_limit_right = prices.limits[prices.bis.T1][1]
        table_T1 = prices._pdata[prices.bis.T1]._table.pt.indexed_left
        table_T1_right = prices._pdata[prices.bis.T1]._table.pt.indexed_right
        table_D1 = prices._pdata[prices.bis.D1]._table
        f = prices.price_at

        ts = open_xlon - one_min
        df = f(ts, UTC)
        values = {
            symb_xhkg: table_T1[symb_xhkg].loc[ts].open,
            symb_xnys: table_T1_right[symb_xnys].loc[prev_close_xnys].close,
            symb_xlon: table_T1_right[symb_xlon].loc[prev_close_xlon].close,
        }
        self.assertions(prices, df, ts, values)

        ts = open_xlon
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_T1[symb_xhkg].loc[ts].open,
            symb_xlon: table_T1[symb_xlon].loc[ts].open,
            symb_xnys: table_T1_right[symb_xnys].loc[prev_close_xnys].close,
        }
        self.assertions(prices, df, ts, values)

        ts = close_xhkg
        df = f(ts, tz=UTC)
        values = {
            symb_xlon: table_T1[symb_xlon].loc[ts].open,
            symb_xhkg: table_T1_right[symb_xhkg].loc[ts].close,
            symb_xnys: table_T1_right[symb_xnys].loc[prev_close_xnys].close,
        }
        self.assertions(prices, df, ts, values)

        ts = open_xnys - one_min
        df = f(ts, tz=UTC)
        values = {
            symb_xlon: table_T1[symb_xlon].loc[ts].open,
            symb_xhkg: table_T1_right[symb_xhkg].loc[close_xhkg].close,
            symb_xnys: table_T1_right[symb_xnys].loc[prev_close_xnys].close,
        }
        self.assertions(prices, df, ts, values)

        ts = open_xnys
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_T1_right[symb_xhkg].loc[close_xhkg].close,
            symb_xlon: table_T1[symb_xlon].loc[ts].open,
            symb_xnys: table_T1[symb_xnys].loc[ts].open,
        }
        self.assertions(prices, df, ts, values)

        ts = close_xlon
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_T1_right[symb_xhkg].loc[close_xhkg].close,
            symb_xlon: table_T1_right[symb_xlon].loc[ts].close,
            symb_xnys: table_T1[symb_xnys].loc[ts].open,
        }
        self.assertions(prices, df, ts, values)

        ts = T1_limit_right - one_min
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_T1_right[symb_xhkg].loc[close_xhkg].close,
            symb_xlon: table_T1_right[symb_xlon].loc[close_xlon].close,
            symb_xnys: table_T1[symb_xnys].loc[ts].open,
        }
        self.assertions(prices, df, ts, values)

        ts = T1_limit_right
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_T1_right[symb_xhkg].loc[close_xhkg].close,
            symb_xlon: table_T1_right[symb_xlon].loc[close_xlon].close,
            symb_xnys: table_T1_right[symb_xnys].loc[ts].close,
        }
        self.assertions(prices, df, ts, values)

        ts = T1_limit_right + one_min
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_D1[symb_xhkg].loc[session].open,
            symb_xlon: table_D1[symb_xlon].loc[prev_session].close,
            symb_xnys: table_D1[symb_xnys].loc[prev_session].close,
        }
        self.assertions(prices, df, open_xhkg, values)

        # NB As the intraday prices availability ends before the end of the xnys session
        # the following requests should be met from daily data
        ts = close_xnys - one_min
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_D1[symb_xhkg].loc[session].open,
            symb_xlon: table_D1[symb_xlon].loc[prev_session].close,
            symb_xnys: table_D1[symb_xnys].loc[prev_session].close,
        }
        self.assertions(prices, df, open_xhkg, values)

        ts = close_xnys
        df = f(ts, tz=UTC)
        values = {
            symb_xhkg: table_D1[symb_xhkg].loc[session].close,
            symb_xlon: table_D1[symb_xlon].loc[session].close,
            symb_xnys: table_D1[symb_xnys].loc[session].close,
        }
        self.assertions(prices, df, close_xnys, values)

    def test_raises_live_prices_unavailable(self, prices):
        """Test raises error when `minute` None and live prices unavailable."""
        with pytest.raises(errors.PriceAtUnavailableLivePricesError):
            prices.price_at()
        with pytest.raises(errors.PriceAtUnavailableLivePricesError):
            prices.price_at(None)
