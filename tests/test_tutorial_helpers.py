"""Tests for market_prices.support.tutorial_helpers module."""

from collections import abc
import re

import pytest
import pandas as pd
from pandas.testing import assert_index_equal
from pandas import Timestamp as T
import exchange_calendars as xcals

from market_prices import intervals, errors
from market_prices.helpers import UTC
from market_prices.utils import calendar_utils as calutils
from market_prices.prices.base import PricesBase
import market_prices.support.tutorial_helpers as m

# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, unused-argument, invalid-name
#   missing-fuction-docstring - doc not required for all tests
#   protected-access not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments,too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name,no-self-use, missing-any-param-doc
#   unused-argument not compatible with pytest fixtures, caught by pylance anyway

# Any flake8 disabled violations handled via per-file-ignores on .flake8


@pytest.fixture(scope="class")
def side() -> abc.Iterator[str]:
    yield "left"


@pytest.fixture(scope="class")
def xlon(side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XLON", side=side)


@pytest.fixture(scope="class")
def xnys(side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XNYS", side=side)


@pytest.fixture(scope="class")
def xhkg(side) -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar("XHKG", side=side)


@pytest.fixture
def PricesMock() -> abc.Iterator[type[PricesBase]]:
    """Mock of PricesBase with T1 and T5 base intervals defined."""

    class PricesMock_(PricesBase):
        """Mock of PricesBase with T1 and T5 base intervals defined."""

        BaseInterval = intervals._BaseInterval(
            "BaseInterval",
            dict(
                T1=intervals.TIMEDELTA_ARGS["T1"],
                T5=intervals.TIMEDELTA_ARGS["T5"],
            ),
        )

        BASE_LIMITS = {
            BaseInterval.T1: T("2021-12-01 00:01", tz=UTC),
            BaseInterval.T5: T("2021-11-01 00:01", tz=UTC),
        }

        def _request_data(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

        def prices_for_symbols(self, *_, **__):
            raise NotImplementedError("mock class, method not implemented.")

    yield PricesMock_


def get_prices_limit_mock(
    Prices: type[PricesBase],
    t1_limit: pd.Timestamp,
    t5_limit: pd.Timestamp,
    calendars: list[xcals.ExchangeCalendar],
) -> PricesBase:
    """Return instance of `Prices` with mocked T1 and T5 limits."""
    limits = {
        Prices.BaseInterval.T1: t1_limit,
        Prices.BaseInterval.T5: t5_limit,
    }

    class PricesMockLimits(Prices):  # type: ignore[valid-type, misc]
        # pylint: disable=missing-class-docstring, too-few-public-methods
        BASE_LIMITS = limits

    symbols = [cal.name for cal in calendars]
    return PricesMockLimits(symbols, calendars)


def test_get_sessions_range_for_bi(PricesMock, xlon, xnys, monkeypatch):
    """Test `_get_sessions_range_for_bi`."""
    f = m.get_sessions_range_for_bi
    calendars = [xlon, xnys]

    def patch_now(ts: pd.Timestamp):
        ts = T(ts, tz=UTC)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *_, **__: ts)

    def get_prices(t1_limit: pd.Timestamp, t5_limit: pd.Timestamp) -> PricesBase:
        t1_limit = pd.Timestamp(t1_limit, tz=UTC)
        t5_limit = pd.Timestamp(t5_limit, tz=UTC)
        return get_prices_limit_mock(PricesMock, t1_limit, t5_limit, calendars)

    # sessions with holidays around

    patch_now("2021-12-31 21:00")
    for prices in (
        get_prices(T("2021-12-24 08:00"), T("2021-11-24 14:30")),
        get_prices(T("2021-12-24 07:59"), T("2021-11-24 14:29")),
    ):
        bi = prices.bis.T1
        end = T("2021-12-31")
        expected = T("2021-12-24"), end
        assert f(prices, bi, xlon) == expected
        expected = T("2021-12-27"), end
        assert f(prices, bi, xnys) == expected
        expected = T("2021-12-24"), end
        assert f(prices, bi) == expected

        bi = prices.bis.T5
        end = T("2021-12-23")
        expected = T("2021-11-25"), end
        assert f(prices, bi, xlon) == expected
        expected = T("2021-11-24"), end
        assert f(prices, bi, xnys) == expected
        expected = T("2021-11-25"), end
        assert f(prices, bi) == expected

    patch_now("2021-12-31 20:54")
    prices = get_prices(T("2021-12-24 08:01"), T("2021-11-24 14:31"))

    bi = prices.bis.T1
    expected = T("2021-12-29"), T("2021-12-31")
    assert f(prices, bi, xlon) == expected
    expected = T("2021-12-27"), T("2021-12-30")
    assert f(prices, bi, xnys) == expected
    expected = T("2021-12-27"), T("2021-12-30")
    assert f(prices, bi) == expected

    bi = prices.bis.T5
    end = T("2021-12-23")
    expected = T("2021-11-25"), end
    assert f(prices, bi, xlon) == expected
    expected = T("2021-11-26"), end
    assert f(prices, bi, xnys) == expected
    expected = T("2021-11-25"), end
    assert f(prices, bi) == expected

    # sessions with no holidays either side

    patch_now("2021-12-30 21:00")
    for prices in (
        get_prices(T("2021-12-15 08:00"), T("2021-11-17 14:30")),
        get_prices(T("2021-12-15 07:59"), T("2021-11-17 14:29")),
    ):
        bi = prices.bis.T1
        expected = T("2021-12-15"), T("2021-12-30")
        assert f(prices, bi, xlon) == expected
        assert f(prices, bi, xnys) == expected
        assert f(prices, bi) == expected

        bi = prices.bis.T5
        end = T("2021-12-14")
        expected = T("2021-11-18"), end
        assert f(prices, bi, xlon) == expected
        expected = T("2021-11-17"), end
        assert f(prices, bi, xnys) == expected
        expected = T("2021-11-18"), end
        assert f(prices, bi) == expected

    patch_now("2021-12-30 20:54")
    for prices in (
        get_prices(T("2021-12-15 08:01"), T("2021-11-17 14:31")),
        get_prices(T("2021-12-15 14:30"), T("2021-11-18 08:00")),
    ):
        bi = prices.bis.T1
        expected = T("2021-12-16"), T("2021-12-30")
        assert f(prices, bi, xlon) == expected
        expected = T("2021-12-15"), T("2021-12-29")
        assert f(prices, bi, xnys) == expected
        expected = T("2021-12-16"), T("2021-12-29")
        assert f(prices, bi) == expected

        bi = prices.bis.T5
        expected = T("2021-11-18"), T("2021-12-14")
        assert f(prices, bi, xlon) == expected
        assert f(prices, bi, xnys) == expected
        assert f(prices, bi) == expected

    patch_now("2021-12-30 14:24")
    for prices in (
        get_prices(T("2021-12-15 14:31"), T("2021-11-18 08:01")),
        get_prices(T("2021-12-15 16:29"), T("2021-11-18 14:30")),
    ):
        bi = prices.bis.T1
        expected = T("2021-12-16"), T("2021-12-29")
        assert f(prices, bi, xlon) == expected
        assert f(prices, bi, xnys) == expected
        assert f(prices, bi) == expected

        bi = prices.bis.T5
        end = T("2021-12-14")
        expected = T("2021-11-19"), end
        assert f(prices, bi, xlon) == expected
        expected = T("2021-11-18"), end
        assert f(prices, bi, xnys) == expected
        expected = T("2021-11-19"), end
        assert f(prices, bi) == expected

    patch_now("2021-12-30 14:24")
    for prices in (
        get_prices(T("2021-12-15 16:30"), T("2021-11-18 14:31")),
        # get_prices(T("2021-12-15 16:29"), T("2021-11-18 14:30")),
    ):
        bi = prices.bis.T1
        expected = T("2021-12-16"), T("2021-12-29")
        assert f(prices, bi, xlon) == expected
        assert f(prices, bi, xnys) == expected
        assert f(prices, bi) == expected

        bi = prices.bis.T5
        start = T("2021-11-19")
        expected = start, T("2021-12-15")
        assert f(prices, bi, xlon) == expected
        expected = start, T("2021-12-14")
        assert f(prices, bi, xnys) == expected
        expected = start, T("2021-12-14")
        assert f(prices, bi) == expected


def test_get_conforming_sessions(xlon, xhkg, xnys):
    """Test `get_conforming_sessions` functions.

    Tests:
        `get_conforming_sessions`
        `get_conforming_sessions_var`
        `get_conforming_cc_sessions`
        `get_conforming_cc_sessions_var`

    Expected returns evaluated from investigation of schedules.
    """
    # pylint: disable=too-complex,too-many-statements
    f = m.get_conforming_sessions
    f_var = m.get_conforming_sessions_var
    f_cc = m.get_conforming_cc_sessions
    f_cc_var = m.get_conforming_cc_sessions_var

    def match(
        cal_param: list[xcals.ExchangeCalendar] | calutils.CompositeCalendar,
        lengths_param: pd.Timedelta | list[pd.Timedelta] | list[list[pd.Timedelta]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        return re.escape(
            "The requested number of consecutive sessions of the requested length(s)"
            f" are not available from {start} through {end}."
            f"\nThe `calendars` or `cc` parameter was receieved as {cal_param}."
            "\nThe `sessions_lengths` or `session_length` parameter was receieved"
            f" as {lengths_param}."
        )

    # single calendar
    start, end = T("2021-12-10"), T("2022")
    cals = [xlon]
    cc = calutils.CompositeCalendar(cals)
    full_session_xlon = pd.Timedelta(hours=8, minutes=30)
    half_session_xlon = pd.Timedelta(hours=4, minutes=30)
    no_session = pd.Timedelta(0)

    length = full_session_xlon
    expected = pd.DatetimeIndex([start])
    num = 1
    assert_index_equal(f(cals, [length], start, end, num), expected)
    assert_index_equal(f_cc(cc, length, start, end, num), expected)
    lengths = [[length]]
    assert_index_equal(f_var(cals, lengths, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length], start, end), expected)

    num = 10  # limit that can be fuflilled
    expected = xlon.sessions_in_range(start, T("2021-12-23"))
    assert_index_equal(f(cals, [length], start, end, num), expected)
    assert_index_equal(f_cc(cc, length, start, end, num), expected)
    lengths = [[length] * num]
    assert_index_equal(f_var(cals, lengths, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length] * num, start, end), expected)

    # beyond limit
    num += 1
    msg = match(cals, [[length] * num], start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        f(cals, [length], start, end, num)
    args = cc, length, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc(*args, num)
    lengths = [[full_session_xlon] * num]
    args = cals, lengths, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_var(*args)
    args = cc, [length] * num, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc_var(*args)

    length = half_session_xlon
    lengths = [[length]]
    num = 1
    expected = pd.DatetimeIndex([T("2021-12-24")])
    assert_index_equal(f(cals, [length], start, end, num), expected)
    assert_index_equal(f_cc(cc, length, start, end, num), expected)
    assert_index_equal(f_var(cals, lengths, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length], start, end), expected)

    num += 1
    msg = match(cals, [[length] * num], start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        f(cals, [length], start, end, num)
    args = cc, length, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc(*args, num)
    lengths = [[length] * num]
    args = cals, lengths, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_var(*args)
    args = cc, [length] * num, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc_var(*args)

    length = pd.Timedelta(4, "h")
    lengths = [[length]]
    num = 1
    msg = match(cals, [[length] * num], start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        assert f(cals, [length], start, end, num)
    args = cc, length, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        assert f_cc(*args, num)
    args = cals, lengths, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        assert f_var(*args)
    args = cc, [length], start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        assert f_cc_var(*args)

    # test var for variatons
    full_expected = pd.DatetimeIndex(
        [T("2021-12-23"), T("2021-12-24"), T("2021-12-29")]
    )
    cc_lengths = [full_session_xlon, half_session_xlon]
    lengths = [cc_lengths]
    assert_index_equal(f_var(cals, lengths, start, end), full_expected[:2])
    assert_index_equal(f_cc_var(cc, cc_lengths, start, end), full_expected[:2])
    cc_lengths = cc_lengths + [full_session_xlon]
    lengths = [cc_lengths]
    assert_index_equal(f_var(cals, lengths, start, end), full_expected)
    assert_index_equal(f_cc_var(cc, cc_lengths, start, end), full_expected)

    # two calendars
    cals = [xlon, xnys]
    cc = calutils.CompositeCalendar(cals)
    full_session_xnys = pd.Timedelta(hours=6, minutes=30)
    full_session_cc = pd.Timedelta(13, "h")

    length_by_cal = [full_session_xlon, full_session_xnys]
    length_cc = full_session_cc
    num = 1
    expected = pd.DatetimeIndex([start])
    assert_index_equal(f(cals, length_by_cal, start, end, num), expected)
    assert_index_equal(f_cc(cc, length_cc, start, end, num), expected)

    lengths_by_cal = [[full_session_xlon], [full_session_xnys]]
    assert_index_equal(f_var(cals, lengths_by_cal, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length_cc], start, end), expected)

    num = 10  # limit that can be fuflilled
    expected = xlon.sessions_in_range(start, T("2021-12-23"))
    assert_index_equal(f(cals, length_by_cal, start, end, num), expected)
    assert_index_equal(f_cc(cc, length_cc, start, end, num), expected)
    lengths_by_cal = [[full_session_xlon] * num, [full_session_xnys] * num]
    assert_index_equal(f_var(cals, lengths_by_cal, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length_cc] * num, start, end), expected)

    # beyond limit
    num += 1
    lengths_msg_arg = [
        [session_length_cal] * num for session_length_cal in length_by_cal
    ]
    msg = match(cals, lengths_msg_arg, start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        f(cals, length_by_cal, start, end, num)
    args = cc, length_cc, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc(*args, num)
    lengths_by_cal = [[full_session_xlon] * num, [full_session_xnys] * num]
    args = cals, lengths_by_cal, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_var(*args)
    args = cc, [length_cc] * num, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc_var(*args)

    lengths = [half_session_xlon, no_session]
    length_cc = half_session_xlon
    num = 1
    expected = pd.DatetimeIndex([T("2021-12-24")])
    assert_index_equal(f(cals, lengths, start, end, num), expected)
    assert_index_equal(f_cc(cc, length_cc, start, end, num), expected)
    lengths_by_cal = [[half_session_xlon], [no_session]]
    assert_index_equal(f_var(cals, lengths_by_cal, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length_cc], start, end), expected)

    num += 1
    lengths_msg_arg = [[session_length_cal] * num for session_length_cal in lengths]
    msg = match(cals, lengths_msg_arg, start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        f(cals, lengths, start, end, num)
    args = cc, length_cc, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc(*args, num)
    lengths_by_cal = [[half_session_xlon] * num, [no_session] * num]
    args = cals, lengths_by_cal, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_var(*args)
    args = cc, [length_cc] * num, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc_var(*args)

    # test var for variatons
    full_expected = pd.DatetimeIndex(
        [T("2021-12-" + str(day)) for day in (23, 24, 27, 28, 29)]
    )
    lengths = [
        [full_session_xlon, half_session_xlon] + [no_session] * 2 + [full_session_xlon],
        [full_session_xnys, no_session] + [full_session_xnys] * 3,
    ]
    lengths_cc = [
        full_session_cc,
        half_session_xlon,
        full_session_xnys,
        full_session_xnys,
        full_session_cc,
    ]
    num_iterations = len(full_expected)

    lengths_all = []
    for i in range(num_iterations):
        lengths_ = []
        for cal_lengths in lengths:
            lengths_.append(cal_lengths[: i + 1])
        lengths_all.append(lengths_)

    for i in range(num_iterations):
        if not i:
            continue  # would find earlier date when just looking for a normal day
        lengths = lengths_all[i]
        assert_index_equal(f_var(cals, lengths, start, end), full_expected[: i + 1])
        assert_index_equal(
            f_cc_var(cc, lengths_cc[: i + 1], start, end), full_expected[: i + 1]
        )

    # three calendars
    cals = [xlon, xnys, xhkg]
    cc = calutils.CompositeCalendar(cals)
    full_session_cc = pd.Timedelta(hours=19, minutes=30)
    full_session_xhkg = full_session_xnys
    half_session_xhkg = pd.Timedelta(hours=2, minutes=30)

    length_by_cal = [full_session_xlon, full_session_xnys, full_session_xhkg]
    length_cc = full_session_cc
    num = 1
    expected = pd.DatetimeIndex([start])
    assert_index_equal(f(cals, length_by_cal, start, end, num), expected)
    assert_index_equal(f_cc(cc, length_cc, start, end, num), expected)

    lengths_by_cal = [[full_session_xlon], [full_session_xnys], [full_session_xhkg]]
    assert_index_equal(f_var(cals, lengths_by_cal, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length_cc], start, end), expected)

    num = 10  # limit that can be fuflilled
    expected = xlon.sessions_in_range(start, T("2021-12-23"))
    assert_index_equal(f(cals, length_by_cal, start, end, num), expected)
    lengths_by_cal = [
        [full_session_xlon] * num,
        [full_session_xnys] * num,
        [full_session_xhkg] * num,
    ]
    assert_index_equal(f_var(cals, lengths_by_cal, start, end), expected)
    assert_index_equal(f_cc_var(cc, [length_cc] * num, start, end), expected)

    # beyond limit
    num += 1
    lengths_msg_arg = [
        [session_length_cal] * num for session_length_cal in length_by_cal
    ]
    msg = match(cals, lengths_msg_arg, start, end)
    with pytest.raises(errors.TutorialDataUnavailableError, match=msg):
        f(cals, length_by_cal, start, end, num)
    args = cc, length_cc, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc(*args, num)
    lengths_by_cal = [
        [full_session_xlon] * num,
        [full_session_xnys] * num,
        [full_session_xhkg] * num,
    ]
    args = cals, lengths_by_cal, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_var(*args)
    args = cc, [length_cc] * num, start, end
    with pytest.raises(errors.TutorialDataUnavailableError, match=match(*args)):
        f_cc_var(*args)

    # test var for variatons
    lengths = [
        [full_session_xlon, half_session_xlon] + [no_session] * 2 + [full_session_xlon],
        [full_session_xnys, no_session] + [full_session_xnys] * 3,
        [full_session_xhkg, half_session_xhkg, no_session] + [full_session_xhkg] * 2,
    ]
    lengths_cc = [
        full_session_cc,
        pd.Timedelta(11, "h"),
        full_session_xnys,
        full_session_cc,
        full_session_cc,
    ]

    num_iterations = len(full_expected)
    lengths_all = []
    for i in range(num_iterations):
        lengths_ = []
        for cal_lengths in lengths:
            lengths_.append(cal_lengths[: i + 1])
        lengths_all.append(lengths_)

    for i in range(num_iterations):
        if not i:
            continue  # would find earlier date when just looking for a normal day
        lengths = lengths_all[i]
        assert_index_equal(f_var(cals, lengths, start, end), full_expected[: i + 1])
        assert_index_equal(
            f_cc_var(cc, lengths_cc[: i + 1], start, end), full_expected[: i + 1]
        )
