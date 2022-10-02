"""Tests for market_prices.intervals module."""

from __future__ import annotations

from collections import abc
from datetime import timedelta
import itertools
import re

import pandas as pd
import pytest

import market_prices.intervals as m
from market_prices.prices import yahoo


# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, unused-argument, invalid-name
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway.
#   invalid-name: names in tests not expected to strictly conform with snake_case.

# Any flake8 disabled violations handled via per-file-ignores on .flake8


def test_constants():
    """Test constants defined as expected."""
    # verify TIMEDELTA_ARGS keys as expected
    all_keys = set("T1 T2 T5 T10 T15 T30 H1 D1".split())
    assert all_keys == set(m.TIMEDELTA_ARGS.keys())

    # verify TIMEDELTA_ARGS values correspond with keys
    mapping = dict(T="minutes", H="hours", D="days")
    for k, v in m.TIMEDELTA_ARGS.items():
        kwargs = {mapping[k[0]]: int(k[1:])}
        assert timedelta(**kwargs) == timedelta(*v)

    assert m.ONE_DAY is m.TDInterval.D1
    assert m.ONE_MIN is m.TDInterval.T1


def test_tdintervals(xlon_calendar):
    """Verify TDInterval members as expected and properties of each member."""
    cal = xlon_calendar
    intraday_daily_separator = 60 * 22
    prev_enum = m.TDInterval.T1  # just to initialise
    for i, en in enumerate(m.TDInterval, 1):
        if i <= intraday_daily_separator:
            if i % 60:
                assert en == getattr(m.TDInterval, "T" + str(i))
                assert en.freq_unit == "T"
                assert en.freq_value == i
                assert en.as_pdfreq == str(i) + "T"
                for c, one_less in itertools.product([cal, None], [True, False]):
                    assert en.as_offset(c, one_less) == pd.tseries.offsets.Minute(i)
                assert en.as_offset() == pd.tseries.offsets.Minute(i)
            else:
                i_ = i // 60
                assert en == getattr(m.TDInterval, "H" + str(i_))
                assert en.freq_unit == "H"
                assert en.freq_value == i_
                assert en.as_pdfreq == str(i_) + "H"
                for c, one_less in itertools.product([cal, None], [True, False]):
                    assert en.as_offset(c, one_less) == pd.tseries.offsets.Hour(i_)
                assert en.as_offset() == pd.tseries.offsets.Hour(i_)

            assert en.is_intraday
            assert en.is_one_minute if i == 1 else not en.is_one_minute
            assert not en.is_daily
            assert not en.is_monthly
            assert not en.is_one_day
            assert not en.is_gt_one_day
            assert en.as_minutes == i

        else:
            i_ = i - intraday_daily_separator
            assert en == getattr(m.TDInterval, "D" + str(i_))
            assert en.freq_unit == "D"
            assert en.freq_value == i_
            assert en.as_pdfreq == str(i_) + "D"
            match = (
                "`calendar` must be passed for intervals representing a number"
                " of trading days."
            )
            with pytest.raises(ValueError, match=match):
                assert en.as_offset()  # requires at least calendar
            assert en.as_offset(cal) == cal.day * (i_ - 1)  # default behaviour
            assert en.as_offset(cal, one_less=True) == cal.day * (i_ - 1)
            assert en.as_offset(cal, one_less=False) == cal.day * i_

            assert not en.is_intraday
            assert en.is_daily
            assert not en.is_monthly
            assert en.is_one_day if i_ == 1 else not en.is_one_day
            assert not en.is_gt_one_day if i_ == 1 else en.is_gt_one_day
            assert not en.is_one_minute

            with pytest.raises(ValueError):
                _ = en.as_minutes

        # verify comparisons with same type
        if i != 1:
            assert prev_enum < en
            assert prev_enum <= en
            assert en > prev_enum
            assert en >= prev_enum

        prev_enum = en


def test_tdintervals_comparion_with_timedelta():
    """Verify comparisons of TDInterval with pd.Timedelta as expected."""
    assert m.TDInterval.T5 == pd.Timedelta(5, "T")
    assert m.TDInterval.T5 > pd.Timedelta(4, "T")
    assert m.TDInterval.T5 >= pd.Timedelta(4, "T")
    assert m.TDInterval.T5 >= pd.Timedelta(5, "T")
    assert m.TDInterval.T5 < pd.Timedelta(6, "T")
    assert m.TDInterval.T5 <= pd.Timedelta(6, "T")
    assert m.TDInterval.T5 <= pd.Timedelta(5, "T")


def test_dointervals(xlon_calendar):
    """Verify DOInterval members as expected and properties of each member."""
    cal = xlon_calendar
    prev_enum = m.DOInterval.M1  # just to initialise with a value, won't be used.
    for i, en in enumerate(m.DOInterval, 1):
        assert en == getattr(m.DOInterval, "M" + str(i))

        assert en.freq_unit == "MS"
        assert en.freq_value == i
        assert en.as_pdfreq == str(i) + "MS"
        for c, one_less in itertools.product([cal, None], [True, False]):
            assert en.as_offset(c, one_less) == pd.DateOffset(months=i)
        assert en.as_offset() == pd.DateOffset(months=i)
        assert en.as_offset_ms == pd.offsets.MonthBegin(i)

        assert not en.is_intraday
        assert not en.is_daily
        assert en.is_monthly
        assert not en.is_one_day
        assert en.is_gt_one_day
        assert not en.is_one_minute

        # verify comparisons with same type
        if i != 1:
            assert prev_enum < en
            assert prev_enum <= en
            assert en > prev_enum
            assert en >= prev_enum

        prev_enum = en


def test_dointervals_operation_with_timestamp():
    """Verify add/sub opperations of DOInterval with pd.Timestamp."""
    ts = pd.Timestamp("2021-10-05")
    interval = m.DOInterval.M2
    assert ts + interval == pd.Timestamp("2021-12-05")
    assert ts - interval == pd.Timestamp("2021-08-05")
    with pytest.raises(TypeError):
        _ = interval + ts
    with pytest.raises(TypeError):
        _ = interval - ts


class TestBaseInterval:
    """Test `m._BaseInterval`."""

    @pytest.fixture(scope="class")
    def BaseInterval(self) -> abc.Iterator[type[m.BI]]:
        class BaseInterval_(m.BI):
            """Base interval enum."""

            T1 = m.TIMEDELTA_ARGS["T1"]
            T2 = m.TIMEDELTA_ARGS["T2"]
            T5 = m.TIMEDELTA_ARGS["T5"]
            H1 = m.TIMEDELTA_ARGS["H1"]
            D1 = m.TIMEDELTA_ARGS["D1"]

        yield BaseInterval_

    @pytest.fixture(scope="class")
    def BaseIntervalIntradayOnly(self) -> abc.Iterator[type[m.BI]]:
        class BaseIntervalIntradayOnly_(m.BI):
            """Base interval enum of only intraday intervals."""

            T1 = m.TIMEDELTA_ARGS["T1"]
            T2 = m.TIMEDELTA_ARGS["T2"]
            T5 = m.TIMEDELTA_ARGS["T5"]
            H1 = m.TIMEDELTA_ARGS["H1"]

        yield BaseIntervalIntradayOnly_

    def test_base_interval(self, BaseInterval, BaseIntervalIntradayOnly):
        """Test _BaseInterval functionality that exceeds _TDIntervalBase."""
        # test previous, next, __getitem__, __contains__
        prev_bi = None
        for i, bi in enumerate(BaseInterval, 0):
            assert bi in BaseInterval
            assert bi is BaseInterval[i]
            assert bi.previous is prev_bi
            if prev_bi is not None:
                assert prev_bi.next is bi
            prev_bi = bi

        assert bi.next is None  # pylint: disable=undefined-loop-variable
        assert m.TDInterval.T5 in BaseInterval
        assert m.TDInterval.T6 not in BaseInterval
        assert pd.Timedelta(5, "T") in BaseInterval
        assert pd.Timedelta(6, "T") not in BaseInterval

        assert BaseInterval.daily_bi() == pd.Timedelta(1, "D")
        assert BaseInterval.intraday_bis() == BaseInterval[:-1]
        assert BaseIntervalIntradayOnly.daily_bi() is None
        assert BaseIntervalIntradayOnly.intraday_bis() == BaseIntervalIntradayOnly[:]

    def test_yahoo_base_interval(self):
        BaseInterval = yahoo.PricesYahoo.BaseInterval
        assert issubclass(BaseInterval, m.BI)
        assert len(BaseInterval) == 5
        assert m.TDInterval.T1 in BaseInterval
        assert m.TDInterval.T2 in BaseInterval
        assert m.TDInterval.T5 in BaseInterval
        assert m.TDInterval.H1 in BaseInterval
        assert m.TDInterval.D1 in BaseInterval


class TestToPTInterval:
    """Test `m.to_ptinterval`."""

    @pytest.fixture(scope="class")
    def f(self) -> abc.Iterator[abc.Callable]:
        yield m.to_ptinterval

    @pytest.fixture(scope="class")
    def components(self) -> abc.Iterator[dict]:
        """Mapping frequency to timedelta kwarg."""
        yield {
            "T": "minutes",
            "MIN": "minutes",
            "H": "hours",
            "D": "days",
            "M": "months",
        }

    def test_type(self, f):
        for invalid_input in [("2T",), ["1D"], 33]:
            with pytest.raises(TypeError):
                f(invalid_input)

    def test_str_input(self, f, components):
        # pylint: disable=too-complex
        match = re.escape("interval/frequency received as")

        valid_units = ["MIN", "T", "H", "D", "M"]
        all_valid_units = valid_units + [unit.lower() for unit in valid_units]
        for unit in all_valid_units:
            with pytest.raises(ValueError, match=match):
                f("0" + unit)
            assert f("1" + unit)

        for invalid_input in ["2s2", "T1", "H2H", "3T4H", "HH2"]:
            with pytest.raises(ValueError, match=match):
                f(invalid_input)

        def match_msg(value: int, unit: str, limit: int) -> str:
            component = components[unit.upper()]
            return re.escape(
                f"An `interval` defined in terms of {component} cannot have a"
                f" value greater than {limit}, although received `interval` as"
                f' "{value}{unit}".'
            )

        limit = 1320
        for unit in ["T", "t", "min", "MIN"]:
            for i in range(1, limit + 1):
                if i % 60:
                    assert f(str(i) + unit) == getattr(m.TDInterval, "T" + str(i))
                else:
                    assert f(str(i) + unit) == getattr(m.TDInterval, "H" + str(i // 60))

        for unit in ["T", "t", "min", "MIN"]:
            for i in range(limit + 1, limit + 6):
                with pytest.raises(ValueError, match=match_msg(i, unit, limit)):
                    _ = f(str(i) + unit)

        def test_unit(unit: str, limit: int, Cls: m.PTInterval):
            for unit_ in [unit.upper(), unit.lower()]:
                for i in range(1, limit + 1):
                    assert f(str(i) + unit_) == getattr(Cls, unit.upper() + str(i))

                for i in range(limit + 1, limit + 6):
                    with pytest.raises(ValueError, match=match_msg(i, unit_, limit)):
                        f(str(i) + unit_)

        test_unit("h", 22, m.TDInterval)
        test_unit("d", 250, m.TDInterval)
        test_unit("m", 36, m.DOInterval)

        assert f("60min") == f("60MIN") == f("1H") == f("1h") == m.TDInterval.H1

    def test_timedelta_input(self, f, components):
        # pylint: disable=too-complex
        match = "`interval` cannot be negative or zero."
        for i in range(0, -3, -1):
            for unit in ["T", "H", "D"]:
                with pytest.raises(ValueError, match=match):
                    f(pd.Timedelta(i, unit))
            for kwarg in ["minutes", "hours", "days"]:
                with pytest.raises(ValueError, match=match):
                    f(timedelta(**{kwarg: i}))

        def match_too_high(td: timedelta | pd.Timedelta, limit: int) -> str:
            td = pd.Timedelta(td)
            component = components[td.resolution_string]
            return re.escape(
                f"An `interval` defined in terms of {component} cannot have a"
                f" value greater than {limit}, although received `interval` as"
                f' "{td}".'
            )

        def match_comps_error(td: pd.Timedelta) -> str:
            td = pd.Timedelta(td)
            return re.escape(
                "An `interval` defined with a timedelta or pd.Timedelta can only"
                " be defined in terms of EITHER minute and/or hours OR days, although"
                f" received as '{td}'."
            )

        def test_unit(unit: str, limit: int, Cls: m.PTInterval):
            for i in range(1, limit + 1):
                for td in (pdtd := pd.Timedelta(i, unit), pdtd.to_pytimedelta()):
                    if unit == "T" and not i % 60:
                        assert f(td) == getattr(Cls, "H" + str(i // 60))
                    else:
                        assert f(td) == getattr(Cls, unit + str(i))

            for i in range(limit + 1, limit + 6):
                for td in (pdtd := pd.Timedelta(i, unit), pdtd.to_pytimedelta()):
                    if unit == "H" and i >= 24:
                        if i == 24:
                            assert f(td) == m.TDInterval.D1
                        if i > 24:
                            with pytest.raises(ValueError, match=match_comps_error(td)):
                                _ = f(td)
                    else:
                        with pytest.raises(ValueError, match=match_too_high(td, limit)):
                            _ = f(td)

        test_unit("T", 1320, m.TDInterval)
        test_unit("H", 22, m.TDInterval)
        test_unit("D", 250, m.TDInterval)

        # test multiple components

        TDClasses = (timedelta, pd.Timedelta)

        # minutes and hours
        for minutes, hours, member in (
            (1, 1, m.TDInterval.T61),
            (60, 1, m.TDInterval.H2),
            (59, 21, m.TDInterval.T1319),
            (60, 21, m.TDInterval.H22),
            (120, 22, m.TDInterval.D1),
        ):
            for TDCls in TDClasses:
                assert f(TDCls(minutes=minutes, hours=hours)) == member

        for minutes, hours in ((1, 22), (59, 22), (1, 23), (59, 23)):
            for TDCls in TDClasses:
                td = TDCls(minutes=minutes, hours=hours)
                with pytest.raises(ValueError, match=match_too_high(td, 1320)):
                    _ = f(td)

        for TDCls in TDClasses:
            td = TDCls(minutes=60, hours=22)
            with pytest.raises(ValueError, match=match_too_high(td, 22)):
                _ = f(td)

            td = TDCls(minutes=60, hours=23)
            assert f(td) == m.TDInterval.D1

            td = TDCls(minutes=1, hours=24)
            with pytest.raises(ValueError, match=match_comps_error(td)):
                _ = f(td)

            # other invalid components / component combinations
            for days, hours, minutes in ((1, 0, 1), (1, 1, 0), (1, 1, 1)):
                td = TDCls(days=days, hours=hours, minutes=minutes)
                with pytest.raises(ValueError, match=match_comps_error(td)):
                    _ = f(td)

        invalid_kwargs = ["seconds", "milliseconds", "microseconds", "nanoseconds"]
        for TDCls in TDClasses:
            for invalid_kwarg in invalid_kwargs:
                if invalid_kwarg == "nanoseconds" and TDCls is timedelta:
                    continue
                d = {invalid_kwarg: 1}
                td = TDCls(**d)
                with pytest.raises(ValueError, match=match_comps_error(td)):
                    _ = f(td)
                for valid_kwarg in ["minutes", "hours", "days"]:
                    td = TDCls(**{**d, **{valid_kwarg: 1}})
                    with pytest.raises(ValueError, match=match_comps_error(td)):
                        _ = f(td)
