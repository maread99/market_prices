"""Tests for market_prices.daterange module.

Notes
-----
Tests assume that `period_kwargs` passed to Getter classes will have been
verified with `market_prices.parsing.verify_period_parameters` and 'start'
and 'end' values will have been parsed initially with
`market_prices.parsing.parse_timestamp` and subsequently with
`market_prices.parsing.parse_start_end`.
"""

from __future__ import annotations

import contextlib
import itertools
import re
from collections import abc

import exchange_calendars as xcals
import hypothesis as hyp
from hypothesis import strategies as sthyp
import pandas as pd
import pytest

import market_prices.daterange as m
import market_prices.utils.calendar_utils as calutils
from market_prices import errors, helpers, intervals
from market_prices.intervals import DOInterval, TDInterval
from market_prices.mptypes import Alignment, Anchor

from .utils import Answers
from . import conftest
from . import hypstrtgy as stmp
from .hypstrtgy import get_pp_default


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


def get_today(calendar: xcals.ExchangeCalendar) -> pd.Timestamp:
    """Return today for a given `calendar`."""
    now_utc = conftest._now_utc
    return calendar.minute_to_session(now_utc, "previous")


@pytest.fixture
def pp_default() -> abc.Iterator[dict]:
    """Default period parameter values."""
    yield {
        "minutes": 0,
        "hours": 0,
        "days": 0,
        "weeks": 0,
        "months": 0,
        "years": 0,
        "start": None,
        "end": None,
        "add_a_row": False,
    }


@contextlib.contextmanager
def add_a_row(pp: dict):
    """Context manager to set pp["add_a_row"] to True during context."""
    prior_value = pp["add_a_row"]
    pp["add_a_row"] = True
    yield
    pp["add_a_row"] = prior_value


@contextlib.contextmanager
def not_add_a_row(pp: dict):
    """Context manager to set pp["add_a_row"] to False during context."""
    prior_value = pp["add_a_row"]
    pp["add_a_row"] = False
    yield
    pp["add_a_row"] = prior_value


class TestGetterDaily:
    """Tests for GetterDaily class."""

    def get_drg(
        self,
        calendar: xcals.ExchangeCalendar,
        pp: dict,
        limit=None,
        ds_interval=None,
        strict=True,
    ) -> m.GetterDaily:
        """Get m.GetterDaily with default arguments unless otherwise passed."""
        if limit is None:
            limit = calendar.first_session
        return m.GetterDaily(calendar, limit, pp, ds_interval, strict)

    def test_constructor_properties(self, xlon_calendar, pp_default):
        """Test properties that expose constructor parameters."""
        cal = xlon_calendar
        limit = cal.first_session

        # required arguments only, options as default
        for drg in (
            m.GetterDaily(cal, limit, pp_default),  # options as default
            # explicitly pass arguments as default values
            m.GetterDaily(cal, limit, pp_default, None, True),
        ):
            assert drg.cal == cal
            assert drg.pp == pp_default
            assert drg.limit == limit
            assert drg.interval is TDInterval.D1
            assert drg.ds_interval is None
            assert drg.strict

        pp = pp_default["start"] = cal.minutes[333]
        limit = cal.minutes[222]
        ds_interval = TDInterval.D5
        strict = False

        drg = self.get_drg(cal, pp, limit, ds_interval, strict)

        assert drg.pp == pp
        assert drg.limit == limit
        assert drg.interval is TDInterval.D1
        assert drg.ds_interval is ds_interval
        assert drg.strict == strict

    def test_intervals(self, xlon_calendar, pp_default):
        """Test interval, ds_interval and final_interval properties."""
        for ds_interval in (TDInterval.T1, TDInterval.T5, TDInterval.H22):
            match_msg = re.escape("`ds_interval` cannot be lower than 'one day'")
            with pytest.raises(ValueError, match=match_msg):
                _ = self.get_drg(xlon_calendar, pp_default, ds_interval=ds_interval)

        drg = self.get_drg(xlon_calendar, pp_default)
        assert drg.ds_interval is None
        assert drg.final_interval == TDInterval.D1

        for ds_interval in (TDInterval.D1, TDInterval.D5, DOInterval.M1):
            drg = self.get_drg(xlon_calendar, pp_default, ds_interval=ds_interval)
            assert drg.ds_interval == ds_interval
            assert drg.final_interval == ds_interval
            if isinstance(ds_interval, TDInterval):
                assert drg.ds_factor == ds_interval.freq_value
            else:
                with pytest.raises(NotImplementedError):
                    _ = drg.ds_factor

    def test_end_now(self, calendars_with_answers, monkeypatch, pp_default):
        calendar, answers = calendars_with_answers

        drg = self.get_drg(calendar, pp_default)
        session = answers.sessions_sample[-4]  # session before last 3
        ts = session.replace(minute=13, hour=15, second=35)
        monkeypatch.setattr("pandas.Timestamp.now", lambda *a, **k: ts)
        assert drg.end_now == (session, session)
        if not answers.non_sessions.empty:
            non_session = answers.non_sessions[-1]
            ts = non_session.replace(minute=13, hour=15, second=35)
            monkeypatch.setattr("pandas.Timestamp.now", lambda *a, **k: ts)
            session = answers.date_to_session(non_session, "previous")
            assert drg.end_now == (session, session)

    def test_get_start(self, calendars_with_answers, pp_default, one_day):
        """Test `get_start`.

        Notes
        -----
        Test assumes value passed to `get_start` will be a date that
        represents a session or a non-session within calendar bounds. This
        will be the case for internal calls (within the GetterDaily class)
        and for input that has been parsed initially with
        `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.
        """
        cal, ans = calendars_with_answers
        drg = self.get_drg(cal, pp_default)

        for session in ans.sessions_sample:
            assert drg.get_start(session) == session

        for non_session in ans.non_sessions[:20]:
            expected = ans.date_to_session(non_session, "next")
            assert drg.get_start(non_session) == expected

        limit = ans.sessions[len(ans.sessions) // 2]
        too_early = limit - one_day
        drg = self.get_drg(cal, pp_default, limit=limit, strict=False)
        assert drg.get_start(too_early) == limit

        drg = self.get_drg(cal, pp_default, limit=limit, strict=True)
        match = re.escape(
            f"Prices unavailable as start ({helpers.fts(too_early)}) is earlier"
            " than the earliest session for which price data is available. The earliest"
            f" session for which prices are available is {helpers.fts(limit)}."
        )
        with pytest.raises(errors.StartTooEarlyError, match=match):
            _ = drg.get_start(too_early)

    def test_get_end(self, calendars_with_answers, pp_default, one_day):
        """Test `get_end`.

        Notes
        -----
        Test assumes value passed to `get_end` will be a date that
        represents a session or a non-session within calendar bounds. This
        will be the case for internal calls (within the GetterDaily class)
        and for input that has been parsed initially with
        `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.
        """
        cal, ans = calendars_with_answers
        drg = self.get_drg(cal, pp_default)

        for session in ans.sessions_sample:
            assert drg.get_end(session) == (session, session)

        for non_session in ans.non_sessions[:20]:
            expected = ans.date_to_session(non_session, "previous")
            assert drg.get_end(non_session) == (expected, expected)

        limit = ans.sessions[len(ans.sessions) // 2]
        too_early = limit - one_day
        match = re.escape(
            f"Prices unavailable as end ({helpers.fts(too_early)}) is earlier"
            " than the earliest session for which price data is available. The earliest"
            f" session for which prices are available is {helpers.fts(limit)}."
        )
        for strict in [True, False]:
            drg = self.get_drg(cal, pp_default, limit=limit, strict=strict)
            with pytest.raises(errors.EndTooEarlyError, match=match):
                _ = drg.get_end(too_early)

    def test_get_end_none(self, calendars_extended, pp_default):
        """Test `get_end` with None input."""
        cal = calendars_extended
        drg = self.get_drg(cal, pp_default)
        today = get_today(cal)
        assert drg.get_end(None) == (today, today)

    def verify_add_a_row(self, cal, ans, ds_interval, start, end, pp):
        """Test 'add_a_row' as True."""
        interval_offset = ds_interval.as_offset(cal)
        if start >= ans.sessions[1] + interval_offset:
            with add_a_row(pp):
                drg = self.get_drg(cal, pp, None, ds_interval, True)
                start_ = start - ds_interval.as_offset(cal, one_less=False)
                if not ds_interval.is_monthly:
                    start_ = ans.date_to_session(start_, "next")
                assert drg.daterange == ((start_, end), end)

    @staticmethod
    def get_latest_start(
        ds_interval: DOInterval,
        end: pd.Timestamp,
        cal: xcals.ExchangeCalendar,
    ) -> pd.Timestamp | None:
        """Return latest viable start date for DOInterval to `end`.

        None if latest start would be earlier than `cal.first_session`.
        """
        last_indice_right = ds_interval.as_offset_ms.rollforward(end)
        latest_start = last_indice_right - ds_interval.as_offset()
        if latest_start < cal.first_session:
            return None
        return latest_start

    @staticmethod
    def match(
        duration: pd.Timedelta,
        ds_interval: TDInterval,
        start: pd.Timestamp,
        end: pd.Timestamp,
        pp: dict,
    ) -> str:
        """Return match str for `PricesUnavailableIntervalPeriodError` error."""
        return re.escape(
            f"Period does not span a full indice of length {ds_interval}."
            f"\nPeriod duration evaluated as {duration}."
            f"\nPeriod start date evaluated as {start}.\nPeriod end"
            f" date evaluated as {end}.\nPeriod dates evaluated from parameters: {pp}."
        )

    @staticmethod
    def match_do(
        drg: m.GetterDaily,
        start: pd.Timestamp,
        end: pd.Timestamp,
        latest_start: pd.Timestamp,
    ) -> str:
        """Return match str for `PricesUnavailableDOIntervalPeriodError` error."""
        return re.escape(
            "Period evaluated as being shorter than one interval at"
            f" {drg.final_interval}.\nPeriod start date evaluates as {start}"
            f" although needs to be no later than {latest_start} to cover one"
            f" interval.\nPeriod end date evaluates to {end}."
            f"\nPeriod evaluated from parameters: {drg.pp}."
        )

    def assertions_dointerval(
        self,
        ds_interval: DOInterval,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cal: xcals.ExchangeCalendar,
        ans: Answers,
        drg: m.GetterDaily,
        pp: dict,
    ):
        """Make assertions for when ds_interval is a DOInterval."""
        latest_start = self.get_latest_start(ds_interval, end, cal)
        if latest_start is None:
            return
        elif start > latest_start:
            msg = self.match_do(drg, start, end, latest_start)
            with pytest.raises(
                errors.PricesUnavailableDOIntervalPeriodError, match=msg
            ):
                _ = drg.daterange
        else:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)

    @staticmethod
    def do_bounds_from_end(
        ds_interval: DOInterval,
        end: pd.Timestamp,
        end_is_now: bool,
        duration: pd.offsets.BaseOffset | None,
        start: pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return expected start and end for DOInterval and duration to end."""
        one_day = helpers.ONE_DAY
        if end_is_now:
            last_indice_right = ds_interval.as_offset_ms.rollforward(end + one_day)
            # end_ is end from which to evalute start as end_ - duration
            end_ = last_indice_right - one_day
        else:
            last_indice_right = ds_interval.as_offset_ms.rollback(end + one_day)
            end = end_ = last_indice_right - one_day
        if duration is not None:
            if isinstance(duration, pd.offsets.CustomBusinessDay) and not duration.n:
                # otherwise else expression would NOT set start to end_ if end_ is not
                # a session
                duration = pd.Timedelta(0)
            start = end_ - duration
        start = ds_interval.as_offset_ms.rollforward(start)
        assert start is not None
        diff_months = (last_indice_right.to_period("M") - start.to_period("M")).n
        _, excess_months = divmod(diff_months, ds_interval.freq_value)
        start += pd.DateOffset(months=excess_months)  # type: ignore[operator]
        return start, end

    @staticmethod
    def do_bounds_from_start(
        ds_interval: DOInterval,
        start: pd.Timestamp,
        duration: pd.Timedelta,
        today: pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return expected start and end for DOInterval and duration from start."""
        one_day = helpers.ONE_DAY
        start = ds_interval.as_offset_ms.rollforward(start)
        if isinstance(duration, pd.offsets.CustomBusinessDay) and not duration.n:
            # otherwise else expression would NOT set start to end_ if end_ is not
            # a session
            duration = pd.Timedelta(0)
        end = start + duration
        last_indice_right = ds_interval.as_offset_ms.rollback(end + one_day)
        diff_months = (last_indice_right.to_period("M") - start.to_period("M")).n
        _, excess_months = divmod(diff_months, ds_interval.freq_value)
        end = last_indice_right - pd.DateOffset(months=excess_months) - one_day
        end = min(end, today)
        return start, end

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    @hyp.settings(deadline=500)
    def test_daterange_start_end(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` for with period parameters as `pp_start_end_sessions`."""
        cal, ans = calendars_with_answers_extended
        pp = data.draw(stmp.pp_start_end_sessions(cal.name))
        start, end = pp["start"], pp["end"]

        # Dedicated check with interval as one day
        drg = self.get_drg(cal, pp, None, TDInterval.D1, True)
        assert drg.daterange == ((start, end), end)

        # Verify according to nature of ds_interval
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        if isinstance(ds_interval, DOInterval):
            start, end = self.do_bounds_from_end(ds_interval, end, False, None, start)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif start + ds_interval.as_offset(cal) <= end:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)
        else:
            duration = cal.sessions_distance(start, end) * cal.day
            msg = self.match(duration, ds_interval, start, end, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    def test_daterange_start_end_ool(self):
        """Test `daterange` with ool 'start'/'end' parameters.

        Assumed covered by tests `test_get_start` and `test_get_end` which
        test dependencies of `daterange` that handle out-of-limit.
        """
        assert True

    @hyp.given(ds_interval=stmp.intervals_non_intraday())
    @hyp.settings(deadline=500)
    @pytest.mark.parametrize("limit_idx", [0, 50])
    def test_daterange_add_a_row_errors(
        self,
        calendars_with_answers_extended,
        limit_idx,
        ds_interval,
        one_day,
    ):
        """Test `daterange` raises expected errors when 'add_a_row' True.

        Notes
        -----
        These errors are not tested for each valid combination of
        period parameters, but rather only a single combination that
        provokes the errors. Why? Assumes implementation of 'add_a_row' is
        independent of the combination of period parameters (i.e. assumes
        that a 'start' value is evaluated from period parameters before
        implementing 'add_a_row').

        Test parameterized to run with limit as left calendar bound (to
        explore possible calendar-related errors at the bound) and to the
        right of the left calendar bound.
        """

        def match_msg(limit: pd.Timestamp) -> str:
            return re.escape(
                "Prices unavailable as start would resolve to an earlier session than"
                " the earliest session for which price data is available. The earliest"
                f" session for which prices are available is {helpers.fts(limit)}.\nNB"
                " range start falls earlier than first available session due only to"
                " 'add_a_row=True'."
            )

        cal, ans = calendars_with_answers_extended
        pp = get_pp_default()

        limit = ans.sessions[limit_idx]
        today = get_today(cal)

        if ds_interval.is_monthly:
            start_exp, _ = self.do_bounds_from_end(
                ds_interval, today, True, None, limit
            )
            earliest_valid = start_exp + one_day
        else:
            earliest_valid = limit + ds_interval.as_offset(cal, one_less=False)
            earliest_valid = ans.date_to_session(earliest_valid, "next")
        pp["start"] = earliest_valid
        pp["add_a_row"] = True
        drg = self.get_drg(cal, pp, limit, ds_interval, strict=True)
        rtrn_start = drg.daterange[0][0]  # on limit

        if ds_interval.is_daily:
            assert rtrn_start == limit
        else:
            assert rtrn_start == start_exp

        if ds_interval.is_monthly:
            prior_session = ans.date_to_session(pp["start"], "previous")
            if prior_session == pp["start"]:
                prior_session = ans.get_prev_session(prior_session)
        else:
            prior_session = ans.get_prev_session(pp["start"])
        pp["start"] = prior_session
        drg = self.get_drg(cal, pp, limit, ds_interval, strict=False)  # left of limit
        with not_add_a_row(pp):
            drg_not_add_a_row = self.get_drg(cal, pp, limit, ds_interval, strict=False)
        # when strict=False should return as if add_a_row=False.
        assert drg.daterange[0][0] == drg_not_add_a_row.daterange[0][0]

        drg = self.get_drg(cal, pp, limit, ds_interval, strict=True)  # left of limit
        with pytest.raises(errors.StartTooEarlyError, match=match_msg(limit)):
            _ = drg.daterange

    @hyp.given(data=sthyp.data(), ds_interval=stmp.intervals_non_intraday())
    @hyp.settings(deadline=500)
    def test_daterange_start_only(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` with 'start' as only non-default period parameter."""
        cal, ans = calendars_with_answers_extended
        pp = get_pp_default()
        today = get_today(cal)

        start = pp["start"] = data.draw(stmp.calendar_session(cal.name, (None, today)))
        drg = self.get_drg(cal, pp, None, ds_interval, True)

        if isinstance(ds_interval, DOInterval):
            start, end = self.do_bounds_from_end(ds_interval, today, True, None, start)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif start + ds_interval.as_offset(cal) <= today:
            assert drg.daterange == ((start, today), today)
            self.verify_add_a_row(cal, ans, ds_interval, start, today, pp)
        else:
            duration = cal.sessions_distance(start, today) * cal.day
            msg = self.match(duration, ds_interval, start, today, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    @hyp.settings(deadline=500)
    def test_daterange_end_only(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` with 'end' as only non-default period parameter."""
        cal, ans = calendars_with_answers_extended
        pp = get_pp_default()

        end = pp["end"] = data.draw(stmp.calendar_session(cal.name, (None, None)))
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        start = cal.first_session

        if ds_interval.is_daily:
            end_session = ans.date_to_session(end, "previous")
            num_sessions = (
                ans.sessions.get_loc(end_session) - ans.sessions.get_loc(start) + 1
            )
            if num_sessions < ds_interval.days:
                with pytest.raises(errors.PricesUnavailableIntervalPeriodError):
                    _ = drg.daterange
            else:
                assert drg.daterange == ((None, end), end)

        if ds_interval.is_monthly:
            start_exp, end_exp = self.do_bounds_from_end(
                ds_interval, end, False, None, start
            )
            if end_exp < start_exp:
                with pytest.raises(errors.PricesUnavailableDOIntervalPeriodError):
                    _ = drg.daterange
            else:
                assert drg.daterange == ((None, end_exp), end_exp)

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    @hyp.settings(deadline=500)
    def test_daterange_duration_days_start(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` for with period parameters as `pp_days_start_session`."""
        cal, ans = calendars_with_answers_extended
        pp = data.draw(stmp.pp_days_start_session(cal.name, start_will_roll_to_ms=True))

        start, days = pp["start"], pp["days"]
        one_interval_in = start + ds_interval.as_offset(cal)

        duration = (days - 1) * cal.day
        end = start + duration

        # Dedicated check with interval as one day
        drg = self.get_drg(cal, pp, None, TDInterval.D1, True)
        assert drg.daterange == ((start, end), end)

        # Verify for nature of ds_interval
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        if isinstance(ds_interval, DOInterval):
            today = get_today(cal)
            start, end = self.do_bounds_from_start(ds_interval, start, duration, today)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif one_interval_in <= end:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)
        else:
            duration = days * cal.day
            msg = self.match(duration, ds_interval, start, end, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    def test_daterange_duration_days_start_ool(self, calendars_extended, pp_default):
        """Test `daterange` ool errors with params as day duration and 'start'."""
        cal, pp = calendars_extended, pp_default
        today = get_today(cal)

        pp["start"] = today
        pp["days"] = 1
        drg = self.get_drg(cal, pp)
        assert drg.daterange == ((today, today), today)  # on today
        for days in range(2, 5):  # right of today
            pp["days"] = days
            drg = self.get_drg(cal, pp)
            assert drg.daterange == ((today, today), today)

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    @hyp.settings(deadline=500)
    def test_daterange_duration_days_end(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` for with period parameters as `pp_days_end_session`.

        Also tests with period parmaeters as simply `pp_days`.
        """
        cal, ans = calendars_with_answers_extended
        strtgy = sthyp.one_of(
            stmp.pp_days(cal.name), stmp.pp_days_end_session(cal.name)
        )
        pp = data.draw(strtgy)

        end, days = pp["end"], pp["days"]
        if end is None:
            end = get_today(cal)
            end_is_now = True
        else:
            end_is_now = False

        duration = (days - 1) * cal.day
        start = end - duration

        # First dedicated check with interval as one day
        drg = self.get_drg(cal, pp, None, TDInterval.D1, True)
        assert drg.daterange == ((start, end), end)

        # Verify for nature of ds_interval.
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        if isinstance(ds_interval, DOInterval):
            start, end = self.do_bounds_from_end(ds_interval, end, end_is_now, duration)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif end - ds_interval.as_offset(cal) >= start:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)
        else:
            duration = days * cal.day
            msg = self.match(duration, ds_interval, start, end, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    def test_daterange_duration_days_end_oolb(
        self, calendars_with_answers_extended, pp_default
    ):
        """Test `daterange` ool errors with params as day duration and 'end'."""
        (cal, ans), pp = calendars_with_answers_extended, pp_default

        bound = ans.first_session
        days = 3
        pp["end"] = end = ans.sessions[days - 1]
        pp["days"] = days

        for strict in [True, False]:  # on left bound
            drg = self.get_drg(cal, pp, strict=strict)
            assert drg.daterange == ((bound, end), end)

        error_msg = re.escape(
            "Prices unavailable as start would resolve to an earlier date than"
            f" the earliest date of calendar '{cal.name}'. The calendar's earliest"
            f" date is {helpers.fts(ans.first_session)} (this bound should coincide"
            " with the earliest date for which daily price data is available)."
        )
        for _ in range(3):  # left of bound
            days += 1
            pp["days"] = days
            drg = self.get_drg(cal, pp, strict=False)
            assert drg.daterange == ((bound, end), end)
            drg = self.get_drg(cal, pp, strict=True)
            with pytest.raises(errors.StartOutOfBoundsError, match=error_msg):
                _ = drg.daterange

        limit = ans.sessions[50]
        days = 3
        pp["end"] = end = ans.sessions[50 + days - 1]
        pp["days"] = days

        for strict in [True, False]:  # on left limit
            drg = self.get_drg(cal, pp, limit=limit, strict=strict)
            assert drg.daterange == ((limit, end), end)

        for i in range(1, 4):  # left of limit
            days += 1
            pp["days"] = days
            drg = self.get_drg(cal, pp, limit=limit, strict=False)
            assert drg.daterange == ((limit, end), end)
            start = ans.sessions[50 - i]
            error_msg = re.escape(
                f"Prices unavailable as start ({helpers.fts(start)}) is earlier than"
                " the earliest session for which price data is available. The earliest"
                f" session for which prices are available is {helpers.fts(limit)}."
            )
            drg = self.get_drg(cal, pp, limit=limit, strict=True)
            with pytest.raises(errors.StartTooEarlyError, match=error_msg):
                _ = drg.daterange

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    @hyp.settings(deadline=500)
    def test_daterange_duration_cal_start(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` for with period parameters as `pp_caldur_start_session`."""
        cal, ans = calendars_with_answers_extended
        pp = data.draw(stmp.pp_caldur_start_session(cal.name))

        start = pp["start"]
        duration = pd.DateOffset(
            days=-1,
            weeks=pp["weeks"],
            months=pp["months"],
            years=pp["years"],
        )
        end = start + duration
        end = ans.date_to_session(end, "previous")

        # Dedicated check with interval as one day
        drg = self.get_drg(cal, pp, None, TDInterval.D1, True)
        assert drg.daterange == ((start, end), end)

        # Verify for nature of interval
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        if isinstance(ds_interval, DOInterval):
            today = get_today(cal)
            start, end = self.do_bounds_from_start(ds_interval, start, duration, today)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif start + ds_interval.as_offset(cal) <= end:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)
        else:
            duration = cal.sessions_distance(start, end) * cal.day
            msg = self.match(duration, ds_interval, start, end, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    def test_daterange_duration_cal_start_ool(self, calendars_extended, pp_default):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in calendar terms and 'start'.
        """
        cal, pp = calendars_extended, pp_default
        today = get_today(cal)

        pp["start"] = start = today - pd.Timedelta(6, "D")
        pp["weeks"] = 1
        drg = self.get_drg(cal, pp)
        assert drg.daterange == ((start, today), today)  # on today

        pp["weeks"] = 2  # right of today
        drg = self.get_drg(cal, pp)
        assert drg.daterange == ((start, today), today)

    @hyp.given(
        data=sthyp.data(),
        ds_interval=stmp.intervals_non_intraday(),
    )
    def test_daterange_duration_cal_end(
        self, calendars_with_answers_extended, data, ds_interval
    ):
        """Test `daterange` for with period parameters as `pp_caldur_end_session`.

        Also tests with period parmaeters as simply `pp_caldur`.
        """
        cal, ans = calendars_with_answers_extended
        strtgy = sthyp.one_of(stmp.pp_caldur(), stmp.pp_caldur_end_session(cal.name))
        pp = data.draw(strtgy)

        end = pp["end"]
        duration = pd.DateOffset(
            days=-1,
            weeks=pp["weeks"],
            months=pp["months"],
            years=pp["years"],
        )
        if end is None:
            end = get_today(cal)
            end_is_now = True
        else:
            end_is_now = False

        start = end - duration
        start = ans.date_to_session(start, "next")

        # First, dedicated check with interval as one day.
        drg = self.get_drg(cal, pp, None, TDInterval.D1, True)
        assert drg.daterange == ((start, end), end)

        # Verify according to nature of interval.
        drg = self.get_drg(cal, pp, None, ds_interval, True)
        if isinstance(ds_interval, DOInterval):
            start, end = self.do_bounds_from_end(ds_interval, end, end_is_now, duration)
            self.assertions_dointerval(ds_interval, start, end, cal, ans, drg, pp)
        elif end - ds_interval.as_offset(cal) >= start:
            assert drg.daterange == ((start, end), end)
            self.verify_add_a_row(cal, ans, ds_interval, start, end, pp)
        else:
            duration = cal.sessions_distance(start, end) * cal.day
            msg = self.match(duration, ds_interval, start, end, pp)
            with pytest.raises(errors.PricesUnavailableIntervalPeriodError, match=msg):
                _ = drg.daterange

    @pytest.mark.parametrize("limit_idx", [0, 100])
    def test_daterange_duration_cal_end_oolb(
        self, calendars_with_answers_extended, pp_default, limit_idx
    ):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in calendar terms and 'end'.

        Notes
        -----
        Does not test for 'start' evaluating to on limit as not feasilble
        to do so with a duration based on calendar terms. Consider setting
        'end' to, say, a week after the limit. It's not possible to know
        if 'end' is a session or a non-session. If it's a non-session then
        'end' will first evaluate to the nearest prior session and 'start'
        will then be evaluated as the duration offset from that prior
        session. Rather than coincide with the bound, the evalutaed 'start'
        would fall before the bound, with the consequence of raising
        StartTooEarlyError.
        """
        (cal, ans), pp = calendars_with_answers_extended, pp_default

        limit = ans.sessions[limit_idx]
        pp["end"] = end = ans.sessions[limit_idx + 1]
        pp["weeks"] = 1  # left of limit
        drg = self.get_drg(cal, pp, limit=limit, strict=False)
        assert drg.daterange == ((limit, end), end)

        start = end - pd.Timedelta(6, "D")
        error_msg = re.escape(
            f"Prices unavailable as start ({helpers.fts(start)}) is earlier than"
            " the earliest session for which price data is available. The earliest"
            f" session for which prices are available is {helpers.fts(limit)}."
        )
        drg = self.get_drg(cal, pp, limit=limit, strict=True)
        with pytest.raises(errors.StartTooEarlyError, match=error_msg):
            _ = drg.daterange


class TestGetterIntraday:
    """Tests for methods and properties of `m.GetterIntraday` class.

    Notes
    -----
    `GetterIntraday.daterange` property tested using GetterIntraday
    methods `get_end`, `get_start` and property `end_now` on the basis that
    these are comprehensively tested in their own right.

    The GetterIntraday.daterange` property is not tested directly for the
    effect of `end_alignment` and `ds_interval` on the end of a daterange.
    Rather, this effect is tested for comprehensively under the
    `test_get_end*` tests that test the `get_end` method on which
    daterange depends for the evaluation of the daterange end.
    """

    def get_drg(
        self,
        calendar: xcals.ExchangeCalendar,
        pp: dict,
        composite_calendar: calutils.CompositeCalendar | None = None,
        delay: pd.Timedelta = pd.Timedelta(0),
        limit: pd.Timestamp | None = None,
        ignore_breaks: bool | dict[intervals.BI, bool] = False,
        interval: TDInterval | None = None,
        ds_interval: TDInterval | None = None,
        anchor: Anchor = Anchor.OPEN,
        end_alignment: Alignment = Alignment.BI,
        strict=True,
    ) -> m.GetterIntraday:
        """Get m.GetterIntraday with default arguments unless otherwise passed."""
        if composite_calendar is None:
            composite_calendar = calutils.CompositeCalendar([calendar])
        limit = calendar.first_minute if limit is None else limit
        return m.GetterIntraday(
            calendar,
            composite_calendar,
            delay,
            limit,
            ignore_breaks,
            pp,
            interval,
            ds_interval,
            anchor,
            end_alignment,
            strict,
        )

    def test_constructor_properties(self, xlon_calendar, pp_default, one_min):
        """Test properties that expose constructor parameters."""
        cal = xlon_calendar
        cc = calutils.CompositeCalendar([cal])
        delay = pd.Timedelta(0)
        limit = cal.first_minute

        # required arguments only, options as default
        for drg in (
            m.GetterIntraday(
                cal, cc, delay, limit, False, pp_default
            ),  # options as default
            # explicitly pass arguments as default values
            m.GetterIntraday(
                cal,
                cc,
                delay,
                limit,
                False,
                pp_default,
                None,
                None,
                Anchor.OPEN,
                Alignment.BI,
                True,
            ),
        ):
            assert drg.cal == cal
            assert drg.pp == pp_default
            assert drg.limit == limit
            assert drg.ds_interval is None
            assert drg.strict
            assert drg.anchor is Anchor.OPEN
            assert drg.alignment is Alignment.FINAL
            assert drg.end_alignment is Alignment.BI
            assert not drg.ignore_breaks
            with pytest.raises(ValueError, match="`interval` has not been set."):
                _ = drg.interval is None
            interval = TDInterval.T5
            drg.interval = interval
            assert drg.interval is interval
            assert drg.ds_factor == 1

        pp = pp_default["start"] = cal.minutes[333]
        cc = calutils.CompositeCalendar([cal])
        limit = cal.minutes[222]
        ignore_breaks = {
            intervals.TDInterval.T1: False,
            intervals.TDInterval.T5: False,
            intervals.TDInterval.H1: True,
        }
        interval = TDInterval.T5
        ds_interval = TDInterval.H2
        anchor = Anchor.WORKBACK
        end_alignment = Alignment.FINAL
        strict = False

        drg = self.get_drg(
            cal,
            pp,
            cc,
            delay,
            limit,
            ignore_breaks,
            interval,
            ds_interval,
            anchor,
            end_alignment,
            strict,
        )

        assert drg.pp == pp
        assert drg.limit == limit
        assert drg.interval is interval
        assert drg.ds_interval is ds_interval
        assert drg.strict == strict
        assert drg.anchor == anchor
        assert drg.alignment == Alignment.BI  # because anchor workback
        assert drg.end_alignment == end_alignment
        assert drg.ds_factor == ds_interval // interval
        assert not drg.ignore_breaks
        drg.interval = intervals.TDInterval.H1
        assert drg.ignore_breaks

        # verify can pass limit as callable
        limit_T1 = limit + one_min

        def mock_limit(interval: intervals.TDInterval) -> pd.Timestamp:
            if interval is TDInterval.T1:
                return limit_T1
            return limit

        drg = m.GetterIntraday(cal, cc, delay, mock_limit, False, pp_default)
        drg.interval = TDInterval.T1
        assert drg.limit == limit_T1
        drg.interval = TDInterval.T2
        assert drg.limit == limit

    def test_intervals(self, xlon_calendar, pp_default):
        """Test interval, ds_interval and final_interval properties."""
        # pylint: disable=too-complex
        cal, pp = xlon_calendar, pp_default

        valid_intervals = [
            TDInterval.T1,
            TDInterval.T1221,
            TDInterval.H1,
            TDInterval.H22,
        ]
        invalid_intervals = [
            TDInterval.D1,
            TDInterval.D111,
            DOInterval.M1,
            DOInterval.M12,
        ]

        # setting interval with no ds_interval

        # setting interval property directly
        drg = self.get_drg(cal, pp)
        with pytest.raises(ValueError, match="`interval` has not been set."):
            _ = drg.interval is None
        for interval in valid_intervals:
            drg.interval = interval
            assert drg.interval == interval
            assert drg.final_interval == interval
            assert drg.end_alignment_interval == interval
        for interval in invalid_intervals:
            match = (
                f"`interval` must be less than one day although receieved '{interval}'."
            )
            with pytest.raises(ValueError, match=match):
                _ = drg.interval = interval

        # setting interval property via constructor
        for interval in valid_intervals:
            drg = self.get_drg(cal, pp, interval=interval)
            assert drg.interval == interval
            assert drg.final_interval == interval
            assert drg.end_alignment_interval == interval

        for interval in invalid_intervals:
            match = (
                f"`interval` must be less than one day although receieved '{interval}'."
            )
            with pytest.raises(ValueError, match=match):
                _ = drg = self.get_drg(cal, pp, interval=interval)

        # setting ds_interval with no interval

        for ds_interval in invalid_intervals:
            match = (
                "`ds_interval` must be an intraday interval although received"
                f" {ds_interval}."
            )
            with pytest.raises(ValueError, match=match):
                _ = self.get_drg(cal, pp, ds_interval=ds_interval)

        for ds_interval in valid_intervals:
            drg = self.get_drg(cal, pp, ds_interval=ds_interval)
            assert drg.final_interval == ds_interval

        # setting interval property with ds_interval set
        ds_interval = TDInterval.T10

        # setting interval property directly
        drg = self.get_drg(cal, pp, ds_interval=ds_interval)
        for interval in (TDInterval.T1, TDInterval.T2, TDInterval.T5, TDInterval.T10):
            drg.interval = interval
            assert drg.interval == interval
            assert drg.final_interval == ds_interval

        for interval in (TDInterval.T3, TDInterval.T4, TDInterval.T7, TDInterval.T9):
            match = re.escape(
                "`interval` must be a factor of `ds_interval` although received"
                f" '{interval}' (ds_interval is '{ds_interval}')."
            )
            with pytest.raises(ValueError, match=match):
                _ = drg.interval = interval

        for interval in (TDInterval.T11, TDInterval.H1):
            match = re.escape(
                "`interval` cannot be higher than ds_interval although received"
                f" '{interval}' (ds_interval is '{ds_interval}')."
            )
            with pytest.raises(ValueError, match=match):
                _ = drg.interval = interval

        # setting interval property via constructor
        for interval in (TDInterval.T1, TDInterval.T2, TDInterval.T5, TDInterval.T10):
            drg = self.get_drg(cal, pp, interval=interval, ds_interval=ds_interval)
            assert drg.interval == interval
            assert drg.final_interval == ds_interval

        for interval in (TDInterval.T3, TDInterval.T4, TDInterval.T7, TDInterval.T9):
            match = re.escape(
                "`interval` must be a factor of `ds_interval` although received"
                f" '{interval}' (ds_interval is '{ds_interval}')."
            )
            with pytest.raises(ValueError, match=match):
                _ = drg = self.get_drg(
                    cal, pp, interval=interval, ds_interval=ds_interval
                )

        for interval in (TDInterval.T11, TDInterval.H1):
            match = re.escape(
                "`interval` cannot be higher than ds_interval although received"
                f" '{interval}' (ds_interval is '{ds_interval}')."
            )
            with pytest.raises(ValueError, match=match):
                _ = self.get_drg(cal, pp, interval=interval, ds_interval=ds_interval)

    def test_end_alignment_properties(self, xlon_calendar, pp_default):
        """Test end_alignment and `end_alignment_interval properties (not effect of)."""
        cal, pp = xlon_calendar, pp_default

        interval, ds_interval = TDInterval.T1, TDInterval.T10
        drg = self.get_drg(
            cal,
            pp,
            interval=interval,
            ds_interval=ds_interval,
            end_alignment=Alignment.BI,
        )
        assert drg.end_alignment == Alignment.BI
        assert drg.end_alignment_interval == interval

        drg = self.get_drg(
            cal,
            pp,
            interval=interval,
            ds_interval=ds_interval,
            end_alignment=Alignment.FINAL,
        )
        assert drg.end_alignment == Alignment.FINAL
        assert drg.end_alignment_interval == ds_interval

    def test_get_start(
        self, calendars_with_answers_extended, one_min, base_ds_interval, pp_default
    ):
        """Test `get_start`.

        Notes
        -----
        Test assumes value passed to `get_start` will be a time that
        represents a trading minute or a non-trading minute within calendar
        bounds. This will be the case for internal calls (within the
        GetterIntraday class) and for input that has been parsed initially
        with `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.
        """
        # pylint: disable=too-complex
        cal, ans = calendars_with_answers_extended
        pp = pp_default
        bi, ds_interval, interval = base_ds_interval
        drg_kwargs = dict(interval=bi, ds_interval=ds_interval)

        drg = self.get_drg(cal, pp, **drg_kwargs)

        # --trading_minutes and close--

        # get a session, with a break if there is one.
        sessions = ans.sessions_sample
        sessions = sessions.intersection(ans.sessions_with_break)
        if sessions.empty:
            sessions = ans.sessions_sample
        session = sessions[len(sessions) // 2]

        # get bounds of session, or bounds of subsessions if session has a break.
        am_open = ans.opens[session]
        pm_close = ans.closes[session]
        next_am_open = ans.opens[ans.get_next_session(session)]
        if ans.session_has_break(session):
            am_close = ans.break_starts[session]
            pm_open = ans.break_ends[session]
            boundss = [
                (am_open, am_close, pm_open),
                (pm_open, pm_close, next_am_open),
            ]
        else:
            boundss = [(am_open, pm_close, next_am_open)]

        # For each interval, test minutes at bounds of each indice.
        interval_offset = interval.as_offset()

        def get_minutes(start: pd.Timestamp) -> list[pd.Timestamp]:
            minutes = [start + one_min]
            if interval is TDInterval.T1:
                return minutes
            if interval is not TDInterval.T2:
                # extra check 'within' bounds
                minutes.append(start + interval_offset - one_min)
            minutes.append(start + interval_offset)
            return minutes

        def verify_starts(open_, close, next_open):
            minute = start = open_
            assert drg.get_start(minute) == start
            next_start = start + interval_offset
            while next_start < close:
                # assert one min earlier than bound does not resolve to next_start
                assert drg.get_start(start) != next_start
                for minute in get_minutes(start):
                    assert drg.get_start(minute) == next_start
                # assert one min later than bound does not resolve to next_start
                assert drg.get_start(minute + one_min) != next_start
                start = next_start
                next_start += interval_offset

            assert drg.get_start(start + one_min) == next_open
            assert drg.get_start(close) == next_open

        for bounds in boundss:
            verify_starts(*bounds)

        today = get_today(cal)
        # --non-trading minutes-- (also includes close minutes)
        for minutes, _, next_session in ans.non_trading_minutes:
            if next_session > today:
                # testing limits covered by `test_get_start_too_late`
                continue
            for minute in minutes:
                assert drg.get_start(minute) == cal.session_open(next_session)

        # --break minutes--
        # -1 becuase `drg.get_start` depends on `cal.next_session`. Normally not
        # an issue as can reasonably expect to get next_session following today,
        # although `ans.break_minutes` includes break minutes for the last session.
        for minutes, session in ans.break_minutes:
            if session >= today:
                # testing limits covered by `test_get_start_too_late`
                continue
            for minute in minutes:
                assert drg.get_start(minute) == ans.first_pm_minutes[session]

        # --out-of-limit--
        # includes verifying effect of hard_strict
        session = ans.sessions[len(ans.sessions) // 2]
        limit = ans.opens[session]
        too_early = limit - one_min
        drg = self.get_drg(cal, pp, limit=limit, strict=False)
        assert drg.get_start(too_early) == limit
        assert drg.get_start(too_early, hard_strict=False) == limit

        match = re.escape(
            f"Prices unavailable as start ({helpers.fts(too_early)}) is earlier"
            " than the earliest minute for which price data is available. The earliest"
            f" minute for which prices are available is {helpers.fts(limit)}."
        )
        with pytest.raises(errors.StartTooEarlyError, match=match):
            _ = drg.get_start(too_early, hard_strict=True)

        drg = self.get_drg(cal, pp, limit=limit, strict=True)
        for hard_strict in (True, False):
            with pytest.raises(errors.StartTooEarlyError, match=match):
                _ = drg.get_start(too_early, hard_strict=hard_strict)

    @hyp.given(ds_interval=stmp.intervals_intraday())
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][0])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][1])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][2])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][3])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][4])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][5])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][6])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][7])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][8])
    @hyp.example(conftest.base_ds_intervals_dict[TDInterval.T1][9])
    @hyp.settings(deadline=None)
    def test_get_start_too_late(
        self, calendars_with_answers_extended, one_min, ds_interval
    ):
        """Test `get_start` raises error when evaluates later than now.

        Notes
        -----
        Test assumes value passed to `get_start` will be a time that
        represents a trading minute or a non-trading minute within calendar
        bounds. This will be the case for internal calls (within the
        GetterIntraday class) and for input that has been parsed initially
        with `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.
        """
        cal, ans = calendars_with_answers_extended
        pp = get_pp_default()

        bi = TDInterval.T1
        interval = ds_interval if ds_interval is not None else bi
        drg_kwargs = dict(interval=bi, ds_interval=ds_interval)
        drg = self.get_drg(cal, pp, **drg_kwargs)
        end_now, end_now_accuracy = drg.end_now
        today = get_today(cal)
        prev_session = ans.get_prev_session(today)
        latest_valid_start = max(end_now - interval, ans.opens[today])

        # test on furthest right valid start

        # get start and end of the last valid subsession
        end_last_valid_subsession = ans.last_minutes[today]
        if ans.first_pm_minutes[today] <= latest_valid_start:
            start_last_valid_subsession = ans.first_pm_minutes[today]
        elif ans.first_minutes[today] <= latest_valid_start:
            start_last_valid_subsession = ans.first_minutes[today]
            if ans.session_has_break(today):
                end_last_valid_subsession = ans.last_am_minutes[today]
        elif ans.first_pm_minutes[prev_session] <= latest_valid_start:
            start_last_valid_subsession = ans.first_pm_minutes[prev_session]
            end_last_valid_subsession = ans.last_minutes[prev_session]
        else:
            start_last_valid_subsession = ans.first_minutes[prev_session]
            if ans.session_has_break(prev_session):
                end_last_valid_subsession = ans.last_am_minutes[prev_session]
            else:
                end_last_valid_subsession = ans.last_minutes[prev_session]

        # get the limit of valid start values within the last valid subsession
        limit = start_last_valid_subsession
        while (limit + interval <= latest_valid_start) and (
            limit + interval <= end_last_valid_subsession
        ):
            limit += interval

        pp["start"] = start = limit
        drg = self.get_drg(cal, pp, **drg_kwargs)
        assert drg.daterange == ((start, end_now), end_now_accuracy)

        # test for error on invalid start (too far right)
        pp["start"] = start + one_min
        drg = self.get_drg(cal, pp, **drg_kwargs)
        match = re.escape(
            f"`start` must evaluate to an earlier time than the latest time for which"
            f" prices are available.\nThe latest time for which prices are available"
            f" for calendar '{cal.name}' is {helpers.fts(latest_valid_start)}, although"
            f" `start` evaluates to"
        )
        with pytest.raises(errors.StartTooLateError, match=match) as err:
            _ = drg.daterange
            assert err.value.start >= end_now

    @pytest.mark.parametrize("end_alignment", [Alignment.BI, Alignment.FINAL])
    def test_get_end(
        self,
        calendars_with_answers_extended,
        pp_default,
        one_min,
        base_ds_interval,
        end_alignment,
    ):
        """Test `get_end` with trading minutes.

        Tested trading minutes all resolve to minutes of a specific session.

        Notes
        -----
        Test assumes value passed to `get_end` will be a time that
        represents a trading minute or a non-trading minute within calendar
        bounds. This will be the case for internal calls (within the
        GetterIntraday class) and for input that has been parsed initially
        with `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.
        """
        # pylint: disable=too-complex
        cal, ans = calendars_with_answers_extended
        cc = calutils.CompositeCalendar([cal])

        bi, ds_interval, interval = base_ds_interval
        pp = pp_default
        interval_offset = interval.as_offset()

        # get a session, with a break if there is one.
        sessions = ans.sessions_sample
        sessions = sessions.intersection(ans.sessions_with_break)
        if sessions.empty:
            sessions = ans.sessions_sample
        session = sessions[len(sessions) // 2]
        break_start = ans.break_starts[session]

        def get_bounds(
            ignore_breaks: bool,
        ) -> (
            tuple[
                tuple[list[pd.Timestamp], ...], bool, pd.Timestamp, pd.Timestamp, bool
            ]
            | None
        ):
            # get bounds of session, or bounds of subsessions if session has a break
            am_open = ans.opens[session]
            pm_close = ans.closes[session]
            next_am_open = ans.opens[ans.get_next_session(session)]
            boundss: tuple[list[pd.Timestamp], ...]
            if ans.session_has_break(session) and not ignore_breaks:
                am_close = ans.break_starts[session]
                pm_open = ans.break_ends[session]
                # length needs to reflect pm subsession as concerned with end alignment
                session_length = pm_close - pm_open
                if interval > session_length or interval > (am_close - am_open):
                    # interval > session_length covered by `test_get_end_high_interval`
                    return None
                boundss = (
                    [am_open, am_close, pm_open],
                    [pm_open, pm_close, next_am_open],
                )
            else:
                session_length = pm_close - am_open
                if interval > session_length:
                    # interval > session_length covered by `test_get_end_high_interval`
                    return None
                boundss = ([am_open, pm_close, next_am_open],)

            if ds_interval is None or end_alignment is Alignment.BI:
                end_alignment_interval = bi
            else:
                end_alignment_interval = ds_interval

            modulus: pd.Timedelta = session_length % end_alignment_interval
            last_indice_is_aligned = not modulus
            end_of_last_aligned_indice = pm_close - modulus
            right_of_unaligned_indice = (
                end_of_last_aligned_indice + end_alignment_interval
            )
            trades_after_close_in_unaligned_indice = not last_indice_is_aligned and (
                right_of_unaligned_indice > next_am_open
                or cc.session_close(session) > pm_close
            )
            return (
                boundss,
                last_indice_is_aligned,
                end_of_last_aligned_indice,
                right_of_unaligned_indice,
                trades_after_close_in_unaligned_indice,
            )

        def get_minutes(
            end: pd.Timestamp, next_end: pd.Timestamp
        ) -> list[pd.Timestamp]:
            minutes = [end]
            if interval is TDInterval.T1:
                return minutes
            if interval is not TDInterval.T2:
                minutes.append(end + one_min)  # extra check 'within' bounds
            minutes.append(next_end - one_min)
            return minutes

        def verify_ends(
            bounds: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
            drg: m.GetterIntraday,
            last_indice_is_aligned: bool,
            end_of_last_aligned_indice: pd.Timestamp,
            right_of_unaligned_indice: pd.Timestamp,
            trades_after_close_in_unaligned_indice: bool,
        ):
            # verify minutes at bounds of each indice that resolve to end of
            # prior indice. Verify from start of second indice until one minute
            # before end of the last indice that falls fully within session.
            open_, close, next_open = bounds
            end = open_ + interval_offset
            next_end = end + interval_offset
            while next_end <= close:
                expected = (end, end)
                minutes = get_minutes(end, next_end)
                # assert one minute earlier than bound does not resolve to end
                assert drg.get_end(minutes[0] - one_min) != expected
                for minute in minutes:
                    # assert boundary minutes expected to resolve to end do so
                    assert drg.get_end(minute) == expected
                # assert one minute later than bound does not resolve to end
                minute += one_min
                assert drg.get_end(minute) != expected
                end = next_end
                next_end += interval_offset

            # verify end of last indice that falls fully within session.
            assert drg.get_end(end) == (end, end)
            if end != close:
                # if end of that indice not aligned with close, verify one minute before
                # right of that unaligned indice - will resolve to close and be adjusted
                # for end alignment.
                if close == break_start:
                    # indices unaligned with end am session resolve to break_start
                    expected = (close, close)
                elif last_indice_is_aligned:
                    # How can last indice be aligned if end != close? Possible if
                    # end_alignment is BI which aligns although ds_interval does not.
                    expected = (close, close)
                elif trades_after_close_in_unaligned_indice:
                    expected = (end_of_last_aligned_indice, end_of_last_aligned_indice)
                else:
                    expected = (right_of_unaligned_indice, close)
                assert drg.get_end(next_end - one_min) == expected
            else:  # end aligned with close
                expected = (end, end)
            # verify bounds of first indice of next session resolve to 'prior close'
            minutes = [next_open, next_open + interval_offset - one_min]
            for minute in minutes:
                assert drg.get_end(minute) == expected
            # verify one minute beyond bound does not resolve to 'prior close'
            assert drg.get_end(minute + one_min) != expected

        drg = self.get_drg(
            cal, pp, cc, ds_interval=ds_interval, end_alignment=end_alignment
        )
        drg.interval = bi

        rtrn = get_bounds(ignore_breaks=False)
        if rtrn is None:
            return
        boundss, *bounds_info = rtrn
        for bounds in boundss:
            verify_ends(bounds, drg, *bounds_info)

        if cal.name != "XHKG" or ds_interval != intervals.TDInterval.H1:
            return

        # test effect of `ignore_breaks`.
        ignore_breaks = True
        drg = self.get_drg(
            cal,
            pp,
            cc,
            ignore_breaks=ignore_breaks,
            ds_interval=ds_interval,
            end_alignment=end_alignment,
        )
        drg.interval = bi
        boundss, *bounds_info = get_bounds(ignore_breaks=ignore_breaks)
        for bounds in boundss:
            verify_ends(bounds, drg, *bounds_info)

    @pytest.mark.parametrize("end_alignment", [Alignment.BI, Alignment.FINAL])
    def test_get_end_non_trading_minutes(
        self,
        calendars_with_answers_extended,
        pp_default,
        one_min,
        base_ds_interval,
        end_alignment,
    ):
        """Test `get_end` with non trading minutes.

        Notes
        -----
        Test assumes value passed to `get_end` will be a time within
        calendar bounds. This will be the case for internal calls (within
        the GetterIntraday class) and for input that has been parsed
        initially with `market_prices.parsing.parse_timestamp` and
        subsequently with `market_prices.parsing.parse_start_end`.
        """
        # pylint: disable=too-complex, too-many-branches
        cal, ans = calendars_with_answers_extended
        pp = pp_default
        bi, ds_interval, interval = base_ds_interval

        cc = calutils.CompositeCalendar([cal])
        if end_alignment is Alignment.BI or ds_interval is None:
            end_alignment_interval = bi
        else:
            end_alignment_interval = ds_interval

        for ignore_breaks in (False, True):
            # pylint: disable=cell-var-from-loop
            if ignore_breaks and ds_interval != intervals.TDInterval.H1:
                # only interested in ignore_breaks when interval is H1
                continue

            drg = self.get_drg(
                cal,
                pp,
                cc,
                ignore_breaks=ignore_breaks,
                ds_interval=ds_interval,
                end_alignment=end_alignment,
            )
            drg.interval = bi

            def get_session_length(session) -> pd.Timedelta:
                """Get length of `session` or pm subsession."""
                session_minutes = ans.get_session_minutes(session, ignore_breaks)[-1]
                return session_minutes[-1] - session_minutes[0] + one_min

            def _modulus(session) -> pd.Timedelta:
                """Difference between end of last indice and session close.

                Last indice considered as last indice that fully represents
                a trading period according to the end_alignment_interval.
                """
                session_length = get_session_length(session)
                return session_length % end_alignment_interval

            def last_indice_is_aligned(session) -> bool:
                return not _modulus(session)

            def end_of_last_aligned_indice(session) -> pd.Timestamp:
                session_close = ans.closes[session]
                return session_close - _modulus(session)

            def end_of_unaligned_indice(session) -> pd.Timestamp:
                return end_of_last_aligned_indice(session) + end_alignment_interval

            def there_is_trading_after_close_within_an_unaligned_indice(
                session,
            ) -> bool:
                if last_indice_is_aligned(session):
                    return False
                next_open = ans.opens[ans.get_next_session(session)]
                return end_of_unaligned_indice(session) > next_open

            # -- non-trading minutes --  will also include close minutes
            for minutes, session, _ in ans.non_trading_minutes:
                close = ans.closes[session]
                for minute in minutes:
                    if last_indice_is_aligned(session):
                        expected = (close, close)
                    elif (
                        ds_interval is not None
                        and end_alignment == Alignment.FINAL
                        and interval > get_session_length(session)
                    ):
                        if not ignore_breaks and ans.session_has_break(session):
                            open_ = ans.break_ends[session]
                        else:
                            open_ = ans.opens[session]
                        right = open_ + interval
                        next_open = ans.opens[ans.get_next_session(session)]
                        if right > next_open:
                            session_open = ans.opens[session]
                            expected = (session_open, session_open)
                        else:
                            expected = (right, close)
                    elif there_is_trading_after_close_within_an_unaligned_indice(
                        session
                    ):
                        end = end_of_last_aligned_indice(session)
                        expected = (end, end)
                    else:
                        right = end_of_unaligned_indice(session)
                        expected = (right, close)
                    assert drg.get_end(minute) == expected

            # -- break minutes --
            # -1 for same reasons as for get_start (see comment there)
            for minutes, session in ans.break_minutes[:-1]:
                close = ans.break_starts[session]
                for minute in minutes:
                    assert drg.get_end(minute) == (close, close)

    def test_get_end_high_interval(
        self, calendars_with_answers_extended, pp_default, one_min
    ):
        """Test final interval > session length and ends before next trading period."""
        cal, ans = calendars_with_answers_extended
        pp = pp_default

        # get a session, with a break if there is one.
        sessions = ans.sessions_sample
        sessions = sessions.intersection(ans.sessions_with_break)
        if sessions.empty:
            sessions = ans.sessions_sample
        session = sessions[len(sessions) // 2]

        # session and next_session must have same length, otherwise a half day can
        # result in setting up test with a short session_length with consequence
        # that the ds_interval is less than the next_session's  length.
        session = sessions[len(sessions) // 2]
        next_session = ans.get_next_session(session)
        i = 1
        # if middle session doesn't fit, work through them until find one that does.
        while (ans.closes[session] - ans.opens[session]) != (
            ans.closes[next_session] - ans.opens[next_session]
        ):
            session = sessions[i]
            next_session = ans.get_next_session(session)
            i += 1

        for ignore_breaks in (False, True):
            # session or, if break, pm subsession length
            _session_minutes = ans.get_session_minutes(session, ignore_breaks)[-1]
            session_length = _session_minutes[-1] - _session_minutes[0] + one_min

            if session_length > TDInterval.H22:
                # not possible for ds_interval > session_length
                return

            ds_interval = intervals.to_ptinterval(session_length + intervals.ONE_MIN)
            drg = self.get_drg(
                cal,
                pp,
                ignore_breaks=ignore_breaks,
                ds_interval=ds_interval,
                end_alignment=Alignment.FINAL,
            )

            # test bounds that are expected to resolve to right of interval

            # earliest minute that will resolve to right of interval is session close
            if not ignore_breaks and ans.session_has_break(session):
                open_ = ans.break_ends[session]
            else:
                open_ = ans.opens[session]
            right = open_ + ds_interval
            close = ans.closes[session]
            expected = (right, close)
            assert drg.get_end(close - one_min) != expected
            assert drg.get_end(close) == expected

            # latest minute that will resolve to interval right is one minute prior to
            # the next session's close or, if next session has a break, one minute prior
            # to the next sessions's break start.
            next_session = ans.get_next_session(session)
            if not ignore_breaks and ans.session_has_break(next_session):
                close = ans.break_starts[next_session]
            else:
                close = ans.closes[next_session]
            assert drg.get_end(close - one_min) == expected
            assert drg.get_end(close) != expected

    def test_get_end_high_interval2(
        self, pp_default, one_min, xlon_calendar, xhkg_calendar
    ):
        """Test final interval > session length and overlaps next trading period."""
        xlon, cal = xlon_calendar, xhkg_calendar
        cc = calutils.CompositeCalendar([cal, xlon])
        session = "2021-11-03"
        open_, close = cal.session_open_close(session)
        duration = close - open_

        # Test on limit of what will resolve to 'right of interval', in this case
        # xhkg session close as there's no gap xhkg close and xlon open, assert true:
        open_xlon = xlon.session_open(session)
        assert open_xlon == close
        ds_interval = intervals.to_ptinterval(duration)

        ignore_breaks = True
        drg = self.get_drg(
            cal,
            pp_default.copy(),
            composite_calendar=cc,
            ignore_breaks=ignore_breaks,
            ds_interval=ds_interval,
            end_alignment=Alignment.FINAL,
        )

        # earliest minute that will resolve to right of interval is session close.
        prev_close = cal.session_close(cal.previous_session(session))
        assert drg.get_end(close - one_min) == (prev_close, prev_close)
        expected = (close, close)
        assert drg.get_end(close) == expected

        # latest minute that will resolve to right of interval is one minute prior to
        # next session's close.
        next_close = cal.session_close(cal.next_session(session))
        assert drg.get_end(next_close - one_min) == expected
        assert drg.get_end(next_close) == (next_close, next_close)

        # Test beyond limit of what will resolve to 'right of interval', such
        # that will resolve to session open.
        ds_interval = intervals.to_ptinterval(duration + one_min)
        drg = self.get_drg(
            cal,
            pp_default.copy(),
            composite_calendar=cc,
            ignore_breaks=ignore_breaks,
            ds_interval=ds_interval,
            end_alignment=Alignment.FINAL,
        )
        # assert ds_interval would take last interval into xlon session
        assert open_ + ds_interval > open_xlon

        # earliest minute that will resolve to open is session close.
        expected = (open_, open_)
        assert drg.get_end(close) == expected
        assert drg.get_end(close - one_min) != expected

        # latest minute that will resolve to open is one minute prior to
        # next session's close.
        assert drg.get_end(next_close - one_min) == expected
        assert drg.get_end(next_close) != expected

    def test_get_end_ool(self, calendars_with_answers_extended, pp_default, one_min):
        """Test `get_end` with ool input."""
        cal, ans = calendars_with_answers_extended
        limit = ans.opens.iloc[len(ans.sessions) // 2]
        too_early = limit - one_min
        match = re.escape(
            f"Prices unavailable as end ({helpers.fts(too_early)}) is earlier"
            " than the earliest minute for which price data is available. The earliest"
            f" minute for which prices are available is {helpers.fts(limit)}."
        )
        for strict in [True, False]:
            drg = self.get_drg(cal, pp_default, limit=limit, strict=strict)
            with pytest.raises(errors.EndTooEarlyError, match=match):
                _ = drg.get_end(too_early)

    def test_end_now_and_get_end_none(
        self, calendars_with_answers, monkeypatch, pp_default, one_min
    ):
        """Test `get_now` property and `get_end` with None input."""
        (cal, ans), pp = calendars_with_answers, pp_default
        interval = TDInterval.T15

        # test trading_minute
        session = ans.sessions_sample[-4]
        open_ = ans.opens[session]
        now = open_ + pd.Timedelta(20, "T")
        monkeypatch.setattr("pandas.Timestamp.now", lambda *a, **k: now)
        drg = self.get_drg(cal, pp, interval=interval)
        end = open_ + (TDInterval.T15 * 2)  # end is end of current live interval
        assert drg.end_now == drg.get_end(None) == (end, end)

        # test trading minute with delay
        drg = self.get_drg(cal, pp, interval=interval, delay=pd.Timedelta(10, "T"))
        end = open_ + TDInterval.T15
        assert drg.end_now == drg.get_end(None) == (end, end)

        # test non-trading minute
        sessions = ans.sessions_sample.intersection(ans.sessions_with_gap_after)
        if sessions.empty:
            return
        session = sessions[-4]
        close = ans.closes[session]
        now = close + pd.Timedelta(10, "T")
        monkeypatch.setattr("pandas.Timestamp.now", lambda *a, **k: now)
        drg = self.get_drg(cal, pp, interval=TDInterval.T1)
        assert drg.end_now == drg.get_end(None) == (close, close)

        drg = self.get_drg(cal, pp, interval=TDInterval.T1, delay=pd.Timedelta(15, "T"))
        end = close - pd.Timedelta(5, "T") + one_min  # returns right of live indice
        assert drg.end_now == drg.get_end(None) == (end, end)

    def test_get_start_get_end_anchor_effect(
        self, calendars_with_answers_extended, pp_default
    ):
        """Test effect of anchor on `get_start` and `get_end`."""
        cal, ans = calendars_with_answers_extended
        pp = pp_default

        # get a session, with a break if there is one.
        sessions = ans.sessions_sample
        sessions = sessions.intersection(ans.sessions_with_break)
        if sessions.empty:
            sessions = ans.sessions_sample
        session = sessions[len(sessions) // 2]

        open_ = ans.opens[session]
        close = ans.closes[session]

        bi = TDInterval.T5
        dsi = TDInterval.T15

        drg_bi = self.get_drg(cal, pp, interval=bi, anchor=Anchor.OPEN)
        drg_wb = self.get_drg(
            cal, pp, interval=bi, ds_interval=dsi, anchor=Anchor.WORKBACK
        )

        # verify that drg_wb behaves as drg_bi, i.e. aligning based on bi, not dsi
        delta = pd.Timedelta(4, "T")
        minutes = pd.date_range(open_ - delta, close + delta, freq=delta)
        delta = pd.Timedelta(7, "T")
        minutes = minutes.union(pd.date_range(open_ - delta, close + delta, freq=delta))

        for minute in minutes:
            assert drg_wb.get_start(minute) == drg_bi.get_start(minute)
            assert drg_wb.get_end(minute) == drg_bi.get_end(minute)

    @hyp.given(
        data=sthyp.data(),
        base_ds_interval=stmp.base_ds_intervals(),
    )
    @hyp.settings(deadline=None)
    def test_daterange_start_end(self, calendars_extended, data, base_ds_interval):
        """Test `daterange` for with period parameters as `pp_start_end_minutes`."""
        cal = calendars_extended
        pp = data.draw(stmp.pp_start_end_minutes(cal.name))
        start, end = pp["start"], pp["end"]
        bi, ds_interval, interval = base_ds_interval
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=ds_interval)

        start = drg.get_start(start)
        end, end_accuracy = drg.get_end(end)

        final_interval = ds_interval if ds_interval is not None else interval
        if start >= end:
            match = re.escape(
                f"Period does not span a full indice of length {final_interval}."
                f"\nPeriod start date evaluated as {start}.\nPeriod end date evaluated"
                f" as {end_accuracy}.\nPeriod dates evaluated from parameters: {pp}."
            )
            with pytest.raises(
                errors.PricesUnavailableIntervalPeriodError, match=match
            ):
                _ = drg.daterange
            return

        assert drg.daterange == ((start, end), end_accuracy)

        with add_a_row(pp):
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=ds_interval)

            start_session = cal.minute_to_session(start, _parse=False)
            i = cal.sessions.get_loc(start_session)
            if start >= cal.first_pm_minutes.iloc[i]:
                session_start = cal.first_pm_minutes.iloc[i]
            else:
                session_start = cal.first_minutes.iloc[i]

            minutes_i = cal.minutes.get_loc(start)
            start_ = cal.minutes[minutes_i - interval.as_minutes]
            if start == session_start:
                start_ = drg.get_start(start_)
                if session_start == cal.first_pm_minutes.iloc[i]:
                    prev_session_start = cal.first_minutes.iloc[i]
                else:
                    prev_session_start = max(
                        cal.first_minutes.iloc[i - 1], cal.first_pm_minutes.iloc[i - 1]
                    )
                start_ = max(prev_session_start, start_)
            else:
                start_ = max(session_start, start_)
            assert drg.daterange == ((start_, end), end_accuracy)

    def test_daterange_start_end_ool(self):
        """Test `daterange` with ool 'start'/'end' parameters.

        Assumed covered by tests `test_get_start` and `test_get_end` which
        test dependencies of `daterange` that handle out-of-limit.
        """
        assert True

    def test_daterange_start_end_intervalperiod_error(
        self, calendars_with_answers_extended, one_min, one_sec
    ):
        """Test raises `errors.PricesUnavailableIntervalPeriodError`.

        Tests when `start` and `end` represent period less than one interval.
        Tests 'open' and 'workback' anchors independently.
        Tests expected error messages.
        """
        (cal, ans) = calendars_with_answers_extended

        def match(
            final_interval: TDInterval,
            start: pd.Timestamp,
            end: pd.Timestamp,
            pp: dict,
            anchor: Anchor,
            duration: int | None = None,
        ) -> str:
            """Match message for errors.PricesUnavailableIntervalPeriodError.

            `duration` only required if anchor is Anchor.WORKBACK. Should be
            passed as integer representing number of minutes.
            """
            duration_insert = ""
            if anchor is Anchor.WORKBACK:
                assert duration is not None
                duration_ = pd.Timedelta(duration, "T")
                duration_insert = f"\nPeriod duration evaluated as {duration_}."
            return re.escape(
                f"Period does not span a full indice of length {final_interval}."
                f"{duration_insert}\nPeriod start date evaluated as {start}.\nPeriod end"
                f" date evaluated as {end}.\nPeriod dates evaluated from parameters:"
                f" {pp}."
            )

        error = errors.PricesUnavailableIntervalPeriodError

        session = ans.sessions_sample[1]
        i = ans.sessions.get_loc(session)
        session_open = ans.opens.iloc[i]
        prev_session_close = ans.closes.iloc[i - 1]

        bi = TDInterval.T5
        dsi = TDInterval.T15

        # Verify for an interval of a session
        pp = get_pp_default()
        pp["start"] = start = session_open + dsi
        end_bi = start + bi
        end_ds = start + dsi

        for anchor in (Anchor.OPEN, Anchor.WORKBACK):
            # at bi limit
            pp["end"] = end_bi
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=None, anchor=anchor)
            assert drg.daterange == ((start, end_bi), end_bi)
            # inside of limit
            pp["end"] -= one_sec
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=None, anchor=anchor)
            # same start and end for both open and wb as period < bi
            msg = match(bi, start, start, pp, anchor, 0)
            with pytest.raises(error, match=msg):
                _ = drg.daterange

            # at ds limit
            pp["end"] = end_ds
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            assert drg.daterange == ((start, end_ds), end_ds)
            # inside of limit
            pp["end"] -= one_sec
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            end_ = start if anchor is Anchor.OPEN else end_ds - bi
            msg = match(dsi, start, end_, pp, anchor, bi * 2)
            with pytest.raises(error, match=msg):
                _ = drg.daterange

        # Verify for an interval crossing sessions
        def assertions_limit(
            dsi: TDInterval | None,
            start: pd.Timestamp,
            end: pd.Timestamp,
            anchor: Anchor,
        ):
            # limit for last interval of prev session and first interval of session
            pp = get_pp_default()
            pp["start"], pp["end"] = start, end
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            assert drg.daterange == ((start, end), end)

            # limits for last interval / first interval only
            pp["start"] += one_sec
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            assert drg.daterange == ((session_open, end), end)

            pp["start"] = start
            pp["end"] -= one_sec
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            assert drg.daterange == ((start, prev_session_close), prev_session_close)

        def assert_raises(
            dsi: TDInterval | None,
            start: pd.Timestamp,
            end: pd.Timestamp,
            anchor: Anchor,
            duration: int | None = None,
        ):
            pp = get_pp_default()
            pp["start"] = start + one_sec
            pp["end"] = end - one_sec
            drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
            intrvl = bi if dsi is None else dsi
            msg = match(intrvl, session_open, prev_session_close, pp, anchor, duration)
            with pytest.raises(error, match=msg):
                _ = drg.daterange

        def assertions(
            dsi: TDInterval | None,
            start: pd.Timestamp,
            end: pd.Timestamp,
            anchor: Anchor,
            duration: int | None = None,
        ):
            assertions_limit(dsi, start, end, anchor)
            assert_raises(dsi, start, end, anchor, duration)

        start = prev_session_close - bi
        end = session_open + bi
        assertions(None, start, end, Anchor.OPEN)

        start_ds = prev_session_close - dsi
        end_ds = session_open + dsi
        assertions(dsi, start_ds, end_ds, Anchor.OPEN)

        assertions(None, start, end, Anchor.WORKBACK, 0)

        # verify for workback with dsi
        # on limit
        pp = get_pp_default()
        pp["start"] = start = prev_session_close - dsi + bi
        pp["end"] = end = session_open + bi
        anchor = Anchor.WORKBACK
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        assert drg.daterange == ((start, end), end)

        # within limit
        pp["start"] += one_sec
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        msg = match(dsi, prev_session_close - bi, end, pp, anchor, bi * 2)
        with pytest.raises(error, match=msg):
            _ = drg.daterange

        pp["start"] = start
        pp["end"] -= one_sec
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        msg = match(dsi, start, prev_session_close, pp, anchor, bi * 2)
        with pytest.raises(error, match=msg):
            _ = drg.daterange

        # verify when single interval would comprise minutes from both end of
        # previous session and start of session
        delta = pd.Timedelta(3, "T")
        pp = get_pp_default()
        pp["start"] = start = prev_session_close - dsi + delta
        pp["end"] = end = session_open + delta
        # when on limit if bi were to be T1 (NB it's not, it's T5)
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        start_ = prev_session_close - dsi + bi
        msg = match(dsi, start_, prev_session_close, pp, anchor, 10)
        with pytest.raises(error, match=msg):
            _ = drg.daterange
        # although if bi is T1
        bi = TDInterval.T1
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        assert drg.daterange == ((start, end), end)
        # within limit
        pp["start"] += one_sec
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        msg = match(dsi, start + one_min, end, pp, anchor, dsi - one_min)
        with pytest.raises(error, match=msg):
            _ = drg.daterange
        pp["start"] = start
        pp["end"] -= one_sec
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        msg = match(dsi, start, end - one_min, pp, anchor, dsi - one_min)
        with pytest.raises(error, match=msg):
            _ = drg.daterange
        pp["start"] += one_sec
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=dsi, anchor=anchor)
        duration = dsi - (one_min * 2)
        msg = match(dsi, start + one_min, end - one_min, pp, anchor, duration)
        with pytest.raises(error, match=msg):
            _ = drg.daterange

    @hyp.given(base_ds_interval=stmp.base_ds_intervals())
    @hyp.settings(deadline=None)
    @pytest.mark.parametrize("session_limit_idx", [0, 50])
    def test_daterange_add_a_row_errors(
        self,
        calendars_with_answers_extended,
        one_min,
        session_limit_idx,
        base_ds_interval,
    ):
        """Test `daterange` raises expected errors when 'add_a_row' True.

        Notes
        -----
        These errors are not tested for each valid combination of
        period parameters, but rather only a single combination that
        provokes the errors. Why? Assumes implementation of 'add_a_row' is
        independent of the combination of period parameters (i.e. assumes
        that a 'start' value is evaluated from period parameters before
        implementing 'add_a_row').

        Test parameterized to run with limit as left calendar bound (to
        explore possible calendar-related errors at the bound) and to the
        right of the left calendar bound.
        """
        cal, ans = calendars_with_answers_extended
        pp = get_pp_default()
        bi, ds_interval, _ = base_ds_interval

        limit_session = ans.sessions[session_limit_idx]
        limit = ans.first_minutes[limit_session]

        drg_kwargs = dict(limit=limit, interval=bi, ds_interval=ds_interval)

        earliest_valid = limit + one_min
        pp["start"] = earliest_valid
        pp["add_a_row"] = True
        drg = self.get_drg(cal, pp, **drg_kwargs)
        rtrn_start = drg.daterange[0][0]  # evaluates to limit
        assert rtrn_start == limit

        pp["start"] = limit
        # would evaluate to left of limit
        drg = self.get_drg(cal, pp, strict=False, **drg_kwargs)
        with not_add_a_row(pp):
            drg_not_add_a_row = self.get_drg(cal, pp, strict=False, **drg_kwargs)
        # when strict=False should return as if add_a_row=False.
        assert drg.daterange[0][0] == drg_not_add_a_row.daterange[0][0]

        drg = self.get_drg(cal, pp, strict=True, **drg_kwargs)

        match = re.escape(
            "Prices unavailable as start would resolve to an earlier minute than"
            " the earliest minute for which price data is available. The earliest"
            f" minute for which prices are available is {helpers.fts(limit)}.\nNB"
            " range start falls earlier than first available minute due only to"
            " 'add_a_row=True'."
        )
        with pytest.raises(errors.StartTooEarlyError, match=match):
            _ = drg.daterange

    @hyp.given(
        data=sthyp.data(),
        base_ds_interval=stmp.base_ds_intervals(),
    )
    @hyp.settings(deadline=None)
    def test_daterange_start_only_end_None(
        self, calendars_extended, data, base_ds_interval
    ):
        """Test `daterange` with 'start' as only non-default period parameter."""
        cal = calendars_extended
        pp = get_pp_default()
        bi, ds_interval, _ = base_ds_interval
        pp["start"] = start = data.draw(stmp.start_minutes(cal.name))
        drg = self.get_drg(cal, pp, interval=bi, ds_interval=ds_interval)
        end_now, end_now_accuracy = drg.end_now
        rtrn_start = drg.get_start(start)
        assert drg.daterange == ((rtrn_start, end_now), end_now_accuracy)

    @hyp.given(
        data=sthyp.data(),
        base_ds_interval=stmp.base_ds_intervals(),
    )
    @hyp.settings(deadline=None)
    def test_daterange_end_only_start_None(
        self, calendars_extended, base_ds_interval, data
    ):
        """Test `daterange` with 'end' as only non-default period parameter."""
        cal = calendars_extended
        pp = data.draw(stmp.pp_end_minute_only(cal.name))
        bi, ds_interval, _ = base_ds_interval

        drg = self.get_drg(cal, pp, interval=bi, ds_interval=ds_interval)
        rtrn_end, rtrn_end_accuracy = drg.get_end(pp["end"])
        assert drg.daterange == ((drg.limit, rtrn_end), rtrn_end_accuracy)

    def test_daterange_start_None_end_None(self, calendars_extended):
        """Test `daterange` with both 'start' and 'end' as None.

        Tests for interval-dependent limit.
        """
        cal = calendars_extended
        pp = get_pp_default()
        today = get_today(cal)
        limits = {
            TDInterval.T1: cal.session_open(cal.session_offset(today, -5)),
            TDInterval.T5: cal.session_open(cal.session_offset(today, -10)),
            TDInterval.H1: cal.session_open(cal.session_offset(today, -15)),
        }

        def limit_f(interval) -> pd.Timestamp:
            return limits[interval]

        drg = self.get_drg(cal, pp, limit=limit_f)

        for interval, exp_start in limits.items():
            drg.interval = interval
            end_now, end_now_accuracy = drg.end_now
            assert drg.daterange == ((exp_start, end_now), end_now_accuracy)

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=500)
    def test_daterange_duration_cal_start_minute(
        self, calendars_extended, data, one_min
    ):
        """Test `daterange` for with period parameters as `pp_caldur_start_minute`."""
        cal = calendars_extended
        pp = data.draw(stmp.pp_caldur_start_minute(cal.name))

        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        start = pp["start"]
        duration = pd.DateOffset(
            weeks=pp["weeks"],
            months=pp["months"],
            years=pp["years"],
        )
        end = start + duration

        try:
            assert drg.daterange == ((start, end), end)
        except AssertionError:
            assert (
                end.value in cal.opens_nanos
                or end.value in cal.break_ends_nanos
                or end.value not in cal.minutes_nanos
            )
            if end.value in cal.break_ends_nanos:
                end = cal.previous_minute(end) + one_min
            else:
                end = max(cal.previous_minute(end), cal.previous_close(end))
            assert drg.daterange == ((start, end), end)

    def test_daterange_duration_cal_start_minute_ool(
        self, calendars_extended, pp_default, one_min
    ):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in calendar terms and 'start'.
        """
        cal, pp = calendars_extended, pp_default
        bi = TDInterval.T1

        drg = self.get_drg(cal, pp, interval=bi)
        now, now_accuracy = end, end_accuracy = drg.end_now

        start = now - pd.DateOffset(weeks=1)
        if start.value in cal.closes_nanos:
            start = cal.previous_minute(start)
            end = end_accuracy = end - one_min

        pp["start"] = start
        pp["weeks"] = 1
        drg = self.get_drg(cal, pp, interval=bi)
        assert drg.daterange == ((start, end), end_accuracy)  # on now, or near abouts

        pp["weeks"] = 2  # right of now
        drg = self.get_drg(cal, pp, interval=bi)
        assert drg.daterange == ((start, now), now_accuracy)

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=None)
    def test_daterange_duration_cal_end_minute(self, calendars_extended, data):
        """Test `daterange` for with period parameters as `pp_caldur_end_minute`.

        Also tests with period parmaeters as simply `pp_caldur`.
        """
        cal = calendars_extended
        strtgy = sthyp.one_of(stmp.pp_caldur(), stmp.pp_caldur_end_minute(cal.name))
        pp = data.draw(strtgy)

        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        end = pp["end"]
        duration = pd.DateOffset(
            weeks=pp["weeks"],
            months=pp["months"],
            years=pp["years"],
        )
        if end is None:
            end = drg.end_now[0]
        start = end - duration

        try:
            assert drg.daterange == ((start, end), end)
        except AssertionError:
            assert (
                start.value in cal.closes_nanos or start.value not in cal.minutes_nanos
            )
            start = cal.next_minute(start)
            assert drg.daterange == ((start, end), end)

    @pytest.mark.parametrize("limit_idx", [0, 100])
    def test_daterange_duration_cal_end_minute_oolb(
        self, calendars_with_answers_extended, pp_default, limit_idx
    ):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in calendar terms and 'end'.

        Notes
        -----
        Does not test for 'start' evaluating to on limit as not feasilble
        to do so with a duration based on calendar terms. See
        `test_daterange_duration_cal_end_oolb.__doc__` for reasoning.
        """
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        bi = TDInterval.T1

        limit = ans.first_minutes[ans.sessions[limit_idx]]
        pp["end"] = end = ans.closes[ans.sessions[limit_idx + 2]]
        pp["weeks"] = 2  # left of limit
        drg = self.get_drg(cal, pp, interval=bi, limit=limit, strict=False)
        assert drg.daterange == ((limit, end), end)

        start = end - pd.DateOffset(weeks=2)
        error_msg = re.escape(
            f"Prices unavailable as start ({helpers.fts(start)}) is earlier than"
            " the earliest minute for which price data is available. The earliest"
            f" minute for which prices are available is {helpers.fts(limit)}."
        )
        drg = self.get_drg(cal, pp, interval=bi, limit=limit, strict=True)
        with pytest.raises(errors.StartTooEarlyError, match=error_msg):
            _ = drg.daterange

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=None)
    def test_daterange_duration_days_start_minute(
        self, calendars_with_answers_extended, data
    ):
        """Test `daterange` for with period parameters as `pp_days_start_minute`."""
        cal, ans = calendars_with_answers_extended
        pp = data.draw(stmp.pp_days_start_minute(cal.name))

        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        start = pp["start"]
        start_session = cal.minute_to_session(start)
        target_session = start_session + (pp["days"] * cal.day)
        if start.value in cal.opens_nanos and start.value not in cal.closes_nanos:
            end = ans.closes[target_session - cal.day]
        elif start.value in cal.break_ends_nanos:
            end = ans.break_starts[target_session]
        else:
            target_day = helpers.to_utc(target_session)
            day_offset = (
                start.normalize() - helpers.to_utc(start_session).normalize()
            ).days
            if day_offset:
                target_day += day_offset * helpers.ONE_DAY
            end = target_day.replace(minute=start.minute, hour=start.hour)

        try:
            assert drg.daterange == ((start, end), end)
        except AssertionError:
            # consider change in open/close/break times between start and end.
            if end.value not in cal.minutes_nanos or end.value in cal.opens_nanos:
                end = cal.previous_close(end)
                assert drg.daterange == ((start, end), end)
            elif pp["days"]:
                target_session = cal.minute_to_session(end)
                distance = cal.sessions_distance(start_session, target_session) - 1
                if distance != pp["days"]:
                    # think change in hour (DST) for 24h calendar.
                    if distance < pp["days"]:
                        end_session = cal.minute_to_future_session(end)
                        end = cal.session_open(end_session)
                    else:
                        end_session = cal.minute_to_past_session(end)
                        end = cal.session_close(end_session)
                    assert drg.daterange == ((start, end), end)
                else:
                    raise
            else:
                raise

    def test_daterange_duration_days_start_minute_ool(
        self, calendars_with_answers_extended, pp_default
    ):
        """Test `daterange` ool errors for params as day duration and 'start' minute."""
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        bi = TDInterval.T1

        today = get_today(cal)

        drg = self.get_drg(cal, pp, interval=bi)
        now, now_accuracy = drg.end_now

        pp["start"] = start = ans.opens[today]
        pp["days"] = 1
        drg = self.get_drg(cal, pp, interval=bi)
        expected = ((start, now), now_accuracy)
        assert drg.daterange == expected  # on now
        for days in range(2, 5):  # right of now
            pp["days"] = days
            drg = self.get_drg(cal, pp, interval=bi)
            assert drg.daterange == expected

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=500)
    def test_daterange_duration_days_end_minute(self, calendars_extended, data):
        """Test `daterange` for with period parameters as `pp_days_end_minute`.

        Also tests with period parmaeters as simply `pp_days`.
        """
        cal = calendars_extended
        strtgy = sthyp.one_of(stmp.pp_days(cal.name), stmp.pp_days_end_minute(cal.name))
        pp = data.draw(strtgy)

        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        end = pp["end"]
        if end is None:
            end = drg.end_now[0]

        end_session = cal.minute_to_session(end, "previous")
        target_session = end_session - (pp["days"] * cal.day)
        target_session_close = cal.session_close(target_session)
        if end.value in cal.closes_nanos and end.value not in cal.opens_nanos:
            start = target_session_close
        else:
            target_day = helpers.to_utc(target_session)
            day_offset = (
                end.normalize() - helpers.to_utc(end_session).normalize()
            ).days
            if day_offset:
                target_day += day_offset * helpers.ONE_DAY
            start = target_day.replace(minute=end.minute, hour=end.hour)

            # start can resolve to earlier than target_session open due to DST changes.
            start = max(start, cal.session_open(target_session))
            # or later than target_session close
            start = min(start, target_session_close)

        try:
            assert drg.daterange == ((start, end), end)
        except AssertionError:
            start_dst = cal.tz.dst(start.tz_convert(None))
            end_dst = cal.tz.dst(end.tz_convert(None))
            assert (
                start.value in cal.closes_nanos
                or start.value not in cal.minutes_nanos
                or start_dst < end_dst
            )
            start = cal.next_minute(start)
            assert drg.daterange == ((start, end), end)

    def test_daterange_duration_days_end_oolb_minute(
        self, calendars_with_answers_extended, pp_default, one_min
    ):
        """Test `daterange` ool errors with params as day duration and 'end' minute."""
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        bi = TDInterval.T1

        drg_kwargs = dict(interval=bi)

        start = ans.opens.iloc[0]
        for i in range(3):
            pp["days"] = i + 1
            pp["end"] = end = ans.closes.iloc[i]

            # on left bound
            for strict in [True, False]:
                drg = self.get_drg(cal, pp, strict=strict, **drg_kwargs)
                assert drg.daterange == ((start, end), end)

            # left of bound
            pp["end"] = end = end - one_min
            drg = self.get_drg(cal, pp, strict=False, **drg_kwargs)
            assert drg.daterange == ((start, end), end)

            drg = self.get_drg(cal, pp, strict=True, **drg_kwargs)
            match = re.escape(
                "Prices unavailable as start would resolve to an earlier minute than"
                f" the earliest minute of calendar '{cal.name}'. The calendar's earliest"
                f" minute is {helpers.fts(ans.first_minute)} (this bound should coincide"
                " with the earliest minute for which daily price data is available)."
            )
            with pytest.raises(errors.StartOutOfBoundsError, match=match):
                _ = drg.daterange

        limit_i = 30
        limit = ans.opens.iloc[limit_i]

        drg_kwargs["limit"] = limit

        for i in range(3):
            pp["days"] = i + 1
            pp["end"] = end = ans.closes.iloc[limit_i + i]

            for strict in [True, False]:  # on left limit
                drg = self.get_drg(cal, pp, strict=strict, **drg_kwargs)
                assert drg.daterange == ((limit, end), end)

            pp["end"] = end = end - one_min
            drg = self.get_drg(cal, pp, strict=False, **drg_kwargs)
            assert drg.daterange == ((limit, end), end)

            drg = self.get_drg(cal, pp, strict=True, **drg_kwargs)
            start = end - (cal.day * pp["days"])
            start = start if cal.is_trading_minute(start) else cal.previous_close(start)
            error_msg = re.escape(
                f"Prices unavailable as start ({helpers.fts(start)}) is earlier than"
                " the earliest minute for which price data is available. The earliest"
                f" minute for which prices are available is {helpers.fts(limit)}."
            )
            with pytest.raises(errors.StartTooEarlyError, match=error_msg):
                _ = drg.daterange

    def test_daterange_duration_days_intervalperiod_error(
        self, calendars_with_answers_extended, pp_default, one_min
    ):
        """Test `daterange` raises period error with params as day duration."""
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        pp = pp_default
        pp["days"] = 1

        if ans.sessions_with_gap_after.empty:
            return

        today = get_today(cal)
        for session in ans.sessions_sample:
            if session > today:
                continue
            i = ans.sessions.get_loc(session)
            open_, close = ans.opens.iloc[i], ans.closes.iloc[i]
            length = close - open_

            if length > TDInterval.H22 or ans.session_has_break(session):
                # breaks not conducive to testing this with 1 day duration as
                # effectively covering two sessions.
                continue

            pp["start"] = open_
            interval = intervals.to_ptinterval(length)
            for anchor in (Anchor.OPEN, Anchor.WORKBACK):
                drg = self.get_drg(
                    cal, pp, interval=interval, anchor=anchor, ignore_breaks=True
                )
                assert drg.daterange == ((pp["start"], close), close)

            interval = intervals.to_ptinterval(length + one_min)
            # verify still returns when anchor open
            drg = self.get_drg(cal, pp, interval=interval, anchor=Anchor.OPEN)
            assert drg.daterange == ((pp["start"], close + one_min), close)

            # but raises error when anchor workback
            drg = self.get_drg(cal, pp, interval=interval, anchor=Anchor.WORKBACK)
            match = re.escape(
                f"Period does not span a full indice of length {interval}."
                f"\nPeriod duration evaluated as {length}."
                f"\nPeriod start date evaluated as {open_}.\nPeriod end date"
                f" evaluated as {close}.\nPeriod dates evaluated from parameters: {pp}."
            )
            with pytest.raises(
                errors.PricesUnavailableIntervalPeriodError, match=match
            ):
                _ = drg.daterange

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=500)
    def test_daterange_duration_intraday_start_minute(self, calendars_extended, data):
        """Test `daterange` for with period parameters as `pp_intraday_start_minute`."""
        cal = calendars_extended
        pp = data.draw(stmp.pp_intraday_start_minute(cal.name))
        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        start = pp["start"]
        minutes = pp["minutes"] + (pp["hours"] * 60)
        i = cal.minutes.get_loc(start)
        end = cal.minutes[i + minutes]
        if end.value in cal.first_minutes_nanos and end.value not in cal.closes_nanos:
            end = cal.previous_close(end)
        elif end.value in cal.break_ends_nanos:
            # to break start (end of am subsession)
            end = cal.previous_minute(end) + helpers.ONE_MIN
        assert drg.daterange == ((start, end), end)

    @hyp.given(data=sthyp.data())
    @hyp.settings(deadline=None)
    def test_daterange_duration_intraday_end_minute(self, calendars_extended, data):
        """Test `daterange` for with period parameters as `pp_intraday_end_minute`.

        Also tests with period parmaeters as simply `pp_intraday`.
        """
        cal = calendars_extended
        strtgy = sthyp.one_of(stmp.pp_intraday(), stmp.pp_intraday_end_minute(cal.name))
        pp = data.draw(strtgy)
        drg = self.get_drg(cal, pp, interval=TDInterval.T1)

        end = pp["end"]
        if end is None:
            end = drg.end_now[0]

        minutes = pp["minutes"] + (pp["hours"] * 60)
        i = cal.minutes.get_indexer([end], method="bfill")[0]
        start = cal.minutes[i - minutes]

        try:
            assert drg.daterange == ((start, end), end)
        except AssertionError:
            assert (
                start.value in cal.closes_nanos or end.value in cal.break_starts_nanos
            )
            start = cal.next_minute(start)
            assert drg.daterange == ((start, end), end)

    def test_daterange_duration_intraday_start_minute_ool(
        self, calendars_with_answers_extended, pp_default
    ):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in trading terms and 'start'.
        """
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        bi = TDInterval.T1

        drg = self.get_drg(cal, pp, interval=bi)
        end, end_accuracy = drg.end_now

        minutes = 5
        pp["start"] = start = end - pd.Timedelta(minutes, "T")

        # on now
        pp["minutes"] = minutes
        drg = self.get_drg(cal, pp, interval=bi)
        assert drg.daterange == ((start, end), end_accuracy)

        # right of now
        pp["hours"] = 72 if not ans.sessions_without_gap_after.empty else 12
        drg = self.get_drg(cal, pp, interval=bi)
        assert drg.daterange == ((start, end), end_accuracy)

    @pytest.mark.parametrize("limit_idx", [0, 100])
    def test_daterange_duration_intraday_end_minute_oolb(
        self, calendars_with_answers_extended, pp_default, limit_idx
    ):
        """Test `daterange` ool errors for specfic arrangement of period parameters.

        Period parameters as duration in trading terms and 'end'.
        """
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        bi = TDInterval.T1

        limit = ans.first_minutes[ans.sessions[limit_idx]]

        # on limit
        minutes = 5
        pp["end"] = end = limit + pd.Timedelta(minutes, "T")
        pp["minutes"] = minutes
        drg = self.get_drg(cal, pp, interval=bi, limit=limit, strict=False)
        assert drg.daterange == ((limit, end), end)

        # left of limit
        for minutes, hours in [(6, 0), (33, 0), (1, 25)]:
            pp["minutes"], pp["hours"] = minutes, hours
            drg = self.get_drg(cal, pp, interval=bi, limit=limit, strict=False)
            assert drg.daterange == ((limit, end), end)

            drg = self.get_drg(cal, pp, interval=bi, limit=limit, strict=True)
            if not limit_idx:
                match = re.escape(
                    "Prices unavailable as start would resolve to an earlier minute"
                    f" than the earliest minute of calendar '{cal.name}'. The"
                    f" calendar's earliest minute is {helpers.fts(limit)} (this bound"
                    " should coincide with the earliest minute for which daily price"
                    " data is available)."
                )
                with pytest.raises(errors.StartOutOfBoundsError, match=match):
                    _ = drg.daterange

            else:
                end_i = cal.minutes.get_loc(end)
                start = cal.minutes[end_i - (minutes + hours * 60)]
                match = (
                    f"Prices unavailable as start ({helpers.fts(start)}) is earlier"
                    " than the earliest minute for which price data is available. The"
                    " earliest minute for which prices are available is"
                    f" {helpers.fts(limit)}."
                )
                with pytest.raises(errors.StartTooEarlyError):  # , match=error_msg):
                    _ = drg.daterange

    def test_daterange_start_ool(self, xlon_calendar_extended):
        """Test effect of start earlier than limit.

        Tests:
            - Raises errors.StartTooEarlyError if strict True.
            - Returns start as limit if strict False
            - Raises errors.StartTooEarlyError if strict False although
            period described in terms of duration.
        """
        xlon = xlon_calendar_extended
        interval = TDInterval.T5
        limit_day = xlon.next_open(get_today(xlon) - pd.Timedelta(30, "D"))
        limit_session = xlon.minute_to_session(limit_day)
        limit = xlon.session_open(limit_session)

        start_session = xlon.session_offset(limit_session, -3)
        start = xlon.session_open(start_session)
        end_session = xlon.session_offset(limit_session, 3)
        end = xlon.session_close(end_session)

        # verify effect when period defined by `start` and `end`
        pp = get_pp_default()
        pp["start"], pp["end"] = start, end

        drg = self.get_drg(xlon, pp, limit=limit, interval=interval, strict=False)
        assert drg.daterange == ((limit, end), end)

        drg = self.get_drg(xlon, pp, limit=limit, interval=interval, strict=True)
        with pytest.raises(errors.StartTooEarlyError):
            _ = drg.daterange

        # verify raises error, regardless of strict, when period defined by
        # `start` and duration
        pp = get_pp_default()
        pp["start"], pp["days"] = start, 7

        for strict in (True, False):
            drg = self.get_drg(xlon, pp, limit=limit, interval=interval, strict=strict)
            with pytest.raises(errors.StartTooEarlyError):
                _ = drg.daterange

    def test_daterange_tight(self, xlon_calendar_extended, pp_default):
        """Test daterange_tight for diverges from daterange as expected."""
        xlon = xlon_calendar_extended
        # from knowledge of schedule
        session = "2021-11-03"
        assert xlon.is_session(session)
        open_, close = xlon.session_open_close(session)
        assert close - open_ == pd.Timedelta(hours=8, minutes=30)

        pp = pp_default
        pp["start"] = open_
        pp["end"] = close

        args = (xlon, pp)
        kwargs = {"interval": TDInterval.H1, "ds_interval": TDInterval.H2}

        exp_start = open_
        exp_end = close + pd.Timedelta(90, "T")
        exp_end_tight = close + pd.Timedelta(30, "T")
        exp_end_accuracy = close

        # verify daterange same as daterange_tight when Alignment BI
        drg = self.get_drg(*args, **kwargs, end_alignment=Alignment.BI)
        assert drg.daterange == ((exp_start, exp_end_tight), exp_end_accuracy)
        assert drg.daterange == drg.daterange_tight

        # verify difference between daterange properties when Alignment FINAL
        drg = self.get_drg(*args, **kwargs, end_alignment=Alignment.FINAL)
        assert drg.daterange == ((exp_start, exp_end), exp_end_accuracy)
        assert drg.daterange_tight == ((exp_start, exp_end_tight), exp_end_accuracy)

    def test_daterange_duration_intraday_intervalduration_error(
        self, calendars_with_answers_extended, pp_default, base_interval
    ):
        """Test `daterange` raises error when intraday duration < final interval.

        Tests raises `errors.PricesUnavailableDurationPeriodError` when expected.
        """
        if base_interval is TDInterval.T1:
            return
        (cal, ans), pp = calendars_with_answers_extended, pp_default
        drg_kwargs = dict(ds_interval=base_interval, interval=TDInterval.T1)

        # on limit, where intraday duration == final interval
        pp["minutes"] = base_interval.as_minutes
        pp["start"] = start = ans.first_minutes.iloc[1]
        end = ans.first_minutes.iloc[1] + base_interval
        drg = self.get_drg(cal, pp, **drg_kwargs)
        assert drg.daterange == ((start, end), end)

        # duration < final interval
        pp["minutes"] = minutes = pp["minutes"] - 1
        drg = self.get_drg(cal, pp, **drg_kwargs)
        duration = pd.Timedelta(minutes, "T")
        match = re.escape(
            f"Period duration shorter than interval. Interval is {base_interval}"
            f" although period duration is only {duration}."
            f"\nDuration evaluated from parameters: {drg.pp}."
        )
        with pytest.raises(errors.PricesUnavailableIntervalDurationError, match=match):
            _ = drg.daterange

    def test_add_a_row_workback(
        self, calendars_with_answers_extended, pp_default, one_min
    ):
        """Test effect of `add_a_row` when anchor is 'workback'.

        NB effect of `add_a_row` with 'open' anchor is also tested for
        within parameterized tests.
        """
        cal, ans = calendars_with_answers_extended
        pp = pp_default
        pp["add_a_row"] = True

        # get a session, with a break if there is one.
        sessions = ans.sessions_sample
        sessions = sessions.intersection(ans.sessions_with_break)
        if sessions.empty:
            sessions = ans.sessions_sample
        session = sessions[len(sessions) // 2]
        prev_session = ans.get_prev_session(session)

        open_ = ans.opens[session]
        prev_session_close = ans.closes[prev_session]

        bi = TDInterval.T1
        dsi = TDInterval.T15
        drg_kwargs = dict(interval=bi, ds_interval=dsi)

        def get_drgs(pp: dict) -> tuple[m.GetterIntraday, m.GetterIntraday]:
            drg_open = self.get_drg(cal, pp, **drg_kwargs, anchor=Anchor.OPEN)
            drg_wb = self.get_drg(cal, pp, **drg_kwargs, anchor=Anchor.WORKBACK)
            return drg_open, drg_wb

        # verify for prior_start in same session as start
        pp["start"] = start = open_ + (2 * dsi)
        # set minutes so that end not on dsi to differentiate between anchors in their
        # effect on the daterange end.
        pp["minutes"] = dsi.as_minutes + 1
        drg_open, drg_wb = get_drgs(pp)
        drg_open_dr, drg_wb_dr = drg_open.daterange, drg_wb.daterange
        # verify that start same for both anchors, and based on dsi
        assert drg_open_dr[0][0] == drg_wb_dr[0][0] == open_ + dsi
        # verify ends differ, in accordance with bi alignment
        assert drg_open_dr[0][1] == drg_open_dr[1] == start + dsi
        assert drg_wb_dr[0][1] == drg_wb_dr[1] == start + dsi + one_min

        # verify for prior_start including last interval of prior session
        pp["start"] = start = open_
        drg_open, drg_wb = get_drgs(pp)
        drg_open_dr, drg_wb_dr = drg_open.daterange, drg_wb.daterange
        assert drg_open_dr[0][0] == drg_wb_dr[0][0] == prev_session_close - dsi
        # verify ends differ, in accordance with bi alignment
        assert drg_open_dr[0][1] == drg_open_dr[1] == start + dsi
        assert drg_wb_dr[0][1] == drg_wb_dr[1] == start + dsi + one_min

        # verify for prior_start that represent an interval that crosses sessions
        delta = pd.Timedelta(7, "T")
        pp["start"] = start = open_ + delta
        drg_open, drg_wb = get_drgs(pp)
        end_ = start + dsi + one_min
        assert drg_wb.daterange == ((prev_session_close - dsi + delta, end_), end_)
        # just to verify how drg_open differs
        end_ = open_ + (dsi * 2)
        assert drg_open.daterange == ((open_, end_), end_)

    def test_daterange_sessions(
        self,
        calendars_with_answers_extended,
        pp_default,
        one_min,
    ):
        (cal, ans), pp = calendars_with_answers_extended, pp_default

        def verify(
            starts: list[pd.Timestamp],
            ends: list[pd.Timestamp],
            start_session: pd.Timestamp,
            end_session: pd.Timestamp,
        ):
            for start, end in itertools.product(starts, ends):
                pp["start"], pp["end"] = start, end
                drg = self.get_drg(cal, pp, interval=TDInterval.T1)
                assert drg.daterange_sessions == (start_session, end_session)

        # test for a few combinations of sample sessions
        today = get_today(cal)
        bv = ans.sessions_sample <= today
        sessions = ans.sessions_sample[bv]
        for start_session, end_session in zip(sessions, reversed(sessions)):
            if start_session > end_session:
                break
            open_ = ans.opens[start_session]
            close = ans.closes[end_session]
            starts = [open_, open_ + one_min]
            ends = [close, close - one_min]
            verify(starts, ends, start_session, end_session)

            start_session_has_break = ans.session_has_break(start_session)
            end_session_has_break = ans.session_has_break(end_session)
            if start_session_has_break and end_session_has_break:
                pm_open = ans.break_ends[start_session]
                am_close = ans.break_starts[end_session]

                verify(
                    starts, [am_close, am_close - one_min], start_session, end_session
                )
                verify([pm_open, pm_open + one_min], ends, start_session, end_session)
