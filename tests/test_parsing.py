"""Tests for market_prices.parsing module."""

from __future__ import annotations

from collections import abc
from collections.abc import Callable
import dataclasses
import datetime
import itertools
import re
import typing
from typing import Annotated, Union, TYPE_CHECKING
import zoneinfo
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from valimp import parse, Parser, Coerce

import market_prices.parsing as m
from market_prices import errors, helpers, mptypes
from market_prices.helpers import UTC

from .utils import Answers

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


def test_verify_period_parameters():
    # pylint: disable=too-complex
    f = m.verify_period_parameters

    dflt = dict(
        minutes=0,
        hours=0,
        days=0,
        weeks=0,
        months=0,
        years=0,
        start=None,
        end=None,
        add_a_row=False,
    )

    assert f(dflt) is None

    def assert_valid(kwargs: dict):
        assert f({**dflt, **kwargs}) is None

    start = pd.Timestamp("2021-11-01")
    end = pd.Timestamp("2021-11-08")

    assert_valid({"start": start})  # start only
    assert_valid({"end": end})  # end only
    assert_valid({"start": start, "end": end})  # start and end only

    trading_kwargs = []
    for minutes, hours in itertools.product([0, 1], [0, 1]):
        if minutes or hours:
            trading_kwargs.append(dict(minutes=minutes, hours=hours))

    days_kwargs = [(dict(days=1))]

    calendar_kwargs = []
    for weeks, months, years in itertools.product([0, 1], [0, 1], [0, 1]):
        if weeks or months or years:
            calendar_kwargs.append(dict(weeks=weeks, months=months, years=years))

    duration_kwargs = trading_kwargs + days_kwargs + calendar_kwargs

    def assert_raises(kwargs: dict):
        with pytest.raises(ValueError):
            f({**dflt, **kwargs})

    for kwargs in duration_kwargs:
        assert_valid(kwargs)  # duration kwargs only
        assert_valid({**kwargs, **dict(start=start)})  # and start
        assert_valid({**kwargs, **dict(end=end)})  # and end

        assert_raises({**kwargs, **dict(start=start, end=end)})  # and start and end

    def assert_raises_inc_with_start_end(kwargs: dict):
        assert_raises(kwargs)
        assert_raises({**kwargs, **dict(start=start)})  # and start
        assert_raises({**kwargs, **dict(end=end)})  # and end
        assert_raises({**kwargs, **dict(start=start, end=end)})  # and start and end

    for kwargs in trading_kwargs:
        # tarding duration kwargs with day duration kwargs
        assert_raises_inc_with_start_end({**kwargs, **days_kwargs[0]})

        # tarding duration kwargs with calendar duration kwargs
        for cal_kwargs in calendar_kwargs:
            assert_raises_inc_with_start_end({**kwargs, **cal_kwargs})

    # calendar duration kwargs with day duration kwargs
    for kwargs in calendar_kwargs:
        assert_raises_inc_with_start_end({**kwargs, **days_kwargs[0]})


def test_parse_timestamp():
    f = m.parse_timestamp

    tzin = ZoneInfo("US/Eastern")

    date = pd.Timestamp("2021-11-02")
    rtrn = f(date, tzin)
    assert rtrn == date  # date as tz-naive timestamp
    assert rtrn.tz is None

    midnight = pd.Timestamp("2021-11-02", tz=UTC)
    rtrn = f(midnight, tzin)
    assert rtrn == midnight
    assert rtrn.tz == UTC  # Does not change tz.

    time = pd.Timestamp("2021-11-02 14:33")
    rtrn = f(time, tzin)
    assert rtrn == pd.Timestamp(time, tz=tzin)  # defines time in terms of tzin...
    assert rtrn.tz == UTC  # ...although tz is UTC
    assert rtrn != pd.Timestamp(time, tz=UTC)
    assert rtrn == time.tz_localize(tzin).tz_convert(UTC)

    time = pd.Timestamp("2021-11-02 14:33", tz=tzin)
    rtrn = f(time, ZoneInfo("Japan"))  # time tz-aware, tzin should be ignored...
    assert rtrn == time
    assert rtrn.tz == UTC  # ...albeit converted to UTC


class TestParseStartEnd:
    """tests for parsing.parse_start_end.

    `start` and `end` each tested against four `calendars` and the unique
    circumnstances of `Answers.sessions_sample`.

    Also verifies expected returns and that expected errors raised when
    `start` and `end` are both passed.
    """

    @pytest.fixture
    def f_with_ans(
        self, calendars_with_answers_extended
    ) -> abc.Iterator[tuple[Callable, Answers]]:
        calendar, answers = calendars_with_answers_extended

        def f(
            s,
            e,
            as_times,
            delay=pd.Timedelta(0),
            strict=True,
            gregorian=False,
            mr_session=None,
            mr_minute=None,
        ) -> mptypes.DateRangeAmb:
            return m.parse_start_end(
                s,
                e,
                as_times,
                calendar,
                delay,
                strict,
                gregorian,
                mr_session,
                mr_minute,
            )

        yield f, answers

    @pytest.fixture(scope="class")
    def delays(self) -> abc.Iterator[tuple[pd.Timedelta, pd.Timedelta]]:
        yield pd.Timedelta(0), pd.Timedelta(15, "T")

    @pytest.fixture(scope="class")
    def as_times(self) -> abc.Iterator[typing.Literal[True]]:
        yield True

    @pytest.fixture(scope="class", autouse=True)
    def as_dates(self) -> abc.Iterator[typing.Literal[False]]:
        yield False

    def test_start_end_as_session(self, f_with_ans, today, as_times, as_dates):
        """Test start and end as sessions within bounds and limits."""
        f, ans = f_with_ans
        for session in ans.sessions_sample:
            if session >= today:
                continue
            session_first_minute = ans.first_minutes[session]
            session_close = ans.closes[session]

            assert f(session, None, as_dates) == (session, None)
            assert f(session, None, as_times) == (session_first_minute, None)
            assert f(None, session, as_dates) == (None, session)
            assert f(None, session, as_times) == (None, session_close)

    def test_start_end_as_non_session(self, f_with_ans, today, as_times, as_dates):
        """Test start and end as non-sessions within bounds and limits."""
        f, ans = f_with_ans
        for date in ans.non_sessions[1:-1:30]:
            if date > today:
                continue
            next_session = ans.date_to_session(date, "next")
            prev_session = ans.date_to_session(date, "previous")
            next_session_first_minute = ans.first_minutes[next_session]
            prev_session_close = ans.closes[prev_session]

            assert f(date, None, as_dates) == (next_session, None)
            assert f(None, date, as_dates) == (None, prev_session)
            assert f(date, None, as_times) == (next_session_first_minute, None)
            assert f(None, date, as_times) == (None, prev_session_close)

    def test_start_end_as_minutes(
        self, f_with_ans, today, one_min, as_times, as_dates, one_sec
    ):
        """Test start/end as trading/non-trading minutes within bounds and limits."""
        f, ans = f_with_ans

        bv = (ans.first_session < ans.sessions_sample) & (today > ans.sessions_sample)
        sessions = ans.sessions_sample[bv]

        first_mins = ans.first_minutes[sessions]
        last_mins = ans.last_minutes[sessions]
        locs = ans.sessions.get_indexer(sessions)
        prev_sessions = ans.sessions[locs - 1]
        next_sessions = ans.sessions[locs + 1]

        for first_min, last_min, session, prev_session, next_session in zip(
            first_mins, last_mins, sessions, prev_sessions, next_sessions
        ):
            assert f(first_min, None, as_dates) == (session, None)
            assert f(None, first_min, as_dates) == (None, prev_session)
            assert f(first_min, None, as_times) == (first_min, None)
            assert f(None, first_min, as_times) == (None, first_min)

            minute = first_min + one_min
            assert f(minute, None, as_dates) == (next_session, None)
            assert f(None, minute, as_dates) == (None, prev_session)
            assert f(minute, None, as_times) == (minute, None)
            assert f(None, minute, as_times) == (None, minute)

            # non-trading minute
            if session in ans.sessions_with_gap_before:
                minute = first_min - one_min
                assert f(minute, None, as_dates) == (session, None)
                assert f(None, minute, as_dates) == (None, prev_session)
                assert f(minute, None, as_times) == (first_min, None)
                assert f(None, minute, as_times) == (None, ans.closes[prev_session])

            assert f(last_min, None, as_dates) == (next_session, None)
            assert f(None, last_min, as_dates) == (None, prev_session)
            assert f(last_min, None, as_times) == (last_min, None)
            assert f(None, last_min, as_times) == (None, last_min)

            minute = last_min - one_min
            assert f(minute, None, as_dates) == (next_session, None)
            assert f(None, minute, as_dates) == (None, prev_session)
            assert f(minute, None, as_times) == (minute, None)
            assert f(None, minute, as_times) == (None, minute)

            if session in ans.sessions_with_gap_after:
                minute = last_min + one_min  # close minute
                assert f(minute, None, as_dates) == (next_session, None)
                assert f(None, minute, as_dates) == (None, session)
                assert f(minute, None, as_times) == (ans.opens[next_session], None)
                assert f(None, minute, as_times) == (None, minute)

                minute += one_min  # non-trading minute
                assert f(minute, None, as_dates) == (next_session, None)
                assert f(None, minute, as_dates) == (None, session)
                assert f(minute, None, as_times) == (ans.opens[next_session], None)
                assert f(None, minute, as_times) == (None, ans.closes[session])

        # verify if start/end are not minute accurate then rounded up/down respectively
        start, end = first_mins.iloc[0], last_mins.iloc[0]
        assert f(start + one_sec, end - one_sec, True) == (
            start + one_min,
            end - one_min,
        )

    def test_start_end_as_break_minute(self, f_with_ans, now_utc, as_times, as_dates):
        """Test start/end as trading/non-trading minutes within bounds and limits."""
        f, ans = f_with_ans
        for mins, session in ans.break_minutes:
            if session == ans.first_session or session >= ans.last_session:
                continue

            next_session = ans.get_next_session(session)
            prev_session = ans.get_prev_session(session)

            for minute in mins:
                if minute > now_utc:
                    continue

                assert f(minute, None, as_dates) == (next_session, None)
                assert f(None, minute, as_dates) == (None, prev_session)
                assert f(minute, None, as_times) == (ans.break_ends[session], None)
                assert f(None, minute, as_times) == (None, ans.break_starts[session])

    def test_start_end_at_left_bound(self, f_with_ans, as_times, as_dates):
        f, ans = f_with_ans

        session = ans.first_session
        assert f(session, None, as_dates) == (session, None)
        assert f(session, None, as_times) == (ans.first_minute, None)
        assert f(None, session, as_dates) == (None, session)
        assert f(None, session, as_times) == (None, ans.closes[session])

        minute = ans.first_minute
        assert f(minute, None, as_dates) == (ans.first_session, None)
        assert f(minute, None, as_times) == (minute, None)
        assert f(None, minute, as_times) == (None, minute)
        # bound for end when time and as_dates is close of first session
        end = ans.closes[ans.first_session]
        assert f(None, end, as_dates) == (None, ans.first_session)

    def test_start_end_oob_left(self, f_with_ans, one_day, one_min, as_times, as_dates):
        f, ans = f_with_ans

        date = ans.first_session - one_day
        match_msg = re.escape(
            f"Prices unavailable as start ({helpers.fts(date)}) is earlier than the"
            f" earliest date of calendar '{ans.name}'. The calendar's earliest date"
            f" is {helpers.fts(ans.first_session)} (this bound should coincide with the"
            " earliest date for which price data is available)."
        )
        for as_ in [as_times, as_dates]:
            with pytest.raises(errors.StartOutOfBoundsError):
                f(date, None, as_)
        assert f(date, None, as_dates, strict=False) == (ans.first_session, None)
        assert f(date, None, as_times, strict=False) == (ans.first_minute, None)

        for as_ in [as_times, as_dates]:
            with pytest.raises(errors.EndOutOfBoundsError):
                f(None, date, as_)

        minute = ans.first_minute - one_min
        match_msg = re.escape(
            f"Prices unavailable as start ({helpers.fts(minute)}) is earlier than the"
            f" earliest minute of calendar '{ans.name}'. The calendar's earliest minute"
            f" is {helpers.fts(ans.first_minute)} (this bound should coincide with the"
            " earliest minute or date for which price data is available)."
        )
        for as_ in [as_times, as_dates]:
            with pytest.raises(errors.StartOutOfBoundsError, match=match_msg):
                f(minute, None, as_)
        assert f(minute, None, as_dates, strict=False) == (ans.first_session, None)
        assert f(minute, None, as_times, strict=False) == (ans.first_minute, None)

        for as_ in [as_times, as_dates]:
            with pytest.raises(errors.EndOutOfBoundsError):
                f(None, minute, as_)

        # additionally, for 'end' should raise error all way to one minute short of
        # first session's close
        with pytest.raises(errors.EndOutOfBoundsError):
            f(None, ans.closes[ans.first_session] - one_min, as_dates)

    def test_start_end_at_right_limit(
        self, f_with_ans, now_utc, monkeypatch, one_min, as_times, as_dates, delays
    ):
        f, ans = f_with_ans

        dti = pd.DatetimeIndex(ans.first_minutes)
        idx = dti.get_slice_bound(now_utc, "left")
        session = ans.sessions[idx - 1]
        session_first_minute = ans.first_minutes[session]
        session_close = ans.closes[session]

        def mock_now_closed(*_, tz=UTC, **__) -> pd.Timestamp:
            now = ans.last_minutes[session] + (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        def mock_now_open(*_, tz=UTC, **__) -> pd.Timestamp:
            now = session_close - (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        for d in delays:
            assert f(None, session, as_dates, d) == (None, None)
            assert f(session, None, as_dates, d) == (session, None)
            assert f(session, None, as_times, d) == (session_first_minute, None)

            if session in ans.sessions_with_gap_after:
                monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
                now_closed = mock_now_closed()
                assert f(None, session, as_times, d) == (None, None)

            monkeypatch.setattr("pandas.Timestamp.now", mock_now_open)
            now_open = mock_now_open()
            now_open_right = now_open + one_min
            assert f(None, session, as_times, d) == (None, None)
            assert f(None, now_open, as_times, d) == (None, None)

            # end should return as None when end >= now
            assert f(None, now_open_right, as_times, d) == (None, None)
            assert f(None, now_open, as_dates, d) == (None, None)
            # bound for start is first minute of last session when time and as_dates
            assert f(session_first_minute, None, as_dates, d) == (session, None)

            if session in ans.sessions_with_gap_after:
                monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
                assert f(None, now_closed, as_times, d) == (None, None)

                assert f(None, now_closed, as_dates, d) == (None, None)

        # with delay, start limit moves back by delay
        monkeypatch.setattr("pandas.Timestamp.now", mock_now_open)
        assert f(now_open, None, as_times, delays[0]) == (now_open, None)
        d = delays[1]
        assert f(now_open - d, None, as_times, d) == (now_open - d, None)

    def test_start_end_ool_right(
        self,
        f_with_ans,
        now_utc,
        one_min,
        one_day,
        monkeypatch,
        as_times,
        as_dates,
        delays,
    ):
        """Test `parse_start_end` for values right of right limit.

        Verifies that end returns as None when end > now.
        """
        # pylint: disable=too-complex
        f, ans = f_with_ans

        idx = pd.DatetimeIndex(ans.first_minutes).get_slice_bound(now_utc, "left")
        ool_session = ans.sessions[idx]  # session to right of last session
        last_session = ans.sessions[idx - 1]
        last_session_close = ans.closes[last_session]

        oob_session = ans.last_session + one_day  # right of right calendar bound
        oob_minute = ans.last_minute + one_min  # right of right calendar bound

        def mock_now_closed(*_, tz=UTC, **__) -> pd.Timestamp:
            now = ans.last_minutes[last_session] + (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        def mock_now_open(*_, tz=UTC, **__) -> pd.Timestamp:
            now = ans.last_minutes[last_session] - (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        for d in delays:
            # input as date
            for session in [ool_session, oob_session]:
                with pytest.raises(errors.StartTooLateError):
                    f(session, None, as_dates, d)
                with pytest.raises(errors.StartTooLateError):
                    f(session, None, as_times, d)

                # when end > now should return end as None
                assert f(None, session, as_dates, d) == (None, None)

            if last_session in ans.sessions_with_gap_after:
                monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
                rtrn = (None, None)  # when end > now should return end as None
                assert f(None, ool_session, as_times, d) == rtrn
                assert f(None, oob_session, as_times, d) == rtrn

            monkeypatch.setattr("pandas.Timestamp.now", mock_now_open)
            for session in [ool_session, oob_session]:
                assert f(None, session, as_times, d) == (None, None)

            # input as time
            ool_minute = mock_now_open() + (5 * one_min)
            for minute in [ool_minute, oob_minute]:
                assert f(None, minute, as_times, d) == (None, None)
                assert f(None, minute, as_dates, d) == (None, None)

                with pytest.raises(errors.StartTooLateError):
                    f(minute, None, as_times, d)

            with pytest.raises(errors.StartTooLateError):
                # bound for start when time and as_dates is first minute of last session
                assert f(ans.first_minutes[last_session] + one_min, None, as_dates, d)

            if last_session in ans.sessions_with_gap_after:
                monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
                now_closed = mock_now_closed()
                now_closed_right = now_closed + one_min
                ool_minute = mock_now_closed() + (7 * one_min)
                if d == delays[0]:
                    rtrn = (None, last_session_close)
                else:
                    rtrn = (None, now_closed_right - d)
                for minute in [ool_minute, oob_minute]:
                    assert f(None, minute, as_times, d) == (None, None)
                    assert f(None, minute, as_dates, d) == (None, None)

                    with pytest.raises(errors.StartTooLateError):
                        f(minute, None, as_times, d)
                    with pytest.raises(errors.StartTooLateError):
                        f(minute, None, as_dates, d)

        # with delay, start limit moves back by delay
        monkeypatch.setattr("pandas.Timestamp.now", mock_now_open)
        d = delays[1]
        ool_minute = mock_now_open() - d + one_min
        with pytest.raises(errors.StartTooLateError):
            f(ool_minute, None, as_times, d)

    def test_start_end_right_limit_not_now(
        self, f_with_ans, one_min, one_day, as_times, as_dates, now_utc, monkeypatch
    ):
        """Tests passing start and end with mr_session and mr_minute."""
        f_, ans = f_with_ans

        dti = pd.DatetimeIndex(ans.first_minutes)
        idx = len(dti) // 2
        session = ans.sessions[idx - 1]
        session_first_minute = ans.first_minutes[session]
        session_close = ans.closes[session]

        def f(*args, **kwargs):
            idx = len(dti) // 2
            session = ans.sessions[idx - 1]
            session_close = ans.closes[session]
            kwargs.setdefault("mr_session", session)
            kwargs.setdefault("mr_minute", session_close)
            return f_(*args, **kwargs)

        # on limits
        assert f(None, session, as_dates) == (None, session)
        assert f(session, None, as_dates) == (session, None)
        assert f(None, session, as_times) == (None, session_close)
        assert f(session, None, as_times) == (session_first_minute, None)
        assert f(None, session_close, as_times) == (None, session_close)

        # over limit
        rol_session = ans.get_next_session(session)
        rol_session_open = ans.opens[rol_session]
        rol_session_close = ans.closes[rol_session]
        rol_minute = session_close + one_min

        # limit should have no effect on end
        assert f(None, rol_session, as_dates) == (None, rol_session)
        assert f(None, rol_session, as_times) == (None, rol_session_close)
        assert f(None, rol_minute, as_dates) == (None, session)
        exp_end = (
            rol_minute if rol_minute == rol_session_open + one_min else session_close
        )
        assert f(None, rol_minute, as_times) == (None, exp_end)

        # should raise when start > limit
        match_D = re.escape(
            "`start` cannot be a later date than the latest date for which prices are"
            " available.\nThe latest date for which prices are available for interval"
            f" '1 day, 0:00:00' is {helpers.fts(session)}, although `start`"
            f" received as {helpers.fts(rol_session)}."
        )
        match_T = re.escape(
            "`start` cannot be a later time than the latest time for which prices"
            " are available.\nThe latest time for which prices are available for"
            f" interval '0:01:00' is {helpers.fts(session_first_minute)}, although"
            f" `start` received as {helpers.fts(rol_minute)}."
        )
        match_TT = re.escape(
            "`start` cannot be a later time than the latest time for which prices"
            " are available.\nThe latest time for which prices are available for"
            f" interval '0:01:00' is {helpers.fts(session_close)}, although"
            f" `start` received as {helpers.fts(rol_minute)}."
        )
        with pytest.raises(errors.StartTooLateError, match=match_D):
            f(rol_session, None, as_dates)
        with pytest.raises(errors.StartTooLateError, match=match_D):
            f(rol_session, None, as_times)
        with pytest.raises(errors.StartTooLateError, match=match_T):
            f(rol_minute, None, as_dates)
        with pytest.raises(errors.StartTooLateError, match=match_TT):
            f(rol_minute, None, as_times)

        # Check behavious as expected when start and end passed to the right of
        # now and the right calendar bound
        # should still raise when start > now
        # should return end as parses when end > now, i.e. should not set to None

        idx = pd.DatetimeIndex(ans.first_minutes).get_slice_bound(now_utc, "left")
        ronow_session = ans.sessions[idx]  # session to right of now
        ronow_session_close = ans.closes[ronow_session]
        last_session = ans.sessions[idx - 1]
        last_session_close = ans.closes[last_session]

        oob_session = ans.last_session + one_day  # right of right calendar bound
        oob_minute = ans.last_minute + one_min  # right of right calendar bound

        def mock_now_closed(*_, tz=UTC, **__) -> pd.Timestamp:
            now = ans.last_minutes[last_session] + (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        def mock_now_open(*_, tz=UTC, **__) -> pd.Timestamp:
            now = ans.last_minutes[last_session] - (5 * one_min)
            if tz is not UTC:
                now = now.tz_convert(tz)
            return now

        # input as date
        for session in [ronow_session, oob_session]:
            with pytest.raises(errors.StartTooLateError):
                f(session, None, as_dates)
            with pytest.raises(errors.StartTooLateError):
                f(session, None, as_times)

        # when end > now should NOT return end as None, but rather just pass end
        assert f(None, ronow_session, as_dates) == (None, ronow_session)
        assert f(None, ronow_session, as_times) == (None, ronow_session_close)
        match = re.escape(
            f"Prices unavailable as end ({helpers.fts(oob_session)}) is later than"
            f" the latest date of calendar '{ans.name}'. The calendar's latest date"
            f" is {helpers.fts(ans.last_session)}."
        )
        with pytest.raises(errors.EndOutOfBoundsRightError, match=match):
            f(None, oob_session, as_dates)
        with pytest.raises(errors.EndOutOfBoundsRightError):
            f(None, oob_session, as_times)

        if last_session in ans.sessions_with_gap_after:
            monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
            # when end > now should return end as session
            assert f(None, ronow_session, as_times) == (None, ronow_session_close)
            assert f(None, ronow_session, as_dates) == (None, ronow_session)
            with pytest.raises(errors.EndOutOfBoundsRightError):
                f(None, oob_session, as_times)
            with pytest.raises(errors.EndOutOfBoundsRightError):
                f(None, oob_session, as_dates)

        monkeypatch.setattr("pandas.Timestamp.now", mock_now_open)
        # should ignore now
        assert f(None, ronow_session, as_times) == (None, ronow_session_close)
        assert f(None, ronow_session, as_dates) == (None, ronow_session)
        with pytest.raises(errors.EndOutOfBoundsRightError):
            f(None, oob_session, as_times)
        with pytest.raises(errors.EndOutOfBoundsRightError):
            f(None, oob_session, as_dates)

        # input as time
        ool_minute = mock_now_open() + (5 * one_min)
        # StartTooLateError should raise regardless of now
        for minute in [ool_minute, oob_minute]:
            with pytest.raises(errors.StartTooLateError):
                f(minute, None, as_times)
            with pytest.raises(errors.StartTooLateError):
                f(minute, None, as_dates)

        # end should return without consideration to now
        assert f(None, ool_minute, as_times) == (None, ool_minute)
        assert f(None, ool_minute, as_dates) == (None, last_session)
        with pytest.raises(errors.EndOutOfBoundsRightError):
            f(None, oob_minute, as_times)
        with pytest.raises(errors.EndOutOfBoundsRightError):
            f(None, oob_minute, as_dates)

        if last_session in ans.sessions_with_gap_after:
            monkeypatch.setattr("pandas.Timestamp.now", mock_now_closed)
            ool_minute = mock_now_closed() + (7 * one_min)

            for minute in [ool_minute, oob_minute]:
                with pytest.raises(errors.StartTooLateError):
                    f(minute, None, as_times)
                with pytest.raises(errors.StartTooLateError):
                    f(minute, None, as_dates)

            assert f(None, ool_minute, as_times) == (None, last_session_close)
            assert f(None, ool_minute, as_dates) == (None, last_session)
            with pytest.raises(errors.EndOutOfBoundsRightError):
                f(None, oob_minute, as_times)
            with pytest.raises(errors.EndOutOfBoundsRightError):
                f(None, oob_minute, as_dates)

    @dataclasses.dataclass
    class StartEnd:
        """Input and expected return values for a `start` or `end` parameter."""

        value: pd.Timestamp
        expected_as_date: pd.Timestamp
        expected_as_time: pd.Timestamp

    @pytest.fixture
    def f_starts_ends(self, f_with_ans, one_min, today) -> abc.Iterator[
        tuple[
            Callable,
            list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
            list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
        ]
    ]:
        # pylint: disable=too-complex
        f, ans = f_with_ans

        def add(lst, *args):
            lst.append(self.StartEnd(*args))  # pylint: disable=no-value-for-parameter

        starts: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

        def add_start(*args):
            add(starts, *args)  # pylint: disable=no-value-for-parameter

        ends: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

        def add_end(*args):
            add(ends, *args)  # pylint: disable=no-value-for-parameter

        bv = ans.sessions < today
        sessions = ans.sessions[bv]

        bv = ans.non_sessions < today
        non_sessions = ans.non_sessions[bv]

        # start dates
        session, next_session = sessions[1], sessions[2]
        session_first_minute = ans.first_minutes[session]
        add_start(session, session, session_first_minute)
        if not non_sessions.empty:
            date = non_sessions[0]
            sesh = ans.date_to_session(date, "next")
            add_start(date, sesh, ans.first_minutes[sesh])

        # start minutes
        minute = session_first_minute
        add_start(minute, session, minute)
        if session in ans.sessions_with_gap_before:
            minute = session_first_minute - one_min
            add_start(minute, session, session_first_minute)

        session_close = ans.closes[session]
        next_session_open = ans.opens[next_session]

        minute = session_first_minute + one_min
        add_start(minute, next_session, minute)

        minutes = [session_close]
        if session in ans.sessions_with_gap_after:
            minutes.append(session_close + one_min)
        for minute in minutes:
            add_start(minute, next_session, next_session_open)

        # end dates
        session, prev_session = sessions[-2], sessions[-3]
        session_close = ans.closes[session]
        session_first_minute = ans.first_minutes[session]
        add_end(session, session, session_close)
        if not non_sessions.empty:
            date = non_sessions[-1]
            sesh = ans.date_to_session(date, "previous")
            add_end(date, sesh, ans.closes[sesh])

        # end minutes
        minute = session_close
        add_end(minute, session, minute)
        if session in ans.sessions_with_gap_after:
            minute = session_close + one_min
            add_end(minute, session, session_close)

        minutes = [session_close - one_min, session_first_minute]
        for minute in minutes:
            add_end(minute, prev_session, minute)

        if session in ans.sessions_with_gap_before:
            minute = session_first_minute - one_min
            add_end(minute, prev_session, ans.closes[prev_session])

        yield f, starts, ends

    def test_start_end(self, f_starts_ends, delays, as_dates, as_times):
        """Test combinations of valid start and end parameters.

        Tests input with different timezones.
        Tests with and without delay.
        """
        f, starts, ends = f_starts_ends

        for start, end in itertools.product(starts, ends):
            expected_as_dates = (start.expected_as_date, end.expected_as_date)
            expected_as_times = (start.expected_as_time, end.expected_as_time)
            for delay in delays:
                assert f(start.value, end.value, as_dates, delay) == expected_as_dates
                assert (
                    f(start.value, end.value, as_times, delay) == expected_as_times
                ), (start.value, end.value)

    def test_start_end_2(self, f_with_ans, as_dates, as_times):
        """Verify where start, end as non-sessions."""
        f, ans = f_with_ans

        if ans.sessions_range_defined_by_non_sessions is not None:
            (start, end), sessions = ans.sessions_range_defined_by_non_sessions
            assert f(start, end, as_dates) == (sessions[0], sessions[-1])
            expected = ans.first_minutes[sessions[0]], ans.closes[sessions[-1]]
            assert f(start, end, as_times) == expected

    def test_start_later_than_end(self, f_with_ans, as_dates, as_times, one_min):
        """Verify raises `errors.StartLaterThanEnd`.

        Verifies error raised if:
            start > end
            start or end defined as a time and start == end

        Also verifies that does not raise error when:
            start == end and start and end are both dates
            start > end.
        """
        f, ans = f_with_ans

        def assert_raises(start, end, as_times: bool):
            with pytest.raises(errors.StartNotEarlierThanEnd):
                f(start, end, as_times)

        session, next_session = ans.sessions[2], ans.sessions[3]
        session_first_minute = ans.first_minutes[session]
        session_second_minute = session_first_minute + one_min
        session_close = ans.closes[session]

        # date / date
        assert f(session, session, as_dates) == (session, session)
        expected = (session_first_minute, session_close)
        assert f(session, session, as_times) == expected

        assert_raises(next_session, session, as_times)
        assert_raises(next_session, session, as_dates)

        # time / time
        minute = session_first_minute
        next_minute = session_second_minute

        assert f(minute, next_minute, as_times) == (minute, next_minute)

        for as_ in [as_times, as_dates]:
            assert_raises(minute, minute, as_)
            assert_raises(next_minute, minute, as_)

        # date / time
        assert f(session, next_minute, as_times) == (minute, next_minute)

        for as_ in [as_times, as_dates]:
            assert_raises(session, minute, as_)
            assert_raises(session, minute - one_min, as_)

        # time / date
        minute = session_close
        prev_minute = session_close - one_min

        assert f(prev_minute, session, as_times) == (prev_minute, minute)

        for as_ in [as_times, as_dates]:
            assert_raises(minute, session, as_)
            assert_raises(minute + one_min, session, as_)

    def test_range_empty(self, f_with_ans, as_dates, as_times, one_min):
        """Verify raises `errors.PricesDateRangeEmpty`.

        Verifies raises error if calendar not open from start through end.
        Verifies against a run of non-sessions. Also verifies that does not
        raise error when start and end on bounds of an empty range.
        """
        f, ans = f_with_ans
        non_sessions_run = ans.non_sessions_run

        if len(non_sessions_run) < 2:
            return

        def assert_raises(start, end, as_times: bool):
            with pytest.raises(errors.PricesDateRangeEmpty):
                f(start, end, as_times)

        first_date, last_date = non_sessions_run[0], non_sessions_run[-1]
        session_before = ans.date_to_session(first_date, "previous")
        session_after = ans.date_to_session(first_date, "next")

        session_before_close = ans.closes[session_before]
        session_before_first_min = ans.first_minutes[session_before]
        session_after_first_min = ans.first_minutes[session_after]
        session_after_close = ans.closes[session_after]

        first_time = session_before_close + one_min
        last_time = session_after_first_min - one_min

        # date / date
        expected = (session_before, session_after)
        assert f(session_before, session_after, as_dates) == expected
        expected = session_before_first_min, session_after_close
        assert f(session_before, session_after, as_times) == expected

        assert_raises(first_date, last_date, as_dates)
        assert_raises(first_date, last_date, as_times)

        # time / time
        #   as_times
        expected = session_before_close - one_min, session_after_first_min
        assert f(*expected, as_times) == expected

        assert_raises(session_before_close, session_after_first_min, as_times)
        assert_raises(session_before_close, last_time, as_times)
        assert_raises(first_time, session_after_first_min, as_times)
        assert_raises(first_time, last_time, as_times)

        #   as_dates
        expected = (session_before, session_after)
        assert f(session_before_first_min, session_after_close, as_dates) == expected

        expected = (session_after, session_after)
        rtrn = f(session_before_first_min + one_min, session_after_close, as_dates)
        assert rtrn == expected

        expected = (session_before, session_before)
        rtrn = f(session_before_first_min, session_after_close - one_min, as_dates)
        assert rtrn == expected

        assert_raises(
            session_before_first_min + one_min,
            session_after_close - one_min,
            as_dates,
        )

        # date / time
        assert_raises(first_date, last_time, as_times)
        assert_raises(first_date, session_after_first_min, as_times)
        expected = (session_after_first_min, session_after_first_min + one_min)
        assert f(first_date, session_after_first_min + one_min, as_times) == expected

        expected = (session_after, session_after)
        assert f(last_date, session_after_close, as_dates) == expected
        assert_raises(last_date, session_after_close - one_min, as_dates)

        # time / date
        assert_raises(first_time, last_date, as_times)
        assert_raises(session_before_close, last_date, as_times)
        expected = (session_before_close - one_min, session_before_close)
        assert f(session_before_close - one_min, last_date, as_times) == expected

        expected = (session_before, session_before)
        assert f(session_before_first_min, first_date, as_dates) == expected
        assert_raises(session_before_first_min + one_min, first_date, as_dates)

    def test_range_empty_2(self, f_with_ans, as_dates, as_times, one_min):
        """Verify raises `errors.PricesDateRangeEmpty`.

        Verifies raises error if calendar not open from start through end.
        Verifies against a gap between consecutive sessions. Also verifies
        that does not raise error when start and end on bounds of an empty
        range.
        """
        f, ans = f_with_ans

        if ans.sessions_with_gap_after.empty:
            return

        def assert_raises(start, end, as_times: bool):
            with pytest.raises(errors.PricesDateRangeEmpty):
                f(start, end, as_times)

        session_before = ans.sessions_with_gap_after[1]
        session_after = ans.get_next_session(session_before)
        close = ans.closes[session_before]
        session_before_open = ans.first_minutes[session_before]
        open_ = ans.first_minutes[session_after]
        session_after_close = ans.closes[session_after]

        first_time = close + one_min
        last_time = open_ - one_min

        # time / time (date input not relevant to gaps between sessions)

        assert_raises(close, open_, as_times)
        assert_raises(close, last_time, as_times)
        assert_raises(first_time, open_, as_times)

        minute_before_close = close - one_min

        assert f(minute_before_close, open_, as_times) == (minute_before_close, open_)
        expected = (minute_before_close, close)
        assert f(minute_before_close, last_time, as_times) == expected

        expected = open_, open_ + one_min
        assert f(first_time, open_ + one_min, as_times) == expected

        expected = (session_before, session_after)
        rtrn = f(session_before_open, session_after_close, as_dates)
        assert rtrn == expected

        expected = (session_after, session_after)
        rtrn = f(session_before_open + one_min, session_after_close, as_dates)
        assert rtrn == expected

        expected = (session_before, session_before)
        rtrn = f(session_before_open, session_after_close - one_min, as_dates)
        assert rtrn == expected

        assert_raises(first_time, last_time, as_times)
        assert_raises(
            session_before_open + one_min,
            session_after_close - one_min,
            as_dates,
        )

    def test_gregorian(self, xlon_calendar_extended, today, one_day, one_min):
        """Test effect of passing `gregorian` as True."""
        kwargs = {
            "as_times": False,
            "calendar": xlon_calendar_extended,
            "delay": pd.Timedelta(0),
            "strict": True,
        }

        def f(start, end, gregorian) -> mptypes.DateRangeAmb:
            return m.parse_start_end(
                start,
                end,
                gregorian=gregorian,
                mr_session=None,
                mr_minute=None,
                **kwargs,
            )

        # verify returns as gregorian dates, not trading calendar sessions
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2021-01-01")
        assert f(start, end, True) == (start, end)
        # check difference when evaluating against trading calendar
        rtrn = f(start, end, False)
        assert rtrn[0] != start
        assert rtrn[1] != end

        # verify end ahead of 'now' still parses to None
        assert f(start, today + one_day, True) == (start, None)

        # verify times pass to gregorian dates
        start_time = (start - one_min).tz_localize(UTC)
        end_time = (end + one_min).tz_localize(UTC)
        assert f(start_time, end_time, True) == (start, end)

        # verify when start/end None, evalute to None
        assert f(None, None, True) == (None, None)


def test_verify_date_not_oob(one_day):
    f = m.verify_date_not_oob

    def too_early_error_msg(ts, bound) -> str:
        return (
            f"`spam` cannot be earlier than the first spam for which"
            f" prices are available. First spam for which prices are"
            f" available is {bound} although `spam` received as {ts}."
        )

    def too_late_error_msg(ts, bound) -> str:
        return (
            f"`spam` cannot be later than the most recent spam for which"
            f" prices are available. Most recent spam for which prices"
            f" are available is {bound} although `spam` received as {ts}."
        )

    date = pd.Timestamp("2021-11-02")
    assert f(date, date, date) is None

    l_bound = date + one_day
    with pytest.raises(
        errors.DatetimeTooEarlyError,
        match=re.escape(too_early_error_msg(date, l_bound)),
    ):
        f(date, l_bound, date, param_name="spam")

    r_bound = date - one_day
    with pytest.raises(
        errors.DatetimeTooLateError, match=re.escape(too_late_error_msg(date, r_bound))
    ):
        f(date, date, r_bound, param_name="spam")


def test_verify_time_not_oob(one_min):
    f = m.verify_time_not_oob

    def too_early_error_msg(ts, bound) -> str:
        return (
            f"`time` cannot be earlier than the first time for which"
            f" prices are available. First time for which prices are"
            f" available is {bound} although `time` received as {ts}."
        )

    def too_late_error_msg(ts, bound) -> str:
        return (
            f"`time` cannot be later than the most recent time for which"
            f" prices are available. Most recent time for which prices"
            f" are available is {bound} although `time` received as {ts}."
        )

    time = pd.Timestamp("2021-11-02 14:33", tz=UTC)

    assert f(time, time, time) is None

    l_bound = time + one_min
    with pytest.raises(
        errors.DatetimeTooEarlyError,
        match=re.escape(too_early_error_msg(time, l_bound)),
    ):
        f(time, l_bound, time)

    r_bound = time - one_min
    with pytest.raises(
        errors.DatetimeTooLateError, match=re.escape(too_late_error_msg(time, r_bound))
    ):
        f(time, time, r_bound)

    midnight = pd.Timestamp("2021-11-02", tz=UTC)
    assert f(midnight, midnight, midnight) is None

    with pytest.raises(errors.DatetimeTooEarlyError):
        f(midnight, midnight + one_min, midnight)
    with pytest.raises(errors.DatetimeTooLateError):
        f(midnight, midnight, midnight - one_min)


# ------------------ tests for valimp.Parser functions --------------------


def test_lead_symbol():
    class MockCls:
        """Mock class to test parsing.lead_symbol."""

        # pylint: disable=too-few-public-methods

        def _verify_lead_symbol(self, symbol: str):
            if symbol != "MSFT":
                raise ValueError(f"{symbol} not in symbols.")

        @parse
        def mock_func(self, arg: Annotated[str, Parser(m.lead_symbol)]) -> str:
            return arg

    f = MockCls().mock_func

    # verify valid inpout
    s = "MSFT"
    assert f(s) is s

    # verify raises error if symbol not valid lead_symbol
    s = "RFT"
    match = f"{s} not in symbols."
    with pytest.raises(ValueError, match=match):
        f("RFT")


def assert_valid_timezone(func: Callable, field: str):
    """Assert `func` arg takes input valid for ZoneInfo.

    Asserts valid input returns as would be returned by ZoneInfo.
    Verifies that invalid input for ZoneInfo raises an error.
    """
    # verify valid input
    assert func("UTC") == UTC
    expected = ZoneInfo("Europe/London")
    assert func("Europe/London") == expected
    assert func(expected) == expected


def test_to_timezone():
    @parse
    def mock_func(
        arg: Annotated[Union[ZoneInfo, str], Parser(m.to_timezone)]
    ) -> ZoneInfo:
        assert isinstance(arg, ZoneInfo)
        return arg

    assert_valid_timezone(mock_func, "Timezone")


def test_to_prices_timezone():
    tz = ZoneInfo("US/Eastern")

    class MockCls:
        """Mock class to test parsing.to_prices_timezone."""

        @property
        def symbols(self) -> list[str]:
            return ["MSFT"]

        @property
        def timezones(self) -> dict:
            return {"MSFT": tz}

        @parse
        def mock_func(
            self,
            arg: Annotated[Union[str, ZoneInfo], Parser(m.to_prices_timezone)],
        ) -> ZoneInfo:
            assert isinstance(arg, ZoneInfo)
            return arg

    f = MockCls().mock_func

    # verify valid input
    assert_valid_timezone(f, "PricesTimezone")

    # verify can take a symbol
    assert f("MSFT") == tz
    # but not any symbol
    with pytest.raises(zoneinfo.ZoneInfoNotFoundError):
        f("HEY")


def test_to_datetimestamp():
    @parse
    def mock_func(
        arg: Annotated[
            Union[pd.Timestamp, str, datetime.datetime, int, float, None],
            Coerce(pd.Timestamp),
            Parser(m.verify_datetimestamp, parse_none=False),
        ] = None,
    ) -> pd.Timestamp:
        if TYPE_CHECKING:
            assert isinstance(arg, pd.Timestamp)
        return arg

    # verify valid input
    assert mock_func() is None
    assert mock_func(None) is None
    expected = pd.Timestamp("2022-03-01")
    assert mock_func("2022-03-01") == expected
    assert mock_func("2022-03") == expected
    assert mock_func(expected) == expected
    assert mock_func(expected.value) == expected

    # verify input cannot be timezone aware
    expected = pd.Timestamp("2022-03-01")
    ts = expected.tz_localize(UTC)
    match = re.escape(f"`arg` must be tz-naive, although receieved as {ts}")
    with pytest.raises(ValueError, match=match):
        mock_func(ts)

    # verify input cannot have a time component
    obj = "2022-03-01 00:01"
    match = re.escape(
        "`arg` can not have a time component, although receieved"
        f" as {pd.Timestamp(obj)}. For an intraday price use .price_at()."
    )
    with pytest.raises(ValueError, match=match):
        mock_func(obj)


def test_to_timetimestamp():
    @parse
    def mock_func(
        arg: Annotated[
            Union[pd.Timestamp, str, datetime.datetime, int, float, None],
            Coerce(pd.Timestamp),
            Parser(m.verify_timetimestamp, parse_none=False),
        ] = None,
    ) -> pd.Timestamp:
        if TYPE_CHECKING:
            assert isinstance(arg, pd.Timestamp)
        return arg

    # verify valid input
    assert mock_func() is None
    assert mock_func(None) is None
    expected = pd.Timestamp("2022-03-01 00:01")
    assert mock_func("2022-03-01 00:01") == expected
    assert mock_func(expected) == expected
    assert mock_func(expected.value) == expected

    # verify input can be midnight if tz aware
    ts = pd.Timestamp("2022-03-01 00:00", tz=UTC)
    assert mock_func(ts) == ts

    # verify input can be timezone naive if not midnight
    ts = pd.Timestamp("2022-03-01 00:01")
    assert mock_func(ts) == ts

    # verify input cannot be midnight and timezone naive
    ts = pd.Timestamp("2022-03-01 00:00")
    match = re.escape(
        "`arg` must have a time component or be tz-aware,"
        f" although receieved as {ts}. To define arg as midnight pass"
        " as a tz-aware pd.Timestamp. For prices as at a session's"
        " close use .close_at()."
    )
    with pytest.raises(ValueError, match=match):
        mock_func(ts)
