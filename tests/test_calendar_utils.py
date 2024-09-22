"""Tests for market_prices.utils.calendar_utils module.

Notes
-----
Exception testing.
    Tests only test for exceptions raised directly by the tested method.
"""

import functools
import itertools
import re
import typing
from collections import abc

import numpy as np
import pandas as pd
from pandas import Timestamp as T
from pandas.testing import assert_index_equal
import pytest
import exchange_calendars as xcals

from market_prices.intervals import TDInterval
from market_prices import errors, helpers
from market_prices.helpers import UTC
import market_prices.utils.calendar_utils as m
import market_prices.utils.pandas_utils as pdutils


# pylint: disable=missing-function-docstring,redefined-outer-name,too-many-public-methods
# pylint: disable=missing-param-doc,missing-any-param-doc,too-many-locals
# pylint: disable=missing-type-doc
# pylint: disable=protected-access,unused-argument,no-self-use,too-many-arguments
#   missing-fuction-docstring, missing-any-param-doc, missing-type-doc:
#       doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments,too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name,no-self-use
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway

# Any flake8 disabled violations handled via per-file-ignores on .flake8


def test_get_exchange_info():
    df = m.get_exchange_info()
    assert isinstance(df, pd.DataFrame) and len(df) > 45
    for column in ["Exchange", "ISO Code", "Country"]:
        assert column in df.columns


def test_minutes_in_period(calendars_with_answers, one_min):
    """Test `minutes_in_period`.

    Assumes all calendars have side "left", as required by `market_prices`.
    """
    # pylint: disable=too-complex, too-many-branches, too-many-statements
    cal, ans = calendars_with_answers
    f = m.minutes_in_period

    for key, sessions in ans.session_blocks.items():
        block_mins = ans.session_block_minutes[key]
        if block_mins.empty:
            continue
        start_session, end_session = sessions[0], sessions[-1]
        num_mins = len(block_mins)
        first_min, last_min = block_mins[0], block_mins[-1]
        assert f(cal, first_min, first_min) == 0
        assert f(cal, last_min, last_min) == 0
        assert f(cal, first_min, last_min) == num_mins - 1
        for start, end in (
            (first_min + one_min, last_min),
            (first_min, last_min - one_min),
        ):
            assert f(cal, start, end) == num_mins - 2
        assert f(cal, first_min + one_min, last_min - one_min) == num_mins - 3

        # when end is session close
        close = last_min + one_min
        assert f(cal, first_min, close) == num_mins

        if start_session == ans.first_session or end_session == ans.last_session:
            pass

        elif (
            start_session in ans.sessions_with_gap_before
            and end_session in ans.sessions_with_gap_after
        ):
            for start, end in (
                (first_min - one_min, close),
                (first_min, close + one_min),
                (first_min - one_min, close + one_min),
            ):
                assert f(cal, start, end) == num_mins

        elif start_session in ans.sessions_with_gap_before:
            start = first_min - one_min
            assert f(cal, start, close) == num_mins
            assert f(cal, start, close + one_min) == num_mins + 1

        elif end_session in ans.sessions_with_gap_after:
            end = close + one_min
            assert f(cal, first_min, end) == num_mins
            assert f(cal, first_min - one_min, end) == num_mins + 1

        else:
            assert f(cal, first_min - one_min, close) == num_mins + 1
            assert f(cal, first_min, close + one_min) == num_mins + 1
            assert f(cal, first_min - one_min, close + one_min) == num_mins + 2

        if len(sessions) > 1 and start_session in ans.sessions_with_gap_after:
            # verify returns 0 with start as close and end as next open
            start_session_close = (
                ans.get_session_minutes(start_session)[-1][-1] + one_min
            )
            assert cal.session_close(start_session) == start_session_close
            next_session_open = ans.get_session_minutes(sessions[1])[0][0]
            for start, end in (
                (start_session_close, next_session_open),
                (start_session_close + one_min, next_session_open - one_min),
            ):
                assert f(cal, start, end) == 0

            # verify one min either side returns a minute
            for start, end in (
                (start_session_close - one_min, next_session_open),
                (start_session_close, next_session_open + one_min),
            ):
                assert f(cal, start, end) == 1
            # ...and that one min on both sides returns 2
            assert (
                f(cal, start_session_close - one_min, next_session_open + one_min) == 2
            )

        if (
            ans.session_has_break(start_session)
            and not start_session == ans.first_session
        ):
            minutes_session = ans.get_session_minutes(start_session)
            mins_am = minutes_session[0]
            open_, break_start = mins_am[0], mins_am[-1] + one_min
            num_mins_am = (break_start - open_) // one_min
            mins_pm = minutes_session[-1]
            break_end, close = mins_pm[0], mins_pm[-1] + one_min
            num_mins_pm = (close - break_end) // one_min

            # verify minues either side of am subsession edges
            assert f(cal, open_, break_start) == num_mins_am
            assert f(cal, open_, break_start + one_min) == num_mins_am
            assert f(cal, open_, break_start - one_min) == num_mins_am - 1

            if start_session in ans.sessions_with_gap_before:
                expected = num_mins_am
            else:
                expected = num_mins_am + 1
            for start, end in (
                (open_ - one_min, break_start),
                (open_ - one_min, break_start + one_min),
            ):
                assert f(cal, start, end) == expected

            # verify minutes over break
            assert f(cal, break_start, break_end) == 0
            assert f(cal, break_start - one_min, break_end) == 1
            assert f(cal, break_start, break_end + one_min) == 1
            assert f(cal, break_start - one_min, break_end + one_min) == 2

            # verify minues either side of pm subsession edges
            assert f(cal, break_end, close) == num_mins_pm
            assert f(cal, break_end - one_min, close) == num_mins_pm
            assert f(cal, break_end + one_min, close) == num_mins_pm - 1

            if start_session in ans.sessions_with_gap_after:
                expected = num_mins_pm
            else:
                expected = num_mins_pm + 1
            for start, end in (
                (break_end, close + one_min),
                (break_end - one_min, close + one_min),
            ):
                assert f(cal, start, end) == expected

    # verify raises error when end < start
    match = re.escape(
        f"`end` cannot be earlier than `start`. Received `start` as {end}"
        f" and `end` as {start}."
    )
    with pytest.raises(ValueError, match=match):
        f(cal, end, start)


def test_get_trading_index():
    """Test `get_trading_index` convenience function.

    `get_trading_index` wraps calendar.trading_index. Test limited to
    verifying that all parameters are passed through together with expected
    values for other options.
    """
    f = m.get_trading_index
    cal = xcals.get_calendar("XHKG", start="2021", side="left")

    end = pd.Timestamp("2021-12-20")
    start = cal.session_offset(end, -10)
    # assert start_s and end_s are regular sessions
    reg_session_length = pd.Timedelta(hours=6, minutes=30)
    start_s_open, start_s_close = cal.session_open_close(start)
    assert start_s_close - start_s_open == reg_session_length
    end_s_open, end_s_close = cal.session_open_close(end)
    assert end_s_close - end_s_open == reg_session_length

    def expected_index(
        interval: TDInterval,
        force: bool = False,
        ignore_breaks: bool = False,
        curtail_overlaps: bool = False,
    ) -> pd.IntervalIndex:
        expected_args = (start, end, interval, True, "left")
        expected_kwargs = dict(
            force=force, ignore_breaks=ignore_breaks, curtail_overlaps=curtail_overlaps
        )
        return cal.trading_index(*expected_args, **expected_kwargs)

    # verify daily interval
    interval = TDInterval.D1
    rtrn = f(cal, interval, start, end)
    expected = expected_index(interval)
    assert_index_equal(rtrn, expected)

    # verify intraday and passing through `ignore_breaks`` and `force``
    for interval in (TDInterval.T1, TDInterval.T5, TDInterval.H1):
        for ignore_breaks, force in itertools.product((True, False), (True, False)):
            expected = expected_index(interval, force, ignore_breaks)
            rtrn = f(cal, interval, start, end, force, ignore_breaks)
            assert_index_equal(rtrn, expected)

    # verify `curtail_overlaps` being passed through
    interval = TDInterval.H2
    expected = expected_index(interval, curtail_overlaps=True)
    rtrn = f(cal, interval, start, end, curtail_overlaps=True)
    assert_index_equal(rtrn, expected)

    with pytest.raises(xcals.errors.IntervalsOverlapError):
        f(cal, interval, start, end, curtail_overlaps=False)


@pytest.fixture(params=[True, False])
def strict_all(request) -> abc.Iterator[bool]:
    yield request.param


@pytest.fixture(params=[True, False])
def to_break_all(request) -> abc.Iterator[bool]:
    yield request.param


def test_subsession_length(calendars_with_answers, strict_all, to_break_all):
    cal, ans = calendars_with_answers
    strict, to_break = strict_all, to_break_all

    def f(session) -> pd.Timedelta:
        return m.subsession_length(cal, session, to_break, strict)

    for session in ans.sessions_sample:
        if ans.session_has_break(session):
            if to_break:
                rtrn = ans.break_starts[session] - ans.opens[session]
                assert f(session) == rtrn
            else:
                rtrn = ans.closes[session] - ans.break_ends[session]
                assert f(session) == rtrn
        elif not strict:
            if to_break:
                with pytest.raises(m.NoBreakError):
                    f(session)
            else:
                rtrn = ans.closes[session] - ans.opens[session]
                assert f(session) == rtrn
        else:
            with pytest.raises(m.NoBreakError):
                f(session)


# Composite Calendar


class CompositeAnswers:
    """Answers for CompositeCalendar methods."""

    ANSWERS_BASE_PATH = (
        "https://raw.github.com/gerrymanoim/exchange_calendars/master/tests/resources/"
    )

    LEFT_SIDES = ["left", "both"]
    RIGHT_SIDES = ["right", "both"]

    def __init__(
        self,
        calendar_names: list[str],
        side: typing.Literal["left", "right", "both", "neither"],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        self._names = calendar_names
        self._side = side
        self._start = start
        self._end = end
        self._dates = pd.date_range(start, end)

        self._session_idx = len(self.sessions) // 2
        assert self._session_idx != 0

    # get and other helper methods

    def _get_csv(self, name: str) -> pd.DataFrame:
        """Get resources .csv file for given calendar `name`."""
        filename = name.replace("/", "-").lower() + ".csv"

        df = pd.read_csv(
            self.ANSWERS_BASE_PATH + filename,
            index_col=0,
            parse_dates=[0, 1, 2, 3, 4],
        )
        # Necessary for csv saved prior to xcals v4.0
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        # Necessary for csv saved prior to xcals v4.0
        for col in df:
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize(UTC)
        return df

    @functools.cached_property
    def _answerss(self) -> dict[str, pd.DataFrame]:
        """Answers file as pd.DataFrame for each calendar."""
        return {name: self._get_csv(name) for name in self._names}

    @property
    def _has_right_side(self) -> bool:
        return self.side in self.RIGHT_SIDES

    @property
    def _has_left_side(self) -> bool:
        return self.side in self.LEFT_SIDES

    def _get_next_session(self, session: pd.Timestamp) -> pd.Timestamp:
        assert session != self.sessions[-1]
        idx = self.sessions.get_loc(session) + 1
        return self.sessions[idx]

    def _get_previous_session(self, session: pd.Timestamp) -> pd.Timestamp:
        assert session != self.sessions[0]
        idx = self.sessions.get_loc(session) - 1
        return self.sessions[idx]

    def _get_session_last_minute(self, session: pd.Timestamp) -> pd.Timestamp:
        close = self.closes[session]
        return close if self._has_right_side else close - helpers.ONE_MIN

    def _get_session_first_minute(self, session: pd.Timestamp) -> pd.Timestamp:
        open_ = self.opens[session]
        return open_ if self._has_left_side else open_ + helpers.ONE_MIN

    @functools.cached_property
    def sessions_with_gap_after(self) -> pd.DatetimeIndex:
        mask = self.closes >= self.opens.shift(-1)
        if self.side == "both":
            closes_plus_min = self.closes + pd.Timedelta(1, "min")
            mask = mask | (closes_plus_min == self.opens.shift(-1))
        return self.sessions[~mask][:-1]

    @functools.cached_property
    def _opens(self) -> pd.Series:
        """Open times for each day of _dates. Value missing if not a session."""
        columns = {name: ans.open for name, ans in self._answerss.items()}
        all_opens = pd.DataFrame(columns, index=self._dates)
        # because pandas has a bug where min will not skipna if values are
        # tz-aware pd.Timestamp
        for col in all_opens:
            all_opens[col] = all_opens[col].dt.tz_convert(None)
        opens_min = all_opens.min(axis=1)
        return opens_min.dt.tz_localize(UTC)

    @functools.cached_property
    def _closes(self) -> pd.Series:
        """Closes times for each day of _dates. Value missing if not a session."""
        columns = {name: ans.close for name, ans in self._answerss.items()}
        all_closes = pd.DataFrame(columns, index=self._dates)
        # because pandas has a bug where min will not skipna if values are
        # tz-aware pd.Timestamp
        for col in all_closes:
            all_closes[col] = all_closes[col].dt.tz_convert(None)
        closes_max = all_closes.max(axis=1)
        return closes_max.dt.tz_localize(UTC)

    # properties of composite calendar

    @property
    def side(self) -> typing.Literal["left", "right", "both", "neither"]:
        """Sides for which answers evaluated."""
        return self._side

    @property
    def always_open(self) -> bool:
        """True if no non-trading minutes."""
        return self.sessions_with_gap_after.empty

    # evaluated properties of all sessions.

    @property
    def opens(self) -> pd.Series:
        return self._opens[self._opens.notna()]

    @property
    def closes(self) -> pd.Series:
        return self._closes[self._closes.notna()]

    @property
    def sessions(self) -> pd.DatetimeIndex:
        sessions_ = self.opens.index
        if typing.TYPE_CHECKING:
            assert isinstance(sessions_, pd.DatetimeIndex)
        return sessions_

    @property
    def not_sessions(self) -> pd.DatetimeIndex:
        """Dates between start and end that are not sessions."""
        return self._opens.index[self._opens.isna()]

    @property
    def first_minutes(self) -> pd.Series:
        """First trading minute of each session."""
        return self.opens if self._has_left_side else self.opens + helpers.ONE_MIN

    @property
    def last_minutes(self) -> pd.Series:
        """Last trading minute of each session."""
        return self.closes if self._has_right_side else self.closes - helpers.ONE_MIN

    @property
    def sessions_length(self) -> pd.Series:
        """Length of each session.

        Returns
        -------
        pd.Series
            Index as `self.sessions`
            Values as pd.Timedelta representing corresponding session length
        """
        return self.closes - self.opens

    # a session and associated properties

    @property
    def session(self) -> pd.Series:
        """A session of answers."""
        return self.sessions[self._session_idx]

    @property
    def session_open(self) -> pd.Series:
        """Open time of `self.session`."""
        return self.opens.iloc[self._session_idx]

    @property
    def session_close(self) -> pd.Series:
        """Close time of `self.session`."""
        return self.closes.iloc[self._session_idx]

    @property
    def next_session(self) -> pd.Series:
        """Next session after `self.session`."""
        return self.sessions[self._session_idx + 1]

    @property
    def previous_session(self) -> pd.Series:
        """Previous session prior to `self.session`."""
        return self.sessions[self._session_idx - 1]

    # overlap properties and helper methods

    @functools.cached_property
    def _overlap_next_mask(self) -> pd.Series:
        """Mask for sessions that overlap with next session.

        Last session will always be False.
        """
        return self.last_minutes >= self.first_minutes.shift(-1)

    @functools.cached_property
    def _overlap_previous_mask(self) -> pd.Series:
        """Mask for sessions that overlap with previous session.

        First session will always be False.
        """
        return self.first_minutes <= self.last_minutes.shift(1)

    @property
    def sessions_overlapping_next_session(self) -> pd.DatetimeIndex:
        """Sessions that overlap with next session."""
        return self.sessions[self._overlap_next_mask]

    @property
    def sessions_overlapping_previous_session(self) -> pd.DatetimeIndex:
        """Sessions that overlap with previous session."""
        return self.sessions[self._overlap_previous_mask]

    @property
    def sessions_overlapping(self) -> pd.DatetimeIndex:
        """Sessions that overlap with another session or sessions."""
        return self.sessions[(self._overlap_next_mask) | (self._overlap_previous_mask)]

    @property
    def sessions_non_overlapping(self) -> pd.DatetimeIndex:
        """Sessions that do not overlap with another session."""
        return self.sessions[
            (~self._overlap_next_mask) & (~self._overlap_previous_mask)
        ]

    @property
    def sessions_double_overlapping(self) -> pd.DatetimeIndex:
        """Sessions that overlap with both next and previous session."""
        return self.sessions[(self._overlap_next_mask) & (self._overlap_previous_mask)]

    @property
    def sessions_overlapping_only_next_session(self) -> pd.DatetimeIndex:
        """Sessions that overlap next session but not previous session.

        Does not include first session.
        """
        return self.sessions[self._overlap_next_mask & (~self._overlap_previous_mask)][
            1:
        ]

    @property
    def sessions_overlapping_only_previous_session(self) -> pd.DatetimeIndex:
        """Sessions that overlap previous session but not next session.

        Does not include last session.
        """
        return self.sessions[self._overlap_previous_mask & (~self._overlap_next_mask)][
            :-1
        ]

    # trading minutes

    @functools.cached_property
    def double_session_minutes(
        self,
    ) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]] | list:
        """Sample of minutes that are trading minutes of two sessions.

        Returns
        -------
        list of tuple[pd.Timestamp, list[pd.Timestamp, pd.Timestamp]]
            where each tuple:
                [0] double session minute.
                [1] list of two pd.Timestamp representing each session of
                    which the double session minute is a trading minute.

            NB outer list empty if there are no overlapping sessions.

        Notes
        -----
        Includes double_session_minutes for one session of each of the
        following if such a session exists:
            session that overlaps with only next session.
            session that overlaps with only previous session.
            session that overlaps with both next and previous session.

        All minutes are on the edge of the range of minutes that are double
        session minutes, such that moving one minute in a certain direction
        will give a minute that is a single session minute. NB these
        single session mintues, on the other side of the edge, are included
        to `single_session_minutes`.
        """
        if self.sessions_overlapping.empty:
            return []

        rtrn = []

        def add_minutes_overlapping_with_next_session(session: pd.Timestamp):
            overlapped_session = self._get_next_session(session)
            sessions = [session, overlapped_session]
            minutes = [
                self._get_session_first_minute(overlapped_session),
                self._get_session_last_minute(session),
            ]
            for minute in minutes:
                rtrn.append((minute, sessions))

        def add_minutes_overlapping_with_previous_session(session: pd.Timestamp):
            overlapped_session = self._get_previous_session(session)
            sessions = [overlapped_session, session]
            minutes = [
                self._get_session_first_minute(session),
                self._get_session_last_minute(overlapped_session),
            ]
            for minute in minutes:
                rtrn.append((minute, sessions))

        overlapping_sessions = self.sessions_overlapping_only_next_session
        if not overlapping_sessions.empty:
            add_minutes_overlapping_with_next_session(overlapping_sessions[0])

        overlapping_sessions = self.sessions_overlapping_only_previous_session
        if not overlapping_sessions.empty:
            add_minutes_overlapping_with_previous_session(overlapping_sessions[0])

        overlapping_sessions = self.sessions_double_overlapping
        if not overlapping_sessions.empty:
            add_minutes_overlapping_with_next_session(overlapping_sessions[0])
            add_minutes_overlapping_with_previous_session(overlapping_sessions[0])

        return rtrn

    @functools.cached_property
    def single_session_minutes(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """List of (single_session_minute, session)."""
        rtrn = []

        non_overlapping_sessions = self.sessions_non_overlapping
        if not non_overlapping_sessions.empty:
            session = non_overlapping_sessions[0]
            rtrn.append((self._get_session_first_minute(session), session))
            rtrn.append((self._get_session_last_minute(session), session))

        # if there are overlapping sessions, include the edge minutes that
        # lie immediately either side of the overlapping periods.

        def add_minutes_around_overlap_with_next_session(session: pd.Timestamp):
            next_session = self._get_next_session(session)
            minute = self._get_session_last_minute(session) + helpers.ONE_MIN
            rtrn.append((minute, next_session))
            minute = self._get_session_first_minute(next_session) - helpers.ONE_MIN
            rtrn.append((minute, session))

        def add_minutes_around_overlap_with_previous_session(session: pd.Timestamp):
            previous_session = self._get_previous_session(session)
            minute = self._get_session_first_minute(session) - helpers.ONE_MIN
            rtrn.append((minute, previous_session))
            minute = self._get_session_last_minute(previous_session) + helpers.ONE_MIN
            rtrn.append((minute, session))

        overlapping_sessions = self.sessions_overlapping_only_next_session
        if not overlapping_sessions.empty:
            add_minutes_around_overlap_with_next_session(overlapping_sessions[0])

        overlapping_sessions = self.sessions_overlapping_only_previous_session
        if not overlapping_sessions.empty:
            add_minutes_around_overlap_with_previous_session(overlapping_sessions[0])

        overlapping_sessions = self.sessions_double_overlapping
        if not overlapping_sessions.empty:
            overlapping_session = overlapping_sessions[0]
            add_minutes_around_overlap_with_next_session(overlapping_session)
            add_minutes_around_overlap_with_previous_session(overlapping_session)

        return rtrn

    # non_trading minutes

    @functools.cached_property
    def non_trading_minutes(
        self,
    ) -> tuple[list[pd.Timestamp], pd.Timestamp, pd.Timestamp] | None:
        if self.always_open:
            return None

        previous_session = self.sessions_with_gap_after[0]
        next_session = self._get_next_session(previous_session)

        non_trading_mins = [
            self._get_session_last_minute(previous_session) + helpers.ONE_MIN,
            self._get_session_first_minute(next_session) - helpers.ONE_MIN,
        ]
        return (non_trading_mins, previous_session, next_session)


class TestCompositeCalendar:
    """Tests for CompositeCalendar."""

    @pytest.fixture(scope="class")
    def calendar_names(self) -> abc.Iterator[tuple[list[str], ...]]:
        yield (
            ["XNYS", "XLON", "XHKG"],
            ["XLON", "XHKG", "CMES"],
            ["XASX", "24/5"],
        )

    @pytest.fixture(scope="class")
    def composite_daterange(
        self,
    ) -> abc.Iterator[tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
        yield (
            (pd.Timestamp("2021-01-04"), pd.Timestamp("2021-12-31")),
            (pd.Timestamp("2016-01-04"), pd.Timestamp("2016-12-30")),
            (pd.Timestamp("2018-01-02"), pd.Timestamp("2018-12-31")),
        )

    @pytest.fixture(scope="class")
    def composite_answers(
        self, composite_daterange, calendar_names
    ) -> abc.Iterator[list[CompositeAnswers]]:
        yield [
            CompositeAnswers(names, "left", *composite_daterange[i])
            for i, names in enumerate(calendar_names)
        ]

    @pytest.fixture(scope="class")
    def calendar_groups(
        self, composite_daterange, calendar_names
    ) -> abc.Iterator[list[list[xcals.ExchangeCalendar]]]:
        calendar_groups = []
        for i, names in enumerate(calendar_names):
            calendars = []
            for name in names:
                calendars.append(
                    xcals.get_calendar(name, *composite_daterange[i], "left")
                )
            calendar_groups.append(calendars)
        yield calendar_groups

    @pytest.fixture(scope="class")
    def composite_calendars(
        self, calendar_groups
    ) -> abc.Iterator[list[m.CompositeCalendar]]:
        comp_cals = []
        for calendars in calendar_groups:
            comp_cals.append(m.CompositeCalendar(calendars))
        yield comp_cals

    @pytest.fixture(scope="class", params=(1, 2))
    def composite_calendars_with_answers(
        self,
        request,
        composite_calendars,
        composite_answers,
    ) -> abc.Iterator[tuple[m.CompositeCalendar, CompositeAnswers]]:
        """Parameterized fixture of composite calendars and answers.

        Offers last two `composite_calendars` and `composite_answers`.
        """
        i = request.param
        yield (composite_calendars[i], composite_answers[i])

    def test_passing_cc_to_parse_date(self, composite_calendars):
        """Verify raising `xcals.DateOutOfBounds`."""
        cc = composite_calendars[0]
        param_name = "date"

        too_early = cc.first_session - pd.Timedelta(1, "D")
        error_msg = (
            f"Parameter `{param_name}` receieved as '{too_early}' although cannot"
            f" be earlier than the first session of calendar '{cc.name}'"
            f" ('{cc.first_session}')."
        )
        with pytest.raises(xcals.errors.DateOutOfBounds, match=re.escape(error_msg)):
            cc._parse_date(too_early)

        too_late = cc.last_session + pd.Timedelta(1, "D")
        error_msg = (
            f"Parameter `{param_name}` receieved as '{too_late}' although cannot"
            f" be later than the last session of calendar '{cc.name}'"
            f" ('{cc.last_session}')."
        )
        with pytest.raises(xcals.errors.DateOutOfBounds, match=re.escape(error_msg)):
            cc._parse_date(too_late)

        # minimal check that returns as expected
        assert cc._parse_date("2021-06-01") == pd.Timestamp("2021-06-01")

    def test_parse_session(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        f = cc._parse_session
        timestamp = answers.sessions[0]
        expected = timestamp
        assert f(timestamp) == expected
        assert f(str(timestamp)) == expected
        not_session = answers.not_sessions[0]
        with pytest.raises(xcals.errors.NotSessionError):
            f(not_session)

        too_early = answers.sessions[0] - pd.Timedelta(1, "D")
        with pytest.raises(xcals.errors.DateOutOfBounds):
            f(too_early)

        too_late = answers.sessions[-1] + pd.Timedelta(1, "D")
        with pytest.raises(xcals.errors.DateOutOfBounds):
            f(too_late)

    def test_is_session(self, composite_calendars_with_answers):
        cc, ans = composite_calendars_with_answers
        f = cc.is_session
        for session in ans.sessions[:3].union(ans.sessions[-3:]):
            assert f(session)

        for date in ans.not_sessions[:3].union(ans.not_sessions[-3:]):
            assert not f(date)

    def test_date_to_session(self, composite_calendars_with_answers, one_day):
        cc, ans = composite_calendars_with_answers
        for session in ans.sessions[:3].union(ans.sessions[-3:]):
            assert cc.date_to_session(session) == session

        for date in ans.not_sessions[:3].union(ans.not_sessions[-3:]):
            match = (
                f"`date` '{date}' does not represent a session. Consider passing"
                " a `direction`."
            )
            with pytest.raises(ValueError, match=match):
                rtrn = cc.date_to_session(date)

            rtrn = cc.date_to_session(date, direction="next")
            non_sessions = pd.date_range(date, rtrn - one_day)
            for date_ in non_sessions:
                assert date_ not in ans.sessions
                assert date_ in ans.not_sessions
            assert rtrn in ans.sessions

            rtrn = cc.date_to_session(date, direction="previous")
            non_sessions = pd.date_range(rtrn + one_day, date)
            for date_ in non_sessions:
                assert date_ not in ans.sessions
                assert date_ in ans.not_sessions
            assert rtrn in ans.sessions

    def test_sessions_in_range(self, composite_calendars_with_answers, one_day):
        cc, ans = composite_calendars_with_answers
        f = cc.sessions_in_range

        assert_index_equal(f(), cc.sessions)

        # verify all combinations of start and end as dates / sessions
        start_date = start_session = ans.not_sessions[5]
        while start_session not in ans.sessions:
            start_session += one_day
        end_date = end_session = ans.not_sessions[-5]
        while end_session not in ans.sessions:
            end_session -= one_day

        slc_start = ans.sessions.get_slice_bound(start_session, "left")
        slc_end = ans.sessions.get_slice_bound(end_session, "right")
        expected = ans.sessions[slc_start:slc_end]

        args = ((start_date, start_session), (end_date, end_session))
        start_ends = itertools.product(*args)
        rtrns = [f(start, end) for start, end in start_ends]
        for rtrn in rtrns:
            assert_index_equal(rtrn, expected)

        # Verify can pass start / end as strings
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        assert_index_equal(f(start, end), expected)

    def test_sessions(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        pd.testing.assert_index_equal(cc.sessions, answers.sessions)
        assert not cc.sessions.isin(answers.not_sessions).any()

    def test_opens_closes(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        pd.testing.assert_series_equal(
            cc.opens, answers.opens, check_names=False, check_freq=False
        )
        pd.testing.assert_series_equal(
            cc.closes, answers.closes, check_names=False, check_freq=False
        )

    def test_first_last_minutes(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        pd.testing.assert_series_equal(
            cc.first_minutes, answers.first_minutes, check_names=False, check_freq=False
        )
        pd.testing.assert_series_equal(
            cc.last_minutes, answers.last_minutes, check_names=False, check_freq=False
        )

    def test_misc_properties(self, composite_calendars_with_answers, calendar_groups):
        cc, answers = composite_calendars_with_answers
        assert cc.first_session == answers.sessions[0]
        assert cc.last_session == answers.sessions[-1]
        assert cc.first_minute == answers.first_minutes.iloc[0]
        assert cc.last_minute == answers.last_minutes.iloc[-1]
        assert cc.side == "left"
        assert len(cc.calendars) in (2, 3)
        i = 1 if len(cc.calendars) == 3 else 2
        cals = calendar_groups[i]
        assert all(cal in cc.calendars for cal in cals)

    def test_next_session(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        assert cc.next_session(answers.session) == answers.next_session

    def test_previous_session(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        assert cc.previous_session(answers.session) == answers.previous_session

    def test_session_open(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        assert cc.session_open(answers.session) == answers.session_open

    def test_session_close(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        assert cc.session_close(answers.session) == answers.session_close

    def test_is_open_on_minute(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        f = cc.is_open_on_minute

        for minute, _ in answers.double_session_minutes:
            assert f(minute) is True

        for minute, _ in answers.single_session_minutes:
            try:
                assert f(minute) is True
            except xcals.errors.MinuteOutOfBounds:
                assert minute < max([cal.first_minute for cal in cc.calendars])

        # test non_trading_minutes
        non_trading_minutes, _, _ = answers.non_trading_minutes
        for minute in non_trading_minutes:
            assert f(minute) is False

    def test_minute_to_sessions(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        f = cc.minute_to_sessions

        for minute, sessions in answers.double_session_minutes:
            expected = pd.DatetimeIndex(sessions)
            rtrn = f(minute, None)
            pd.testing.assert_index_equal(expected, rtrn)

        for minute, session in answers.single_session_minutes:
            expected = pd.DatetimeIndex([session])
            rtrn = f(minute, None)
            pd.testing.assert_index_equal(expected, rtrn)

        # test non_trading_minutes
        non_trading_minutes, prev_session, next_session = answers.non_trading_minutes
        for minute in non_trading_minutes:
            assert f(minute, None).empty

            rtrn = f(minute, "next")
            assert len(rtrn) == 1
            assert rtrn[0] == next_session

            rtrn = f(minute, "previous")
            assert len(rtrn) == 1
            assert rtrn[0] == prev_session

    def test_sessions_overlap(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        expected = answers.sessions_overlapping_next_session
        rtrn = cc.sessions[cc.sessions_overlap()]
        pd.testing.assert_index_equal(expected, rtrn)

    def test_sessions_length(self, composite_calendars_with_answers):
        cc, answers = composite_calendars_with_answers
        expected = answers.sessions_length
        rtrn = cc.sessions_length()
        pd.testing.assert_series_equal(expected, rtrn, check_freq=False)

    @pytest.mark.parametrize(
        "factor", [pd.Timedelta(v, "min") for v in [5, 7, 11, 300]]
    )
    def test_is_factor_of_sessions(self, composite_calendars_with_answers, factor):
        cc, answers = composite_calendars_with_answers
        rtrn = cc.is_factor_of_sessions(factor)
        assert isinstance(rtrn, pd.Series)
        assert pd.api.types.is_bool_dtype(rtrn)
        assert len(rtrn) == len(answers.sessions)

    def test_non_trading_index1(self, composite_calendars):
        """Test `non_trading_index` for multiple calendars, unusual timings."""
        cc = composite_calendars[0]
        f = cc.non_trading_index

        # test full index, with no arguments
        full_index = f()
        assert isinstance(full_index, pd.IntervalIndex)
        assert cc.closes.iloc[0] in full_index[:6].left
        assert cc.opens.iloc[-1] in full_index[-6:].right

        # test utc option
        args = ("2021-02", "2021-03")
        index_utc = f(*args, utc=True)
        assert index_utc.left.tz == UTC and index_utc.right.tz == UTC
        index_naive = f(*args, utc=False)
        assert index_naive.left.tz is None and index_naive.right.tz is None
        assert_index_equal(index_naive.left, index_utc.left.tz_convert(None))
        assert_index_equal(index_naive.right.tz_localize(UTC), index_utc.right)

        # Compare with expected from manual inspection of period with unusual timings.
        # NB also tests passing end as last session of composite calendar
        rtrn = f("2021-12-24", "2021-12-31", utc=False)

        def intrvl(left: str, right: str) -> pd.Interval:
            return pd.Interval(
                pd.Timestamp("2021-12-" + left),
                pd.Timestamp("2021-12-" + right),
                "left",
            )

        # From manual inspection of a period with unusual timings.
        expected = pd.IntervalIndex(
            [
                intrvl("24 04:00", "24 08:00"),  # xhkg early close to xlon open
                intrvl("24 12:30", "27 14:30"),  # xlon early close to next xnys open
                intrvl("27 21:00", "28 01:30"),  # xnys close to xhkg open
                intrvl("28 04:00", "28 05:00"),  # xhkg break
                intrvl("28 08:00", "28 14:30"),  # xhkg close to xnys open
                intrvl("28 21:00", "29 01:30"),  # xnys close to xhkg open
                intrvl("29 04:00", "29 05:00"),  # xhkg break
                intrvl("29 21:00", "30 01:30"),  # xnys close to xhkg open
                intrvl("30 04:00", "30 05:00"),  # xhkg break
                intrvl("30 21:00", "31 01:30"),  # xnys close to xhkg open
                intrvl("31 04:00", "31 08:00"),  # xhkg early close to xlon open
                intrvl("31 12:30", "31 14:30"),  # xlon early close to next xnys open
            ]
        )

        assert_index_equal(rtrn, expected)

    def test_non_trading_index2(self):
        """Test `non_trading_index` for specific unusual circumstances.

        Tests for:
            24h calendars
            combinations of calendars where composite sessions overlap
        """
        start, end = "2020", "2021-12-31"
        x245 = xcals.get_calendar("24/5", start, end, side="left")
        x247 = xcals.get_calendar("24/7", start, end, side="left")
        cmes = xcals.get_calendar("CMES", start, end, side="left")
        xasx = xcals.get_calendar("XASX", start, end, side="left")

        # Verify 24/7 calendar has no non-trading indices.
        cc = m.CompositeCalendar([x247])
        rtrn = cc.non_trading_index()
        assert isinstance(rtrn, pd.IntervalIndex)
        assert rtrn.empty

        # Verify 24/5 calendar non-trading indices fully align with weekends.
        cc = m.CompositeCalendar([x245])
        rtrn = cc.non_trading_index(utc=False)

        dates = pd.date_range(start, end)
        left = dates[dates.weekday == 5]
        right = dates[dates.weekday == 0]
        expected = pd.IntervalIndex.from_arrays(left, right, "left")

        assert_index_equal(rtrn, expected)

        # Verify no non-trading indices where composite sessions overlap and 24h cal.
        cc = m.CompositeCalendar([x247, cmes])
        rtrn = cc.non_trading_index()
        assert isinstance(rtrn, pd.IntervalIndex)
        assert rtrn.empty

        # Verify where most composite sessions overlap, some non-trading periods.
        cc = m.CompositeCalendar([x245, xasx])
        rtrn = cc.non_trading_index("2021-12", "2021-12-31", utc=False)

        def intrvl(left: str, right: str) -> pd.Interval:
            return pd.Interval(
                pd.Timestamp("2021-12-" + left),
                pd.Timestamp("2021-12-" + right),
                "left",
            )

        # From manual inspection of a period with unusual timings.
        expected = pd.IntervalIndex(
            [
                intrvl("04 00:00", "05 23:00"),  # x245 fri close to xasx 'mon' open
                intrvl("11 00:00", "12 23:00"),  # x245 fri close to xasx 'mon' open
                intrvl("18 00:00", "19 23:00"),  # x245 fri close to xasx 'mon' open
                intrvl(
                    "25 00:00", "27 00:00"
                ),  # x245 fri to x245 'mon' open (xasx closed)
            ]
        )

        assert_index_equal(rtrn, expected)


class TestCCTradingIndex:
    """Tests for CompositeCalendar.trading_index."""

    @pytest.fixture(scope="class")
    def cal_start(self) -> abc.Iterator[pd.Timestamp]:
        yield T("2021")

    @pytest.fixture(scope="class")
    def cal_end(self) -> abc.Iterator[pd.Timestamp]:
        yield T("2022")

    @pytest.fixture(scope="class")
    def side(self) -> abc.Iterator[str]:
        yield "left"

    @pytest.fixture(scope="class")
    def xlon(self, cal_start, cal_end, side) -> abc.Iterator[xcals.ExchangeCalendar]:
        yield xcals.get_calendar("XLON", cal_start, cal_end, side)

    @pytest.fixture(scope="class")
    def xnys(self, cal_start, cal_end, side) -> abc.Iterator[xcals.ExchangeCalendar]:
        yield xcals.get_calendar("XNYS", cal_start, cal_end, side)

    @pytest.fixture(scope="class")
    def xhkg(self, cal_start, cal_end, side) -> abc.Iterator[xcals.ExchangeCalendar]:
        yield xcals.get_calendar("XHKG", cal_start, cal_end, side)

    @pytest.fixture(scope="class")
    def bvmf(self, cal_start, cal_end, side) -> abc.Iterator[xcals.ExchangeCalendar]:
        yield xcals.get_calendar("BVMF", cal_start, cal_end, side)

    @pytest.fixture
    def intervals(self) -> abc.Iterator[list[TDInterval]]:
        yield [
            TDInterval.T1,
            TDInterval.T5,
            TDInterval.T13,
            TDInterval.H1,
            TDInterval.H2,
            TDInterval.H4,
            TDInterval.H10,
        ]

    def index_for_session(
        self,
        interval: TDInterval,
        start: pd.Timestamp,
        end: pd.Timestamp,
        add_interval: bool = False,
        utc: bool = True,
    ) -> pd.IntervalIndex:
        """Get expected index for or within a single trading (sub)session."""
        if add_interval:
            end += interval
        tz = UTC if utc else None
        dti = pd.date_range(start, end, freq=interval.as_pdfreq, tz=tz)
        return pd.IntervalIndex.from_arrays(dti[:-1], dti[1:], "left")

    def _add_interval(
        self, interval: TDInterval, open_: pd.Timestamp, close: pd.Timestamp
    ) -> bool:
        """Query if an extra indice should be added with right after close."""
        return bool((close - open_) % interval)

    def get_expected(
        self,
        interval: TDInterval,
        opens: list[pd.Timestamp],
        closes: list[pd.Timestamp],
        utc: bool = True,
    ) -> pd.IntervalIndex:
        open_, close = opens[0], closes[0]
        add_interval = self._add_interval(interval, open_, close)
        index = self.index_for_session(interval, open_, close, add_interval, utc)
        for open_, close in zip(opens[1:], closes[1:]):
            add_interval = self._add_interval(interval, open_, close)
            index_ = self.index_for_session(interval, open_, close, add_interval, utc)
            index = index.union(index_)
        return index

    def test_xlon(self, xlon, intervals):
        """Test for composite calendar comprising single calendar.

        Also verifies effect of:
            `utc` parameter
            passing `start` and `end` as dates and, separately, times.
        """
        cc = m.CompositeCalendar([xlon])
        f = cc.trading_index

        # from inspection of schedule
        opens = [T("2021-12-24 08:00"), T("2021-12-29 08:00")]
        closes = [T("2021-12-24 12:30"), T("2021-12-29 16:30")]
        last_session_duration = closes[-1] - opens[-1]

        for interval in intervals:
            # passing as sessions
            start, end = T("2021-12-24"), T("2021-12-29")
            rtrn = f(interval, start, end, curtail_calendar_overlaps=False)
            index = self.get_expected(interval, opens, closes)
            assert_index_equal(rtrn, index)

            # verify curtail_calendar_overlaps has no effect when no overlaps
            rtrn_ = f(interval, start, end, curtail_calendar_overlaps=True)
            assert_index_equal(rtrn, rtrn_)

            # passing as times
            start = T("2021-12-24 08:01", tz=UTC)
            end = T("2021-12-29 16:30", tz=UTC)
            rtrn = f(interval, start, end)
            unaligned = last_session_duration % interval != pd.Timedelta(0)
            expected = index[1:-1] if unaligned else index[1:]
            assert_index_equal(rtrn, expected)

            # passing as times
            start = T("2021-12-24 08:01", tz=UTC)
            end = T("2021-12-29 16:29", tz=UTC)
            rtrn = f(interval, start, end)
            assert_index_equal(rtrn, index[1:-1])

        # test `utc` option
        interval = TDInterval.T5
        start, end = T("2021-12-24"), T("2021-12-29")
        rtrn = f(interval, start, end, utc=False)
        expected = self.get_expected(interval, opens, closes, utc=False)
        assert_index_equal(rtrn, expected)

    @pytest.fixture
    def match_xhkg(self) -> abc.Iterator[str]:
        """Error message for `CompositeIndexCalendarConflict` triggered by xhkg."""
        yield (
            "Unable to create a composite trading index as indices of calendar"
            " 'XHKG' would overlap. This can occur when the interval is"
            " longer than a break or the gap between one session's close and the"
            " next session's open."
        )

    def test_xhkg(self, xhkg, intervals, match_xhkg):
        """Test for composite calendar comprising single calendar with breaks.

        Verifies effect of `ignore_breaks`.
        """
        cc = m.CompositeCalendar([xhkg])
        f = cc.trading_index

        # if ignoring breaks
        opens = [T("2021-12-23 01:30"), T("2021-12-24 01:30"), T("2021-12-28 01:30")]
        closes = [T("2021-12-23 08:00"), T("2021-12-24 04:00"), T("2021-12-28 08:00")]

        for interval in intervals:
            start, end = T("2021-12-23"), T("2021-12-28")
            rtrn = f(interval, start, end, ignore_breaks=True)
            index = self.get_expected(interval, opens, closes)
            assert_index_equal(rtrn, index)

        # if including breaks
        opens = [
            T("2021-12-23 01:30"),
            T("2021-12-23 05:00"),
            T("2021-12-24 01:30"),
            T("2021-12-28 01:30"),
            T("2021-12-28 05:00"),
        ]

        closes = [
            T("2021-12-23 04:00"),
            T("2021-12-23 08:00"),
            T("2021-12-24 04:00"),
            T("2021-12-28 04:00"),
            T("2021-12-28 08:00"),
        ]

        start, end = T("2021-12-23"), T("2021-12-28")
        ignore_breaks = False
        for interval in intervals:
            args = (interval, start, end, ignore_breaks)
            for curtail in (False, True):
                if curtail or interval < TDInterval.H2:
                    rtrn = f(*args, curtail_calendar_overlaps=curtail)
                    index = self.get_expected(interval, opens, closes)
                    if curtail and interval >= TDInterval.H2:
                        index = pdutils.make_non_overlapping(index)
                    assert_index_equal(rtrn, index)
                else:
                    with pytest.raises(
                        errors.CompositeIndexCalendarConflict, match=match_xhkg
                    ):
                        f(*args, curtail_calendar_overlaps=curtail)

    def test_overlapping(self, xnys, xlon):
        """Test for composite calendar with two overlappiong calendars.

        Also verifies effect of `raise_overlapping`
        """
        cc = m.CompositeCalendar([xlon, xnys])
        f = cc.trading_index

        start, end = T("2021-12-23"), T("2021-12-27")
        opens = [T("2021-12-23 08:00"), T("2021-12-24 08:00"), T("2021-12-27 14:30")]
        closes = [T("2021-12-23 21:00"), T("2021-12-24 12:30"), T("2021-12-27 21:00")]

        for interval in [TDInterval.T1, TDInterval.T5, TDInterval.T30]:
            rtrn = f(interval, start, end, curtail_calendar_overlaps=False)
            index = self.get_expected(interval, opens, closes)
            assert_index_equal(rtrn, index)

            # verify curtail_calendar_overlaps has no effect when no calendar-specific
            # overlaps
            rtrn_ = f(interval, start, end, curtail_calendar_overlaps=True)
            assert_index_equal(rtrn, rtrn_)

        # verify raises `errors.CompositeIndexConflict`
        match = (
            "At least one indice of the trading index would partially overlap"
            " another. Pass `raise_overlapping` as False to supress this error."
        )
        for interval in [TDInterval.H1, TDInterval.H4]:
            with pytest.raises(errors.CompositeIndexConflict, match=match):
                rtrn = f(interval, start, end)

        # verify effect of raise_overlapping=False returns overlapping index
        for interval in [TDInterval.H1, TDInterval.H4]:
            xlon_index = xlon.trading_index(start, end, interval)
            xnys_index = xnys.trading_index(start, end, interval)
            index = xlon_index.union(xnys_index, sort=False).sort_values()
            assert index.is_overlapping
            rtrn = f(interval, start, end, raise_overlapping=False)
            assert_index_equal(rtrn, index)

    def test_detached(self, xhkg, bvmf, intervals, match_xhkg):
        """Test for composite calendar with two detached calendars.

        Also verifies effect of `ignore_breaks`
        """
        cc = m.CompositeCalendar([xhkg, bvmf])
        f = cc.trading_index

        # if ignoring breaks
        opens = [
            T("2021-12-23 01:30"),
            T("2021-12-23 13:00"),
            T("2021-12-24 01:30"),
            T("2021-12-27 13:00"),
            T("2021-12-28 01:30"),
            T("2021-12-28 13:00"),
        ]

        closes = [
            T("2021-12-23 08:00"),
            T("2021-12-23 21:00"),
            T("2021-12-24 04:00"),
            T("2021-12-27 21:00"),
            T("2021-12-28 08:00"),
            T("2021-12-28 21:00"),
        ]

        for interval in intervals:
            start, end = T("2021-12-23"), T("2021-12-28")
            rtrn = f(interval, start, end, ignore_breaks=True)
            index = self.get_expected(interval, opens, closes)
            assert_index_equal(rtrn, index)

        # if including breaks
        opens = [
            T("2021-12-23 01:30"),
            T("2021-12-23 05:00"),
            T("2021-12-23 13:00"),
            T("2021-12-24 01:30"),
            T("2021-12-27 13:00"),
            T("2021-12-28 01:30"),
            T("2021-12-28 05:00"),
            T("2021-12-28 13:00"),
        ]

        closes = [
            T("2021-12-23 04:00"),
            T("2021-12-23 08:00"),
            T("2021-12-23 21:00"),
            T("2021-12-24 04:00"),
            T("2021-12-27 21:00"),
            T("2021-12-28 04:00"),
            T("2021-12-28 08:00"),
            T("2021-12-28 21:00"),
        ]

        start, end = T("2021-12-23"), T("2021-12-28")
        ignore_breaks = False
        for interval in intervals:
            args = (interval, start, end, ignore_breaks)
            for curtail in (True, False):
                if not curtail and interval > TDInterval.H1:
                    with pytest.raises(
                        errors.CompositeIndexCalendarConflict, match=match_xhkg
                    ):
                        f(*args, curtail_calendar_overlaps=curtail)
                elif curtail and interval is TDInterval.H10:
                    with pytest.raises(errors.CompositeIndexConflict):
                        f(*args, curtail_calendar_overlaps=curtail)
                else:
                    rtrn = f(*args, curtail_calendar_overlaps=curtail)
                    index = self.get_expected(interval, opens, closes)
                    if curtail and interval > TDInterval.H1:
                        index = pdutils.make_non_overlapping(index)
                    assert_index_equal(rtrn, index)

    def test_minute_methods(self, one_min):
        """Tests methods that query minutes.

        Tests methods:
            `minute_to_trading_minute`
            `next_minute`
            `previous_minute`
            `is_open_on_minute` (also tested for via dedicated method)
        """

        def match(minute: pd.Timestamp) -> str:
            return re.escape(
                f"`minute` '{minute}' is not a trading minute. Consider passing"
                " `direction` as 'next' or 'previous'"
            )

        def assertions(
            cc,
            minute: pd.Timestamp,
            is_open: bool,
            prev_: pd.Timestamp,
            next_: pd.Timestamp,
        ):
            assert cc.next_minute(minute) == next_
            assert cc.previous_minute(minute) == prev_
            assert cc.is_open_on_minute(minute) is is_open
            if is_open:
                assert cc.minute_to_trading_minute(minute) == minute
                assert cc.minute_to_trading_minute(minute, "next") == minute
                assert cc.minute_to_trading_minute(minute, "previous") == minute
            else:
                with pytest.raises(ValueError, match=match(minute)):
                    cc.minute_to_trading_minute(minute)
                assert cc.minute_to_trading_minute(minute, "next") == next_
                assert cc.minute_to_trading_minute(minute, "previous") == prev_

        # verify for composite calendar comprising detached calendars.
        # from knowledge of schedules
        xnys = xcals.get_calendar("XNYS", "2021", "2021-12-31", "left")
        xhkg = xcals.get_calendar("XHKG", "2021", "2021-12-31", "left")
        cc = m.CompositeCalendar([xnys, xhkg])

        session_prev = pd.Timestamp("2021-12-22")
        session = pd.Timestamp("2021-12-23")
        session_next_xhkg = pd.Timestamp("2021-12-24")

        open_xhkg = xhkg.session_open(session)
        break_start = xhkg.session_break_start(session)
        break_end = xhkg.session_break_end(session)
        close_xhkg = xhkg.session_close(session)
        open_next_xhkg = xhkg.session_open(session_next_xhkg)

        close_prev_xnys = xnys.session_close(session_prev)
        open_xnys = xnys.session_open(session)
        close_xnys = xnys.session_close(session)

        # verify prior to xhkg open
        minute = open_xhkg - one_min
        assertions(cc, minute, False, close_prev_xnys - one_min, open_xhkg)

        # verify xhkg open
        minute = open_xhkg
        assertions(cc, minute, True, close_prev_xnys - one_min, minute + one_min)

        # verify after xhkg open
        minute = open_xhkg + one_min
        assertions(cc, minute, True, open_xhkg, minute + one_min)

        # verify prior to break start
        minute = break_start - one_min
        assertions(cc, minute, True, minute - one_min, break_end)

        # verify break start to before break end
        for minute in (break_start, break_end - one_min):
            assertions(cc, minute, False, break_start - one_min, break_end)

        # verify break end
        minute = break_end
        assertions(cc, minute, True, break_start - one_min, minute + one_min)

        # verify from after break end to before before xhkg close
        for minute in (break_end + one_min, close_xhkg - one_min - one_min):
            assertions(cc, minute, True, minute - one_min, minute + one_min)

        # verify prior to xhkg close
        minute = close_xhkg - one_min
        assertions(cc, minute, True, minute - one_min, open_xnys)

        # verify xhkg close, after close, before xnys open
        for minute in (close_xhkg, close_xhkg + one_min, open_xnys - one_min):
            assertions(cc, minute, False, close_xhkg - one_min, open_xnys)

        # verify xnys open
        minute = open_xnys
        assertions(cc, minute, True, close_xhkg - one_min, minute + one_min)

        # verify after xnys open to before before xnys close
        for minute in (open_xnys + one_min, close_xnys - one_min - one_min):
            assertions(cc, minute, True, minute - one_min, minute + one_min)

        # verify xnys close
        minute = close_xnys
        assertions(cc, minute, False, minute - one_min, open_next_xhkg)

        # verify xnys closed 24 and xhkg closed on 27, so...
        minute = xhkg.session_close(session_next_xhkg)
        next_ = pd.Timestamp("2021-12-27 14:30", tz=UTC)
        assertions(cc, minute, False, minute - one_min, next_)

    def test_minute_properties(self):
        """Tests properties that return minutes and minute nanos.

        Tests properties:
            `minutes_nanos`
            `minutes`
            `opens_nanos`
            `closes_nanos`
        """
        xnys = xcals.get_calendar("XNYS", "2021", "2021-12-31", "left")
        xhkg = xcals.get_calendar("XHKG", "2021", "2021-12-31", "left")
        xlon = xcals.get_calendar("XLON", "2021", "2021-12-31", "left")

        cc = m.CompositeCalendar([xnys, xhkg, xlon])

        def assertions(
            cc_nanos: np.ndarray,
            xnys_nanos: np.ndarray,
            xlon_nanos: np.ndarray,
            xhkg_nanos: np.ndarray,
        ):
            assert pd.DatetimeIndex(cc_nanos).is_monotonic_increasing

            # verify all minute of each constituent calendar included
            assert (np.intersect1d(cc_nanos, xnys_nanos) == xnys_nanos).all()
            assert (np.intersect1d(cc_nanos, xlon_nanos) == xlon_nanos).all()
            assert (np.intersect1d(cc_nanos, xhkg_nanos) == xhkg_nanos).all()

            # ...and that no minute is included that is not a minute of a const. cal.
            diff = np.setdiff1d(cc_nanos, xnys_nanos)
            assert len(diff)
            diff = np.setdiff1d(diff, xlon_nanos)
            assert len(diff)
            diff = np.setdiff1d(diff, xhkg_nanos)
            assert not diff.size

        assertions(
            cc.minutes_nanos, xnys.minutes_nanos, xlon.minutes_nanos, xhkg.minutes_nanos
        )
        assertions(
            cc.closes_nanos, xnys.closes_nanos, xlon.closes_nanos, xhkg.closes_nanos
        )
        assertions(cc.opens_nanos, xnys.opens_nanos, xlon.opens_nanos, xhkg.opens_nanos)

        # verify cc.minutes
        assert cc.minutes.tz == UTC
        assert (cc.minutes.tz_convert(None) == pd.DatetimeIndex(cc.minutes_nanos)).all()
