"""Hypothesis strategies.

All strategies return parameters that are within, or evaluate to a
session or minute within, the following calendar bounds:
    start: as calendar default start
    end: as defined by `_calendar_end` which coincides with conftest.today

Accordingly these strategies are NOT suitable for testing for errors raised
when a parameter is, or evaluates to, a session or minute that lies outside
of these bounds.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING
import copy

import exchange_calendars as xcals
import pandas as pd
import pytz
from hypothesis import assume
from hypothesis import strategies as st

from market_prices import helpers
from market_prices.intervals import DOInterval, TDInterval

from . import conftest

# NOTE typing. Functions returning strategies are typed to return the SearchStrategy
# object returned by the decorated function (as opposed to the undecorated return).
# The resulting [return-value] mypy errors can be ignored so long as the "got" type
# matches the type expected to be contained in the SearchStartegy.

# pylint: disable=missing-param-doc


def noneify(value: typing.Any) -> st.SearchStrategy:
    """Return strategy to return either given value or None."""
    return st.sampled_from([value, None])


_calendar_cache: dict[str, xcals.ExchangeCalendar] = {}
_calendar_end = pd.Timestamp("2021-11-17")
_24h_calendars = ["CMES", "24/7"]


def _add_to_cache(calendar_name: str) -> xcals.ExchangeCalendar:
    # pylint: disable=protected-access, invalid-name
    CalendarCls = xcals.calendar_utils._default_calendar_factories[calendar_name]
    calendar = CalendarCls(end=_calendar_end, side="left")
    _calendar_cache[calendar_name] = calendar
    return calendar


def _get_from_cache(calendar_name) -> xcals.ExchangeCalendar:
    cached = _calendar_cache.get(calendar_name, False)
    if not cached:
        cached = _add_to_cache(calendar_name)
    return cached


def get_calendar(calendar_name: str) -> xcals.ExchangeCalendar:
    """Get a calendar."""
    return _get_from_cache(calendar_name)


@st.composite
def calendar_session(
    draw,
    calendar_name: str,
    limit: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None),
) -> st.SearchStrategy[pd.Timestamp]:
    """Return strategy to generate a session for a given calendar.

    Parameters
    ----------
    limit
        Define limits of returned session. Pass as tuple defining
        (start_limit, end_limit), and defining a limit as None to use the
        default.
    """
    calendar = get_calendar(calendar_name)
    sessions = calendar.sessions
    l_limit = limit[0] if limit[0] is not None else sessions[0]
    r_limit = limit[1] if limit[1] is not None else sessions[-1]
    slc = sessions.slice_indexer(l_limit, r_limit)
    session = draw(
        st.sampled_from(calendar.sessions[slc].values)
    )  # can't sample from a dti
    return pd.Timestamp(session)


@st.composite
def calendar_start_end_sessions(
    draw,
    calendar_name: str,
    limit: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None),
    min_dist: int | pd.Timedelta = 0,
    max_dist: int | pd.Timedelta | None = None,
) -> st.SearchStrategy[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return strategy to generate a start and end session for a given calendar.

    Parameters
    ----------
    limit
        Define limits of returned session. Pass as tuple defining
        (start_limit, end_limit), and defining a limit as None to use the
        default.

    min_dist : default: 0
        Minimum distance between start and end sessions. Pass as 'int' to
        define as a number of sessions, or pd.Timedelta to define as a
        delta. Default 0 such that start session can be same as end
        session.

    max_dist : default: length of calendar sessions
        Maximum distance between start and end sessions. Pass as 'int' to
        define as a number of sessions, or pd.Timedelta to define as a
        delta.
    """
    calendar = get_calendar(calendar_name)
    sessions = calendar.sessions
    l_limit = limit[0] if limit[0] is not None else sessions[0]
    r_limit = r_limit_ = limit[1] if limit[1] is not None else sessions[-1]

    if min_dist:
        if isinstance(min_dist, int):
            i = sessions.get_indexer([r_limit], method="ffill")[0]
            r_limit_ = sessions[i - min_dist]
        else:
            r_limit_ -= min_dist

    s = draw(calendar_session(calendar.name, limit=(l_limit, r_limit_)))

    l_limit = l_limit_ = s
    if min_dist:
        if isinstance(min_dist, int):
            i = sessions.get_loc(l_limit)
            l_limit_ = sessions[i + min_dist]
        else:
            l_limit_ += min_dist

    if max_dist is not None:
        if isinstance(max_dist, int):
            i = sessions.get_loc(l_limit)
            _r_limit_ = sessions[i + max_dist]
        else:
            _r_limit_ = l_limit + max_dist
        r_limit = min(_r_limit_, r_limit)

    assert l_limit_ <= r_limit, (l_limit_, r_limit)

    e = draw(calendar_session(calendar_name, limit=(l_limit_, r_limit)))
    return s, e


def nano_to_min(nano: int) -> pd.Timestamp:
    """Convert a 'nano' to a utc pd.Timestamp."""
    return pd.Timestamp(nano).tz_localize(pytz.UTC)


@st.composite
def start_minutes(
    draw,
    calendar_name: str,
    limit: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None),
) -> st.SearchStrategy[pd.Timestamp]:
    """Return strategy to generate a 'start' minute for a given calendar.

    Minute will represent a trading minute (not a close).

    Parameters
    ----------
    limit
        Define limits of returned minute. Pass as tuple defining
        (left_limit, right_limit), and defining a limit as None to use the
        defaults.

        Default right limit:
            last trading minute of second-to-last calendar session.
        Default left limit:
            24h calendars: 6 months earlier than right limit, or first
                calendar minute if earlier.
            others: 2 years earlier than right limit, or first calendar
                minute if earlier.
    """
    calendar = get_calendar(calendar_name)

    l_limit, r_limit = limit

    if r_limit is None:
        r_limit = calendar.last_minutes[-2]

    if l_limit is None:
        if calendar_name in _24h_calendars:
            offset = pd.DateOffset(months=6)
        else:
            offset = pd.DateOffset(years=2)
        # typing - can subtract a DateOffset from a Timestamp
        l_limit = max(r_limit - offset, calendar.first_minute)  # type: ignore[operator]

    nanos = calendar.minutes_nanos

    start = None if l_limit is None else nanos.searchsorted(l_limit.value, side="left")
    stop = None if r_limit is None else nanos.searchsorted(r_limit.value, side="right")
    nano = draw(st.sampled_from(nanos[start:stop]))
    return nano_to_min(nano)


@st.composite
def end_minutes(
    draw,
    calendar_name: str,
    limit: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None),
) -> st.SearchStrategy[pd.Timestamp]:
    """Return strategy to generate an 'end' minute for a given calendar.

    Minute will represent a trading minute, excluding open minutes, or a
    close.

    Parameters
    ----------
    limit
        Define limits of returned minute. Pass as tuple defining
        (start_limit, end_limit), and defining a limit as None to use the
        defaults.

        Default right limit:
            close of second-to-last calendar close.
        Default left limit:
            24h calendars: 6 months earlier than right limit, or first
                calendar close if earlier.
            others: 2 years earlier than right limit, or first calendar
                close if earlier.
    """
    # pylint: disable=too-many-locals
    calendar = get_calendar(calendar_name)

    l_limit, r_limit = limit

    if r_limit is None:
        r_limit = calendar.closes[-2]

    if l_limit is None:
        if calendar_name in _24h_calendars:
            offset = pd.DateOffset(months=6)
        else:
            offset = pd.DateOffset(years=2)
        last_close = calendar.closes[0]
        alt_limit = r_limit - offset  # type: ignore[operator]  # is a valid operation
        l_limit = max(last_close, alt_limit)

    start = calendar.closes_nanos.searchsorted(l_limit.value, side="left")
    stop = calendar.closes_nanos.searchsorted(r_limit.value, side="right")

    opens = calendar.opens_nanos[start:stop]
    break_starts = calendar.break_starts_nanos[start:stop]
    break_ends = calendar.break_ends_nanos[start:stop]
    closes = calendar.closes_nanos[start:stop]
    nanos = xcals.calendar_helpers.compute_minutes(
        opens, break_starts, break_ends, closes, side="right"
    )
    nano = draw(st.sampled_from(nanos))
    return nano_to_min(nano)


@st.composite
def calendar_start_end_minutes(
    draw,
    calendar_name: str,
) -> st.SearchStrategy[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return strategy to generate a start and end minute for a given calendar.

    'start' will be a trading minute.
    'end' will be either a trading minute or close minute.
    """
    s = draw(start_minutes(calendar_name))
    e = draw(end_minutes(calendar_name, limit=(s + helpers.ONE_MIN, None)))
    return s, e


_pp_default = {
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


def get_pp_default() -> dict[str, typing.Any]:
    """Get copy of dictionary expressing deafult period parameters."""
    return copy.copy(_pp_default)


@st.composite
def pp_start_end_sessions(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate period parameters with 'start' and 'end' only.

    'start' and 'end' will both be sessions of `calendar`.

    All other parameters will be as default values.
    """
    pp = get_pp_default()
    pp["start"], pp["end"] = draw(calendar_start_end_sessions(calendar_name))
    return pp


@st.composite
def pp_end_minute_only(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate period parameters with 'end' only.

    'end' will be a trading minute or a close minute of `calendar_name`.

    All other period parameters will be as default values.
    """
    pp = get_pp_default()
    pp["end"] = draw(end_minutes(calendar_name))
    return pp


@st.composite
def pp_start_end_minutes(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate period parameters with 'start' and 'end' only.

    'start' will be a trading minute, 'end' will be a trading minute or a
    close minute of `calendar_name`.

    All other period parameters will be as default values.
    """
    pp = get_pp_default()
    strtgy = calendar_start_end_minutes(calendar_name)
    pp["start"], pp["end"] = draw(strtgy)
    return pp


@st.composite
def pp_days(draw, calendar_name: str) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'days'.
        all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = get_pp_default()
    calendar = get_calendar(calendar_name)
    pp["days"] = draw(st.integers(1, len(calendar.sessions) - 1))
    return pp


@st.composite
def pp_days_start_session(
    draw,
    calendar_name: str,
    start_will_roll_to_ms: bool = False,
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'days'.
        'start' defined as a session.
        all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.

    Parameters
    ----------
    start_will_roll_to_ms
        Pass true if start will be rolled forward to the next month start,
        i.e. if ds_interval is monthly.
    """
    pp = draw(pp_days(calendar_name))
    calendar = get_calendar(calendar_name)
    sessions = calendar.sessions
    limit_r = sessions[-pp["days"]]
    if start_will_roll_to_ms:
        offset = pd.tseries.frequencies.to_offset("M")
        if TYPE_CHECKING:
            assert offset is not None
        limit_r = offset.rollback(limit_r)
    cal_l_limit = calendar.first_session
    assume(limit_r > cal_l_limit)
    limit = (None, limit_r)
    start = draw(calendar_session(calendar_name, limit))
    pp["start"] = start
    return pp


@st.composite
def pp_days_end_session(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'days'.
        'end' defined as a session.
        all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_days(calendar_name))
    calendar = get_calendar(calendar_name)
    sessions = calendar.sessions
    l_limit = sessions[pp["days"] - 1]
    # account for DOInterval
    l_limit = pd.offsets.MonthBegin().rollforward(l_limit + helpers.ONE_DAY)
    cal_r_limit = calendar.last_session
    assume(l_limit < cal_r_limit)
    limit = (l_limit, None)
    end = draw(calendar_session(calendar_name, limit))
    pp["end"] = end
    return pp


@st.composite
def pp_days_start_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'days'.
        'start' defined as a start minute.
        all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = get_pp_default()
    calendar = get_calendar(calendar_name)
    start = draw(start_minutes(calendar_name, (None, calendar.last_minutes[-3])))
    start_session_i = calendar.sessions.get_loc(calendar.minute_to_session(start))
    max_days = len(calendar.sessions) - 2 - start_session_i
    pp["days"] = draw(st.integers(1, max_days))
    pp["start"] = start
    return pp


@st.composite
def pp_days_end_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'days'.
        'end' defined as an end minute.
        all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = get_pp_default()
    pp["days"] = draw(st.integers(1, 1000))
    end = draw(end_minutes(calendar_name))
    pp["end"] = end
    return pp


@st.composite
def pp_caldur(draw) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        duration defined in 'weeks' and/or 'months' and/or 'years' (i.e.
        calendar terms) all other parameters take default values.

    The daterange corresponding with period parameters will hvae a duration
    less than 2 years.
    """
    pp = get_pp_default()

    pp["years"] = years = draw(st.integers(0, 1))
    pp["months"] = months = draw(st.integers(0, 6))
    pp["weeks"] = weeks = draw(st.integers(0, 25))
    assume(sum([years, months, weeks]))
    return pp


@st.composite
def pp_caldur_start_session(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'weeks' and/or 'months' and/or 'years' (i.e.
            calendar terms).
        - 'start' defined as a session.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_caldur())
    calendar = get_calendar(calendar_name)
    duration = pd.DateOffset(
        weeks=pp["weeks"],
        months=pp["months"],
        years=pp["years"],
    )
    last_session = calendar.last_session
    limit = (None, last_session - duration)  # type: ignore[operator]  # is valid op.
    start = draw(calendar_session(calendar_name, limit))
    # See `pp_caldur_end_session` for note on need for this assume guard
    assume(start + duration <= last_session)
    pp["start"] = start
    return pp


@st.composite
def pp_caldur_end_session(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'weeks' and/or 'months' and/or 'years' (i.e.
            calendar terms)
        - 'end' defined as a session.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_caldur())
    calendar = get_calendar(calendar_name)
    duration = pd.DateOffset(
        days=1,
        weeks=pp["weeks"],
        months=pp["months"],
        years=pp["years"],
    )
    first_session = calendar.first_session
    l_limit = first_session + duration  # type: ignore[operator]  # is valid operation.
    # account for DOInterval
    l_limit = pd.offsets.MonthBegin().rollforward(l_limit + helpers.ONE_DAY)
    limit = (l_limit, None)
    end = draw(calendar_session(calendar_name, limit))
    # Following guard is necessary as first_session + duration - duration != duration,
    # or at least might not be (pd.DateOffset arithmetic operates on components
    # from largest to smallest, notwithstanding type of operation. So if adding
    # pd.DateOffset(months=1, weeks=4) to a Timestamp will add 1 month then 4 weeks,
    # although if subtracting will take away 1 month, then 4 weeks, not the same.)
    assume(end - duration >= first_session)
    pp["end"] = end
    return pp


@st.composite
def pp_caldur_start_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'weeks' and/or 'months' and/or 'years' (i.e.
            calendar terms)
        - 'start' defined as a trading minute.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_caldur())
    calendar = get_calendar(calendar_name)

    duration = pd.DateOffset(
        weeks=pp["weeks"],
        months=pp["months"],
        years=pp["years"],
    )
    limit = (None, calendar.last_minutes[-2] - duration)
    start = draw(start_minutes(calendar_name, limit))
    # See `pp_caldur_end_session` for note on need for this assume guard
    assume(start + duration <= calendar.last_minutes[-2])
    pp["start"] = start
    return pp


@st.composite
def pp_caldur_end_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'weeks' and/or 'months' and/or 'years' (i.e.
            calendar terms)
        - 'end' defined as a trading or close minute.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_caldur())
    # no need for limit as by default end_minutes will be limited to most recent two
    # years, such that resulting daterange will start later than left calendar bound.
    pp["end"] = draw(end_minutes(calendar_name))
    return pp


@st.composite
def pp_intraday(draw) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'minutes' and/or 'hours' (i.e. in trading
            terms)
        - all other parameters take default values.

    The daterange corresponding with period parameters will have a
    duration.
    """
    pp = get_pp_default()
    pp["hours"] = hours = draw(st.integers(0, 72))
    pp["minutes"] = minutes = draw(st.integers(0, 59))
    assume(sum([hours, minutes]))
    return pp


@st.composite
def pp_intraday_start_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'minutes' and/or 'hours' (i.e. in trading
            terms)
        - 'start' defined as a trading minute.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_intraday())
    calendar = get_calendar(calendar_name)
    i = calendar.minutes.get_loc(calendar.last_minutes[-2])
    i -= pp["minutes"] + (pp["hours"] * 60)
    limit = (None, calendar.minutes[i])
    pp["start"] = draw(start_minutes(calendar_name, limit))
    return pp


@st.composite
def pp_intraday_end_minute(
    draw, calendar_name: str
) -> st.SearchStrategy[dict[str, typing.Any]]:
    """Return strategy to generate specific arrangment of period parameters.

        - duration defined in 'minutes' and/or 'hours' (i.e. in trading
            terms)
        - 'end' defined as a trading minute.
        - all other parameters take default values.

    The daterange corresponding with period parameters will fall within
    `calendar` bounds.
    """
    pp = draw(pp_intraday())
    # no need for limit as by default end_minutes will be limited to most recent two
    # years, such that resulting daterange will start later than left calendar bound.
    pp["end"] = draw(end_minutes(calendar_name))
    return pp


def intervals_non_intraday() -> st.SearchStrategy[TDInterval | DOInterval]:
    """Return strategy to generate non-intraday interval.

    Strategy returns an interval of TDInterval that is 1D or higher or an
    interval of DOInterval.
    """
    intervals_list = TDInterval.daily_intervals() + list(DOInterval)
    return st.sampled_from(intervals_list)


def intervals_intraday() -> st.SearchStrategy[TDInterval]:
    """Return strategy to generate intraday interval."""
    intervals_list = TDInterval.intraday_intervals()
    return st.sampled_from(intervals_list)


def base_intervals() -> st.SearchStrategy[TDInterval]:
    """Return strategy to generate a sample base interval."""
    return st.sampled_from(conftest.base_intervals_sample)


def base_ds_intervals() -> (
    st.SearchStrategy[tuple[TDInterval, TDInterval | None, TDInterval]]
):
    """Return strategy for a sample base interval and valid ds_intervals."""
    return st.sampled_from(conftest.base_ds_intervals_list)
