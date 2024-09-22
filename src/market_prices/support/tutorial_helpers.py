"""Helper functions to identify data to use in tutorials or tests."""

import exchange_calendars as xcals
import pandas as pd

from market_prices import intervals, errors
from market_prices.prices.base import PricesBase
from market_prices.utils import calendar_utils as calutils


def get_sessions_range_for_bi(
    prices: PricesBase,
    bi: intervals.BI,
    calendar: xcals.ExchangeCalendar | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get range over which data available only at intervals >= `bi`.

    Range will start/end on first full session following/preceeding the
    corresponding limit.

    Parameters
    ----------
    prices
        Prices instance from which to evaluate limits of data availability.

    bi
        Range will cover sessions for which data is only available at
        intervals > `bi`.

    calendar
        If passed, range will be defined with sessions of `calendar`.
        Otherwise range will be defined with sessions of `prices.cc`.
    """
    # pylint: disable=too-complex,too-many-branches
    start_limit, end_limit = prices.limits[bi]
    if not bi.is_one_minute:
        bi_previous = bi.previous
        assert bi_previous is not None
        prev_left_limit = prices.limits[bi_previous][0]
        assert prev_left_limit is not None
        end_limit = prev_left_limit

    if calendar is not None:
        try:
            start_session = calendar.minute_to_session(start_limit, "none")
        except ValueError:
            start_session = calendar.minute_to_future_session(start_limit, 1)
        else:
            if not calendar.session_open(start_session) == start_limit:
                start_session = calendar.next_session(start_session)

        try:
            end_session = calendar.minute_to_session(end_limit, "none")
        except ValueError:
            end_session = calendar.minute_to_past_session(end_limit, 1)
        else:
            end_session = calendar.previous_session(end_session)

    else:
        cc = prices.cc
        try:
            start_session = cc.minute_to_sessions(start_limit)[-1]
        except IndexError:
            start_session = cc.minute_to_sessions(start_limit, "next")[-1]
        else:
            assert start_limit is not None
            if start_limit > cc.session_open(start_session):
                start_session = cc.next_session(start_session)

        try:
            end_session = cc.minute_to_sessions(end_limit)[0]
        except IndexError:
            end_session = cc.minute_to_sessions(end_limit, "previous")[0]
        else:
            if end_limit < cc.session_close(end_session):
                end_session = cc.previous_session(end_session)

    return start_session, end_session


def _required_session_lengths(
    session: pd.Timestamp,
    calendars: list[xcals.ExchangeCalendar],
    session_lengths: list[pd.Timedelta],
) -> bool:
    """Query if `calendars` have required `session_lengths` for `session`."""
    for calendar, session_length in zip(calendars, session_lengths):
        if not session_length:
            if not calendar.is_session(session):
                continue
            return False
        elif not calendar.is_session(session):
            return False
        open_, close = calendar.session_open_close(session)
        duration = close - open_
        if duration != session_length:
            return False
    return True


def _required_sessions_lengths(
    sessions: pd.DatetimeIndex,
    calendars: list[xcals.ExchangeCalendar],
    sessions_lengths: list[list[pd.Timedelta]],
) -> bool:
    for session, session_lengths in zip(sessions, zip(*sessions_lengths)):
        if not _required_session_lengths(
            session, calendars, session_lengths  # type: ignore[arg-type]  # as req.
        ):
            return False
    return True


def get_conforming_sessions_var(
    calendars: list[xcals.ExchangeCalendar],
    sessions_lengths: list[list[pd.Timedelta]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Get consecutive sessions of varied known lengths.

    Parameters
    ----------
    calendars
        Calendars against which to evaluate sessions_lengths.

    sessions_lengths
        list of same length as `calendars` with each item representing th
        required session_lengths for that calendar.

        Items should be lists of equal length which in turn represents the
        number of required sessions. Each value should represent the
        required session length for the corresponding session.

        pd.Timedelta(0) will be treated as a request that the date should
        be a non-session for the corresponding calendar.

    start
        Earliest first session of returned sessions.

    end
        Latest last session of returned sessions.

    Raises
    ------
    ValueError
        If no run of sessions from `start` through `end` fulfills the
        defined `sessions_lengths`.
    """
    cc = calutils.CompositeCalendar(calendars)
    available_sessions = cc.sessions_in_range(start, end)
    num_available_sessions = len(available_sessions)
    number = len(sessions_lengths[0])
    for sessions_lengths_cal in sessions_lengths:
        # assert same number of sessions_lengths defined for each calendar.
        assert len(sessions_lengths_cal) == number
    for i in range(num_available_sessions):
        stop = i + number
        if stop > num_available_sessions:
            break
        sessions = available_sessions[i : i + number]
        if _required_sessions_lengths(sessions, calendars, sessions_lengths):
            return pd.DatetimeIndex(sessions)
    raise errors.TutorialDataUnavailableError(start, end, calendars, sessions_lengths)


def get_conforming_sessions(
    calendars: list[xcals.ExchangeCalendar],
    session_length: list[pd.Timedelta],
    start: pd.Timestamp,
    end: pd.Timestamp,
    number: int = 1,
) -> pd.DatetimeIndex:
    """Get consecutive sessions of same known length.

    All sessions of each calendar will have same length.

    Parameters
    ----------
    number
        Number of consecutive sessions required.

    calendars
        Calendars against which to evaluate session_length.

    session_length
        list of same length as `calendars` with each item representing the
        required session_length for that calendar.

        pd.Timedelta(0) will be treated as requiring that the calendar is
        not open.

    start
        Earliest first session of returned sessions.

    end
        Latest last session of returned sessions.

    Raises
    ------
    errors.TutorialDataUnavailableError
        If no run of `number` sessions from `start` through `end` fulfill
        the defined `session_length`.
    """
    sessions_lengths = [
        [session_length_cal] * number for session_length_cal in session_length
    ]
    return get_conforming_sessions_var(calendars, sessions_lengths, start, end)


def get_conforming_cc_sessions_var(
    cc: calutils.CompositeCalendar,
    sessions_lengths: list[pd.Timedelta],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Get consecutive composite calendar sessions of known length.

    Required composite session length can vary from one session to the
    next.

    Composite session length considered difference latest calendar close
    and earliest calendar open.

    Parameters
    ----------
    cc
        Composite calendar against which to evaluate composite session
        length.

    sessions_lengths
        List of length equal to the number of required sessions, value as
        required length of corresopnding composite session. All values
        should be higher than pd.Timedelta(0).

    start
        Earliest first session of returned composite sessions.

    end
        Latest last session of returned composite sessions.

    Raises
    ------
    ValueError
        If no run of composite sessions from `start` through `end` fulfills
        the defined `sessions_lengths`.
    """
    available_sessions = cc.sessions_in_range(start, end)
    num_available_sessions = len(available_sessions)
    number = len(sessions_lengths)
    for i in range(num_available_sessions):
        stop = i + number
        if stop > num_available_sessions:
            break
        sessions = available_sessions[i : i + number]
        if cc.sessions_length(sessions[0], sessions[-1]).to_list() == sessions_lengths:
            return sessions
    raise errors.TutorialDataUnavailableError(start, end, cc, sessions_lengths)


def get_conforming_cc_sessions(
    cc: calutils.CompositeCalendar,
    session_length: pd.Timedelta,
    start: pd.Timestamp,
    end: pd.Timestamp,
    number: int = 1,
) -> pd.DatetimeIndex:
    """Get consecutive composite calendar sessions of same known length.

    Composite session length considered difference latest calendar close
    and earliest calendar open.

    Parameters
    ----------
    cc
        Composite calendar against which to evaluate composite session
        length.

    session_length
        Composite session length.

    start
        Earliest first session of returned composite sessions.

    end
        Latest last session of returned composite sessions.

    number
        Number of consecutive sessions requried.

    Raises
    ------
    ValueError
        If there is no run of `number` composite sessions of
        `session_length` from `start` through `end`.
    """
    srs = cc.sessions_length(start, end) == session_length
    if not srs.any():
        raise errors.TutorialDataUnavailableError(start, end, cc, session_length)
    sessions = xcals.utils.pandas_utils.longest_run(srs)
    if len(sessions) < number:
        raise errors.TutorialDataUnavailableError(start, end, cc, session_length)
    return sessions[:number]
