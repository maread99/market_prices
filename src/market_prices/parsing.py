"""Functions to parse public input.

Supplements parsing functionality employed via pydantic.

Covers:
    Verification.
    Conversion.
    Assignment of default values.
"""

from __future__ import annotations

import typing

import exchange_calendars as xcals
import pandas as pd
import pytz

from market_prices import errors, helpers, mptypes, intervals


def verify_period_parameters(pp: mptypes.PP):
    """Raise appropriate error if period parameters are not valid.

    Parameters
    ----------
    pp
        Period parameters to verify.
    """
    minutes, hours, days, weeks, months, years, start, end = (
        pp["minutes"],
        pp["hours"],
        pp["days"],
        pp["weeks"],
        pp["months"],
        pp["years"],
        pp["start"],
        pp["end"],
    )

    duration_values = [minutes, hours, days, weeks, months, years]
    has_duration = bool(sum(duration_values))

    if end is not None and start is not None and has_duration:
        msg = "If pass start and end then cannot pass a duration component."
        raise ValueError(msg)

    if sum([minutes, hours]) > 0:
        if sum([days, weeks, months, years]):
            msg = (
                "`hours` and `minutes` cannot be combined with other duration"
                " components."
            )
            raise ValueError(msg)

    if days > 0:
        if sum([weeks, months, years]):
            msg = "`days` cannot be combined with other duration components."
            raise ValueError(msg)


def parse_timestamp(ts: pd.Timestamp, tzin: pytz.BaseTzInfo) -> pd.Timestamp:
    """Parse timestamp to date or UTC time.

    Parameters
    ----------
    ts
        Timestamp to be parsed

    tzin
        Timezone of `ts` if `ts` is tz-naive and represents a time.
    """
    if helpers.is_date(ts):
        return ts
    if ts.tz is None:
        ts = ts.tz_localize(tzin)  # type: ignore[unreachable]  # 'tis reachable
    return ts.tz_convert(pytz.UTC)


def _parse_start(
    start: pd.Timestamp,
    is_date: bool,
    as_time: bool,
    cal: xcals.ExchangeCalendar,
) -> pd.Timestamp:
    """Parse `start` to a time if `as_time` or date otherwise."""
    if is_date:
        session = cal.date_to_session(start, "next")
        if as_time:
            return cal.session_first_minute(session)
        return session
    elif as_time:
        return cal.minute_to_trading_minute(start, "next")
    # is time required as date
    elif start.value in cal.first_minutes_nanos:
        return cal.minute_to_session(start)
    else:
        return cal.minute_to_future_session(start)


def _mr_minute_left(cal: xcals.ExchangeCalendar, delay: pd.Timedelta) -> pd.Timestamp:
    """Left side of most recent minute."""
    now = helpers.now() - delay
    if cal.is_open_on_minute(now, ignore_breaks=True):
        return now
    else:
        return cal.previous_minute(now)


def _mr_minute_right(cal: xcals.ExchangeCalendar, delay: pd.Timedelta) -> pd.Timestamp:
    """Right side of most recent minute."""
    return _mr_minute_left(cal, delay) + helpers.ONE_MIN


def _mr_session(cal: xcals.ExchangeCalendar, delay: pd.Timedelta) -> pd.Timestamp:
    """Most recent session."""
    return cal.minute_to_session(_mr_minute_left(cal, delay), "none")


def _parse_end(
    end: pd.Timestamp,
    is_date: bool,
    as_time: bool,
    cal: xcals.ExchangeCalendar,
    delay: pd.Timedelta,
) -> pd.Timestamp:
    """Parse `end` to a time if `as_time` or date otherwise."""
    # pylint: disable=too-many-return-statements
    if is_date:
        end = min(end, _mr_session(cal, delay))
        session = cal.date_to_session(end, "previous")
        if as_time:
            return min(cal.session_close(session), _mr_minute_right(cal, delay))
        return session

    else:
        end = min(end, _mr_minute_right(cal, delay))
        if as_time:
            if end.value in cal.closes_nanos:
                return end
            minute = cal.minute_to_trading_minute(end, "previous")
            # advance to close/break_start if end was not a trading minute
            return minute if minute == end else minute + helpers.ONE_MIN
        else:  # is time required as date
            if end.value in cal.closes_nanos:  # pylint: disable=else-if-used
                return cal.minute_to_session(end - helpers.ONE_MIN)
            elif end > _mr_minute_left(cal, delay) - helpers.ONE_SEC:
                # return live session if market open, otherwise prior session
                return cal.minute_to_session(end, "previous")
            else:
                return cal.minute_to_past_session(end)


def _parse_start_end(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    as_times: bool,
    cal: xcals.ExchangeCalendar,
    delay: pd.Timedelta,
    strict: bool,
) -> mptypes.DateRangeAmb:
    """Parse `start` and `end` values received from client.

    Parameters as for `parse_start_end`.

    Returns
    -------
    mptypes.DateRangeAmb
        [0]: None if `start` None, otherwise `start` parsed to a trading
        minute if `as_times` or to a session otherwise.

        [1]: None if `end` None, otherwise `end` parsed to either a trading
        minute or session close if `as_times` or to a session otherwise. If
        `end` is later than 'now' (adjusted for `delay`) then range end
        will be None.
    """
    # pylint: disable=too-complex, too-many-arguments, too-many-locals,
    # pylint: disable=too-many-branches, too-many-statements
    if start is None and end is None:
        return None, None

    # establish if dates or times, if time and has second, round to appropriate minute.
    if start is not None:
        start_is_date = helpers.is_date(start)
        if not start_is_date:
            # do not include any incomplete minute
            start = start.ceil("T")
    if end is not None:
        end_is_date = helpers.is_date(end)
        if not end_is_date:
            # do not include incomplete minute
            end = end.floor("T")
        # if end > now, set to None.
        now_interval = intervals.ONE_DAY if end_is_date else intervals.ONE_MIN
        now = helpers.now(now_interval, "left")
        if end >= now - delay:
            end = None

    # Code order dictated by having to verify that `start`/`end` are within calendar
    # bounds before trying to parse or verify anything that requires a calendar method.

    # Check start isn't later than latest date/time for which prices available.
    # Has side-effect of ensuring `start` is to the left of right calendar bound.
    if start is not None:

        if start_is_date:
            mrs = _mr_session(cal, delay)
            if start > mrs:
                raise errors.StartTooLateError(start, mrs, cal, delay)
        elif not as_times:  # pylint: disable=confusing-consecutive-elif
            # start is time and require as_date
            mrs_start = cal.session_first_minute(_mr_session(cal, delay))
            if start > mrs_start:
                raise errors.StartTooLateError(start, mrs_start, cal, delay)
        else:
            mrm = _mr_minute_left(cal, delay)
            if start > mrm:
                raise errors.StartTooLateError(start, mrm, cal, delay)

    # if `start` or `end` earlier than left calendar bound
    end_parsed: pd.Timestamp | None
    if end is not None:
        if end_is_date:
            bound = cal.first_session
        else:
            if as_times:  # pylint: disable=else-if-used
                bound = cal.first_minute
            else:
                bound = cal.closes[cal.first_session]
        if end < bound:
            raise errors.EndOutOfBoundsError(cal, end)
        # if end to the right of right calendar bound then will be parsed to 'now'.
        end_parsed = _parse_end(end, end_is_date, as_times, cal, delay)
    else:
        end_parsed = None

    start_parsed: pd.Timestamp | None
    if start is not None:
        bound = cal.first_session if start_is_date else cal.first_minute
        if start < bound:
            if strict:
                raise errors.StartOutOfBoundsError(cal, start)
            else:
                start = bound
        start_parsed = _parse_start(start, start_is_date, as_times, cal)
    else:
        start_parsed = None

    # all further checks require both start and end, so can return if either is None.
    if start is None or end is None:
        return start_parsed, end_parsed

    # As at here, only unknown is if `end`` is to left of right calendar bound.

    # Verify `start`` not > `end`
    # NB comparisons are made using non-parsed values. Not comparing parsed values
    # avoids raising error where `start` is < `end` (as received) although the calendar
    # is not open between these values (a circumstance better handled by the later
    # 'check calendar open' verification). For example, if `start` and `end` are dates
    # within a run of non-sessions then they will parse to the closest session
    # after/before the run respectively, hence checking against parsed values would
    # consider start > end, although the situation is better described as there being
    # no sessions between `start` and `end` as received.
    if start_is_date == end_is_date:  # both dates or both times
        if start > end or (start == end and not start_is_date):
            raise errors.StartNotEarlierThanEnd(start, end)

    elif start_is_date:  # pylint: disable=confusing-consecutive-elif
        # start is date, end is time
        if not as_times:  # as_dates
            session_start = cal.session_first_minute(start_parsed)
            if session_start >= end:
                raise errors.StartNotEarlierThanEnd(start, end)
        else:  # as_times
            if cal.is_session(start):
                _start_ = cal.session_first_minute(start)
            else:
                prev_session = cal.date_to_session(start, "previous")
                _start_ = cal.session_close(prev_session)
            if _start_ >= end:
                raise errors.StartNotEarlierThanEnd(start, end)

    else:  # pylint: disable=else-if-used
        # start is time, end is date
        if as_times:
            if cal.is_session(end):
                _end_ = cal.session_close(end)
            else:
                next_session = cal.date_to_session(end, "next")
                _end_ = cal.session_first_minute(next_session) - helpers.ONE_MIN
            if start >= _end_:
                raise errors.StartNotEarlierThanEnd(start, end)
        else:  # as_dates
            session_end = cal.session_close(end_parsed)
            if start >= session_end:
                raise errors.StartNotEarlierThanEnd(start, end)

    # Check calendar open between start and end.
    # Note: verified after start > end check as this check would otherwise
    # raise PricesDateRangeEmpty if start > end (i.e. a circumstance better
    # handled by StartNotEarlierThanEnd).
    s, e = start, end
    if as_times:
        if end_is_date:
            end = min(end, cal.last_session)
            session = cal.date_to_session(end, "next")
            e = (
                cal.session_close(session)
                if end == session
                else cal.session_close(cal.previous_session(session))
            )
        if start_is_date:
            session = cal.date_to_session(start, "previous")
            s = (
                cal.session_first_minute(session)
                if start == session
                else cal.session_close(session)
            )
        num_mins = cal.minutes_distance(s, e)
        # if len 1 then not valid if s is a close or a non-trading minute
        if not num_mins or (
            num_mins == 1
            and (s.value not in cal.minutes_nanos or s.value in cal.closes_nanos)
        ):
            raise errors.PricesDateRangeEmpty(start, end, True, cal)

    else:
        if not start_is_date:
            s = cal.minute_to_session(start, "previous")
            if start.value not in cal.first_minutes_nanos:
                s += helpers.ONE_DAY
        if not end_is_date:
            end = min(end, cal.last_minute)
            if typing.TYPE_CHECKING:
                assert end is not None
            e = cal.minute_to_session(end, "next")
            if end.value not in cal.closes_nanos:
                e -= helpers.ONE_DAY
        if cal.sessions_in_range(s, e).empty:
            raise errors.PricesDateRangeEmpty(start, end, False, cal)

    return start_parsed, end_parsed


def parse_start_end(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    as_times: bool,
    calendar: xcals.ExchangeCalendar,
    delay: pd.Timedelta,
    strict: bool,
    gregorian: bool,
) -> mptypes.DateRangeAmb:
    """Parse `start` and `end` values received from client.

    Parameters
    ----------
    start
        Formal parameter assigned to start parameter of
        `market_prices.prices.PricesBase.get` and subsequently parsed by
        `parse_timestamp`.

    end
        Formal parameter assigned to end parameter of
        `market_prices.prices.PricesBase.get` and subsequently parsed by
        `parse_timestamp`.

    as_times
        True: Return parsed timestamps as trading minutes (as opposed to
            sessions).
        False: Return parsed timestamps as sessions (as opposed to trading
            times).

    calendar
        Calendar against which to evaluate start and end.

    delay
        Price delay associated with lead_symbol.

    strict
        Determines behaviour if `start` is earlier that the left calendar
        bound.
            True: raise `errors.StartOutOfBoundsError`
            False: return start as left calendar bound.

    gregorian
        True: Evaluate start/end in terms of the Gregorian calendar
        False: Evaluate start/end in terms of the trading `calendar`.

    Returns
    -------
    mptypes.DateRangeAmb
        [0]: None if `start` None, otherwise `start` parsed to a trading
        minute if `as_times` or to a session or date otherwise.

        [1]: None if `end` None, otherwise `end` parsed to either a trading
        minute or session close if `as_times` or to a session or date
        otherwise. If `end` is later than 'now' (adjusted for `delay`) then
        range end will be None.
    """
    # pylint: disable=too-many-arguments
    start_, end_ = _parse_start_end(start, end, as_times, calendar, delay, strict)
    if not gregorian:
        return start_, end_

    # If `gregorian` then start/end should be parsed in terms of gregorian calendar
    # dates/times (rather than in terms of the trading `calendar`, as evaluated above).
    # Executes above first in order to raise any errors found there, although this is a
    # bit of a hack that may reult in, at least, a few incoherent errors. There should
    # really be independent parsing for gregorian / trading calendar terms, with any
    # common parsing code factored out to a common base.
    assert not as_times
    if end is not None and end_ is None:  # `end` is later than None
        end = None
    # if times then roll to dates
    if start is not None and not helpers.is_date(start):
        start = (start.normalize() + helpers.ONE_DAY).tz_convert(None)
    if end is not None and not helpers.is_date(end):
        end = (end.normalize()).tz_convert(None)
    return start, end


def verify_date_not_oob(
    date: pd.Timestamp,
    l_bound: pd.Timestamp,
    r_bound: pd.Timestamp,
    param_name: str = "date",
) -> None:
    """Verify client input to describe a date is within bounds.

    NB: Does NOT verify that `date` represents a date (as opposed to a
    time). To verify that input represents a date annotate the parameter of
    the public method with mptypes.DateTimestamp.

    Parameters
    ----------
    date
        Timestamp to be verified. Should represent a date.

    l_bound
        Left bound. Raises errors.DatetimeTooEarly if `date` is earlier
        than bound.

    r_bound
        Right bound. Raises errors.DatetimeTooLate if `date` is later than
        bound.

    param_name
        Name of parameter being verfied. Included in any error message.
    """
    if date < l_bound:
        raise errors.DatetimeTooEarlyError(date, l_bound, param_name)
    if date > r_bound:
        raise errors.DatetimeTooLateError(date, r_bound, param_name)


def verify_time_not_oob(
    time: pd.Timestamp, l_limit: pd.Timestamp, r_limit: pd.Timestamp
) -> None:
    """Verify client input to describe a time is within limits.

    NB: Does NOT verify that `time` represents a time (as opposed to a
    date). To verify that input represents a time annotate the parameter of
    the public method with mptypes.TimeTimestamp.

    Parameters
    ----------
    time
        Timestamp to verify

    l_limit
        Left limit. Earliest valid timestamp.

    r_limit
        Right limit. Latest valid timestamp.
    """
    if time < l_limit:
        raise errors.DatetimeTooEarlyError(time, l_limit, "time")
    if time > r_limit:
        raise errors.DatetimeTooLateError(time, r_limit, "time")
