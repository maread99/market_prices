"""Utility functions and classes for exchange calendars.

Of note:
    `CompositeCalendar`. Create a composite calendar reflecting the open
    times of multiple underlying calendars.
"""

import functools
from collections import abc
from typing import TYPE_CHECKING, Literal

import exchange_calendars as xcals
from exchange_calendars.calendar_helpers import Date, Minute, Session, TradingMinute
import numpy as np
import pandas as pd
from valimp import parse

from market_prices import helpers, intervals, errors
from market_prices.helpers import UTC
from market_prices.utils import pandas_utils as pdutils


def get_exchange_info() -> pd.DataFrame:
    """Retrieve information on exchanges for which calendars available.

    Returns
    -------
    DataFrame
        Information on exchanges for which calendars are available.
        Calendar codes given by column 'ISO Code'.
    """
    # 25/01/14 - for whatever reason seems that tables cannot currently be fetched
    # from the prior address "https://pypi.org/project/exchange-calendars/".
    return pd.read_html("https://github.com/gerrymanoim/exchange_calendars")[2]


def minutes_in_period(
    calendar: xcals.ExchangeCalendar, start: TradingMinute, end: Minute
) -> int:
    """Return number of minutes in a period.

    Parameters
    ----------
    calendar
        Calanedar against which to evaluate minute in period.

    start
        Start of period. Must be a trading minute.

    end
        End of period. Must be a trading minute or a session close.

    Returns
    -------
    int
        Number of minutes in period.
    """
    if end < start:
        raise ValueError(
            f"`end` cannot be earlier than `start`. Received `start` as {start}"
            f" and `end` as {end}."
        )
    distance = calendar.minutes_distance(start, end, _parse=False)
    if end.value in calendar.minutes_nanos:
        # end represents the right of the prior minute, not a minute in its own right.
        distance -= 1
    return distance


class NoBreakError(ValueError):
    """A session assumed as having a break does not have a break.

    Parameters
    ----------
    calendar
        Calendar for which `session` assumed as having a break.

    session
        Session assumed as having a break.
    """

    def __init__(self, calendar: xcals.ExchangeCalendar, session: pd.Timestamp):
        # pylint: disable=super-init-not-called
        self.calendar = calendar
        self.session = session

    def __str__(self) -> str:
        msg = (
            f"Session '{self.session}' of calendar '{self.calendar.name}'"
            " does not have a break."
        )
        return msg


@parse
def subsession_length(
    calendar: xcals.ExchangeCalendar,
    session: Session,
    to_break: bool = False,
    strict: bool = False,
) -> pd.Timedelta:
    """Duration of a subsession.

    Parameters
    ----------
    calendar
        Calendar against which to evaluate `session`.

    session
        Session of the subsession that require duration of.

    to_break : default: False
        True: get duration of subsession from open to break-start.
        False: get duration of subsession from break-end to close.

    strict : default: False
        Defines behaviour if session does not have a break.

        If True, raise NoBreakError.

        If False, raise NoBreakError if `to_break` is True, otherwise
        return full session length.

    Raises
    ------
    market_prices.utils.calendar_utils.NotSession
        If `session` is not a session of `calendar`.

    market_prices.utils.calendar_utils.NoBreak
        If `session` does not have a break and either `strict` or
        `to_break` is True.
    """
    # pylint: disable=missing-param-doc
    xcals.calendar_helpers.parse_session(calendar, session)
    if calendar.session_has_break(session):
        if to_break:
            open_ = calendar.session_open(session)
            close_ = calendar.session_break_start(session)
        else:
            open_ = calendar.session_break_end(session)
            close_ = calendar.session_close(session)
    elif to_break or strict:
        raise NoBreakError(calendar, session)
    else:
        open_, close_ = calendar.session_open_close(session)
    return close_ - open_


def get_trading_index(
    calendar: xcals.ExchangeCalendar,
    interval: intervals.BI | intervals.TDInterval,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force: bool = False,
    ignore_breaks: bool = False,
    curtail_overlaps: bool = False,
) -> pd.IntervalIndex | pd.DatetimeIndex:
    """Return trading index for a calendar.

    Convenience function that returns `calendar.trading_index` with
    'intervals' and 'closed' options as required for market_prices.

    Parameters
    ----------
    calendar
        Calendar against which to evaluate trading index.

    interval
        Time delta between each indice. Cannot be higher than one day.

    All other parameters as calendar.trading_index.
    """
    # pylint: disable=missing-param-doc, too-many-arguments
    return calendar.trading_index(
        start,
        end,
        interval,
        intervals=True,
        closed="left",
        force=force,
        ignore_breaks=ignore_breaks,
        curtail_overlaps=curtail_overlaps,
    )


def composite_schedule(calendars: abc.Sequence[xcals.ExchangeCalendar]) -> pd.DataFrame:
    """Composite Open/Close schedule for multiple calendars.

    Note:
        Schedule ignores breaks within a session of any specific calendar.
        Schedule ignores any gaps between same-day sessions of different
        calendars. For example, if composite comprises two calendars, one
        closes at 08:00 UTC and the other does not open until 09:00 UTC,
        then the composite exchange will be considered open from 08:00
        through 09:00.

    Parameters
    ----------
    calendars
        Calendars that composite schedule is to be composed from.
    """
    first_session = max([c.first_session for c in calendars])
    last_session = min([c.last_session for c in calendars])

    schedules = [c.schedule[first_session:last_session] for c in calendars]
    index = pdutils.index_union(schedules)
    schedules = [sch.reindex(index) for sch in schedules]
    columns = pd.Index(["open", "close"])
    opens = pd.DataFrame([sch[columns[0]] for sch in schedules]).min()
    closes = pd.DataFrame([sch[columns[1]] for sch in schedules]).max()
    schedule = pd.concat([opens, closes], axis=1)
    schedule.columns = columns
    return schedule


class CompositeCalendar:
    """Calendar emulator evaluted from multiple calendars.

    Parameters
    ----------
    calendars
        Calendars from which to compose composite calendar. Composite
        schedule will include all sessions from the latest first session
        of any calendar to the earliest last session.
        Note: Calendars must all have same 'side' value.

    Notes
    -----
    Offers only selected properties / methods of ExchangeCalendar.

    Unless otherwise stated in the method/proprerty documentation, methods
    and properties can be assumed to treat as a non-trading minute any
    minute that is not a trading minute of any underlying calendar. These
    can include break minutes and minutes that fall in any 'intra-session'
    gap between the close of one underlying calendar and the open of the
    next.
    """

    # pylint: disable=too-many-public-methods

    LEFT_SIDES = ["left", "both"]
    RIGHT_SIDES = ["right", "both"]
    ONE_MIN = pd.Timedelta(1, "min")

    def __init__(self, calendars: set | abc.Sequence[xcals.ExchangeCalendar]):
        self._cals = calendars
        sides = {cal.side for cal in calendars}
        if len(sides) != 1:
            raise ValueError("All `calendars` must have the same side.")
        self._side = sides.pop()

    # NB method called by errors raised by parsing methods
    @property
    def name(self) -> str:
        """Composite Calendar name."""
        return f"CS {[cal.name for cal in self.calendars]}"

    @property
    def calendars(self) -> list[xcals.ExchangeCalendar]:
        """Calendars (unique) that comprise composite calendar."""
        return list(self._cals)

    @property
    def side(self) -> Literal["left", "right", "both", "neither"]:
        """Composite Calendar side."""
        return self._side

    @property
    def _has_right_side(self) -> bool:
        return self.side in self.RIGHT_SIDES

    @property
    def _has_left_side(self) -> bool:
        return self.side in self.LEFT_SIDES

    # called by xcals.calendar_helpers.parse_timestamp
    def _minute_oob(self, minute: Minute) -> bool:
        """Is `minute` out-of-bounds."""
        return minute < self.first_minute or minute > self.last_minute

    # called by xcals.calendar_helpers.parse_date
    def _date_oob(self, date: Date) -> bool:
        """Is `date` out-of-bounds."""
        return date < self.first_session or date > self.last_session

    @functools.cached_property
    def schedule(self) -> pd.DataFrame:
        """Composite schedule.

        Only composite session open and close are given. The schedule does
        not include any breaks or intra-session gaps between underlying
        calendars.
        """
        return composite_schedule(self.calendars)

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """Sessions that comprise composite schedule."""
        return self.schedule.index

    @functools.cached_property
    def _sessions_nanos(self) -> np.ndarray:
        return self.sessions.asi8

    @staticmethod
    def _combine_nanos(nanos: list[np.ndarray]) -> np.ndarray:
        """Combine sorted nanos to as single array of unique values."""
        # https://stackoverflow.com/questions/12427146/combine-two-arrays-and-sort/12427633#12427633
        arr = np.concatenate(nanos)
        arr.sort(kind="stable")
        flag = np.ones(len(arr), dtype=bool)
        np.not_equal(arr[1:], arr[:-1], out=flag[1:])
        return arr[flag]

    @functools.cached_property
    def minutes_nanos(self) -> np.ndarray:
        """Nano version of `minutes`."""
        # May 22. Investigated creating the nanos for opens, closes and break bounds,
        # similar to xcals.calendar_helpers.compute_minutes. Was a bit (say 10%) slower
        # than combining as below.
        return self._combine_nanos([cal.minutes_nanos for cal in self.calendars])

    @functools.cached_property
    def opens_nanos(self) -> np.ndarray:
        """Open times of every underlying calendar, as nanos.

        NOTE: These are NOT the open times of only the composite session,
        but rather the open times of every underlying calendar.
        """
        return self._combine_nanos([cal.opens_nanos for cal in self.calendars])

    @functools.cached_property
    def closes_nanos(self) -> np.ndarray:
        """Close times of every underlying calendar, as nanos.

        NOTE: These are NOT the close times of only the composite session
        closes, but rather the close times of every underlying calendar.
        """
        return self._combine_nanos([cal.closes_nanos for cal in self.calendars])

    @property
    def opens(self) -> pd.Series:
        """Open time for each composite session."""
        return self.schedule["open"]

    @property
    def closes(self) -> pd.Series:
        """Close time for each composite session."""
        return self.schedule["close"]

    @functools.cached_property
    def minutes(self) -> pd.DatetimeIndex:
        """All trading minutes.

        Excludes minutes when no underlying calendar is open.
        """
        return pd.DatetimeIndex(self.minutes_nanos, tz=UTC)

    @property
    def first_minutes(self) -> pd.Series:
        """First minute of each composite session."""
        return self.opens if self._has_left_side else self.opens + self.ONE_MIN

    @property
    def last_minutes(self) -> pd.Series:
        """Last minute of each composite session."""
        return self.closes if self._has_right_side else self.closes - self.ONE_MIN

    @property
    def first_session(self) -> pd.Timestamp:
        """First composite session."""
        return self.sessions[0]

    @property
    def last_session(self) -> pd.Timestamp:
        """Last composite session."""
        return self.sessions[-1]

    @property
    def first_minute(self) -> pd.Timestamp:
        """First composite calendar minute."""
        return self.first_minutes.iloc[0]

    @property
    def last_minute(self) -> pd.Timestamp:
        """Last composite calendar minute."""
        return self.last_minutes.iloc[-1]

    def _parse_session(self, session: Session) -> pd.Timestamp:
        """Parse client input representing a session."""
        param_name = "session"
        ts = xcals.calendar_helpers.parse_date(session, param_name, calendar=self)
        if ts not in self.sessions:
            raise xcals.errors.NotSessionError(self, ts, param_name)
        return ts

    def next_session(self, session: Session) -> pd.Timestamp:
        """Return session that immediately follows a given session.

        Parameters
        ----------
        session
            Session from which to evaluate next session.
        """
        session = self._parse_session(session)
        i = self.sessions.get_loc(session) + 1
        return self.sessions[i]

    def previous_session(self, session: Session) -> pd.Timestamp:
        """Return session that immediately preceeds a given session.

        Parameters
        ----------
        session
            Session from which to evaluate previous session.
        """
        session = self._parse_session(session)
        i = self.sessions.get_loc(session) - 1
        return self.sessions[i]

    def session_open(self, session: Session) -> pd.Timestamp:
        """Return open time of a given composite session.

        Parameters
        ----------
        session
            Composite session for which requireopen time.
        """
        session = self._parse_session(session)
        open_ = self.opens[session]
        if TYPE_CHECKING:
            assert isinstance(open_, pd.Timestamp)
        return open_

    def session_close(self, session: Session) -> pd.Timestamp:
        """Return close time of a given composite session.

        Parameters
        ----------
        session
            Composite session for which require close time.
        """
        session = self._parse_session(session)
        close = self.closes[session]
        if TYPE_CHECKING:
            assert isinstance(close, pd.Timestamp)
        return close

    def next_session_open(self, session: Session) -> pd.Timestamp:
        """Return open time of next composite session.

        Parameters
        ----------
        session
            Session from which to evaluate the next session for which
            require open time.
        """
        return self.session_open(self.next_session(session))

    def is_open_on_minute(
        self, minute: Minute, ignore_breaks: bool = False, _parse: bool = False
    ) -> bool:
        """Query if a given minute is a trading minute of any session.

        Note: a session close is NOT considered to be a tradng minute.

        Parameters
        ----------
        minute
            Minute to query.

        ignore_breaks
            False: (default) Treat break minutes as non-trading minutes.
            True: Treat break minutes as trading minutes.

        Returns
        -------
        bool
            If `ignore_breaks` False:
                True if `minute` is a trading minute of any of the
                underlying calendars. False otherwise.

            If `ignore_breaks` True:
                True if `minute` is a trading minute or a break minute
                of any of the underlying calendars. False otherwise.
        """
        if _parse:
            minute = xcals.calendar_helpers.parse_timestamp(minute, "minute", self)
        return any(
            cal.is_open_on_minute(minute, ignore_breaks=ignore_breaks, _parse=False)
            for cal in self.calendars
        )

    def next_minute(self, minute: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return trading minute that immediately follows a given minute.

        Parameters
        ----------
        minute
            Minute for which to get next trading minute. Minute can be a
            trading or a non-trading minute.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the next minute.
        """
        if _parse:
            minute = xcals.calendar_helpers.parse_timestamp(minute, "minute", self)
        try:
            idx = xcals.calendar_helpers.next_divider_idx(
                self.minutes_nanos, minute.value
            )
        except IndexError:
            # minute > last_minute handled via parsing
            if minute == self.last_minute:
                raise ValueError(
                    "Minute cannot be the last trading minute or later"
                    f" (received `minute` parsed as '{minute}'.)"
                ) from None
        return self.minutes[idx]

    def previous_minute(self, minute: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return trading minute that immediately preceeds a given minute.

        Parameters
        ----------
        minute
            Minute for which to get previous trading minute. Minute can be
            a trading or a non-trading minute.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the previous minute.
        """
        if _parse:
            minute = xcals.calendar_helpers.parse_timestamp(minute, "minute", self)
        try:
            idx = xcals.calendar_helpers.previous_divider_idx(
                self.minutes_nanos, minute.value
            )
        except ValueError:
            # dt < first_minute handled via parsing
            if minute == self.first_minute:
                raise ValueError(
                    "Minute cannot be the first trading minute or earlier"
                    f" (received `minute` parsed as '{minute}'.)"
                ) from None
        return self.minutes[idx]

    def minute_to_trading_minute(
        self,
        minute: Minute,
        direction: Literal["next", "previous", "none"] = "none",
        _parse: bool = True,
    ) -> pd.Timestamp:
        """Resolve a minute to a trading minute.

        Differs from `previous_minute` and `next_minute` by returning
        `minute` unchanged if `minute` is a trading minute.

        Parameters
        ----------
        minute
            Timestamp to be resolved to a trading minute.

        direction:
            How to resolve `minute` if does not represent a trading minute:
                "next" - return trading minute that immediately follows
                `minute`.

                "previous" - return trading minute that immediately
                preceeds `minute`.

                "none" - raise KeyError.

        Returns
        -------
        pd.Timestamp
            Returns `minute` if `minute` is a trading minute otherwise
            first trading minute that, in accordance with `direction`,
            either immediately follows or preceeds `minute`.

        Raises
        ------
        ValueError
            If `minute` is not a trading minute and `direction` is None.

        See Also
        --------
        next_mintue
        previous_minute
        """
        if _parse:
            minute = xcals.calendar_helpers.parse_timestamp(minute, "minute", self)
        if self.is_open_on_minute(minute, _parse=False):
            return minute
        elif direction == "next":
            return self.next_minute(minute, _parse=False)
        elif direction == "previous":
            return self.previous_minute(minute, _parse=False)
        else:
            raise ValueError(
                f"`minute` '{minute}' is not a trading minute. Consider passing"
                " `direction` as 'next' or 'previous'."
            )

    def minute_to_sessions(
        self,
        minute: Minute,
        direction: Literal["next", "previous", None] = None,
        _parse: bool = True,
    ) -> pd.DatetimeIndex:
        """Get sessions of which a given minute is a trading minute.

        Parameters
        ----------
        minute
            Minute against which to eveluate session(s).

        direction : default: None
            In event that `minute` is not a minute of any session, return:
                "next" - first session that follows `minute`
                "previous" -  first session that preceeds `minute`
                None - empty DatetimeIndex

        Returns
        -------
        pd.DatetimeIndex
            All sessions that `minute` is a minute of. Will have length 2
            if `minute` is a minute of each of two overlapping sessions,
            otherwise will have length 1.
        """
        # pylint: disable=missing-param-doc
        if _parse:
            ts = xcals.calendar_helpers.parse_timestamp(minute, calendar=self)
        else:
            ts = minute
        bv = (self.first_minutes <= ts) & (ts <= self.last_minutes)
        if bv.any() or direction is None:
            return self.sessions[bv]
        elif direction == "next":
            return self.sessions[ts < self.first_minutes][:1]
        elif direction == "previous":
            return self.sessions[ts >= self.last_minutes][-1:]
        raise ValueError(
            "`direction` must be in ['next', 'previous'] or None although"
            f" received as {direction}."
        )

    def _parse_date(self, date: Date) -> pd.Timestamp:
        """Parse a parameter defining a date."""
        ts = xcals.calendar_helpers.parse_date(date, "date", calendar=self)
        return ts

    def _parse_start(self, start: Date) -> pd.Timestamp:
        """Parse a parameter defining start of a date range."""
        ts = xcals.calendar_helpers.parse_date(start, "start", calendar=self)
        return ts

    def _parse_end(self, end: Date) -> pd.Timestamp:
        """Parse a parameter defining end of a date range."""
        ts = xcals.calendar_helpers.parse_date(end, "end", calendar=self)
        return ts

    def _get_date_idx(self, date: pd.Timestamp) -> int:
        """Index position of a date.

        Parameters
        ----------
        date
            Date to query. Must be a tz-naive timestamp with time component
            as midnight.

        Returns
        -------
        int
            Index position of session if `date` represents a session,
            otherwise index position of session that immediately
            follows `date`.
        """
        return self.sessions.get_indexer([date], method="bfill")[0]

    def is_session(self, date: Date) -> bool:
        """Query if a date is a valid session.

        Parameters
        ----------
        date
            Date to be queried.

        Returns
        -------
        bool
            True if `date` is a session, False otherwise.
        """
        date = self._parse_date(date)
        idx = self._get_date_idx(date)
        return self.sessions[idx] == date

    def date_to_session(
        self,
        date: Date,
        direction: Literal["none", "previous", "next"] = "none",
    ) -> pd.Timestamp:
        """Return a session corresponding to a given date.

        Parameters
        ----------
        date
            Date for which require session. Can be a date that does not
            represent an actual session (see `direction`).

        direction : default: "none"
            Defines behaviour if `date` does not represent a session:
                "next" - return first session following `date`.
                "previous" - return first session prior to `date`.
                "none" - raise ValueError.
        """
        # pylint: disable=missing-param-doc
        date = self._parse_date(date)
        if self.is_session(date):
            return date
        elif direction in ["next", "previous"]:
            idx = self._get_date_idx(date)
            if direction == "previous":
                idx -= 1
            return self.sessions[idx]
        elif direction == "none":
            raise ValueError(
                f"`date` '{date}' does not represent a session. Consider passing"
                " a `direction`."
            )
        else:
            raise ValueError(
                f"'{direction}' is not a valid `direction`. Valid `direction`"
                ' values are "next", "previous" and "none".'
            )

    def sessions_in_range(
        self, start: Date | None = None, end: Date | None = None
    ) -> pd.DatetimeIndex:
        """Return sessions that fall within a range.

        Parameters
        ----------
        start : default: first calendar session
            Start of date range to query.

        end : default: last calendar session
            End of date range to query.
        """
        # pylint: disable=missing-param-doc
        if start is None:
            start = self.first_session
        elif not (isinstance(start, pd.Timestamp) and helpers.is_date(start)):
            start = self._parse_start(start)
        if end is None:
            end = self.last_session
        elif not (isinstance(end, pd.Timestamp) and helpers.is_date(end)):
            end = self._parse_end(end)
        slc_start = self._sessions_nanos.searchsorted(start.value, "left")
        slc_stop = self._sessions_nanos.searchsorted(end.value, "right")
        return self.sessions[slc_start:slc_stop]

    def sessions_overlap(
        self, start: Date | None = None, end: Date | None = None
    ) -> pd.Series:
        """Query if sessions overlap following sessions.

        Query if minutes of a session overlap minutes of the next session.

        Parameters
        ----------
        start : default: first calendar session
            Start of date range covering sessions to query.

        end : default: last calendar session
            End of date range covering sessions to query.

        Returns
        -------
        pd.Series
            index : pd.DatetimeIndex
                Sessions from `start` through `end`.

            value : bool
                True if session's last minute is later than next session's
                first minute.
        """
        # pylint: disable=missing-param-doc
        start = self._parse_start(start) if start is not None else None
        end = self._parse_end(end) if end is not None else None

        # if end not last session then set to next session after end in
        # order to evalute if end overlaps with following session.
        if end == self.last_session:
            end = None
        if end is not None:
            end = self.next_session(end)

        # typing - can index with slice defined in terms of pd.Timestamp
        first_minutes = self.first_minutes[start:end]  # type: ignore[misc]
        last_minutes = self.last_minutes[start:end]  # type: ignore[misc]
        bv = last_minutes >= first_minutes.shift(-1)
        mask = pd.Series(bv, index=first_minutes.index)
        return mask if end is None else mask[:-1]

    def sessions_length(
        self, start: Date | None = None, end: Date | None = None
    ) -> pd.Series:
        """Get length (duration) of each session.

        Parameters
        ----------
        start : default: first calendar session
            Start of date range covering sessions to query.

        end : default: last calendar session
            End of date range covering sessions to query.

        Returns
        -------
        pd.Series
            index : pd.DatetimeIndex
                Sessions from `start` through `end`.
            value : pd.Timedelta
                Session length.
        """
        # pylint: disable=missing-param-doc
        start = self.first_session if start is None else self._parse_start(start)
        end = self.last_session if end is None else self._parse_end(end)
        return self.closes[start:end] - self.opens[start:end]  # type: ignore[misc]

    def is_factor_of_sessions(
        self,
        factor: pd.Timedelta,
        start: Date | None = None,
        end: Date | None = None,
    ) -> pd.Series:
        """Query if a given factor is a factor of given sessions durations.

        Parameters
        ----------
        factor
            Factor to query.

        start : default: first calendar session
            Start of date range covering sessions to query.

        end : default: last calendar session
            End of date range covering sessions to query.

        Returns
        -------
        pd.Series
            index : pd.DatetimeIndex
                Sessions from `start` through `end`.
            value : bool
                True if `factor` is a factor of session's durations.
        """
        # pylint: disable=missing-param-doc
        lengths = self.sessions_length(start, end)
        return (lengths % factor).eq(pd.Timedelta(0))

    def non_trading_index(
        self,
        start: Date | None = None,
        end: Date | None = None,
        utc: bool = True,
    ) -> pd.IntervalIndex:
        """Index of times during which composite calendar is closed.

        Calendar considered closed whenever none of the underlying
        calendars are open. Where a calendar has a break, that calendar is
        considered closed during the break.

        Parameters
        ----------
        start : default: first calendar session
            Start of date range covering sessions to query.

        end : default: last calendar session
            End of date range covering sessions to query.

        utc
            False to return index of IntervalIndex as tz-naive, True to
            return index with tz as UTC.

        Returns
        -------
        pd.IntervalIndex
            Indices represent each continuous period during which the
            composite calendar is closed.
        """
        # pylint: disable=missing-param-doc
        start = self.first_session if start is None else self._parse_start(start)
        start = self.date_to_session(start, "next")
        end = self.last_session if end is None else self._parse_end(end)
        end = self.date_to_session(end, "previous")
        cnti = _CompositeNonTradingIndex(self)
        return cnti.non_trading_index(start, end, utc)

    def trading_index(
        self,
        interval: intervals.BI | intervals.TDInterval,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ignore_breaks: bool | dict[xcals.ExchangeCalendar, bool] = False,
        raise_overlapping: bool = True,
        curtail_calendar_overlaps: bool = False,
        utc: bool = True,
    ) -> pd.IntervalIndex:
        """Return intraday trading index for the composite calendar.

        Index will include indices only for intervals during which at least
        one calendar trades at least some of the interval.

        Parameters
        ----------
        interval
            Time delta between each indice. Must be intraday. For a daily
            index use `sessions_in_range`.

        start
            Timestamp from which index is to commence (can be inside or
            outside of trading hours). Can be a time or date (where date
            represnted by tz-naive timestamp with no time component).

        end
            Timestamp at which index is to end (can be inside or outside
            of trading hours). All indicies with right side before or on
            timestamp will be included. Can be a time or date (where date
            represnted by tz-naive timestamp with no time component).

        ignore_breaks
            True: include indices through any break, as if the session
            were continuous.

            False: do not include any indices that would otherwise commence
            during a break. Start afternoon session with an indice with
            left side on pm subsession open.

            Can be passed as a dictionary to specify by calendar:
                key: xcals.ExchangeCalendar of `calendars`.
                value: bool (as above)

        raise_overlapping
            True: raise `errors.CompositeIndexConflict` if any indice
            would partially overlap with another.

            False: return trading index with any partially overlapping
            indices.

        curtail_calendar_overlaps
            How to treat indices of the calendar-specific trading indexes
            which collectively comprise the composite trading index.

                True: Curtail right of earlier overlapping indice to left
                of latter indice.

                False: Raise `CompositeIndexCalendarConflict`.

            NB this option is concerned with overlapping indices of the
            trading indexes of each constituent calendar, NOT with indices
            that may overlap as a result of consolidating these trading
            indexes into a composite trading index. (NB `raise_overlapping`
            determines action to take if any composite trading index
            indices are overlapping).

        utc
            True: return index with timezone as "UTC".
            False: return index as timezone-naive.
        """
        # pylint: disable=too-many-arguments
        ignore_breaks_ = ignore_breaks
        curtail_overlaps = curtail_calendar_overlaps
        cal = self.calendars[0]
        if isinstance(ignore_breaks_, dict):
            ignore_breaks = ignore_breaks_[cal]
        assert isinstance(ignore_breaks, bool)
        try:
            index = get_trading_index(
                cal, interval, start, end, False, ignore_breaks, curtail_overlaps
            )
        except xcals.errors.IntervalsOverlapError as e:
            raise errors.CompositeIndexCalendarConflict(cal) from e

        # work in tz-naive for quicker .union
        index = pdutils.interval_index_new_tz(index, None)
        for cal in self.calendars[1:]:
            if isinstance(ignore_breaks_, dict):
                ignore_breaks = ignore_breaks_[cal]
            assert isinstance(ignore_breaks, bool)
            try:
                index_ = get_trading_index(
                    cal, interval, start, end, False, ignore_breaks, curtail_overlaps
                )
            except xcals.errors.IntervalsOverlapError as e:
                raise errors.CompositeIndexCalendarConflict(cal) from e
            index_ = pdutils.interval_index_new_tz(index_, None)
            index = index.union(index_, sort=False)
        index = index.sort_values()
        if raise_overlapping and not index.is_non_overlapping_monotonic:
            raise errors.CompositeIndexConflict()
        if utc:
            index = pdutils.interval_index_new_tz(index, UTC)
        return index


class _CompositeNonTradingIndex:
    """Create an index of times that a composite calendar is closed."""

    def __init__(self, cc: CompositeCalendar):
        self.cc = cc
        self.calendars = cc.calendars

        self._index: pd.IntervalIndex
        self._start_session: pd.Timestamp | None
        self._end_session: pd.Timestamp | None
        self._opens: pd.DataFrame | None
        self._closes: pd.DataFrame | None
        self._reset()

    def _reset(
        self,
        start_session: pd.Timestamp | None = None,
        end_session: pd.Timestamp | None = None,
    ):
        self._index = pd.IntervalIndex(
            [], closed="left", dtype="interval[datetime64[ns], left]"
        )
        self._start_session = start_session
        self._end_session = end_session
        if start_session is not None:
            self._closes = self._get_closes()
            self._opens = self._get_opens()
        else:
            self._opens = None
            self._closes = None

    @staticmethod
    def _adjust_tz(srs: pd.Series) -> pd.Series:
        srs = srs.copy()
        # work in tz-naive for speed and to order missing values last
        if srs.dt.tz is not None:
            srs = srs.dt.tz_convert(None)
        return srs

    @property
    def _sessions_slice(self) -> slice:
        return slice(self._start_session, self._end_session)

    def _get_closes(self) -> pd.DataFrame:
        d = {}
        i = 0

        def add_to_dict(srs: pd.Series):
            nonlocal i
            d[i] = srs
            i += 1

        for cal in self.calendars:
            if cal.sessions_has_break(self._start_session, self._end_session):
                break_starts = self._adjust_tz(cal.break_starts)
                add_to_dict(break_starts[self._sessions_slice])
            closes = self._adjust_tz(cal.closes)
            add_to_dict(closes[self._sessions_slice])

        df = pd.DataFrame(d)
        return pd.DataFrame(np.sort(df.values))

    def _get_opens(self) -> pd.DataFrame:
        d = {}
        i = 0

        def add_to_dict(srs: pd.Series):
            nonlocal i
            d[i] = srs

            i += 1

        for cal in self.calendars:
            opens = self._adjust_tz(cal.opens)
            add_to_dict(opens[self._sessions_slice])
            if cal.sessions_has_break(self._start_session, self._end_session):
                break_ends = self._adjust_tz(cal.break_ends)
                add_to_dict(break_ends[self._sessions_slice])

        df = pd.DataFrame(d)
        opens_df = pd.DataFrame(np.sort(df.values))

        next_day_open = opens_df.min(axis=1).shift(-1)
        assert self.closes is not None
        day_close = self.closes.max(axis=1)
        bv_sessions_overlap = day_close > next_day_open
        next_day_open[bv_sessions_overlap] = day_close[bv_sessions_overlap]
        opens_df["next_day"] = next_day_open

        try:
            last_day_next_session = self.cc.next_session(self._end_session)
        except IndexError:
            # last calendar session, so set next open to last close
            last_day_next_open = self.closes.iat[-1, -1]
        else:
            last_day_next_open = max(
                c.session_open(last_day_next_session)
                for c in self.cc.calendars
                if c.is_session(last_day_next_session)
            )
            last_day_next_open = last_day_next_open.tz_convert(None)
        opens_df.iat[-1, -1] = last_day_next_open
        return opens_df

    @property
    def closes(self) -> pd.DataFrame | None:
        """Close times for each session and subsession of each calendar.

        Period covered relates to last call to `non_trading_index`.
        """
        return self._closes

    @property
    def opens(self) -> pd.DataFrame | None:
        """Open times for each session and subsession of each calendar.

        Period covered relates to last call to `non_trading_index`.
        """
        return self._opens

    def _get_next_close(self, rnd: int, last_open: pd.Series) -> pd.Series:
        assert self.closes is not None and self.opens is not None
        next_close = self.closes[self.closes.gt(last_open, axis=0)].min(axis=1)
        for i in range(rnd, len(self.closes.columns)):
            bv = next_close >= self.opens[i]
            next_close[bv] = self.closes[i]
        return next_close

    def _get_next_open(self, last_close: pd.Series) -> pd.Series:
        assert self.opens is not None
        return self.opens[self.opens.ge(last_close, axis=0)].min(axis=1)

    def _add_to_index(self, last_close: pd.Series, next_open: pd.Series):
        try:
            index = pd.IntervalIndex.from_arrays(
                last_close.dropna(), next_open.dropna(), "left"
            )
        except ValueError:
            last_close_ = last_close.dropna()
            # last value of last close is last calendar close (there is no next open)
            if last_close_.iloc[-1] == self.cc.closes.iloc[-1].tz_convert(None):
                index = pd.IntervalIndex.from_arrays(
                    last_close_.iloc[:-1], next_open.dropna(), "left"
                )
            else:
                raise
        self._index = self._index.union(index, sort=False)

    def non_trading_index(
        self, start: pd.Timestamp, end: pd.Timestamp, utc: bool = True
    ) -> pd.IntervalIndex:
        """Index of times during which composite calendar is closed.

        Calendar considered closed whenever none of the underlying
        calendars are open. Where a calendar has a break, that calendar is
        considered closed during the break.

        Parameters
        ----------
        start
            First session of period over which to evaluate index.

        end
            Last session of period over which to evaluate index.

        utc
            False to return index of IntervalIndex as tz-naive, True to
            return index with tz as UTC.

        Returns
        -------
        pd.IntervalIndex
            Indices represent each continuous period during which the
            composite calendar is closed.
        """
        self._reset(start, end)
        assert self.closes is not None and self.opens is not None
        last_open = self.opens[0]
        for i in range(1, len(self.closes.columns) + 1):
            last_close = self._get_next_close(i, last_open)  # noqa: F841
            next_open = last_open = self._get_next_open(last_close)
            drop_bv = last_close == next_open  # drop where not gap between sessions
            if drop_bv.any():
                last_close = last_close[~drop_bv]
                next_open = next_open[~drop_bv]
            self._add_to_index(last_close, next_open)

        index = self._index.sort_values()
        if utc:
            left = index.left.tz_localize(UTC)
            right = index.right.tz_localize(UTC)
            return pd.IntervalIndex.from_arrays(left, right, "left")
        else:
            return index
