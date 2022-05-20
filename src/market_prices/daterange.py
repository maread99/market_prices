"""Classes to get date ranges over which to request prices."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_prices import errors, helpers, intervals, mptypes
from market_prices.utils import calendar_utils as calutils


class _Getter(abc.ABC):
    """Evaluate a date range over which to request prices.

    Evaluates a date range for given period parameters, calendar and
    interval.

    Parameters
    ----------
    calendar
        Calendar against which to evaluate date range. Earliest calendar
        session / minute should be no later than `limit`.

    limit
        Earliest session/minute from which prices are available.

        If limits differ by interval then can be receievd as a callable.
        Callable should takes one argument, the interval, and return the
        limit corresponding with that interval.

    pp
        Period parameters against which to evaluate date range.
        `pp` should have been previously verified with
        `market_prices.parsing.verify_period_parameters` and 'start' and
        'end' values should have been parsed with initially with
        `market_prices.parsing.parse_timestamp` and subsequently with
        `market_prices.parsing.parse_start_end`.

        Provision made for passing None only to allow for creating a
        Getter to only use `get_end` (which is not dependent on period
        parameters). NOTE: If `pp` is not passed then other public methods
        will fail.

    ds_interval
        Downsample interval. Any higher interval to which data is to be
        downsampled.

    strict
        Determines behaviour in event daterange start falls before
        `limit`:
            True: raises errors.StartTooEarlyError.
            False: sets daterange start to `limit`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        calendar: xcals.ExchangeCalendar,
        limit: pd.Timestamp | Callable[[intervals.BI], pd.Timestamp],
        pp: mptypes.PP | None = None,
        ds_interval: intervals.PTInterval | None = None,
        strict=True,
    ):
        self._cal = calendar
        self._limit = limit
        self._pp = pp
        self._verify_ds_interval(ds_interval)
        self._ds_interval = ds_interval
        self._strict = strict

    @property
    def cal(self) -> xcals.ExchangeCalendar:
        """Calendar against which to evaluate date range."""
        return self._cal

    @property
    def ds_interval(self) -> intervals.PTInterval | None:
        """Any interval to which data will be subsequently downsampled."""
        return self._ds_interval

    @property
    @abc.abstractmethod
    def interval(self) -> intervals.BI | None:
        """Time delta represented by each indice."""

    @abc.abstractmethod
    def _verify_ds_interval(self, ds_interval: intervals.PTInterval | None):
        """Verify `ds_interval` is valid."""

    @property
    def final_interval(self) -> intervals.PTInterval:
        """Higher of `interval` and `ds_interval`."""
        if self.ds_interval is not None:
            return self.ds_interval
        else:
            return self.interval  # type: ignore[return-value]  #...
            # will raise error if interval is None

    @property
    def pp(self) -> mptypes.PP:
        """Period parameters."""
        if self._pp is None:
            raise ValueError("`pp` was not passed to the constructor.")
        return self._pp

    @property
    def pp_start_end(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Period parameters 'start' and 'end'."""
        return self.pp["start"], self.pp["end"]

    @property
    def limit(self) -> pd.Timestamp:
        """Left limit."""
        if callable(self._limit):
            interval = self.interval
            assert interval is not None
            return self._limit(interval)
        return self._limit

    @property
    def strict(self) -> bool:
        """Query if `self.limit` applied strictly."""
        return self._strict

    @property
    def ds_factor(self) -> int:
        """Downsample factor. Interval as factor of `ds_interval`."""
        if isinstance(self.ds_interval, intervals.DOInterval):
            raise NotImplementedError(
                "`ds_factor` is not implemented when `self.ds_interval` is a DOInterval"
            )
        assert not isinstance(self.final_interval, intervals.DOInterval)
        # mypy ignore as if self.interval None then will raise error advising
        return self.final_interval // self.interval  # type: ignore[operator]

    @property
    @abc.abstractmethod
    def end_now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return live end value and accuracy."""

    @abc.abstractmethod
    def _get_end(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return `ts` aligned with last indice / right of last indice."""

    @abc.abstractmethod
    def _get_start(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Return `ts` aligned with first indice / left of first indice."""

    def get_end(  # pylint: disable=missing-param-doc
        self, ts: pd.Timestamp | None
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return right side of indice representing end of a date range.

        For intraday indices, indices based on session opens of
        `self.calendar`.

        Parameters
        ----------
        ts : , default: 'now'
            End of period to be represented by date range.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            [0]: right side of last indice of date range.
            [1]: accuracy of [0]. Will only differ from [0] if all of:
                - `interval` is intraday.
                - date range end is an unaligned session close.
                - no symbol trades in the period from the session close
                to the end of the unaligned indice.

                In this case [0] will represent the end of the unaligned
                indice* and [1] will represnt the session close.

        Notes
        -----
        * this allows requested prices to include the session close in the
        data in the knowledge that doing so will not introduce prices
        beyond the session close.
        """
        if ts is None:
            return self.end_now
        if ts < self.limit:
            raise errors.EndTooEarlyError(self, self.limit, ts)
        return self._get_end(ts)

    def get_start(self, ts: pd.Timestamp, hard_strict: bool = False) -> pd.Timestamp:
        """Return left side of indice representing start of a date range.

        For intraday indices, indices based on session opens of
        `self.calendar`.

        Parameters
        ----------
        ts
            Start of period to be represented by date range.

        hard_strict
            Require start to be less than `limit` regardless of
            `self.strict`.
        """
        if ts < self.limit:
            if hard_strict or self.strict:
                raise errors.StartTooEarlyError(self, self.limit, ts)
            else:
                return self.limit
        return self._get_start(ts)

    def _get_start_end(
        self, hard_strict: bool = False
    ) -> tuple[mptypes.DateRangeAmb, pd.Timestamp | None]:
        """Return start and end as left/right side of first/last indice.

        Parameters
        ----------
        hard_strict
            Require start to be less than `limit` regardless of
            `self.strict`.

        Returns
        -------
        tuple[ReqDateRange, pd.Timestamp]
            [0] daterange
            [1] end accuracy
        """
        start, end = self.pp_start_end
        start = self.get_start(start, hard_strict) if start is not None else None

        if end is None and start is not None and self._has_duration:
            # end not required
            return (start, end), end  # [0] is mptypes.DateRangeAmb
        end, end_accuracy = self.get_end(end)
        return (start, end), end_accuracy  # [0] is mptypes.DateRangeReq

    @property
    def _has_duration(self) -> bool:
        pp = self.pp
        return bool(
            sum(
                [
                    pp["minutes"],
                    pp["hours"],
                    pp["days"],
                    pp["weeks"],
                    pp["months"],
                    pp["years"],
                ]
            )
        )

    @property
    @abc.abstractmethod
    def daterange(self) -> tuple[mptypes.DateRangeReq, pd.Timestamp]:
        """Date range over which to request prices, and end accuracy."""


class GetterDaily(_Getter):
    """Get date range defined as sessions (as opposed to times)."""

    @property
    def interval(self) -> intervals.BI:
        """Time delta represented by each indice. Always one day."""
        bi = intervals.ONE_DAY
        if TYPE_CHECKING:
            assert isinstance(bi, intervals.BI)  # pylint: disable=protected-access
        return bi

    def _verify_ds_interval(self, ds_interval: intervals.PTInterval | None):
        if ds_interval is not None and ds_interval.is_intraday:
            raise ValueError(
                "`ds_interval` cannot be lower than 'one day' although"
                f" received {ds_interval}."
            )

    def _get_end(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.final_interval.is_monthly:
            assert isinstance(self.final_interval, intervals.DOInterval)
            # Move foward ts one day as working to MS, i.e. if ts last day of month
            # then want to rollback to start of next month.
            end = self.final_interval.as_offset_ms.rollback(ts + helpers.ONE_DAY)
            # Take off one day to return end last day of data required, not first day
            # of following month
            end -= helpers.ONE_DAY
        else:
            # TODO xcals 4.0 lose wrapper
            end = helpers.to_tz_naive(self.cal.date_to_session(ts, "previous"))
        return end, end

    def _get_start(self, ts: pd.Timestamp) -> pd.Timestamp:
        if self.final_interval.is_monthly:
            assert isinstance(self.final_interval, intervals.DOInterval)
            # are there any sessions between ts and the start of the current month?
            start = self.final_interval.as_offset_ms.rollforward(ts)
        else:
            # TODO xcals 4.0 lose wrapper
            start = helpers.to_tz_naive(self.cal.date_to_session(ts, "next"))
        return start

    @property
    def end_now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return live end value and accuracy.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            [0]: Current session if 'now' falls in a session, otherwise
            previous session.
            [1]: Accuarcy. As [0].
        """
        # TODO xcals 4.0 lose wrapper
        end = helpers.to_tz_naive(self.cal.minute_to_session(helpers.now(), "previous"))
        return end, end

    def _prior_start(self, start: pd.Timestamp) -> pd.Timestamp | None:
        """Return start prior to `start`.

        Returns None if prior start would fall before `self.limit`.
        """
        assert self.ds_interval is not None
        if self.ds_interval.is_monthly:
            prior_start = start - self.ds_interval
            if prior_start < self.limit:
                return None
        else:
            try:
                window = self.cal.sessions_window(start, -self.ds_factor)
            except ValueError:
                sessions = self.cal.sessions
                # TODO xcals 4.0 del if clause.
                if sessions.tz is not None:
                    sessions = self.cal.sessions.tz_convert(None)
                limit_idx = sessions.get_loc(self.limit)
                idx = sessions.get_loc(start) - self.ds_factor
                if idx < limit_idx:
                    # prior indice lies to left of limit
                    return None
                else:
                    raise
            prior_start = helpers.to_tz_naive(window[0])
            if prior_start < self.limit:
                return None
        assert isinstance(prior_start, pd.Timestamp)
        return prior_start

    def _offset_days(self, ts: pd.Timestamp, days: int) -> pd.Timestamp:
        days = days - 1 if days > 0 else days + 1
        if days < 0:
            try:
                session = self.cal.session_offset(ts, days)
            except xcals.errors.RequestedSessionOutOfBounds:
                if self.strict:
                    raise errors.StartOutOfBoundsError(self.cal, is_date=True) from None
                else:
                    session = self.cal.first_session
        else:
            session = self.cal.session_offset(ts, days)
        return helpers.to_tz_naive(session)  # TODO xcals 4.0 lose wrapper

    def _daterange_add_a_row_adjustment(self, start: pd.Timestamp) -> pd.Timestamp:
        if not self.pp["add_a_row"]:
            return start

        start_pre_add_a_row = start
        prior_start = self._prior_start(start)
        if prior_start is None:
            if self.strict:
                raise errors.StartTooEarlyError(self, self.limit, add_a_row=True)
            else:
                # Do not set to limit, i.e. do not include 'some of a ds_interval'
                prior_start = start_pre_add_a_row
        return prior_start

    def _check_period_covers_one_monthly_interval(
        self, start: pd.Timestamp, end: pd.Timestamp
    ):
        """Raise error if period is shorter than final interval.

        Period evaluated as `start` through `end`.
        """
        assert isinstance(self.final_interval, intervals.DOInterval)
        last_indice_right = self.final_interval.as_offset_ms.rollforward(end)
        latest_start = last_indice_right - self.final_interval.as_offset()
        if start > latest_start:
            raise errors.PricesUnavailableDOIntervalPeriodError(
                latest_start, self, start, end
            )

    def _daterange_is_monthly(self) -> tuple[mptypes.DateRangeReq, pd.Timestamp]:
        """Date range over which to request prices with monthly ds_interval."""
        # pylint: disable=too-complex, too-many-branches
        # pylint: disable=too-many-locals, too-many-statements
        assert isinstance(self.final_interval, intervals.DOInterval)
        # if end is None will be assigned now although this will in turn be
        # overwritten if start is not None and there's a duration.
        (start, end), end_accuracy = self._get_start_end()

        if self._has_duration:

            if end == self.end_now[0]:
                # evaluate duration from period end, i.e. right of last indice.
                # end will be reset to now after evaluating start.
                one_day = helpers.ONE_DAY
                end = self.final_interval.as_offset_ms.rollforward(end + one_day)
                end -= one_day  # type: ignore[operator,assignment]

            if self.pp["days"] > 0:
                days = self.pp["days"]
                if start is None:
                    assert end is not None
                    if days == 1:
                        start = end
                    else:
                        end_ = end
                        # "next" as want to count back from end of month
                        end_ = self.cal.date_to_session(end, "next")
                        end_ = helpers.to_tz_naive(end_)  # TODO xcals.4.0 lose line
                        start = self._offset_days(end_, -days)
                else:
                    if days == 1:  # pylint: disable=else-if-used
                        end = start
                    else:
                        start_ = start
                        # "previous" as want to count from start of month
                        start_ = self.cal.date_to_session(start, "previous")
                        # TODO xcals.4.0 lose line
                        start_ = helpers.to_tz_naive(start_)
                        end = self._offset_days(start_, days)

            else:
                # days -1 as start/end both considered days of period.
                duration = pd.DateOffset(
                    days=-1,
                    weeks=self.pp["weeks"],
                    months=self.pp["months"],
                    years=self.pp["years"],
                )
                if start is None:
                    assert end is not None
                    # mypy ignored because Timestamp -/+ DateOffset is valid
                    start = end - duration  # type: ignore[operator]
                else:
                    end = start + duration  # type: ignore[operator]

            start = self.get_start(start)
            end, end_accuracy = self.get_end(end)

            end_now, end_now_accuracy = self.end_now
            if end > end_now:
                end, end_accuracy = end_now, end_now_accuracy

        assert end is not None and end_accuracy is not None

        # `start_` exists just to be able to check if period covers one monthly
        # interval to the best that knowledge allows. Initiated here to provide
        # for following adjustment to align the start.
        start_ = start if start is not None else self.get_start(self.limit)
        end_ = self.final_interval.as_offset_ms.rollforward(end)
        diff_months = (
            end_.to_period("M") - start_.to_period("M")  # type: ignore[operator]
        ).n
        _, excess_months = divmod(diff_months, self.final_interval.freq_value)
        if excess_months:
            excess = pd.DateOffset(months=excess_months)
            pp_start, pp_end = self.pp_start_end
            if pp_start is not None and pp_end is None and self._has_duration:
                # align end
                end = (end_ - excess) - helpers.ONE_DAY  # type: ignore[operator]
                end, end_accuracy = self.get_end(end)
            elif start is not None:  # align start
                start = start_ = start + excess  # type: ignore[operator]
            else:
                start_ += excess  # type: ignore[operator]

        self._check_period_covers_one_monthly_interval(start_, end)

        if start is not None:
            start = self._daterange_add_a_row_adjustment(start)
        return (start, end), end_accuracy

    def _check_period_covers_one_daily_interval(
        self, start: pd.Timestamp, end: pd.Timestamp
    ):
        """Raise error if period is shorter than final interval.

        Period evaluated as `start` through `end`.
        """
        assert isinstance(self.final_interval, (pd.Timedelta, intervals.TDInterval))
        num_sessions = self.cal.sessions_distance(start, end)
        if pd.Timedelta(num_sessions, "D") < self.final_interval:
            cbdays = num_sessions * self.cal.day
            raise errors.PricesUnavailableIntervalPeriodError(self, start, end, cbdays)

    @property
    def daterange(self) -> tuple[mptypes.DateRangeReq, pd.Timestamp]:
        """Date range over which to request prices, and end accuracy.

        Returns
        -------
        tuple[mptypes.ReqDateRange, pd.Timestamp]
            [0]: date range over which to request price data:
                [0]: Start of range as left side of an indice based on
                    `self.interval`.
                [1]: End of range as right side of an indice based on
                    `self.final_interval`
            [1]: Accuracy of range end. As daterange daily, same as date
            range end.

        Raises
        ------
        errors.PricesUnavailableIntervalPeriodError
            Evaluated period is shorter than `self.final_interval`.
        """
        # pylint: disable=too-complex,too-many-branches
        if self.final_interval.is_monthly:
            return self._daterange_is_monthly()

        # if end is None will be assigned now although this will in turn be
        # overwritten if start is not None and there's a duration.
        (start, end), end_accuracy = self._get_start_end()

        if self._has_duration:

            if self.pp["days"] > 0:
                days = self.pp["days"]
                if start is None:
                    assert end is not None
                    if days == 1:
                        start = end
                    else:
                        start = self._offset_days(end, -days)
                else:
                    if days == 1:  # pylint: disable=else-if-used
                        end = start
                    else:
                        end = self._offset_days(start, days)

            else:
                # days -1 as start/end both considered days of period.
                duration = pd.DateOffset(
                    days=-1,
                    weeks=self.pp["weeks"],
                    months=self.pp["months"],
                    years=self.pp["years"],
                )
                if start is None:
                    assert end is not None
                    # mypy ignored because Timestamp -/+ DateOffset is valid
                    start = end - duration  # type: ignore[operator]
                else:
                    end = start + duration  # type: ignore[operator]

            start = self.get_start(start)
            end, end_accuracy = self.get_end(end)

            end_now, end_now_accuracy = self.end_now
            if end > end_now:
                end, end_accuracy = end_now, end_now_accuracy

        assert end is not None and end_accuracy is not None

        # check period against interval as best can from available information.
        start_ = self.get_start(self.limit) if start is None else start
        self._check_period_covers_one_daily_interval(start_, end)

        if start is not None:
            start = self._daterange_add_a_row_adjustment(start)
        return (start, end), end_accuracy


class GetterIntraday(_Getter):
    """Get date range defined by times (as opposed to sessions).

    Parameters (in addition to those defined on base)
    ----------
    composite_calendar
        CompositeCalendar for calendars associated with all symbols to be
        included to prices.

    delay
        Any value by which to reduce 'now' by when evaluting dates to
        the current time.

    ignore breaks
        Should breaks be ignored when evaluating the daterange. Interval
        specific.

    interval
        Time delta represented by each indice. Must be lower than one
        day. If not passed then an interval must be assigned to
        `interval` attribute prior to calling `daterange` or any other
        property dependent on the interval.

    end_alignment
        Which interval to align end of daterange to, either:
            mptypes.Alignment.BI: `self.interval`
            mptypes.Alignment.FINAL: `self.final_interval`
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        calendar: xcals.ExchangeCalendar,
        composite_calendar: calutils.CompositeCalendar,
        delay: pd.Timedelta,
        limit: pd.Timestamp | Callable,
        ignore_breaks: dict[intervals.BI, bool] | bool,
        pp: mptypes.PP | None = None,
        interval: intervals.BI | None = None,
        ds_interval: intervals.TDInterval | None = None,
        anchor: mptypes.Anchor = mptypes.Anchor.OPEN,
        end_alignment: mptypes.Alignment = mptypes.Alignment.BI,
        strict=True,
    ):
        self._cc = composite_calendar
        self._delay = delay
        self._anchor = anchor
        self._end_alignment = end_alignment
        self._ignore_breaks = ignore_breaks
        super().__init__(calendar, limit, pp, ds_interval, strict)
        if interval is not None:
            self.interval = interval
        else:
            self._interval: intervals.BI | None = None

    def _verify_ds_interval(self, ds_interval: intervals.PTInterval | None):
        if ds_interval is not None and not ds_interval.is_intraday:
            raise ValueError(
                "`ds_interval` must be an intraday interval although"
                f" received {ds_interval}."
            )

    @property
    def interval(self) -> intervals.BI:
        """Time delta represented by each indice."""
        if self._interval is None:
            raise ValueError("`interval` has not been set.")
        return self._interval

    @interval.setter
    def interval(self, interval: intervals.BI):
        """Time delta represented by each indice.

        Parameters
        ----------
        interval
            Base interval representing time delta covered by each indice.
        """
        if not interval.is_intraday:
            raise ValueError(
                f"`interval` must be less than one day although receieved '{interval}'."
            )
        if self.ds_interval is not None:
            if interval > self.ds_interval:
                raise ValueError(
                    "`interval` cannot be higher than ds_interval although received"
                    f" '{interval}' (ds_interval is '{self.ds_interval}')."
                )
            if self.ds_interval % interval:  # type: ignore[operator]  # % is supported
                raise ValueError(
                    "`interval` must be a factor of `ds_interval` although received"
                    f" '{interval}' (ds_interval is '{self.ds_interval}')."
                )
        self._interval = interval

    @property
    def _interval_total_minutes(self) -> int:
        return int(self.interval.total_seconds() // 60)

    @property
    def final_interval(self) -> intervals.BI | intervals.TDInterval:
        """Higher of `interval` and `ds_interval`."""
        # Included to narrow type from base.
        interval = super().final_interval
        assert isinstance(interval, (intervals.BI, intervals.TDInterval))
        return interval

    @property
    def anchor(self) -> mptypes.Anchor:
        """Basis on which prices will be anchored over the date range."""
        return self._anchor

    @property
    def alignment(self) -> mptypes.Alignment:
        """Alignment for indices."""
        if self.anchor is mptypes.Anchor.OPEN:
            return mptypes.Alignment.FINAL
        return mptypes.Alignment.BI

    @property
    def alignment_interval(self) -> intervals.BI | intervals.TDInterval:
        """Interval against which to align indices."""
        if self.alignment == mptypes.Alignment.BI:
            return self.interval
        assert self.alignment == mptypes.Alignment.FINAL
        return self.final_interval

    @property
    def end_alignment(self) -> mptypes.Alignment:
        """Alignment for an unaligned session close end."""
        return self._end_alignment

    @property
    def end_alignment_interval(self) -> intervals.BI | intervals.TDInterval:
        """Interval against which to align unaligned session close end."""
        if self.end_alignment == mptypes.Alignment.BI:
            return self.interval
        assert self.end_alignment == mptypes.Alignment.FINAL
        return self.final_interval

    @property
    def ignore_breaks(self) -> bool:
        """Query if breaks should be ignored at the current `interval`."""
        if isinstance(self._ignore_breaks, dict):
            return self._ignore_breaks[self.interval]
        return self._ignore_breaks

    @property
    def end_now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return live end value and accuracy.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            [0]: end of current indice if 'now' falls within an indice,
            otherwise end of prior indice. In turn, this end may be
            adjusted if it is an unaligned session close.

            [1]: Accuracy, as for --get_end()--.
        """
        now = helpers.now(self.interval) - self._delay
        ts = min(now + self.alignment_interval, self.cal.next_open(now))
        return self._get_end(ts)

    def _trading_index(
        self,
        start_session: pd.Timestamp,
        end_session: pd.Timestamp,
        closed: Literal["left", "right"],
    ) -> pd.DatetimeIndex:
        force = closed == "right"
        return self.cal.trading_index(
            start=start_session,
            end=end_session,
            period=self.alignment_interval.as_pdfreq,
            intervals=False,
            closed=closed,
            force=force,
            ignore_breaks=self.ignore_breaks,
        )

    def _prior_start(self, start: pd.Timestamp) -> pd.Timestamp | None:
        """Return start prior to `start`.

        Returns None if prior start would fall before `self.limit` or
        calendar left bound.
        """
        end_session = self.cal.minute_to_session(start, "none")
        minutes = self.final_interval.as_minutes
        try:
            start_offset = self.cal.minute_offset(start, -minutes)
        except xcals.errors.RequestedMinuteOutOfBounds:
            # before calendar left bound. Unless `start` first minute then prior_start
            # would resolve to (sub)session start that immediately preceeds start
            if start == self.cal.first_minute:
                return None
            # so that resolves to previous (sub)session start
            start_offset = self.cal.previous_minute(start)

        start_session = self.cal.minute_to_session(start_offset, "none")
        index = self._trading_index(start_session, end_session, "left")
        i = index.get_loc(start)
        diff = self.ds_factor if self.final_interval != self.alignment_interval else 1
        prior_start = index[i - diff]
        if prior_start < self.limit:
            return None
        return prior_start

    def _end_unaligned_close_adj(
        self, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Apply required adjustment if `end` is an unaligned session close.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            [0]: timestamp through which to request data.
            [1]: accuracy of [0].

            If `end` is aligned to an indice then both items will be
            returned as `end`.

            If `end` is an unaligned session close:
                if no symbol trades during the part of the final indice
                (based on `end_alignment_interval`) that falls after the
                session close:
                    [0]: right side of final indice (beyond session close).
                    [1]: session close.

                if any symbol trades during the part of the final indice
                that falls after the session close:
                    [0]: left side of final indice
                    [1]: left side of final indice
                NB In this case interval will be maintained although
                session close will not be represented.

                exception: If the end alignment interval is longer than the
                length of the session / pm subsession (if session has
                break) the both [0] and [1] will be returned as the session
                / pm subsession open.
        """
        if end.value not in self.cal.closes_nanos:
            # not a session close
            return end, end

        close = end

        i = self.cal.closes_nanos.searchsorted(close.value)
        session = self.cal.sessions[i]

        if self.ignore_breaks:
            session_open, session_close = self.cal.session_open_close(session)
            length = session_close - session_open
        else:
            length = calutils.subsession_length(self.cal, session, False, False)

        modulus = length % self.end_alignment_interval
        if not modulus:  # final indice aligns
            return close, close

        nti = self._cc.non_trading_index(session, session)

        if modulus == length:
            # end alignment interval is >= (sub)session length

            if not self.ignore_breaks and self.cal.session_has_break(session):
                open_ = self.cal.session_break_end(session)
            else:
                open_ = self.cal.session_open(session)
            right = open_ + self.end_alignment_interval
            for non_trading_indice in nti:
                if all(
                    ts in non_trading_indice for ts in (close, right - helpers.ONE_MIN)
                ):
                    # right of interval would not include prices registered after close.
                    return right, close

            # NOTE 03/12/21 - Returns open, rather than prior close, in order to prevent
            # any second call from daterange (when there's a duration) from taking off
            # another day (which it would do if this were receieve the close back). I
            # don't like it, it's the only circumstance that doesn't return end as the
            # right side of an indice. It's necessity is almost certainly the result of
            # not having a better daterange implementation, but believe it's benign
            # and, at least with the current daterange implementation, necessary.
            # NOTE 13/04/22 - would be an issue if a request for data is made only for
            # the period at the end of the range that doesn't represent a trading period
            #  would result in error being raised on not receiving any prices back.
            open_ = self.cal.session_open(session)
            return open_, open_

        # unaligned indice needs to be resolved to one side or the other
        right = close - modulus + self.end_alignment_interval
        for non_trading_indice in nti:
            if all(ts in non_trading_indice for ts in (close, right - helpers.ONE_MIN)):
                # final indice would not include prices registered after close
                return right, close
        # final indice would include prices registered after session close,
        # so return end of prior indice
        end = close - modulus
        return end, end

    def _get_end(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        session = self.cal.minute_to_session(ts, "previous")
        try:
            prev_session = self.cal.previous_session(session)
        except ValueError:
            if session == self.cal.first_session:
                prev_session = session
            else:
                raise

        # Force close True so that last indice is not excluded for intervals
        # that are not factors of session length
        index = self._trading_index(prev_session, session, "right")
        i = index.get_indexer([ts], method="ffill")[0]
        end = index[i]
        return self._end_unaligned_close_adj(end)

    def _get_start(self, ts: pd.Timestamp) -> pd.Timestamp:
        session = self.cal.minute_to_session(ts, "next")
        # index includes next session to have value when ts is within final interval
        # of trading index (i.e. when needs to roll forwards to next open).
        next_session = self.cal.next_session(session)
        index = self._trading_index(session, next_session, "left")
        start = index[index.get_indexer([ts], method="bfill")[0]]
        end_now = self.end_now[0]
        last_valid_start = end_now - self.final_interval
        if start > last_valid_start:
            # last reprieve, only evaluate if could be saved by it...
            # open of current session is a valid start
            last_valid_start = max(last_valid_start, self.cal.previous_open(end_now))
            if start > last_valid_start:
                assert isinstance(last_valid_start, pd.Timestamp)
                raise errors.StartTooLateError(
                    start, last_valid_start, self.cal, self._delay, evaluated=True
                )
        return start

    def _offset_days(self, ts: pd.Timestamp, days: int) -> pd.Timestamp:
        # pylint: disable=too-complex,too-many-branches

        # If ts is a session open, close, break_start or break_end then offset to
        # same bound of target session.

        # "previous" to cover ts as close (not a trading minute).
        session = self.cal.minute_to_session(ts, "previous")
        schedule_vals = self.cal.schedule.loc[session].dropna().view(np.int64).values
        if ts.value in schedule_vals:
            target_i = self.cal.sessions.get_loc(session) + days

            if target_i < 0:
                if target_i == -1 and ts.value in self.cal.closes_nanos:
                    # because can resolve within calendar bounds although standard
                    # resolution would need to call calendar methods on minutes to
                    # left of calendar bound.
                    return self.cal.first_minute
                elif not self.strict:
                    return self.limit
                else:
                    raise errors.StartOutOfBoundsError(self.cal)

            target_session = self.cal.sessions[target_i]
            # TODO XCALS 4.0 del if clause.
            if self.cal.opens.dt.tz is None:
                ts = ts.tz_convert(None)
            if ts == self.cal.opens[session]:
                minute = self.cal.opens[target_session]
            elif ts == self.cal.closes[session]:
                minute = self.cal.closes[target_session]
                if days < 0:
                    # if offset to get 'start', return next trading minute,
                    # not a close that cannot represent 'start'.
                    minute = self.cal.minute_to_trading_minute(minute, "next")
            elif ts == self.cal.break_ends[session]:
                minute = self.cal.break_ends[target_session]
            elif ts == self.cal.break_starts[session]:
                minute = self.cal.break_starts[target_session]
                if days < 0:
                    # if offset to get 'start', return next trading minute,
                    # not a close that cannot represent 'start'.
                    minute = self.cal.minute_to_trading_minute(minute, "next")
            return helpers.to_utc(minute)  # TODO XCALS 4.0 lose wrapper

        # If `ts` not a bound, offset according to minute.

        try:
            minute = self.cal.minute_offset_by_sessions(ts, days)
        except xcals.errors.RequestedMinuteOutOfBounds:
            if not self.strict:
                return self.limit
            else:
                raise errors.StartOutOfBoundsError(self.cal) from None

        if minute.value in self.cal.last_minutes_nanos and (
            minute.minute,
            minute.hour,
        ) != (ts.minute, ts.hour):
            # offset to non-trading minute which was forced to last trading minute.
            # Want as close.
            minute += helpers.ONE_MIN

        return minute

    @property
    def daterange(self) -> tuple[mptypes.DateRange, pd.Timestamp]:
        """Date range over which to request prices, and end accuracy.

        Returns
        -------
        tuple[mptypes.DateRange, pd.Timestamp]
            [0]: date range over which to request price data:
                [0]: Start of range as left side of an indice based on
                    `self.interval`.
                [1]: End of range as right side of an indice based on
                    `end_alignment_interval`.
            [1]: Accuracy of range end. Will be as end of date range unless
            range end is an unaligned session close that was advanced to
            the right of the unaligned indice, in which case accuracy will
            be as session close. Note: end will only have been advanced if
            no symbol traded between session close and the right of the
            unaligned indice.

        Raises
        ------
        errors.PricesUnavailableIntervalPeriodError
            Evaluated period does not span at least one full indice.
        """
        # pylint: disable=too-complex,too-many-branches,too-many-statements

        # raise error if start is before limit and end to be evaluated from start.
        hard_strict = self._has_duration and self.pp["start"] is not None
        # if end is None will be assigned now although this will in turn be
        # overwritten if start is not None and there's a duration.
        (start, end), end_accuracy = self._get_start_end(hard_strict=hard_strict)

        if self._has_duration:

            intraday_duration = self.pp["hours"] * 60 + self.pp["minutes"]
            if intraday_duration:
                if intraday_duration < self.final_interval.as_minutes:
                    raise errors.PricesUnavailableIntervalDurationError(
                        pd.Timedelta(intraday_duration, "T"), self
                    )
                if start is None:
                    end_ = end_accuracy
                    assert end_ is not None
                    if (
                        end_.value in self.cal.closes_nanos
                        and end_.value not in self.cal.opens_nanos
                    ) or (end_.value in self.cal.break_starts_nanos):
                        # minute_offset takes a trading minute
                        end_ = self.cal.next_minute(end_)
                    try:
                        start = self.cal.minute_offset(end_, -intraday_duration)
                    except xcals.errors.RequestedMinuteOutOfBounds:
                        mins = len(self.cal.minutes_in_range(self.limit, end)) - 1
                        if mins < intraday_duration:
                            if self.strict:
                                raise errors.StartOutOfBoundsError(self.cal) from None
                            else:
                                start = self.limit
                        else:
                            raise
                else:
                    end = self.cal.minute_offset(start, intraday_duration)

            elif self.pp["days"] > 0:
                days = self.pp["days"]
                if start is None:
                    assert end_accuracy is not None
                    start = self._offset_days(end_accuracy, -days)
                else:
                    end = self._offset_days(start, days)

            else:
                duration = pd.DateOffset(
                    weeks=self.pp["weeks"],
                    months=self.pp["months"],
                    years=self.pp["years"],
                )
                if start is None:
                    # mypy ignored because Timestamp -/+ DateOffset is valid
                    start = end - duration  # type: ignore[operator]
                else:
                    end = start + duration  # type: ignore[operator]

            start = self.get_start(start)
            end, end_accuracy = self.get_end(end)

            end_now, end_now_accuracy = self.end_now
            if end > end_now:
                end, end_accuracy = end_now, end_now_accuracy

        elif start is None:  # pylint: disable=confusing-consecutive-elif
            start = self.get_start(self.limit)

        assert start is not None
        assert end is not None
        assert end_accuracy is not None

        if self.anchor is mptypes.Anchor.OPEN:
            if start >= end_accuracy:
                # If start < end then there's >= one interval between them.
                # Start will be > end if end resolves to close of previous session
                # and start resolves to open of next session.
                raise errors.PricesUnavailableIntervalPeriodError(
                    self, start, end_accuracy
                )
        else:
            # minutes_in_period to account for crossing (sub)sessions.
            if start >= end_accuracy:
                minutes = 0
            else:
                minutes = calutils.minutes_in_period(self.cal, start, end_accuracy)
            if self.final_interval.as_minutes > minutes:
                period_duration = pd.Timedelta(minutes, "T")
                raise errors.PricesUnavailableIntervalPeriodError(
                    self, start, end_accuracy, period_duration
                )

        if self.pp["add_a_row"]:
            start_pre_add_a_row = start
            start = self._prior_start(start)
            if start is None:
                if self.strict:
                    raise errors.StartTooEarlyError(self, self.limit, add_a_row=True)
                else:
                    # Do not set to limit, i.e. don't include 'some of a ds_interval'
                    start = start_pre_add_a_row

        return (start, end), end_accuracy

    @property
    def daterange_tight(self) -> tuple[mptypes.DateRangeReq, pd.Timestamp]:
        """Tight date range over which to request prices.

        Use to request prices and query requested prices. For base
        intervals range will include all required indices whilst avoiding
        periods beyond the last requested indice for which prices will not
        be available. (This is not an issue if only every requesting
        prices for the full range, although will be if only request prices
        for part of the range and that part happens to be the big at the
        end for which prices are not available.)

        As `daterange` although will tighten range 'end' to less than one
        `interval` from 'end_accuracy'. Accordingly, only different from
        `daterange` in event of all of the following:
            - `ds_interval` is > `interval`.
            - end is an unaligned end.
            - difference between 'end' and 'end_accuracy' is greater than
                `interval`.

        For example, if `interval` were 1H, `ds_interval` were 2H and
        `daterange` were to return:
            ((Timestamp('2022-03-08 14:30', tz='UTC'),
              Timestamp('2022-03-10 22:30', tz='UTC')),
            Timestamp('2022-03-10 21:00', tz='UTC'))

        ...then `daterange_tight` would return:
            ((Timestamp('2022-03-08 14:30', tz='UTC'),
              Timestamp('2022-03-10 21:30', tz='UTC')),
            Timestamp('2022-03-10 21:00', tz='UTC'))
        """
        (start, end), end_accuracy = self.daterange
        if end - end_accuracy >= self.interval:
            _, remainder = divmod(end - end_accuracy, self.interval)
            end = end_accuracy + remainder
        return (start, end), end_accuracy

    @property
    def daterange_sessions(self) -> mptypes.DateRange:
        """Date range defined by start and end sessions."""
        (start, _), end = self.daterange
        start_session = self.cal.minute_to_session(start)
        if end.value in self.cal.closes_nanos or end.value in self.cal.break_starts:
            # to make a trading minute of end session. Also, for 24h calendars
            # avoids `end` being treated as first minute of next session.
            end -= helpers.ONE_MIN
        end_session = self.cal.minute_to_session(end)
        # TODO xcals 4.0 del clause
        if start_session.tz is not None:
            start_session = start_session.tz_convert(None)
            end_session = end_session.tz_convert(None)
        return start_session, end_session
