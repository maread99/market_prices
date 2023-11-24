"""Error classes."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import exchange_calendars as xcals
from exchange_calendars.exchange_calendar import ExchangeCalendar
import pandas as pd

from market_prices import helpers, intervals, mptypes
from market_prices.utils import calendar_utils as calutils
from market_prices.helpers import fts
from market_prices.intervals import BI

if TYPE_CHECKING:
    from market_prices.daterange import _Getter, GetterDaily, GetterIntraday
    from market_prices.prices.base import PricesBase
    from market_prices.pt import PTIntraday

# pylint: disable=super-init-not-called


class MarketPricesError(Exception):
    """Base error class."""

    def __str__(self) -> str:
        return getattr(self, "_msg", "Prices unavailable.")

    def __unicode__(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()


class APIError(MarketPricesError):
    """Error when accessing an API endpoint."""


class NoGetPricesParams(MarketPricesError):
    """Instance of `PricesBase.GetPricesParams` is not yet available."""

    _msg = (
        "An instance of `PricesBase.GetPricesParams` is not available as"
        " the `get` method has not been called."
    )


class PricesUnavailableError(MarketPricesError):
    """Prices unavailable for passed parameters."""


class PricesDateRangeEmpty(PricesUnavailableError):
    """Requested DateRange contains no trading sessions/minutes."""

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        intraday: bool,
        calendar: xcals.ExchangeCalendar,
    ):
        insert = "is closed" if intraday else "has no sessions"
        self._msg = (
            f"Calendar '{calendar.name}' {insert} from '{fts(start)}'"
            f" through '{fts(end)}'."
        )


# --- 'start' or 'end' earlier than left bound (calendar first session/minute). ---


class _OutOfBoundsError(PricesUnavailableError):
    """`start` or `end` parameter earlier than calendar's left bound.

    Parameters
    ----------
    calendar
        Calendar against which `ts` was evaluated as out-of-bounds.

    ts
        Timestamp evaluated as out-of-bounds. None if not able to evaluate
        as would have been out-of-bounds.

    is_date
        Only required if `ts` is None.
        True: `ts` would have represented a date.
        False: `ts` would have represente a time.
    """

    param: Literal["start", "end"]

    def __init__(
        self,
        calendar: xcals.ExchangeCalendar,
        ts: pd.Timestamp | None = None,
        is_date: bool | None = None,
    ):
        is_date = is_date if ts is None else helpers.is_date(ts)
        self._c = calendar
        self._ts = ts
        self._ts_type = "date" if is_date else "minute"
        self._bound = calendar.first_session if is_date else calendar.first_minute

    def __str__(self) -> str:
        if self._ts is not None:
            insert = f"({helpers.fts(self._ts)}) is earlier"
        else:
            insert = f"would resolve to an earlier {self._ts_type}"
        msg = (
            f"Prices unavailable as {self.param} {insert} than the"
            f" earliest {self._ts_type} of calendar '{self._c.name}'."
            f" The calendar's earliest {self._ts_type} is {helpers.fts(self._bound)}"
            f" (this bound should coincide with the earliest {self._ts_type} for which"
            " daily price data is available)."
        )
        return msg


class StartOutOfBoundsError(_OutOfBoundsError):
    """'start' parameter earlier than calendar's left bound."""

    param: Literal["start"] = "start"


class EndOutOfBoundsError(_OutOfBoundsError):
    """'end' parameter earlier than calendar's left bound."""

    param: Literal["end"] = "end"


# - 'start' later than right limit (latest session/minute for which prices available) -


class StartTooLateError(PricesUnavailableError):
    """start parameter later than right limit for which prices available.

    Parameters
    ----------
    start
        Either start parameter as received, adjusted only for `tzin`, or
        start of daterange as subsequently evaluated.

    evaluated
        True: `start` is start of an evaluated date range.
        False: `start` is start parameter as receieved (default).
    """

    def __init__(
        self,
        start: pd.Timestamp,
        limit: pd.Timestamp,
        calendar: xcals.ExchangeCalendar,
        delay: pd.Timedelta,
        evaluated: bool = False,
    ):
        # pylint: disable=too-many-arguments
        self.start = start  # inspected by tests.

        time_date = "date" if helpers.is_date(start) else "time"
        cannot_must = "cannot" if helpers.is_date(start) else "must"
        later_earlier = "a later" if helpers.is_date(start) else "an earlier"
        evaluate_be = "evaluate to" if evaluated else "be"
        evaluates_received = "evaluates to" if evaluated else "received as"

        self._msg = (
            f"`start` {cannot_must} {evaluate_be} {later_earlier} {time_date} than the"
            f" latest {time_date} for which prices are available.\nThe latest"
            f" {time_date} for which prices are available for calendar"
            f" '{calendar.name}' is {fts(limit)}, although `start`"
            f" {evaluates_received} {fts(start)}."
        )
        if time_date == "time" and delay:
            self._msg += f"\nNote: lead_symbol has a delay of {delay}."


# --- 'start'/'end' earlier than earliest session/minute for which prices available ---


class _TooEarlyError(PricesUnavailableError):
    """start/end is or resolves to session/minute before first available."""

    param: Literal["start", "end"]

    def __init__(
        self,
        drg: _Getter,
        limit: pd.Timestamp,
        ts: pd.Timestamp | None = None,
    ):
        if ts is not None:
            is_session = helpers.is_date(ts)
        else:
            if TYPE_CHECKING:
                assert isinstance(drg.interval, BI)
                assert isinstance(intervals.ONE_DAY, BI)
            is_session = drg.interval is intervals.ONE_DAY
        self._ts_type = "session" if is_session else "minute"
        self._limit = limit
        self._ts = ts
        self._c = drg.cal

    def __str__(self) -> str:
        if self._ts is not None:
            insert = f"({helpers.fts(self._ts)}) is earlier"
        else:
            insert = f"would resolve to an earlier {self._ts_type}"
        msg = (
            f"Prices unavailable as {self.param} {insert} than the"
            f" earliest {self._ts_type} for which price data is available."
            f" The earliest {self._ts_type} for which prices are available"
            f" is {helpers.fts(self._limit)}."
        )
        return msg


class StartTooEarlyError(_TooEarlyError):
    """start is or resolves to session/minute before first available."""

    param: Literal["start"] = "start"

    def __init__(
        self,
        drg: _Getter,
        limit: pd.Timestamp,
        ts: pd.Timestamp | None = None,
        add_a_row: bool | None = False,
    ):
        # add_a_row True if known that start before first available
        # session/datetime due solely to add_a_row option being True.
        self._add_a_row = add_a_row
        super().__init__(drg, limit, ts)

    def __str__(self) -> str:
        msg = super().__str__()
        if self._add_a_row:
            msg += (
                "\nNB range start falls earlier than first available"
                f" {self._ts_type} due only to 'add_a_row=True'."
            )
        return msg


class EndTooEarlyError(_TooEarlyError):
    """end of daterange resolves to session/minute before first available."""

    param: Literal["end"] = "end"


# 'start' later than 'end'.


class StartNotEarlierThanEnd(PricesUnavailableError):
    """`start` parameter not earlier than `end` (and should be).

    Parameters
    ----------
    start
        `start` as received, adjusted only for `tzin`.

    end
        `end` as received, adjusted only for `tzin`.
    """

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        self._msg = (
            "`start` should be, or evaluate to, a value earlier than `end`, although"
            f" receieved `start` as '{fts(start)}' and `end` as '{fts(end)}'."
        )


# Datetime too early or too late.


def _datetime_ool_msg(
    ts: pd.Timestamp,
    limit: pd.Timestamp,
    side: Literal["left", "right"],
    param_name: str,
) -> str:
    """Error message advising a date or time is out-of-limit."""
    insert = "earlier" if side == "left" else "later"
    insert2 = "first" if side == "left" else "most recent"
    return (
        f"`{param_name}` cannot be {insert} than the {insert2} {param_name} for which"
        f" prices are available. {insert2.capitalize()} {param_name} for which"
        f" prices are available is {limit} although `{param_name}` received as {ts}."
    )


class DatetimeTooEarlyError(PricesUnavailableError):
    """Client input earlier than earliest valid date or time."""

    def __init__(self, ts: pd.Timestamp, limit: pd.Timestamp, param_name: str):
        self._msg = _datetime_ool_msg(ts, limit, "left", param_name)


class DatetimeTooLateError(PricesUnavailableError):
    """Client input later than latest valid date or time."""

    def __init__(self, ts: pd.Timestamp, limit: pd.Timestamp, param_name: str):
        self._msg = _datetime_ool_msg(ts, limit, "right", param_name)


class PricesIntradayUnavailableError(PricesUnavailableError):
    """Prices unavailable to evaluate at an intraday interval."""

    def __init__(self, prices: PricesBase):
        self._prices = prices
        self.anchor = self._prices.gpp.anchor
        self.interval = self._prices.gpp.ds_interval

        bis = self._prices._bis_valid
        if self.anchor is mptypes.Anchor.WORKBACK:
            bis = self._prices._bis_no_partial_indices(bis)
        self.bis = bis

        self._drg = self._prices.gpp.drg_intraday_no_limit
        if self.bis:
            self._bi = self.bis[-1]
            self._drg.interval = self._bi
            self._dr = self._drg.daterange[0]

    @property
    def _earliest_minute_available(self) -> pd.Timestamp:
        limit = self._prices.limits[self._bi][0]
        cal = self._drg.cal
        return cal.minute_to_trading_minute(limit, "next")

    @property
    def _earliest_availability(self) -> str:
        msg = (
            f"\nThe earliest minute from which data is available"
            f" at {self._bi} is {self._earliest_minute_available}, although"
            f" at this base interval the requested period evaluates to"
            f" {self._dr}."
        )
        return msg

    @property
    def _strict_advice(self) -> str:
        msg = (
            f"\nData is available from {self._earliest_minute_available} through to"
            f" the end of the requested period. Consider passing `strict` as False"
            f" to return prices for this part of the period."
        )
        return msg

    @property
    def _s1(self) -> str:
        insert = (
            f"interval {self.interval}"
            if self.interval is not None
            else "an inferred interval"
        )
        s = (
            "Data is unavailable at a sufficiently low base interval to"
            f" evaluate prices at {insert} anchored '{self.anchor}'."
        )
        return s

    @property
    def _s2(self) -> str:
        s2_ = "intervals"
        insert = ""
        if self.interval is not None:
            s2_ += f" that are a factor of {self.interval}"
            insert = " and"
        if self.anchor is mptypes.Anchor.WORKBACK:
            s2_ += f"{insert} that have no partial trading indices"
        elif not self._prices.has_single_calendar:
            s2_ += f"{insert} for which timestamps of all calendars are synchronised"

        if self.bis:
            s = f"\nBase {s2_}:\n\t{self.bis}."
            s += self._earliest_availability
        else:
            s = f"There are no base {s2_}."
        return s

    @property
    def _pp(self) -> str:
        return f"\nPeriod evaluated from parameters: {self._prices.gpp.pp_raw}."

    def __str__(self) -> str:
        s = self._s1 + self._s2 + self._pp
        if self._earliest_minute_available < self._dr[1]:
            s += self._strict_advice
        return s


class LastIndiceInaccurateError(PricesIntradayUnavailableError):
    """Cannot serve full period with greatest possible end accuracy."""

    def __init__(
        self,
        prices: "PricesBase",
        bis_period: list[BI],
        bis_accuracy: list[BI],
    ):
        self._prices = prices
        self.bis_period = bis_period
        self.bis_accuracy = bis_accuracy

        self._bi = self.bis_accuracy[-1]
        self._drg = self._prices.gpp.drg_intraday_no_limit
        self._drg.interval = self._bi
        # getting this with strict as False.
        self._dr = self._drg.daterange[0]

    @property
    def _strict_advice(self) -> str:
        msg = (
            "\nData that can express the period end with the greatest possible"
            f" accuracy is available from {self._earliest_minute_available}."
            " Pass `strict` as False to return prices for this part of the period."
        )
        return msg

    def __str__(self) -> str:
        if self.bis_period:
            insert = "" if self._prices.has_single_calendar else " synchronised"
            msg = (
                f"Full period available at the following{insert}"
                " intraday base intervals although these do not allow for"
                " representing the end indice with the greatest possible"
                f" accuracy:\n\t{self.bis_period}.\n"
            )
        else:
            msg = (
                "Full period not available at any synchronised intraday"
                " base interval. "
            )

        msg += (
            "The following base intervals could represent the end indice with"
            " the greatest possible accuracy although have insufficient"
            f" data available to cover the full period:\n\t{self.bis_accuracy}."
        )
        msg += self._earliest_availability
        msg += self._pp
        msg += self._strict_advice
        msg += "\nAlternatively, consider"
        if self._prices.gpp.anchor is mptypes.Anchor.OPEN:
            msg += " creating a composite table (pass `composite` as True) or"
        msg += " passing `priority` as 'period'."
        return msg


class PricesUnavailableIntervalError(PricesUnavailableError):
    """Base for interval related price availability errors."""


class PricesUnavailableIntervalDurationError(PricesUnavailableIntervalError):
    """Interval greater than period duration.

    Interval greater than a period duration defined in terms of trading
    time.
    """

    def __init__(
        self,
        duration: pd.Timedelta,
        drg: GetterIntraday,
    ):
        self._msg = (
            f"Period duration shorter than interval. Interval is {drg.final_interval}"
            f" although period duration is only {duration}."
            f"\nDuration evaluated from parameters: {drg.pp}."
        )


class PricesUnavailableIntervalPeriodError(PricesUnavailableIntervalError):
    """Interval longer than evaluted period."""

    def __init__(
        self,
        drg: _Getter,
        start: pd.Timestamp,
        end: pd.Timestamp,
        duration: pd.Timedelta | None = None,
    ):
        duration_insert = (
            "" if duration is None else f"\nPeriod duration evaluated as {duration}."
        )
        self._msg = (
            f"Period does not span a full indice of length {drg.final_interval}."
            f"{duration_insert}\nPeriod start date evaluated as {start}.\nPeriod end"
            f" date evaluated as {end}.\nPeriod dates evaluated from parameters:"
            f" {drg.pp}."
        )


class PricesUnavailableDOIntervalPeriodError(PricesUnavailableIntervalPeriodError):
    """DOInterval longer than evaluted period."""

    def __init__(
        self,
        latest_start: pd.Timestamp,
        drg: GetterDaily,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        self._msg = (
            "Period evaluated as being shorter than one interval at"
            f" {drg.final_interval}.\nPeriod start date evaluates as {start} although"
            f" needs to be no later than {latest_start} to cover one interval."
            f"\nPeriod end date evaluates to {end}."
            f"\nPeriod evaluated from parameters: {drg.pp}."
        )


class PricesUnvailableDurationConflict(PricesUnavailableIntervalError):
    """Interval daily although duration in hours and/or minutes."""

    def __init__(self):
        self._msg = (
            "Duration cannot be defined in terms of hours and/or"
            " minutes when interval is daily or higher."
        )


class PricesUnvailableInferredIntervalError(PricesUnavailableIntervalError):
    """Interval inferred as intraday although only daily data available.

    Interval inferred as intraday although request can only be met with
    daily data.
    """

    def __init__(self, pp: mptypes.PP):
        self._msg = (
            "Given the parameters receieved, the interval was inferred as intraday"
            " although the request can only be met with daily data. To return daily"
            " prices pass `interval` as a daily interval, for example '1D'."
            "\nNB. The interval will only be inferred as daily if `end` and `start` are"
            " defined (if passed) as sessions (timezone naive and with time component"
            " as 00:00) and any duration is defined in terms of either `days` or"
            " `weeks`, `months` and `years`. Also, if both `start` and `end` are passed"
            " then the distance between them should be no less than 6 sessions."
            f"\nPeriod parameters were evaluted as {pp}."
        )


class PricesUnavailableIntervalResampleError(PricesUnavailableIntervalError):
    """Insufficient data to resample to single indice."""

    def __init__(
        self,
        offset: pd.offsets.BaseOffset,  # type: ignore[name-defined]
        pt: PTIntraday,
    ):
        self._msg = (
            f"Insufficient data to resample to {offset}.\nData from"
            f" {pt.first_ts} to {pt.last_ts} is insufficient to"
            f" create a single indice at {offset} when data is"
            f" anchored to 'open'. Try anchoring on 'workback'."
        )


class CompositePricesCalendarError(PricesUnavailableError):
    """Unable to create composite table."""

    def __str__(self) -> str:
        return (
            "Unable to create a composite table from a daily and an"
            " intraday table due to a calendar conflict. Exchange times of"
            " consecutive sessions of different calendars overlap (the end"
            " of a session for one calendar falls after the start of the"
            " next session of another calendar.)"
        )


class PricesUnavailableFromSourceError(PricesUnavailableError):
    """Prices unavailable from source for passed parameters.

    Attributes
    ----------
    params: dict
        Parameters passed to source, for which no prices returned.

    rtrn: Any
        Object returned by source to prices request.
    """

    def __init__(self, params, rtrn):
        self.params = params
        self.rtrn = rtrn
        self._msg = f"Prices unavailable from source for parameters: {params}."
        if self.rtrn is not None:
            self._msg += f" Request returned: {rtrn}."


class CalendarError(MarketPricesError):
    """Error getting or accessing calendar."""

    def __init__(self, msg: str):
        self._msg = msg


class CalendarExpiredError(CalendarError):
    """Calendar's right bound is earlier than required."""

    def __init__(
        self,
        calendar: xcals.ExchangeCalendar,
        bound: Any,
        required: Any,
    ):
        self._msg = (
            f"Calendar '{calendar.name}' has expired. The calendar's right bound is"
            f" {bound} although operation requires calendar to be valid through to at"
            f" least {required}."
        )


class CalendarTooShortError(CalendarError):
    """Calendar is too short to support intraday price history."""

    def __init__(self, calendar: xcals.ExchangeCalendar, limit: pd.Timestamp):
        self._msg = (
            f"Calendar '{calendar.name}' is too short to support all available intraday"
            f" price history. Calendar starts '{calendar.first_minute}' whilst earliest"
            f" minute for which intraday price history is available is '{limit}'"
            " (all calendars must be valid over at least the period for which intraday"
            " price data is available)."
        )


class CompositeIndexConflict(MarketPricesError):
    """Some indices of composite trading index are misaligned."""

    _msg = (
        "At least one indice of the trading index would partially overlap another."
        " Pass `raise_overlapping` as False to supress this error."
    )


class CompositeIndexCalendarConflict(MarketPricesError):
    """Indices of a composite calendar conflict with other indices of same calendar."""

    def __init__(self, calendar: ExchangeCalendar):
        self._msg = (
            "Unable to create a composite trading index as indices of calendar"
            f" '{calendar.name}' would overlap. This can occur when the interval is"
            " longer than a break or the gap between one session's close and the"
            " next session's open."
        )


class IndexConflictError(MarketPricesError):
    """Indices of same interval are misaligned."""

    def __init__(
        self, calendar: ExchangeCalendar, non_compat_sessions: pd.DatetimeIndex
    ):
        self._msg = (
            f"Table index not compatible with calendar {calendar.name}. At least one"
            " table indice would conflict with a calendar indice for each of the"
            f" following sessions: \n{non_compat_sessions}."
        )


class TutorialDataUnavailableError(MarketPricesError):
    """No sessions conform with the requested restrictions.

    There are an insufficient number of consecutive sessions of the
    requested length between the requested limits.

    Parameters
    ----------
    start
        Start limit from which can evaluate conforming sessions.

    end
        End limit to which can evaluate conforming sessions.

    cal_param
        Calendars with sessions that are required at lengths descibed
        by `lengths_param`.

    lengths_param
        Required lengths of sessions of calendars receieved as
        `cal_param`.
    """

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cal_param: list[xcals.ExchangeCalendar] | calutils.CompositeCalendar,
        lengths_param: pd.Timedelta | list[pd.Timedelta] | list[list[pd.Timedelta]],
    ):
        self._msg = (
            "The requested number of consecutive sessions of the requested length(s)"
            f" are not available from {start} through {end}."
            f"\nThe `calendars` or `cc` parameter was receieved as {cal_param}."
            "\nThe `sessions_lengths` or `session_length` parameter was receieved"
            f" as {lengths_param}."
        )


class PricesWarning(UserWarning):
    """User warning to advise of operations undertaken to price data."""

    _msg = "Prices Warning"  # subclass should override

    def __str__(self) -> str:
        return self._msg


class PricesMissingWarning(PricesWarning):
    """Intraday prices missing for one or more sessions."""

    def __init__(self, symbol: str, bi: BI, sessions: pd.DatetimeIndex, source: str):
        date_format = "%Y-%m-%d"
        sessions = sessions.format(date_format=date_format)
        self._msg = (
            f"Prices from {source} are missing for '{symbol}' at the"
            f" base interval '{bi}' for the following sessions: {sessions}."
        )


class DuplicateIndexWarning(PricesWarning):
    """Warning advising user of removal of duplicate indice(s)."""

    def __init__(self, duplicates: pd.DataFrame, symbol: str):
        self._msg = (
            f"\nThe following indices for symbol '{symbol}' have"
            f" been removed as they were duplicated in data receieved from"
            f" source:\n{duplicates}."
        )


class IntervalIrregularWarning(PricesWarning):
    """Advises user that interval is irregular due to indice curtailing."""

    def __str__(self) -> str:
        return (
            "\nPriceTable interval is irregular. One or more indices were"
            " curtailed to prevent the last indice assigned to a (sub)session"
            " from overlapping with the first indice of the following"
            " (sub)session.\nUse .pt.indices_length and .pt.by_indice_length"
            " to interrogate."
        )


class CalendarTooShortWarning(PricesWarning):
    """Warning that calendar is too short to support all price history."""

    def __init__(self, calendar: xcals.ExchangeCalendar, limit: pd.Timestamp):
        self._msg = (
            f"Calendar '{calendar.name}' is too short to support all available price"
            f" history. Calendar starts '{calendar.first_session}' whilst earliest date"
            f" for which price history is available is '{limit}'. Prices will not be"
            f" available for any date prior to {calendar.first_session}."
        )


warnings.simplefilter("always", DuplicateIndexWarning)
warnings.simplefilter("always", PricesMissingWarning)
warnings.simplefilter("default", IntervalIrregularWarning)
warnings.simplefilter("always", CalendarTooShortWarning)
