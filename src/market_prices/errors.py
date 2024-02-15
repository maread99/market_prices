"""Error classes."""

from __future__ import annotations

import functools
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
    """`start` or `end` parm earlier/later than calendar's left/right bound.

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

    param: Literal["start", "end"] = "end"
    left_bound: bool = True

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
        if self.left_bound:
            self._bound = calendar.first_session if is_date else calendar.first_minute
        else:
            self._bound = calendar.last_session if is_date else calendar.last_minute

    def __str__(self) -> str:
        earlier_later = "earlier" if self.left_bound else "later"
        earliest_latest = "earliest" if self.left_bound else "latest"
        if self._ts is not None:
            insert = f"({fts(self._ts)}) is {earlier_later}"
        else:
            insert = f"would resolve to an {earlier_later} {self._ts_type}"
        insert2 = "date" if self._ts_type == "date" else "minute or date"
        msg = (
            f"Prices unavailable as {self.param} {insert} than the"
            f" {earliest_latest} {self._ts_type} of calendar '{self._c.name}'."
            f" The calendar's {earliest_latest} {self._ts_type} is {fts(self._bound)}"
        )
        if self.left_bound:
            msg += (
                f" (this bound should coincide with the earliest {insert2} for"
                " which price data is available)."
            )
        else:
            msg += "."
        return msg


class StartOutOfBoundsError(_OutOfBoundsError):
    """'start' parameter earlier than calendar's left bound."""

    param = "start"


class EndOutOfBoundsError(_OutOfBoundsError):
    """'end' parameter earlier than calendar's left bound."""


class EndOutOfBoundsRightError(_OutOfBoundsError):
    """'end' parameter earlier than calendar's left bound."""

    left_bound = False


# 'start'/'end' later than right limit (latest session/minute for which prices available)


class _TooLateError(PricesUnavailableError):
    """start/end is or resolves to session/minute after last available.

    Parameters
    ----------
    ts
        'start'/'end' parameter either as received, adjusted only for
        `tzin`, or start/end of daterange as subsequently evaluated.

    evaluated
        True: `ts` is start/end of an evaluated date range.
        False: `ts` is start/end parameter as receieved (default).
    """

    param: Literal["start", "end"]

    def __init__(
        self,
        ts: pd.Timestamp,
        limit: pd.Timestamp,
        interval: intervals.TDInterval,
        delay: pd.Timedelta | None = None,
        evaluated: bool = False,
    ):
        # pylint: disable=too-many-arguments
        self.ts = ts  # inspected by tests.
        time_date = "date" if helpers.is_date(ts) else "time"
        evaluate_be = "evaluate to" if evaluated else "be"
        evaluates_received = "evaluates to" if evaluated else "received as"

        self._msg = (
            f"`{self.param}` cannot {evaluate_be} a later {time_date} than the"
            f" latest {time_date} for which prices are available.\nThe latest"
            f" {time_date} for which prices are available for interval"
            f" '{interval}' is {fts(limit)}, although `{self.param}`"
            f" {evaluates_received} {fts(ts)}."
        )
        if time_date == "time" and delay:
            self._msg += f"\nNote: lead_symbol has a delay of {delay}."


class StartTooLateError(_TooLateError):
    """'start' param later than right limit for which prices available."""

    param = "start"


class EndTooLateError(_TooLateError):
    """'end' param later than right limit for which prices available."""

    param = "end"


# --- 'start'/'end' earlier than earliest session/minute for which prices available ---


class _TooEarlyError(PricesUnavailableError):
    """start/end evaluates to session/minute earlier than first available."""

    param: Literal["start", "end"]

    def __init__(
        self,
        ts: pd.Timestamp,
        limit: pd.Timestamp,
    ):
        is_session = helpers.is_date(ts)
        self._ts_type = "session" if is_session else "minute"
        self._limit = limit
        self._ts = ts

    def __str__(self) -> str:
        msg = (
            f"Prices unavailable as {self.param} evaluates to {fts(self._ts)}"
            f" which is earlier than the earliest {self._ts_type} for which price"
            f" data is available. The earliest {self._ts_type} for which prices are"
            f" available is {fts(self._limit)}."
        )
        return msg


class StartTooEarlyError(_TooEarlyError):
    """start evaluates to session/minute earlier than first available."""

    param: Literal["start"] = "start"

    def __init__(
        self,
        ts: pd.Timestamp,
        limit: pd.Timestamp,
        add_a_row: bool | None = False,
    ):
        # add_a_row True if known that start before first available
        # session/datetime due solely to add_a_row option being True.
        self._add_a_row = add_a_row
        super().__init__(ts, limit)

    def __str__(self) -> str:
        msg = super().__str__()
        if self._add_a_row:
            msg += (
                "\nNB The evaluated start falls earlier than first available"
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


class PricesDailyIntervalError(PricesUnavailableError):
    """Raises if request daily price table when daily not a base interval."""

    _msg = (
        "Daily and monthly prices unavailable as prices class does not have a"
        " daily base interval defined."
    )

    def __init__(self, msg: str | None = None):
        if msg is not None:
            self._msg = msg


class PricesIntradayIntervalError(PricesUnavailableError):
    """Raises if request intraday price table although no intraday base intervals."""

    _msg = (
        "Intraday prices unavailable as prices class does not have any intraday"
        " base intervals defined."
    )


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

    @functools.cached_property
    def _earliest_minute_available(self) -> pd.Timestamp:
        limit = self._prices.limits[self._bi][0]
        cal = self._drg.cal
        return cal.minute_to_trading_minute(limit, "next")

    @functools.cached_property
    def _latest_minute_available(self) -> pd.Timestamp:
        limit = self._prices.limits[self._bi][1]
        cal = self._drg.cal
        if cal.is_trading_minute(limit):
            return limit
        return cal.minute_to_trading_minute(limit, "previous") + helpers.ONE_MIN

    @functools.cached_property
    def end_before_ll(self) -> bool:
        return self._dr[1] < self._earliest_minute_available

    @functools.cached_property
    def start_after_rl(self) -> bool:
        return self._latest_minute_available < self._dr[0]

    @property
    def _dr_available(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self._earliest_minute_available, self._latest_minute_available)

    @property
    def _availability(self) -> str:
        msg = (
            f"\nThe period over which data is available at {self._bi} is"
            f" {self._dr_available}, although at this base interval the"
            f" requested period evaluates to {self._dr}."
        )
        return msg

    def _get_strict_advices(self) -> str:
        """Advices of part of requested period for which prices are available."""
        if self.end_before_ll or self.start_after_rl:
            return ""

        start_before_ll = self._dr[0] < self._earliest_minute_available
        end_after_rl = self._dr_available[1] < self._dr[1]
        if start_before_ll and end_after_rl:
            s = (
                f"from {fts(self._earliest_minute_available)} through"
                f" {fts(self._latest_minute_available)}."
            )
        elif start_before_ll:
            s = (
                f"from {fts(self._earliest_minute_available)} through to the end"
                " of the requested period."
            )
        else:
            assert end_after_rl
            s = (
                "from the start of the requested period through to"
                f" {fts(self._latest_minute_available)}."
            )
        s += (
            " Consider passing `strict` as False to return prices for this"
            " part of the period."
        )
        return s

    @property
    def _strict_advices(self) -> str:
        advices = self._get_strict_advices()
        if advices:
            advices = "\nData is available " + advices
        return advices

    @property
    def _s0(self) -> str:
        if self.interval is None:
            insert = "an inferred interval"
        else:
            insert = f"interval {self.interval}"
        s = (
            "Data is unavailable at a sufficiently low base interval to"
            f" evaluate prices at {insert} anchored '{self.anchor}'."
        )
        return s

    @property
    def _s0_no_data_available(self) -> str:
        start_end = "start" if self.start_after_rl else "end"
        later_earlier = "later" if self.start_after_rl else "earlier"
        latest_earliest = "latest" if self.start_after_rl else "earliest"
        return (
            f"The {start_end} of the requested period is {later_earlier}"
            f" than the {latest_earliest} timestamp at which intraday data"
            " is available for any base interval."
        )

    @property
    def _s1(self) -> str:
        if self.bis and (self.start_after_rl or self.end_before_ll):
            return self._s0_no_data_available
        return self._s0

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
            s += self._availability
        else:
            s = f"There are no base {s2_}."
        return s

    @property
    def _pp(self) -> str:
        return f"\nPeriod evaluated from parameters: {self._prices.gpp.pp_raw}."

    def __str__(self) -> str:
        return self._s1 + self._s2 + self._pp + self._strict_advices


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
    def _strict_advices(self) -> str:
        advices = self._get_strict_advices()
        if advices:
            start = (
                "\nData that can express the period end with the greatest"
                " possible accuracy is available "
            )
            advices = start + advices
        return advices

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
        msg += self._availability
        msg += self._pp
        msg += self._strict_advices
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


class PricesUnavailableInferredIntervalError(PricesUnavailableIntervalError):
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


class MethodUnavailableNoDailyInterval(PricesUnavailableError):
    """Called method requires daily data.

    Raised by a method that requires daily data although daily interval is
    not available for the prices subclass.
    """

    def __init__(self, name: str):
        self._msg = (
            f"`{name}` is not available as this method requires daily data although"
            " a daily base interval is not available to this prices class."
        )


class PriceAtUnavailableError(PricesUnavailableError):
    """Prices unavailable to serve a `price_at` request."""

    def __init__(self, minute: pd.Timestamp, num_sessions: int):
        self._msg = (
            "`price_at` cannot return data as intraday data is not available"
            f" at '{minute}' and the sessions of the underlying symbols"
            " continuously overlap (it is not possible to evaluate prices"
            f" at any specific minute during a period covering the {num_sessions}"
            f" composite sessions that immediately preceed '{minute}')."
        )


class PriceAtUnavailableDailyIntervalError(PricesUnavailableError):
    """To serve `price_at` daily prices required but not available."""

    def __init__(self, minute: pd.Timestamp):
        self._msg = (
            "`price_at` cannot return prices as intraday data is not available"
            f" at '{minute}' (for at least one symbol) and daily data is not"
            " available to the prices class."
        )


class PriceAtUnavailableLimitError(PricesUnavailableError):
    """Prices unavailable to serve a `price_at` request due to left limit."""

    def __init__(self, minute: pd.Timestamp, limit: pd.Timestamp):
        self._msg = (
            "`price_at` cannot return data as intraday data is not available"
            f" at '{minute}' and to evaluate a price at '{minute}' would require,"
            " for least one symbol, data ealier than the earliest session for"
            f" which daily data is available ({limit})"
        )


class PriceAtUnavailableLivePricesError(PricesUnavailableError):
    """Live prices unavailable to serve a `price_at` request for 'now'."""

    def __init__(self):
        self._msg = "'minute' parameter can only be None if live prices are available."


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


class PricesUnavailableForExport(PricesUnavailableError):
    """Prices unavailable to export."""

    def __init__(self, interval: intervals.PTInterval, kwargs: dict | None = None):

        msg = (
            "It was not possible to export prices as an error was raised"
            f" when prices were requested for interval {interval}. The"
            " error is included at the top of the traceback."
        )

        if kwargs is not None:
            msg += f" Prices were requested with the following kwargs: {kwargs}"

        msg += "\nNB prices have not been exported for any interval."
        self._msg = msg


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
        sessions_ = sessions.strftime(date_format).tolist()
        self._msg = (
            f"Prices from {source} are missing for '{symbol}' at the"
            f" base interval '{bi}' for the following sessions: {sessions_}."
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
