"""Base implementation for price processing.

PricesBase(metaclass=abc.ABCMeta):
    ABC for serving price data obtained from a source.
"""

import abc
import collections
import contextlib
import copy
import dataclasses
import datetime
import functools
import itertools
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from valimp import Coerce, Parser, parse

from market_prices import data
from market_prices import daterange as dr
from market_prices import errors, helpers, intervals, mptypes, parsing, pt
from market_prices.helpers import UTC
from market_prices.intervals import (
    BI,
    PTInterval,
    TDInterval,
    _BaseIntervalMeta,
    to_ptinterval,
)
from market_prices.mptypes import Alignment, Anchor, OpenEnd, Priority, Symbols
from market_prices.utils import calendar_utils as calutils
from market_prices.utils import pandas_utils as pdutils

# pylint: disable=too-many-lines

# Helper functions available to subclasses


def fill_reindexed(
    df: pd.DataFrame,
    calendar: xcals.ExchangeCalendar,
    bi: intervals.BI,
    symbol: str,
    source: str,
) -> tuple[pd.DataFrame, list[errors.PricesMissingWarning]]:
    """Fill missing intraday prices in reindexed data for a single symbol.

    Parameters
    ----------
    df
        Prices to be filled.

    calendar
        Calendar against which prices were reindexed.

    bi
        Intraday interval represented by each indice.

    symbol
        Symbol corresponding to price data. Will by included to any
        warning advising of missing price data.

    source
        Source of price data. Value be included to any warning advising
        of missing price data.

    Returns
    -------
    tuple[pd.DataFrame, list[errors.PricesMissingWarning]]
        [0] Reindexed price data
        [1] List of any missing prices warnings.
    """
    warnings = []
    # fill index grouped by day or session to avoid filling a session's
    # initial values with prior session's close value.
    # pylint: disable=too-many-locals
    na_rows = df.close.isna()
    if not na_rows.any():
        return df, warnings

    first_session = calendar.minute_to_session(df.index[0], "next")
    last_session = calendar.minute_to_session(df.index[-1], "previous")
    opens = calendar.opens.loc[first_session:last_session]
    closes = calendar.closes.loc[first_session:last_session]

    grouper: pd.Series | pd.Grouper
    if (opens.dt.day != closes.dt.day).any():
        grouper = pd.Series(index=df.index, dtype="int32")
        for i, (open_, close) in enumerate(zip(opens, closes)):
            grouper[(grouper.index >= open_) & (grouper.index < close)] = i
    else:
        # shortcut
        grouper = pd.Grouper(freq="D")

    close_groupby = df["close"].groupby(by=grouper)
    adj_close = close_groupby.ffill()
    bv = adj_close.isna()
    if bv.any():
        open_groupby = df["open"].groupby(by=grouper)
        # bfill open to fill any missing initial rows for a session with
        # session's first registered price
        adj_close[bv] = open_groupby.bfill()[bv]

    if adj_close.isna().any():
        # at least one whole session missing prices
        bv = adj_close.isna()
        missing_sessions_groupby = adj_close[bv].groupby(by=grouper)
        # assert all values for missing sessions are missing
        assert missing_sessions_groupby.value_counts().empty
        group_sizes = missing_sessions_groupby.size()
        bv_sizes = group_sizes != 0  # pylint: disable=compare-to-zero
        sessions = group_sizes[bv_sizes].index
        if isinstance(sessions[0], (float, int)):
            sessions = pd.DatetimeIndex([opens.index[int(i)] for i in sessions])
        # fill forwards in first instance
        adj_close = adj_close.ffill()
        if adj_close.isna().any():
            # Values missing for at least one session at start of df, fill backwards
            bv = adj_close.isna()
            adj_close[bv] = df.open.bfill()[bv]
        warnings.append(errors.PricesMissingWarning(symbol, bi, sessions, source))

    df.loc[:, "close"] = adj_close
    for col in ["open", "high", "low"]:
        df.loc[na_rows, [col]] = adj_close[na_rows]

    df["volume"] = df["volume"].fillna(value=0)
    return df, warnings


def fill_reindexed_daily(
    df: pd.DataFrame,
    cal: xcals.ExchangeCalendar,
    mindate: pd.Timestamp,
    delay: pd.Timedelta,
    symbol: str,
    source: str,
) -> tuple[pd.DataFrame, list[errors.PricesMissingWarning]]:
    """Fill missing daily prices in reindexed data for a single symbol.

    Parameters
    ----------
    df
        Prices to be filled.

    calendar
        Calendar against which prices were reindexed.

    mindate
        Earliest session for which prices must be included

    delay
        Real time delay for symbol (used to ascertain whether prices are
        missing for any currently live session).

    symbol
        Symbol corresponding to price data. Will by included to any
        warning advising of missing price data.

    source
        Source of price data. Value be included to any warning advising
        of missing price data.

    Returns
    -------
    tuple[pd.DataFrame, list[errors.PricesMissingWarning]]
        [0] Reindexed price data
        [1] List of any missing prices warnings.
    """
    warnings = []
    na_rows = df.close.isna() & (df.index > mindate)
    if not na_rows.any():
        return df, warnings

    # do not fill prices for any most recent sessions for which prices would not be
    # expected to be available. NB this considers more than one session as for some
    # funds prices many not be available for a number of days (in which case can pass
    # the corresponding delay as a very high value).
    i = 1
    len_df = len(df)
    while i <= len_df:
        if na_rows.iloc[-i] and helpers.now() <= cal.session_open(df.index[-i]) + delay:
            na_rows.iloc[-i] = False
            if not na_rows.any():
                return df, warnings
        else:
            break
        i += 1

    # fill
    adj_close = df["close"].ffill()
    bv = adj_close.isna()
    if bv.any():
        # bfill open to fill any missing initial rows with next available
        # session's open
        adj_close[bv] = df["open"].bfill()[bv]

    df.loc[na_rows, ["open", "high", "low", "close"]] = adj_close[na_rows]
    df.loc[na_rows, "volume"] = 0
    warnings.append(
        errors.PricesMissingWarning(
            symbol, TDInterval.D1, na_rows.index[na_rows], source
        )
    )
    return df, warnings


def adjust_high_low(df) -> pd.DataFrame:
    """Adjust data so that ohlc values are congruent.

    Assumes close values over incongruent high or low.
    Assumes high or low values over incongruent open.

    For any row:
        If 'close' higher than 'high', sets 'high' to 'close'.
        If 'close' lower than 'low', sets 'low' to 'close'.
        If 'open' higher than 'high', sets 'open' to 'high'.
        If 'open' lower than 'low', sets 'open' to 'low'.

    Notes
    -----
    Yahoo API has a nasty bug where occassionally the open price of the
    most recent day takes the same value as the prior day (as of
    06/11/20 only noticed issue on daily data). However, the high
    and low values will register the actual high or low of the day.
    Consequently it's possible for the day high to otherwise be lower
    than the open (observed) or the day low to be higher than the open
    (assumed). This in turn causes bqplot OHLC plots to fail (as the
    corresponding bar can not be created.)
    """
    bv = df.open > df.high
    if bv.any():
        df.loc[bv, "open"] = df.loc[bv, "high"]

    bv = df.open < df.low
    if bv.any():
        df.loc[bv, "open"] = df.loc[bv, "low"]

    bv = df.close > df.high
    if bv.any():
        df.loc[bv, "high"] = df.loc[bv, "close"]

    bv = df.close < df.low
    if bv.any():
        df.loc[bv, "low"] = df.loc[bv, "close"]

    return df


def get_columns_multiindex(
    symbol: str, columns: pd.Index | None = None
) -> pd.MultiIndex:
    """Get Multiindex to represent columns for a single symbol.

    Parameters
    ----------
    symbol
        Symbol to be represented in multiindex.

    index
        Any existing columns index

    Returns
    -------
    multiindex
        Multiindex with level 0 as `pd.Index` representing 'symbol' and
        level 1 as any existing `index`. If `index` not passed then level 1
        will be populated by a `pd.Index` representing default ordered
        columns.
    """
    if columns is None:
        columns = pd.Index(helpers.AGG_FUNCS.keys())
    parts = [[symbol], columns]
    return pd.MultiIndex.from_product(parts, names=["symbol", ""])


# Functions employed by PricesBase


def create_composite(
    first: tuple[pd.DataFrame, pd.Interval], second: tuple[pd.DataFrame, pd.Interval]
) -> pd.DataFrame:
    """Create composite DataFrame from rows of two intraday price tables.

    Composite price table comprises indices from specified interval
    of an intraday prices table to specified interval of another.

    If interval length of each price table is not the same then index of
    composite table will not have a regular interval length.

    Row indices of composite table are unique with no duplication or
    overlap of any interval of either table.

    Parameters
    ----------
    first : tuple[pd.DataFrame, pd.Interval]
        tuple[0]
            First price table
        tuple[1]
            Indice of first price table to serve as first indice of the
            composite table.

    second: tuple[pd.DataFrame, pd.Interval]
        tuple[0]
            Second price table
        tuple[1]
            Indice of second price table to serve as last indice of the
            composite table.

    Raises
    ------
    ValueError
        If `first` does not preceed and partially overlap `second` (i.e.
        ValueError will be raises if either table fully overlaps the
        other).

    IndexError
        If a common indice can not be identified.
    """
    df1 = first[0]
    df1_start = first[1]
    df2 = second[0]
    df2_end = second[1]
    df1_table_end, df2_table_end = df1.index[-1].left, df2.index[-1].left

    if not df1.pt.first_ts < df2.pt.first_ts < df1_table_end < df2_table_end:
        raise ValueError("`first` table must preceed and partially overlap `second`.")

    common_indices = df1.index.left.isin(df2.index.left)
    if not common_indices.any():
        raise IndexError("Unable to identify a common indice.")
    split = df1[common_indices].iloc[-1].name.left  # split on last common timestamp
    # typing - pd.Interval does have attribute .left
    df1_subset = df1.loc[df1_start.left : split][:-1]
    df2_subset = df2.loc[split : df2_end.left]
    df = pd.concat([df1_subset, df2_subset])
    assert not df.index.has_duplicates
    assert not df.index.is_overlapping
    return df


class PricesBase(metaclass=abc.ABCMeta):
    """Abstract Base Class for serving price data obtained from a source.

    Subclasses should implement the following abstractions:

        - Abstract Class Attributes -

        BaseInterval : Type[intervals._BaseInterval]
            Enumeration with members as TDInterval that represent an
            interval for which subclass requests price data from the
            provider (see 'Serving Price Data' section for notes on which
            intervals should be assigned as base intervals).

            The value of enum members should be defined as a tuple of
            values that would be passed to `timedelta` as positional
            arguments to define the corresponding interval. For
            convenience, common tuples are defined at
            `intervals.TIMEDELTA_ARGS`.

            Example BaseInterval definition:
                BaseInterval = intervals._BaseInterval(
                    "BaseInterval",
                    dict(
                        T1=intervals.TIMEDELTA_ARGS["T1"],  # 1 minute interval
                        T2=intervals.TIMEDELTA_ARGS["T2"],  # 2 minute interval
                        T5=intervals.TIMEDELTA_ARGS["T5"],  # 5 minute interval
                        H1=intervals.TIMEDELTA_ARGS["H1"],  # 1 hour interval
                        D1=intervals.TIMEDELTA_ARGS["D1"],  # 1 day interval
                    ),
                )

            Note: On base class BaseInterval is implemented as a type only.
            This type is not enforced at runtime.

            Note: Alternatively, if available base intervals can only be
            ascertained at runtime, the `_define_base_intervals` can be
            called from the subclasses constructor to define the base
            intervals for the specific instance. If base intervals will be
            defined in this way for all instances then it is not necessary
            to separately define a `BaseInterval` class attribute.

        BASE_LIMITS :
        dict[BI, pd.Timestamp | pd.Timedelta | None]
            key: BaseInterval
                Every base interval should be represented.

            value : pd.Timedelta | pd.Timestamp, optional
                Limit of earliest availability of historal data for the
                base interval, as either timedelta to now or absolute
                timestamp.

                For a daily interval:
                    The timedelta / timestamp must be day accurate (no
                        time component).
                    If defined as a timestamp then must be timezone naive.
                    If limit is unknown then value can take None.

                For intraday intervals:
                    Timestamps must have timezone as UTC.
                    Limits must be defined for all intraday base intervals.

            Example, if only 60 days of data are available for data at the
                5 minute base interval, although there is no limit on daily
                data:
                    {BaseInterval.T5: pd.Timedelta(days=60),
                     BaseInterval.D1: None}

            Note: Subclass instances can call `_update_base_limits` to
            override BASE_LIMITS with instance-specific limits for one,
            many or all intervals. This can be used when the limit for an
            interval can only be ascertained at runtime, for example if
            data availability for the daily interval is dependent on the
            specific set of symbols. It can alternatively be used to define
            the limits for all intervals. If used to define the limits for
            all intervals for all instances then it is not necessary to
            define the BASE_LIMITS class attribute.
                If used, `_update_base_limits` must be called from the
                subclass constructor before executing the constructor as
                defined on the base class.

            Note: Instance specific base limits are exposed via the
            `base_limits` property.

        BASE_LIMITS_RIGHT :
        dict[BI, pd.Timestamp | None]
            Note: If price data is available through to 'now' for all base
            intervals then it is NOT necessary to define BASE_LIMITS_RIGHT.

            key: BaseInterval
                If BASE_LIMITS_RIGHT is defined then every base interval
                must be represented.

            value : pd.Timestamp, optional
                Limit of most recent availability of historal data for the
                base interval, as either an absolute timestamp or None
                if price data is available through to 'now'. If the
                interval is daily then the timestamp should be day accurate
                (no time component) and be timezone naive, otherwise the
                timestamp should have timezone as UTC.

            Note: Subclass instances can call `_update_base_limits_right`
            to override BASE_LIMITS_RIGHT with instance-specific limits for
            one, many or all intervals. This can be used when the limit for
            an interval can only be ascertained at runtime, for example if
            the data source are local .csv files. If the method is used to
            define the right limits for all intervals for all instances
            then it is not necessary to define the BASE_LIMITS_RIGHT class
            attribute.
                If used, `_update_base_limits_right` must be called from
                the subclass constructor before executing the constructor
                as defined on the base class.

            Note: Instance specific base limits are exposed via the
            `base_limits` property.

        - Abstract Methods -

        _request_data(self, interval: BaseInterval,
                      start: pd.Timestamp | None,
                      end: pd.Timestamp | None) -> pd.DataFrame:
            Request data from source with `interval` from `start` to
            `end`.

            Return should include all indices between start and end for
            which any symbol could have traded (regardless of whether a
            trade was placed or not). DataFrame should have columns as
            "open", "high", "low" and "close", all of dtype "np.float64".
            DataFrame should have index as:
                `pd.DatetimeIndex` for daily data
                `pd.IntervalIndex` for intraday data, with both left and
                right sides as `pd.DatetimeIndex` of dtype
                'datetime64[ns, UTC]'.

            Parameters as abstract method doc.

    Subclasses can optionally override the following class attributes:

        PM_SUBSESSION_ORIGIN : Literal["open", "break_end"]: default "open"
            Where sessions have a break, declares whether the source
            indexes prices for the pm session based on the morning open
            or the post break pm open, i.e. does the source provide indices
            through the break, as if trading were continuous, or does it
            treat the am and pm sessions distinctly, such that prices for
            the pm subsession commence from the break end.

        SOURCE_LIVE : bool : default True
            Declares whether the data source is dynamic and offers live
            prices (True), or static (False) such that the same prices will
            always be returned for any given indice.

    Subclasses can optionally extend the following methods:

        __init__():
            All attributes and properties created by the base constructor
            should be preserved.

    Subclasses can optionally override or extend the following methods:

        prices_for_symbols(self, symbols: str) -> Type[PricesBase]
            Returns an instance of the prices class for a received subset
            of symbols.

            Can be overriden by subclass if the default implementation is
            not vaiable. (Also, see `_get_class_instance`.)

        _get_class_instance(self, symbols: list[str], **kwargs)
            Called by `prices_for_symbols` to create instance of subclass.

            If subclass needs to make changes to implement
            `prices_for_symbols` then it may be possible to simply extend
            this method to pass through any additional arguments that the
            constructor requires.

        limit_intraday_bi(self, bi: intervals.BI) -> pd.Timestamp
            Can be overriden to make any adjustments to the data source's
            standard left limit of availability for a given interval.

    Other than any noted above, it is not intended that private methods
    (prefixed with single underscore) are overriden or extended by
    subclasses.

    The base class also defines a host of public properties and methods.
    Use `help(PricesBase)` for a listing.

    Notes
    -----
                        --- Serving Price Data ---

    This section offers notes on the internals of how price data is
    served.

    All calls are served from data of a base interval (of `BaseInterval`).
    Data is served at either a base interval or a higher interval to which
    the base interval data is resampled. The base intervals represent the
    fewest intervals that collectively provide for all data available from
    the source to be evaluated. For example, if the source limits
    availability of both 5 mintute and 15 minute data to the last three
    months then only the 5 minute interval will be included to base
    intervals, with requests for prices at 15 minute intervals served from
    resampled 5 minute data.

    Base intervals can represent any intraday frequency, for example 1
    minute, 5 minutes etc, or a 'day' in which case the interval represents
    a full trading day. `market_prices` does not support base intervals
    higher than one day (all requests for prices at intervals longer than
    daily are met from resampled daily data).

    All data requested from source is stored locally. Client calls will be
    served from stored data whenever the call can be fully met from stored
    data at a SINGLE base interval, be that by serving the data 'as is' or
    resampling it to a higher interval. Where stored data at a single base
    interval can only partially meet a call, a further request from source
    at that base interval will only be made for the difference between
    stored and required data.

        NB user requests that return price data at a specific interval will
        only be served from price data at a SINGLE base interval, i.e. such
        requests will NOT be met by way of consolidating data from two or
        more base intervals. For example, say base intervals include 1
        minute and 5 minutes and a continuous 10 trading sessions of data
        is stored at the 1 minute interval. If the client requests 12
        trading days at a 5 minute interval and those 12 days include the
        10 days stored at 1 minute then data may (see next paragraphs) be
        requested from source for the full 12 days at the 5 minute
        interval, i.e. notwithstanding that the request could have been met
        by resampling the 10 days stored at 1 minute, requesting only the
        2 days difference at 5 minutes.

    Which base interval is used to evalute prices at a non base interval?

        In the first instance a call will be served from a base interval
        for which data is available over the full required period and which
        can represent the period end as accurately as possible. If there is
        more than one such base interval then will use the highest of these
        for which stored data covers the required period or, if there is no
        such stored data, the highest. NB In certain circumstances where
        consistency of return would be otherwise affected (e.g. where a
        request does not define the required interval) no consideration is
        given as to any data that may be stored and the highest of the
        'most accurate' base intervals is used regardless.

        If there is no base interval for which data is available over the
        full required period and which can represent the period end as
        accurately as possible then a base interval will be used that
        prioritises either the period length or the accuracy of the period
        end. The client can determine this priority with the get() method's
        'priority' parameter, which can take "period" or "end".
            If priority is "period" then of all base intervals that can
            fulfill the period, will use the one which provides for the
            greatest period end accuracy.

            If priority is "end" then of all base intervals that can
            represent the end with greatest accuracy, will use the one that
            provides for the greatest length of period to be fulfilled.

            In either case, if more than one base interval is equally able
            then, as in the first instance, will use the highest of these
            for which stored data is available or, if there is no such
            stored data or consistency of return a consideration, the
            highest.

    Any indice is usually only ever requested once from the source.
    Exceptions:

        If a request includes the current indice of an open exchange
        then further requests will re-request this live indice, so
        as to return the most up-to-date price, until the first
        occasion the indice becomes historic (as opposed to live).

        If all symbols do not share the same delay then any period
        over which price data is not available for all symbols
        (due to the delay) will be re-requested until the first
        request when price data is available for all symbols.

    - Always unavailable base intervals -

    Where prices are obtained for symbols trading on different exchanges,
    base intervals will only be available if indices are aligned. For
    example, if one exchange opens on the hour, say 08.00, and another
    opens at an offset from the hour, say 09.30, then data with a 1 hour
    interval will not align (in this case the first exchange will have
    an indice from 09.00 - 10.00 and the second from 09.30 - 10.30).

    Indices will always be considered to align when they do not conflict.
    For example, indices evaluated from a calendar representing the Hong
    Kong Stock Exchange and indices evaluated from a calendar representing
    the New York Stock Exchange will not be considered unaligned simply
    because these exchanges opening hours do not overlap.

    Non-aligned indices are more common for higher base intervals which
    in turn are those that offer longer periods of price data. Accordingly,
    a longer period of intraday data may be available to a symbol or
    symbols that share the same calendar than if mixed with symbols that
    trade on other calendars which overlap with the symbol's opening hours.

        Related methods:
            .has_single_calendar -> bool
                Query if all symbols share the same calendar.

            ._indices_aligned -> dict[BaseInterval, pd.Series]:
                Query if indices are aligned. Return by base interval
                and session.

            ._indices_aligned_for_drg -> bool
                Query if all indices are aligned over the daterange
                evalauted by a given 'Getter' instance.

    When data is anchored 'workback' a base interval will be unavailable
    for any period that includes an indice that partially covers a period
    during which no symbol traded (i.e. during the last indice of a
    calendar's session which is unaligned with that session's close). This
    avoids introducing non-trading periods into the indices.

        It is more likely for higher base intervals to be unavailable due
        to the greater likelyhood that indices will be unaligned with the
        session close. In turn, higher base intervals are more likely to
        have longer periods of data available to them. Consequently
        anchoring 'workback' can return price data for a significantly
        lesser period than anchoring 'open' (when strict=False). See the
        `data_availability.ipynb` tutorial for an example.

    - All or nothing -

    Data requests are not administered 'by-symbol' but rather on a
    collective basis. Accordingly all requests to source for prices
    will request prices for all symbols (notwithstanding 'include'
    or 'exclude' being passed to `get`).

    For each base interval a DataFrame is used to store requested data.
    Each of these include all symbols.
    """

    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    # See class documentation for implementation of abstract class attributes
    BaseInterval: _BaseIntervalMeta
    BASE_LIMITS: dict[BI, pd.Timedelta | pd.Timestamp | None]
    BASE_LIMITS_RIGHT: dict[BI, pd.Timestamp | None]

    PM_SUBSESSION_ORIGIN: Literal["open", "break_end"] = "open"
    SOURCE_LIVE: bool = True

    def __init__(
        self,
        symbols: mptypes.Symbols,
        calendars: mptypes.Calendars,
        lead_symbol: str | None = None,
        delays: int | list[int] | dict[str, int] = 0,
    ):
        """Create class instance.

        Parameters
        ----------
        symbols
            Symbols of instruments for which require price data. (provider
            specific, see subclass constructor documentation).

        calendars : [
            mptypes.Calendar
            | list[mptypes.Calendar]
            | dict[str, mptypes.Calendar
        ]
            Calendar(s) defining trading times and timezones.

            A single calendar representing all `symbols` can be passed as
            an mptype.Calendar, specifically any of:
                Instance of a subclass of `xcals.ExchangeCalendar`.
                Calendar 'side' must be "left".

                `str` of ISO Code of an exchange for which the
                `exchange_calendars` package maintains a calendar. See
                https://github.com/gerrymanoim/exchange_calendars#calendars
                or call
                `market_prices.utils.calendar_utils.get_exchange_info`.

                `str` of any other calendar name supported by
                `exchange_calendars`, as returned by
                `exchange_calendars.get_calendar_names`.

            Multiple calendars, each representing one or more symbols, can
            be passed as any of:
                list of objects, of same length as symbols, where each
                element is a mptypes.Calendar (i.e. defined as for a single
                calendar) that relates to the symbol at the corresponding
                index.

                dict:
                    key: str
                        symbol.
                    value: mptypes.Calendar (i.e. as for a single calendar)
                        Calendar corresponding with symbol.

            All calendars should have a first session no later than
            `base_limits` for the daily interval.

        lead_symbol
            Symbol with calendar that should be used as the default
            calendar to evaluate period from period parameters. If not
            passed default calendar will be defined as the most common
            calendar.

        delays
            Real-time price delay for each symbol, in minutes. For multiple
            symbols that do not all have the same delay pass as either:
                list of int of same length as symbols where each item
                relates to symbol at corresponding index.

                dict:
                    key: str
                        symbol.
                    value: int
                        Delay corresponding with symbol.
        """
        self.verify_base_intervals()
        self._symbols = helpers.symbols_to_list(symbols)
        self._base_intervals: _BaseIntervalMeta
        self._base_limits: dict[BI, pd.Timedelta | pd.Timestamp | None]
        self._verify_base_limits()
        self._base_limits_right: dict[BI, pd.Timestamp | None]
        self._set_base_limits_right()
        self._verify_base_limits_right()
        self._verify_lead_symbol(lead_symbol)
        self._calendars: dict[str, xcals.ExchangeCalendar]
        self._set_calendars(calendars)
        self._lead_calendar: xcals.ExchangeCalendar
        self._set_lead_calendar(lead_symbol)
        self._lead_symbol: str
        self._set_lead_symbol(lead_symbol)
        self._delays: dict[str, pd.Timedelta]
        self._set_delays(delays)
        self._cc = calutils.CompositeCalendar(self.calendars_unique)
        self._pdata: dict[BI, data.Data]
        self._set_pdata()
        self._trading_indexes_: dict[BI, pd.IntervalIndex]
        self._set_trading_indexes()
        self._indices_aligned_: dict[BI, pd.Series]
        self._set_indices_aligned()
        self._indexes_status_: dict[BI, pd.Series]
        self._set_indexes_status()
        self._gpp: "PricesBase.GetPricesParams" | None = None

    @property
    def symbols(self) -> list[str]:
        """List of symbols for which price data can be obtained."""
        return self._symbols

    def _dict_for_all_symbols(self, param: str, value: Any | list | dict) -> dict:
        """Create dictionary for by-symbol attribute.

        Parameters
        ----------
        param: str
            Parameter name.

        value: Any | list | dict:
            Any:
                scalar to apply to every symbol

            list:
                List of values where list has same length as symbols
                and each item relates to symbol at corresponding index.

            dict:
                Dictionary as return, either complete or can be missing
                any number of symbols.

        Returns
        -------
        dict
            Each item represents a symbol, all symbols represented.
                key: str
                    symbol.
                value: Any | None
                    value corresponding with symbol.
        """
        if not isinstance(value, (list, dict)):
            d = {s: value for s in self.symbols}
        elif isinstance(value, dict):
            if not set(value.keys()) == set(self.symbols):
                msg = (
                    f"If passing {param} as a dict then dict must have"
                    f" same keys as symbols, although receieved {param} as"
                    f" {value} for symbols {self.symbols}."
                )
                raise ValueError(msg)
            d = value.copy()
        elif isinstance(value, list):
            if not len(value) == len(self.symbols):
                msg = (
                    f"If passing {param} as a list then list must have same"
                    f" length as symbols, although receieved {param} as"
                    f" {value} for symbols {self.symbols}."
                )
                raise ValueError(msg)
            d = dict(zip(self.symbols, value))
        return d

    @parse
    def _define_base_intervals(self, update: _BaseIntervalMeta):
        """Define base intervals.

        Parameters
        ----------
        update
            `intervals._BaseInterval` definition with which to define the
            class `BaseInterval` attribute.

        Notes
        -----
        `_define_base_intervals` should only be called from a subclass
        constructor and only before executing the constructor as defined
        on the base class.
        """
        self._base_intervals = update

    @property
    def bis(self) -> _BaseIntervalMeta:
        """Base intervals."""
        try:
            return self._base_intervals
        except AttributeError:
            return self.BaseInterval

    @property
    def base_limits(self) -> dict[BI, pd.Timedelta | pd.Timestamp | None]:
        """Return availabilty of earliest data by base interval."""
        try:
            return self._base_limits
        except AttributeError:
            pass

        try:
            return self.BASE_LIMITS
        except AttributeError:
            pass

        raise AttributeError(
            "Base limits are not defined. Subclasses of `PricesBase` must define"
            " base limits via the BASE_LIMITS class attribute or the"
            " `_update_base_limits` method."
        )

    @property
    def base_limits_right(self) -> dict[BI, pd.Timestamp | None]:
        """Return availabilty of most recent data by base interval."""
        return self._base_limits_right

    # Parsing methods

    def verify_base_intervals(self):
        """Verify that base intervals have been defined."""
        if getattr(self, "_base_intervals", None) is not None:
            return
        elif getattr(self, "BaseInterval", None) is not None:
            return
        raise AttributeError(
            "Base intervals are not defined. Subclasses of `PricesBase` must define"
            " base intervals via the BaseInterval class attribute or the"
            " `_define_base_intervals` method."
        )

    def _verify_base_limits(self):
        """Verify type of base limits values."""
        valid_types = (pd.Timedelta, pd.Timestamp)
        for bi, limit in self.base_limits.items():
            if bi.is_intraday:
                if not isinstance(limit, valid_types):
                    raise ValueError(
                        "Intraday base interval limits must be of type pd.Timedelta or"
                        f" pd.Timestamp, although limit for {bi} would be defined"
                        f" as {limit}."
                    )
            else:
                if not (isinstance(limit, valid_types) or limit is None):
                    raise ValueError(
                        "Daily base interval limits must be None or of type"
                        f" pd.Timedelta or pd.Timestamp, although limit for {bi}"
                        f" would be defined as {limit}."
                    )
                if isinstance(limit, pd.Timestamp) and not helpers.is_date(limit):
                    raise ValueError(
                        "If limit of daily base interval is defined as a pd.Timestamp"
                        " then timestamp must represent a date, although being"
                        f" defined as {limit}."
                    )

        base_limits_keys = self.base_limits.keys()
        all_keys_valid = all(key in self.bis for key in base_limits_keys)
        if not all_keys_valid or len(base_limits_keys) != len(self.bis):
            raise ValueError(
                "Base limits do not accurately represent base intervals. Base intervals"
                f" are {self.bis.__members__} although base limit keys would be"
                f" {base_limits_keys}."
            )

    def _set_base_limits_right(self):
        """Set `_base_limits_right` to default if not otherwise defined."""
        if getattr(self, "_base_limits_right", None) is not None:
            return
        if getattr(self, "BASE_LIMITS_RIGHT", None) is not None:
            self._base_limits_right = self.BASE_LIMITS_RIGHT.copy()
        else:
            self._base_limits_right = {bi: None for bi in self.bis}

    def _verify_base_limits_right(self):
        """Verify type of right base limits values."""
        for bi, limit in self.base_limits_right.items():
            if limit is None:
                continue
            if not isinstance(limit, pd.Timestamp):
                raise ValueError(
                    "Base interval right limits must be None or of type pd.Timestamp,"
                    f" although right limit for {bi} would be defined as {limit}."
                )
            if bi.is_daily and not helpers.is_date(limit):
                raise ValueError(
                    "If right limit of daily base interval is defined then timestamp"
                    f" must represent a date, although would be defined as {limit}."
                )

        base_limits_keys = self.base_limits_right.keys()
        all_keys_valid = all(key in self.bis for key in base_limits_keys)
        if not all_keys_valid or len(base_limits_keys) != len(self.bis):
            raise ValueError(
                "Base right limits do not accurately represent base intervals. Base"
                f" intervals are {self.bis.__members__} although base limit keys would"
                f" be {base_limits_keys}."
            )

    def _update_base_limits(self, update: dict[BI, pd.Timedelta | pd.Timestamp | None]):
        """Update data availability for one or more base intervals.

        Parameters
        ----------
        update
            Dictionary with which to update existing `base_limits`.

        Notes
        -----
        `_update_base_limits` should only be called from a subclass
        constructor and only before executing the constructor as defined
        on the base class.
        """
        if getattr(self, "_base_limits", None) is None:
            if getattr(self, "BASE_LIMITS", None) is None:
                self._base_limits = {}
            else:
                self._base_limits = self.BASE_LIMITS.copy()
        prev_limits = self._base_limits.copy()
        self._base_limits.update(update)
        try:
            self._verify_base_limits()
        except ValueError:
            self._base_limits = prev_limits
            raise

    def _update_base_limits_right(self, update: dict[BI, pd.Timestamp | None]):
        """Update data availability for one or more base intervals.

        Parameters
        ----------
        update
            Dictionary with which to update existing `base_limits`.

        Notes
        -----
        `_update_base_limits` should only be called from a subclass
        constructor and only before executing the constructor as defined
        on the base class.
        """
        if getattr(self, "_base_limits_right", None) is None:
            self._set_base_limits_right()
        prev_limits = self._base_limits_right.copy()
        self._base_limits_right.update(update)
        try:
            self._verify_base_limits_right()
        except ValueError:
            self._base_limits_right = prev_limits
            raise

    def _verify_lead_symbol(self, symbol: str | None):
        """Verify public input to a `lead_symbol` parameter."""
        if symbol is None:
            return
        elif symbol not in self.symbols:
            msg = (
                f"`lead_symbol` received as '{symbol}' although must be None"
                f" or in {self.symbols}."
            )
            raise ValueError(msg)

    # Calendars
    def _set_calendars(self, calendars: mptypes.Calendars):
        """Set and verify calendars."""
        d = self._dict_for_all_symbols("calendars", calendars)

        ll = None if self.bi_daily is None else self.base_limits[self.bi_daily]
        if isinstance(ll, pd.Timedelta):
            ll = helpers.now(intervals.BI_ONE_DAY) - ll
        # margin to ensure calendar's first session is not later than limit.
        kwargs = {"start": ll - pd.Timedelta(14, "D")} if ll is not None else {}
        for k, v in d.items():
            if not isinstance(v, xcals.ExchangeCalendar):
                cal = xcals.get_calendar(v, side="left", **kwargs)
                d[k] = cal
            elif v.side != "left":
                msg = (
                    f"All calendars must have side 'left', although received calendar"
                    f" '{v.name}' with side '{v.side}'."
                )
                raise ValueError(msg)

        required_bound = helpers.now() + helpers.ONE_DAY
        for cal in set(d.values()):
            if cal.last_minute < required_bound:
                raise errors.CalendarExpiredError(cal, cal.last_minute, required_bound)

            # raise error if calendar does not cover period over which any intraday
            # prices are available. Could lose this restriction although would add
            # complexity. Cleaner to restrict the calendars here.
            if self.bis_intraday:
                intraday_ll = self.base_limits[self.bis_intraday[-1]]
                if isinstance(intraday_ll, pd.Timedelta):
                    intraday_ll = helpers.now() - intraday_ll
                if cal.first_minute > intraday_ll:
                    assert isinstance(intraday_ll, pd.Timestamp)
                    raise errors.CalendarTooShortError(cal, intraday_ll)

            assert isinstance(ll, pd.Timestamp) or ll is None
            if ll is not None and cal.first_session > ll:
                # raise warning if calendar does not cover period over which daily
                # prices available.
                warnings.warn(errors.CalendarTooShortWarning(cal, ll))

        self._calendars = d

    @property
    def calendars(self) -> dict[str, xcals.ExchangeCalendar]:
        """Calendar associated with each symbol.

        See Also
        --------
        calendars_symbols
        """
        return self._calendars

    @functools.cached_property
    def calendars_symbols(self) -> dict[xcals.ExchangeCalendar, list[str]]:
        """Symbols associated with each calendar.

        See Also
        --------
        calendars
        """
        d = collections.defaultdict(list)
        for s, cal in self.calendars.items():
            d[cal].append(s)
        return dict(d)

    def _set_lead_calendar(self, lead_symbol: str | None):
        if lead_symbol is not None:
            cal = self.calendars[lead_symbol]
        else:
            cal = pdutils.most_common(list(self.calendars.values()))
        self._lead_calendar = cal

    @property
    def calendar_default(self) -> xcals.ExchangeCalendar:
        """Default calendar used to evaluate sessions and timezone."""
        return self._lead_calendar

    @functools.cached_property
    def calendars_unique(self) -> list[xcals.ExchangeCalendar]:
        """List of calendars associated with symbols."""
        return list(set(self.calendars.values()))

    @functools.cached_property
    def calendars_names(self) -> list[str]:
        """Names of all calendars."""
        return [c.name for c in self.calendars_unique]

    @property
    def has_single_calendar(self) -> bool:
        """Query if all symbols are assigned same calendar."""
        return len(self.calendars_unique) == 1

    def _set_lead_symbol(self, symbol: str | None = None):
        # set as `symbol` or first symbol with default calendar.
        if symbol is None:
            for symbol in self.symbols:  # pylint: disable=redefined-argument-from-local
                if self.calendars[symbol] == self.calendar_default:
                    break
        self._lead_symbol = symbol

    @property
    def lead_symbol_default(self) -> str:
        """Default lead symbol."""
        return self._lead_symbol

    def _set_delays(self, delays: int | list[int] | dict[str, int]):
        d = self._dict_for_all_symbols("delays", delays)
        self._delays = {k: pd.Timedelta(v, "min") for k, v in d.items()}

    @property
    def delays(self) -> dict[str, pd.Timedelta]:
        """Real-time price delay for each symbol."""
        return self._delays

    @property
    def min_delay(self) -> pd.Timedelta:
        """Minimum real-time price delay of any symbol."""
        return min(self.delays.values())

    @property
    def max_delay(self) -> pd.Timedelta:
        """Maximum real-time price delay of any symbol."""
        return max(self.delays.values())

    def _calendars_delay(self, f) -> dict[xcals.ExchangeCalendar, pd.Timedelta]:
        d = {}
        for cal, symbols in self.calendars_symbols.items():
            d[cal] = f([self.delays[s] for s in symbols])
        return d

    @functools.cached_property
    def calendars_min_delay(self) -> dict[xcals.ExchangeCalendar, pd.Timedelta]:
        """Minimum delay by calendar."""
        return self._calendars_delay(min)

    @functools.cached_property
    def calendars_max_delay(self) -> dict[xcals.ExchangeCalendar, pd.Timedelta]:
        """Maximum delay by calendar."""
        return self._calendars_delay(max)

    @functools.cached_property
    def timezones(self) -> dict[str, ZoneInfo]:
        """Timezones, by symbol. Evaluated from calendars."""
        return {k: c.tz for k, c in self.calendars.items()}

    @functools.cached_property
    def tz_default(self) -> ZoneInfo:
        """Default timezone."""
        return self.timezones[self.lead_symbol_default]

    @property
    def composite_calendar(self) -> calutils.CompositeCalendar:
        """Composite calendar evaluated from all associated calendars."""
        return self._cc

    @property
    def cc(self) -> calutils.CompositeCalendar:
        """Alias for composite_calendar."""
        return self.composite_calendar

    # Base intervals
    @property
    def bi_daily(self) -> BI | None:
        """Daily base interval, or None if all base intervals intraday."""
        return self.bis.daily_bi()

    @functools.cached_property
    def bis_intraday(self) -> list[BI]:
        """Intraday base intervals."""
        return self.bis.intraday_bis()

    @functools.cached_property
    def _calendars_latest_first_session(self) -> pd.Timestamp:
        """Latest first session of any calendar."""
        return self.cc.first_session

    @functools.cached_property
    def _calendars_latest_first_minute(self) -> pd.Timestamp:
        """Latest first minute of any calendar."""
        return max(cal.first_minute for cal in self.calendars_unique)

    def _set_pdata(self):
        """Set --_pdata-- to dict with key as bi and value as `data.Data`."""
        d = {}
        max_delay = self.max_delay
        for bi in self.bis:
            ll = self.base_limits[bi]
            if bi.is_daily:
                if ll is not None:
                    # Consider any calendar limitations
                    if isinstance(ll, pd.Timedelta):
                        ll = helpers.now(bi) - ll
                    ll = max(ll, self._calendars_latest_first_session)
                else:
                    ll = self._calendars_latest_first_session
            d[bi] = data.Data(
                request=self._request_data,
                cc=self.cc,
                bi=bi,
                delay=max_delay,
                left_limit=ll,
                right_limit=self.base_limits_right[bi],
                source_live=self.SOURCE_LIVE,
            )
        self._pdata = d

    @property
    def live_prices(self) -> bool:
        """Query if live prices are avaiable (as opposed to only historic).

        NB Method is not concerned with whether any live prices are
        available real-time or delayed.
        """
        return set(self._base_limits_right.values()) == {None}

    def _minute_to_session(
        self,
        ts: pd.Timestamp,
        extreme: Literal["earliest", "latest"],
        direction: Literal["previous", "next"],
    ) -> pd.Timestamp:
        """Get a session corresponding with a minute.

        Session will be a session of one of the associated calendars.

        Note:
            Only considers calendars where `ts` falls within schedule.

            Calendars closed "left" such that a session close is not
            considered a session minute.

            For each calendar, the evaluated minute will be the earlier of
            `ts` or the latest minute at which a price is available for any
            symbol to which the calendar is assigned.

        Parameters
        ----------
        ts
            Minute to be evaluated

        extreme
            From the set of corresponding sessions (one for each calendar),
            return the "earliest"/"latest".

        direction
            Where `ts` is not a trading minute of a calendar, evaluate the
            minute as the "previous"/"next" trading minute before/after
            `ts`.
        """
        sessions = []
        if ts < helpers.now() - self.max_delay:
            for cal in self.calendars_unique:
                try:
                    sessions.append(cal.minute_to_session(ts, direction))
                except ValueError:
                    continue
        else:
            for cal in self.calendars_unique:
                avail_to = helpers.now() - self.calendars_min_delay[cal]
                ts_ = min([avail_to, ts])
                try:
                    sessions.append(cal.minute_to_session(ts_, direction))
                except ValueError:
                    continue
        f = min if extreme == "earliest" else max
        session = f(sessions)
        return session

    def _minute_to_latest_next_trading_minute(
        self, minute: pd.Timestamp
    ) -> pd.Timestamp:
        """Return latest next trading minute of all calendars.

        Returns latest trading minute of all calendars that is or
        immediately follows `minute`.
        """
        cals = self.calendars_unique
        return max(cal.minute_to_trading_minute(minute, "next") for cal in cals)

    def _minute_to_earliest_previous_trading_minute(
        self, minute: pd.Timestamp
    ) -> pd.Timestamp:
        """Return earliest preivous trading minute of all calendars.

        Returns earliest trading minute of all calendars that is or
        immediately preceeds `minute`.
        """
        cals = self.calendars_unique
        return min(cal.minute_to_trading_minute(minute, "previous") for cal in cals)

    @property
    def limits(self) -> dict[BI, mptypes.DateRangeReq]:
        """Left and right limits of data availability by base interval.

        NB The left limits are the earliest timestamps for which price data
        MAY be avaiable from the source. The actual left limit may be later
        for any specific symbol or as a result of any peculiarities with
        the data source (see See Also section).

        Returns
        -------
        dict
            key:
                Base interval
            value:
                [0] left limit, or (daily base interval only) None if limit
                    yet to be ascertained.
                [1] right limit

        See Also
        --------
        limit_intraday_bi
            Left limit for a specific base interval, accounting for any
            source pecularities.

        limit_intraday_bi_calendar
            Left limit for a specific base interval calibrated to a
            specific calendar or to all calendars.
        """
        return {bi: (pd_.ll, pd_.rl) for bi, pd_ in self._pdata.items()}

    @property
    def limit_daily(self) -> pd.Timestamp | None:
        """Earliest date for which daily prices can be requested.

        None if earliest date is yet to be ascertained or if daily price
        data is not available.
        """
        if self.bi_daily is None:
            return None
        return self._pdata[self.bi_daily].ll

    @property
    def limit_right_daily(self) -> pd.Timestamp | None:
        """Latest date for which daily prices can be requested.

        None if no daily interval.
        """
        if not self.bi_daily:
            return None
        return self._pdata[self.bi_daily].rl

    def limit_intraday_bi(self, bi: intervals.BI) -> pd.Timestamp:
        """Left limit for a specific intrday base interval.

        Parameters
        ----------
        bi
            Base interval for which require intraday left limit.
        """
        return self.limits[bi][0]

    def limit_intraday_bi_calendar(
        self,
        bi: intervals.BI,
        calendar: xcals.ExchangeCalendar | None = None,
    ) -> pd.Timestamp:
        """Earliest minute of a calendar that can be requested for a bi.

        Parameters
        ----------
        bi
            Base interval for which require intraday left limit.

        calendar
            Calendar against which to calibrate left limit. Return will be
            the earliest trading minute of calendar that follows the
            absolute left limit for the `bi`.

            If None the return will be the earliest minute, following the
            absolute left limit for the `bi`, by which all calendars have
            registered a trading minute.
        """
        limit_abs = self.limit_intraday_bi(bi)
        if calendar is not None:
            return calendar.minute_to_trading_minute(limit_abs, "next")
        return self._minute_to_latest_next_trading_minute(limit_abs)

    def limit_intraday(
        self, calendar: xcals.ExchangeCalendar | None = None
    ) -> pd.Timestamp:
        """Earliest minute that intraday prices can be requested.

        Note: intraday prices will not be considered available for any
        base interval for which indices are not aligned over the full
        period for which intraday data is available at that interval.

        Parameters
        ----------
        calendar
            Calendar for which intraday prices should be available.

            By default (None) intraday prices will be available for all
            calendars from the returned timestamp.
        """
        if not self.bis_intraday:
            raise NotImplementedError(
                "`limit_intraday` is not implemented when no intraday interval is"
                " defined."
            )
        limits = []
        for bi in self.bis_intraday:
            limit_minute = self.limit_intraday_bi_calendar(bi, calendar)
            limit_session = self.cc.minute_to_sessions(limit_minute, "next")[0]
            if self._indices_aligned[bi][limit_session:].all():
                assert limit_minute is not None
                limits.append(limit_minute)
        return min(limits)

    @property
    def limit_right_intraday(self) -> pd.Timestamp:
        """Latest minute that intraday prices can be requested.

        'now' if live prices are available.

        Note: intraday prices will not be considered available for any
        base interval for which indices are not aligned over the full
        period for which intraday data is available at that interval.
        """
        if not self.bis_intraday:
            raise NotImplementedError(
                "`limit_right_intraday` is not implemented when no intraday interval"
                " is defined."
            )

        if self.live_prices:
            return helpers.now()
        limits_raw = []
        for bi in self.bis_intraday:
            limit_minute = self.limits[bi][1]
            limit_session = self.cc.minute_to_sessions(limit_minute, "previous")[-1]
            if self._indices_aligned[bi][:limit_session].all():
                assert limit_minute is not None
                limits_raw.append(limit_minute)
        limit_raw = max(limits_raw)
        assert isinstance(limit_raw, pd.Timestamp)
        return limit_raw

    @property
    def limits_sessions(self) -> dict[BI, mptypes.DateRangeReq]:
        """Earliest and latest sessions for which price data available.

        For daily, earliest session will be None if yet to ascertain
        earliest date of daily price data availability.
        """
        d = {}
        for bi, (ll, rl) in self.limits.items():
            if bi.is_daily:
                if ll is None:
                    start = None
                else:
                    start = self.cc.date_to_session(ll, "next")
                end = self.cc.date_to_session(rl, "previous")
            else:
                assert ll is not None
                start = self._minute_to_session(ll, "earliest", "next")
                end = self._minute_to_session(rl, "latest", "previous")
            d[bi] = (start, end)
        return d

    def _earliest_requestable_calendar_session(
        self, calendar: xcals.ExchangeCalendar
    ) -> pd.Timestamp:
        """Earliest calendar session for which prices can be requested.

        Returns
        -------
        pd.Timestamp
            If the left limit of the daily base interval is known, or base
            intervals do not include daily, then return will be `calendar`
            session corresponding with earliest left limit of all base
            intervals.

            If daily left limit is yet to be ascertained, return will be
            the first calendar session.
        """
        if self.bi_daily is not None and self.limit_daily is None:
            session = calendar.first_session
        elif self.limit_daily is not None:
            session = calendar.date_to_session(self.limit_daily, "next")
        else:
            session = calendar.minute_to_session(self.limit_intraday(calendar))
        return session

    @property
    def earliest_requestable_session(self) -> pd.Timestamp:
        """Earliest session that prices can be requested for any calendar.

        Returns
        -------
        pd.Timestamp
            If base intervals do not include daily, return will be earliest
            session (of any calendar) corresponding with earliest left
            limit of all intraday base intervals for which prices align
            (over the period for which data available at that base
            interval).

            If base intervals include daily and daily left limit is known,
            return will be first session of any calendar on or after the
            daily left limit.

            If base intervals include daily although daily left limit is
            yet to be ascertained, return will be the latest
            'first_session' of all calendars.
        """
        if self.bi_daily is not None and self.limit_daily is None:
            session = max([cal.first_session for cal in self.calendars_unique])
        elif self.bi_daily is not None:
            session = self.limits_sessions[self.bi_daily][0]
        else:
            session = min(
                [
                    ll
                    for bi, (ll, rl) in self.limits_sessions.items()
                    if self._indices_aligned[bi][ll:rl].all()
                ]
            )
        return session

    def _earliest_requestable_calendar_minute(
        self, calendar: xcals.ExchangeCalendar
    ) -> pd.Timestamp:
        """Earliest calendar minute for which prices can be requested.

        Returns
        -------
        pd.Timestamp
            If intervals do not include daily, return will be first
            `calendar` trading minute following the earliest left limit of
            all base intervals.

            If intervals include daily and the left limit for the daily
            interval is known, return will be the open of the session
            corresponding with left limit.

            If intervals include daily although the left limit for the
            daily interval is yet to be ascertained, return will be the
            `calendar` first minute.
        """
        if self.bi_daily is not None and self.limit_daily is None:
            minute = calendar.first_minute
        elif self.limit_daily is not None:
            session = self._earliest_requestable_calendar_session(calendar)
            minute = calendar.session_open(session)
        else:
            minute = self.limit_intraday(calendar)
        return minute

    @property
    def earliest_requestable_minute(self) -> pd.Timestamp:
        """Earliest minute that prices can be requested for ALL calendars.

        Returns
        -------
        pd.Timestamp
            If there is no daily base interval, return will be latest of
            the earliest minute of each calendar that follows the
            earliest left limit of all base intervals.

            If the left limit of the daily base interval is known, return
            will be latest of the earliest session open of each calendar
            that falls on or after the limit.

            If daily left limit is yet to be ascertained, return will be
            the latest 'first_minute' of all calendars.
        """
        calendars = self.calendars_unique
        minutes = [self._earliest_requestable_calendar_minute(c) for c in calendars]
        return max(minutes)

    @property
    def _minute_for_last_requestable_session(self) -> pd.Timestamp:
        """Return minute that can be used to get last requestable session."""
        if self.bi_daily is not None:
            minute = pd.Timestamp(
                self.limit_right_daily + helpers.ONE_DAY - helpers.ONE_MIN,
                tz=helpers.UTC,
            )
        else:
            minute = self.limit_right_intraday
        return min(helpers.now(), minute)

    @property
    def last_requestable_session_all(self) -> pd.Timestamp:
        """Most recent session available for all calendars.

        Considers only daily data.
        """
        minute = self._minute_for_last_requestable_session
        return self._minute_to_session(minute, "earliest", "previous")

    @property
    def last_requestable_session_any(self) -> pd.Timestamp:
        """Most recent session available for any calendar.

        Considers only daily data.
        """
        minute = self._minute_for_last_requestable_session
        return self._minute_to_session(minute, "latest", "previous")

    @property
    def latest_requestable_minute(self) -> pd.Timestamp:
        """Latest minute for which prices can be requested.

        Returns
        -------
        pd.Timestamp
            `now` if live prices are available.

            Otherwise, the later of the latest minute for which intraday
            price data is available at any interval or the latest session
            close of any calendar on the latest session for which daily
            price data is available.
        """
        if self.live_prices:
            return helpers.now()

        minutes = []
        if self.bis_intraday:
            minutes.append(self.limit_right_intraday)

        if self.bi_daily is not None:
            session = self.cc.date_to_session(self.limit_right_daily, "previous")
            minutes.append(self.cc.session_close(session))

        return max(minutes)

    def _set_trading_indexes(self):
        indexes = {}
        for bi in self.bis_intraday:
            if bi.is_one_minute:
                continue
            start, end = self.limits_sessions[bi]
            ignore_breaks = self._ignore_breaks(bi)
            indexes[bi] = self.cc.trading_index(
                bi, start, end, ignore_breaks, raise_overlapping=False, utc=False
            )
        self._trading_indexes_ = indexes

    @property
    def _trading_indexes(self) -> dict[BI, pd.IntervalIndex]:
        """Trading indexes.

        Trading indexes comprise the sorted union of the trading indexes of
        each of the underlying calendars. Indexes cover the period for
        which prices are available for the corresponding base interval.

        Trading indexes are not provided for any T1 or D1 base interval.

        Trading indexes are timezone naive.

        Returns
        -------
        dict
            keys : Baseinterval
            values: pd.IntervalIndex
                Left and right arrays are tz-naive. Intervals are close on
                "left".
        """
        return self._trading_indexes_

    def _set_indices_aligned(self):
        # pylint: disable=too-many-locals
        aligned = {}
        for bi in self.bis_intraday:
            start, end = self.limits_sessions[bi]
            slc = slice(start, end)

            if bi.is_one_minute:
                # shortcut, T1 cannot have conflicts
                sessions = self.cc.opens[slc].index
                aligned[bi] = pd.Series(True, index=sessions)
                continue

            index = self._trading_indexes[bi]
            nano_index = index.left.asi8
            opens = self.cc.opens[slc].astype("int64")
            closes = self.cc.closes[slc].astype("int64")

            sessions = opens.index
            srs = pd.Series(True, index=sessions)
            for session, open_, close in zip(sessions, opens, closes):
                bv = (nano_index >= open_) & (nano_index < close)
                srs[session] = index[bv].is_non_overlapping_monotonic

            aligned[bi] = srs

        self._indices_aligned_ = aligned

    @property
    def _indices_aligned(self) -> dict[BI, pd.Series]:
        """Query if trading indices of calendars are aligned.

        Trading indices are considered unaligned, for a specific session
        and at a specific base interval, if any indice pertaining to
        one calendar would partially overlap an indice pertaining to
        another. For example, an indice from 16.00 - 17.00 would conflict
        conflict with another, based on another calendar, from 16.30 -
        17.30.

        Returns
        -------
        dict
            keys : Baseinterval

            values: pd.Series
                index : pd.DatetimeIndex
                    Union of sessions of all calendars over period for
                    which prices available at base interval.

                value : bool
                    Boolean indicates if at least one of the session's
                    trading indices would be unaligned.
        """
        return self._indices_aligned_

    def _indices_aligned_for_drg(self, drg: dr.GetterIntraday) -> bool:
        """Query if all indices aligned for drg.

        True if all trading indices, for period interval associated with
        drg, are aligned.
        """
        try:
            start, end = drg.daterange_sessions
        except (
            errors.EndTooEarlyError,
            errors.EndTooLateError,
            errors.StartTooEarlyError,
            errors.StartTooLateError,
        ):
            return False
        return self._indices_aligned[drg.interval][slice(start, end)].all()

    def _set_indexes_status(self):
        if not self.bis_intraday:
            self._indexes_status_ = {}
            return

        # pylint: disable=too-many-locals
        highest_intraday_bi = self.bis_intraday[-1]
        start, end = self.limits_sessions[highest_intraday_bi]
        nti = self.cc.non_trading_index(start, end, utc=False)

        statuses: dict[BI, pd.Series] = {}
        for bi in self.bis_intraday:
            start_session, end_session = self.limits_sessions[bi]
            sessions = self.cc.sessions_in_range(start_session, end_session)
            status = pd.Series(True, index=sessions, dtype="object")

            if bi.is_one_minute:
                # shortcut, cannot have partial indices or conflicts at T1
                statuses[bi] = status
                continue

            # Get trading index
            trading_index = self._trading_indexes[bi]

            # Get non-trading index
            start = trading_index[0].left
            end_session_next = self.cc.next_session(end_session)
            end = self.cc.session_open(end_session_next).tz_convert(None)
            slc = nti.slice_indexer(start, end)
            nti_subset = nti[slc]

            # Union them
            all_in = trading_index.union(nti_subset, sort=False).sort_values()

            # interrogate by composite session
            all_in_left_nanos = all_in.left.asi8
            all_in_right_nanos = all_in.right.asi8

            session_opens = self.cc.opens[start_session:end_session]
            start_session_next = self.cc.next_session(start_session)
            session_opens_next = self.cc.opens[start_session_next:end_session_next]

            indices_aligned = self._indices_aligned[bi]
            for session, start, end in zip(sessions, session_opens, session_opens_next):
                if not indices_aligned[session]:
                    status[session] = np.nan
                    continue
                bv = (all_in_left_nanos >= start.value) & (
                    all_in_right_nanos <= end.value
                )
                day_index = all_in[bv]
                status[session] = day_index.is_non_overlapping_monotonic
            statuses[bi] = status

        self._indexes_status_ = statuses

    @property
    def _indexes_status(self) -> dict[BI, pd.Series]:
        """Status of intraday trading indexes.

        Returns
        -------
        dict
            keys : intervals.BI
            values : pd.Series
                index : pd.DatetimeIndex
                    Union of sessions of all calendars.
                value : bool | pd.NaN
                    pd.NaN: The trading indices for different calendars
                    overlap and are not aligned.

                    True: Session trading indices for each calendar do not
                    conflict (they are either aligned or do not overlap)
                    and no indice includes a period during which no
                    calendar is open.

                    False: Session trading indices for each calendar do not
                    conflict (they are either aligned or do not overlap)
                    and at least one indice is a partial trading indice
                    during a period of which no calendar is open.
        """
        return self._indexes_status_

    def _has_valid_fully_trading_indices(self, drg: dr.GetterIntraday) -> bool:
        """Query if indices associated with `drg` are fully trading.

        Returns
        -------
        bool
            True : Trading indices do not conflct and all cover periods
            during which at least one calendar is always open.

            False : Trading indices either conflct or include at least one
            indice that covers a period during part of which no calendar is
            open.
        """
        try:
            start, end = drg.daterange_sessions
        except errors.PricesUnavailableIntervalError:
            return False
        indexes_status = self._indexes_status[drg.interval][start:end]
        return not indexes_status.isna().any() and indexes_status.all()

    @property
    def _pdata_ranges(self) -> dict[BI, list[pd.Interval]]:
        """Date ranges of requested data, by base interval."""
        return {bi: pdata.ranges for bi, pdata in self._pdata.items()}

    @property
    def has_data(self) -> bool:
        """Query if any data is stored for any datetime interval."""
        # NB May 22. KEEP this `has_data` method even though it has no internal clients.
        for pdata in self._pdata.values():
            if pdata.has_data:
                return True
        return False

    def _subsessions_synced(self, calendar: xcals.ExchangeCalendar, bi: BI) -> bool:
        """Query if open of am and pm subsessions are synchronised.

        Returns
        -------
        bool
            Considering date range over which data at `interval` is
            available...

            True: no session has a break or open times of subsessions of
            each session are synchronised. Open times for a session
            considered synchronised if `interval` is a factor of
            difference between am open and pm open.

            False: otherwise.
        """
        if not calendar.has_break:
            return True
        ll, rl = self.limits[bi]
        ll_session = calendar.minute_to_session(ll, "next")
        rl_session = calendar.minute_to_session(rl, "previous")
        if not calendar.sessions_has_break(ll_session, rl_session):
            return True
        phase_diff = (calendar.break_ends - calendar.opens) % bi
        synced = (phase_diff == pd.Timedelta(0)) | phase_diff.isna()
        return synced.all()

    def _ignore_breaks_cal(
        self, calendar: xcals.ExchangeCalendar, bi: intervals.BI
    ) -> bool:
        """Query if breaks should be ignored in a trading index for `calendar`."""
        return self.PM_SUBSESSION_ORIGIN == "open" and not self._subsessions_synced(
            calendar, bi
        )

    def _ignore_breaks(self, bi: intervals.BI) -> dict[xcals.ExchangeCalendar, bool]:
        """Query if breaks should be ignored when creating a trading index."""
        return {cal: self._ignore_breaks_cal(cal, bi) for cal in self.calendars_unique}

    @property
    def _ignore_breaks_any(self) -> bool:
        """Query if breaks are ignored for any calendar for any interval."""
        for bi in self.bis_intraday:
            ignore_a_break = any(self._ignore_breaks(bi).values())
            if ignore_a_break:
                return True
        return False

    def _get_trading_index(  # pylint: disable=too-many-arguments
        self,
        calendar: xcals.ExchangeCalendar,
        interval: BI,
        start: pd.Timestamp,
        end: pd.Timestamp,
        force: bool = False,
    ) -> pd.IntervalIndex | pd.DatetimeIndex:
        """Return index covering only trading hours.

        Intraday indices are included for all trading times from open
        through close. Where a session has a break, indices are included
        through the break, as if the session were continuous, in the event
        that the open and the pm open (i.e. break end) are unaligned at
        `interval` and `PM_SUBSESSION_ORIGIN` is "open", otherwise (i.e.
        if the open and pm open are aligned or `PM_SUBSESSION_ORIGIN` is
        "break_end") indices will not be included that would otherwise
        commence during the break.

        Parameters as calutils.get_trading_index.
        """
        if interval.is_daily:
            ignore_breaks = True
        else:
            ignore_breaks = self._ignore_breaks_cal(calendar, interval)
        return calutils.get_trading_index(
            calendar, interval, start, end, force, ignore_breaks
        )

    @abc.abstractmethod
    def _request_data(
        self,
        interval: BI,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Request data.

        Parameters
        ----------
        interval
            Interval covered by each row.

        start
            Timestamp from which data required. Can only take None if
            interval is daily, in which case will assume data required from
            first date available.

        end : pd.Timestamp
            Timestamp to which data required.

        Returns
        -------
        pd.DataFrame
            .index:
                If `interval` intra-day:
                    pd.IntervalIndex, closed on left, UTC times indicating
                    interval covered by row.

                If `interval` daily:
                    pd.DatetimeIndex of dates represening session covered
                    by row.

            .columns: MultiIndex:
                level-0: symbol.
                level-1: ['open', 'high', 'low', 'close', 'volume'] for
                    each symbol.

        Raises
        ------
        errors.PricesUnavailableFromSourceError
            Prices unavailable.

        ValueError
            If `start` None although interval is intraday.
        """

    @property
    def gpp(self) -> "PricesBase.GetPricesParams":
        """Last created instance of `PricesBase.GetPricesParams`.

        Instance of `PricesBase.GetPricesParams` that stores parameters
        corresponding to last call to `get()` method.

        Raises
        ------
        errors.NoGetPricesParams
            If `get` has not been called on instance.
        """
        if self._gpp is None:
            raise errors.NoGetPricesParams
        else:
            return self._gpp

    # Base intervals that meet a condition

    @property
    def _bis_valid(self) -> list[BI]:
        """Return intraday bis that could serve data at `interval`.

        Returned base intervals could serve data at interval either 'as is'
        or by downsampling.

        Only returns base intervals that would be aligned for all
        calendars.

        Method makes no claims on whether price data is available for the
        returned base intervals over the requested period.
        """
        ds_interval = self.gpp.ds_interval
        assert ds_interval is None or isinstance(ds_interval, (TDInterval, BI))
        bis_factors = []
        ends_too_early = 0
        i = 0
        for bi in self.bis_intraday:
            if ds_interval is not None and ds_interval % bi != pd.Timedelta(0):
                continue
            i += 1
            drg = self.gpp.drg_intraday_no_limit
            drg.interval = bi
            try:
                drg.daterange
            except errors.PricesUnavailableIntervalError:
                # reject bi and all higher bis if interval is longer than duration.
                # believe this is only a consideration if ds_interval is None.
                if ds_interval is None:
                    break
            except errors.EndTooEarlyError:
                # as drg_no_limit, will only occur if both start and end are to
                # the left of the calendar bound
                ends_too_early += 1
                continue
            # Note no analogous provision is made for start and end being to the right
            # of the calendar bounds (if were to consider there's a need to implement
            # then would require at least a new `EndOutOfBoundsRightError`).
            bis_factors.append(bi)

        if not bis_factors and ends_too_early == i:
            raise errors.EndOutOfBoundsError(self.calendar_default, None, False)

        if self.has_single_calendar:
            return bis_factors
        else:
            drg = self.gpp.drg_intraday_no_limit
            bis_aligned = []
            for bi in bis_factors:
                drg.interval = bi
                if not self._indices_aligned_for_drg(drg):
                    continue
                bis_aligned.append(bi)
            return bis_aligned

    @property
    def _bis_stored(self) -> list[BI]:
        """Return bis for which data stored for full requested period."""
        bis_valid = self._bis_valid
        drg = self.gpp.drg_intraday
        rtrn = []
        for bi in bis_valid:
            drg.interval = bi
            try:
                daterange = drg.daterange_tight[0]
            except (
                errors.StartTooEarlyError,
                errors.EndTooEarlyError,
                errors.PricesUnavailableIntervalError,
            ):
                continue
            if self._pdata[bi].requested_range(daterange):
                rtrn.append(bi)
        return rtrn

    @property
    def _bis_available_all(self) -> list[BI]:
        """Return bis for which data available over all of period.

        Returns bis for which data is available over the full requested
        period as defined by the current gpp parameters.
        """
        bis_valid = self._bis_valid
        if self.gpp.request_all_available_data:
            return bis_valid

        # Why use the 'no_limit' drg? Becuase do not want dateranges to be
        # curtailed to limits, which would preclude excluding bis where start
        # before left limit or end after right limit.
        # Have looked at other approaches (including a strict drg and considering
        # the errors raised) and conlcluded this is the cleanest way.
        drg = self.gpp.drg_intraday_no_limit
        rtrn: list[BI] = []
        interval_period_errors = []
        starts_too_late = 0
        ends_too_early = 0
        for bi in bis_valid:
            drg.interval = bi
            try:
                start, end = drg.daterange_tight[0]
            except errors.PricesUnavailableIntervalError as err:
                interval_period_errors.append(err)
                continue
            if self._pdata[bi].available_range((start, end)):
                rtrn.append(bi)
            elif start > self.limits[bi][1]:
                starts_too_late += 1
            elif end < self.limits[bi][0]:
                ends_too_early += 1

        if rtrn:
            return rtrn

        len_bis = len(bis_valid)
        if starts_too_late == len_bis or ends_too_early == len_bis:
            raise errors.PricesIntradayUnavailableError(self)

        if not interval_period_errors:
            return rtrn  # return empty list

        # no rtrn and interval_period_errors. Should an interval error
        # be raised or data availability error? If a no_limit drg with T1
        # interval doesn't raise a period / interval error then its AT
        # LEAST a data availability issue (could be a period / interval
        # issue as well). Go with the definite known issue by returning
        # as empty list.
        drg_no_limit = self.gpp.drg_intraday_no_limit
        drg_no_limit.interval = intervals.BI_ONE_MIN
        try:
            drg_no_limit.daterange
        except errors.PricesUnavailableIntervalError:
            raise interval_period_errors[0] from None
        return rtrn

    @property
    def _bis_available_end(self) -> list[BI]:
        """Return bis for which data available at end of period for given parameters.

        Makes no claim on whether data available at start of period for
        returned bis.
        """
        bis_valid = self._bis_valid
        if self.gpp.request_all_available_data:
            return bis_valid

        # Why use the 'no_limit' drg? See comment to `_bis_available_all`
        drg = self.gpp.drg_intraday_no_limit

        rtrn = []
        starts_too_late = 0
        ends_too_early = 0
        for bi in bis_valid:
            drg.interval = bi
            try:
                start, end = drg.daterange_tight[0]
            except errors.PricesUnavailableIntervalError:
                if bi == intervals.BI_ONE_MIN:
                    # ds_interval < interval
                    raise
                continue
            if self._pdata[bi].available(end):
                rtrn.append(bi)
            elif end < self.limits[bi][0]:
                ends_too_early += 1

            if start > self.limits[bi][1]:
                starts_too_late += 1

        if rtrn:
            return rtrn

        len_bis = len(bis_valid)
        if starts_too_late == len_bis or ends_too_early == len_bis:
            raise errors.PricesIntradayUnavailableError(self)

        return rtrn

    @property
    def _bis_available_any(self) -> list[BI]:
        """Return bis for which any data available.

        Returns bis for which any data is available during the period as
        defined by the current gpp parameters.

        Makes no claim on whether data is available throughout the period
        or at either extreme (rather merely that some data is available at
        some point during the period).
        """
        bis_valid = self._bis_valid
        if self.gpp.request_all_available_data:
            return bis_valid

        # Why use the 'no_limit' drg? See comment to `_bis_available_all`
        drg = self.gpp.drg_intraday_no_limit

        rtrn = []
        starts_too_late = 0
        ends_too_early = 0
        for bi in bis_valid:
            drg.interval = bi
            try:
                start, end = drg.daterange_tight[0]
            except errors.PricesUnavailableIntervalError:
                if bi == intervals.BI_ONE_MIN:
                    # ds_interval < interval
                    raise
                continue

            if self._pdata[bi].available_any((start, end)):
                rtrn.append(bi)
            if start > self.limits[bi][1]:
                starts_too_late += 1
            elif end < self.limits[bi][0]:
                ends_too_early += 1

        len_bis = len(bis_valid)
        if starts_too_late == len_bis or ends_too_early == len_bis:
            raise errors.PricesIntradayUnavailableError(self)

        return rtrn

    def _bis_no_partial_indices(self, bis: list[BI]) -> list[BI]:
        """Return those bis of `bis` for which data would have no partial indices.

        Returned bis would serve price data with all indices over the
        requried period fully representing trading periods for a least one
        calendar of `calendars`.

        Notes
        -----
        Method makes no claims on whether price data is available for the
        returned base intervals over the requested period.

        See Also
        --------
        `_bis_available_no_partial_indices`
        `_bis_available_end_no_partial_indices`
        """
        drg = self.gpp.drg_intraday_no_limit
        rtrn = []
        for bi in bis:
            drg.interval = bi
            if not self._has_valid_fully_trading_indices(drg):
                continue
            rtrn.append(bi)
        return rtrn

    @property
    def _bis_available_no_partial_indices(self) -> list[BI]:
        """Return bis for which data available with no partial indices.

        As for `_bis_no_partial_indices` with additional constraint that price data
        available for returned bis over the requested period.
        """
        bis = self._bis_available_all
        bis = self._bis_no_partial_indices(bis)
        return bis

    @property
    def _bis_available_end_no_partial_indices(self) -> list[BI]:
        """Return bis for which data available at end of period with no partial indices.

        As for `_bis_no_partial_indices` save for additional constraint that
        returned bis can serve end of `drg.daterange`. Makes no claims on
        whether returned bis could serve start of `drg.daterange`.
        """
        bis = self._bis_available_end
        bis = self._bis_no_partial_indices(bis)
        return bis

    @property
    def _bis_available_any_no_partial_indices(self) -> list[BI]:
        """Return bis for which data available during period with no partial indices.

        As for `_bis_no_partial_indices` save for additional constraint that
        returned bis can serve data during `drg.daterange`. Makes no claims on
        whether returned bis could serve data throughout all of `drg.daterange`
        or at either extreme.
        """
        bis = self._bis_available_any
        bis = self._bis_no_partial_indices(bis)
        return bis

    def _bis_period_end_now(self, bis: list[BI]) -> list[BI]:
        """Return bis of `bis` for which period end and 'now' evaluate to same value."""
        # Feb 22. Believe this can only ever return all `bis` as received or an empty
        # list. Either the prices were requested to 'now' or they weren't.
        drg = self.gpp.drg_intraday_no_limit
        rtrn = []
        for bi in bis:
            drg.interval = bi
            if drg.daterange[1] == drg.end_now[1]:
                rtrn.append(bi)
        return rtrn

    # Get base intervals that could serve price data for current gpp.

    @property
    def _bis(self) -> list[BI]:
        """Base intervals able to serve prices throughout requested period.

        Base intervals can serve prices throughout the requested period
        with these prices based on the requested anchor. Method makes no
        claim as to the accuracy with which these base intervals may align
        with the end of the requested period.
        """
        if self.gpp.anchor is Anchor.OPEN:
            bis = self._bis_available_all
        else:
            bis = self._bis_available_no_partial_indices
        if not bis:
            raise errors.PricesIntradayUnavailableError(self)
        return bis

    @property
    def _bis_end(self) -> list[BI]:
        """Base intervals able to serve prices at end of requested period.

        Base intervals can serve prices at end the requested period with
        these prices based on the requested anchor. Method makes no
        claim as to the accuracy with which these base intervals may align
        with the end of the requested period.
        """
        if self.gpp.anchor is Anchor.OPEN:
            bis = self._bis_available_end
        else:
            bis = self._bis_available_end_no_partial_indices
        if not bis:
            raise errors.PricesIntradayUnavailableError(self)
        return bis

    @property
    def _bis_any(self) -> list[BI]:
        """Base intervals able to serve prices during requested period.

        Base intervals can serve prices during the requested period with
        these prices based on the requested anchor. Method makes no
        claim as to whether the base intervals can serve prices during the
        whole of the requested period or at either extreme or as to the
        accuracy with which these base intervals may align with the end of
        the requested period.
        """
        if self.gpp.anchor is Anchor.OPEN:
            bis = self._bis_available_any
        else:
            bis = self._bis_available_any_no_partial_indices
        if not bis:
            raise errors.PricesIntradayUnavailableError(self)
        return bis

    # Get a base interval from given base intervals

    def _bis_most_accurate(self, bis: list[BI]) -> list[BI]:
        """Return bis of `bis` able to serve period end with greatest accuracy."""
        bis = copy.copy(bis)

        if not bis:
            return bis

        if self.gpp.anchor is Anchor.OPEN and bis == self._bis_period_end_now(bis):
            return bis  # shortcut

        drg = self.gpp.drg_intraday_no_limit
        ends = []
        for bi in bis:
            drg.interval = bi
            (_, end), end_accuracy = drg.daterange
            ends.append(
                drg.get_end_as_trading_minute_or_nearest_close(end, end_accuracy)
            )

        drg.interval = intervals.BI_ONE_MIN
        most_accurate_end = drg.daterange[1]
        diff = np.abs(pd.DatetimeIndex(ends).asi8 - most_accurate_end.value)
        min_diff = np.min(diff)
        bv = diff == min_diff
        ma_bis = [bis[i] for i, b in enumerate(bv) if b]
        return ma_bis

    @property
    def _bis_end_most_accurate(self) -> list[BI]:
        """Return bis able to represent period end with greatest accuracy.

        Of bis able to serve end of period, return those able to represent
        the period end with the greatest accuracy.
        """
        return self._bis_most_accurate(self._bis_end)

    def _get_stored_bi_from_bis(self, bis: list[BI]) -> BI | None:
        """Return first bi of `bis` able to serve full period from stored data.

        Returns None if full period cannot be served by any bi of `bis`.
        """
        bis_stored = self._bis_stored
        for bi in bis:
            if bi in bis_stored:
                return bi
        return None

    def _get_bis_not_ignoring_breaks(
        self, bis: list[BI], calendar: xcals.ExchangeCalendar
    ) -> list[BI]:
        """Return bis of `bis` that do not ignore breaks for `calendar`."""
        rtrn = []
        for bi in bis:
            if not self._ignore_breaks_cal(calendar, bi):
                rtrn.append(bi)
        return rtrn

    def _get_bi_from_bis(
        self, bis: list[BI], priority: mptypes.Priority | None = None
    ) -> BI:
        """Return a bi of `bis` to serve prices for given `priority`.

        Where more than one bi is equally able to serve prices for the
        given `priority` precedence is given, in order:
            to bis that can serve prices without ignore breaks
            to the highest bis if ds_interval is None or the request is for
                all available data at a specific interval (this provides
                for consistency of return in these circumstances).
            to bis that can serve prices from stored data
            to the highest base interval

        Parameters
        ----------
        bis
            Base intervals that can serve prices over required period (be
            that all, any or only end of requested period).

        priority (default: as `self.gpp.priority`)
            Priority.END: return a bi of `bis` that is able to represent
            the period end with the greatest accuracy.

            Priority.PERIOD: treat accuracy of period end as irrelvant.
        """
        priority = self.gpp.priority if priority is None else priority
        if priority is Priority.END:
            bis = self._bis_most_accurate(bis)
        bis = copy.copy(bis)
        cal = self.gpp.calendar
        pref_bis = self._get_bis_not_ignoring_breaks(bis, cal)
        if pref_bis:
            bis = pref_bis
        if self.gpp.request_earliest_available_data or self.gpp.ds_interval is None:
            # to ensure consistency of return in these circumstances, return highest
            # bi (i.e. ASSUMED as bi with longest available data) regardless of store.
            return bis[-1]
        bis.sort(reverse=True)
        bi = self._get_stored_bi_from_bis(bis)
        return bi if bi is not None else bis[0]

    # Get a table

    def _get_bi_table(self, bi: BI, daterange: mptypes.DateRangeReq) -> pd.DataFrame:
        """Return price table for base interval `bi` over `daterange`.

        Parameters
        ----------
        bi
            Price table base interval.

        daterange
            Date range that price data should cover. NB date range should
            always be evaluated via `daterange.GetterDaily` or
            `daterange.GetterIntraday`.
        """
        table = self._pdata[bi].get_table(daterange)
        assert table is not None  # assuming daterange evaluated from Getter.
        return table

    def _get_bi_table_intraday(self) -> tuple[pd.DataFrame, BI]:
        """Return base interval table for `get` and passed parameters."""
        strict = self.gpp.strict
        # Force strict to True (whilst evaluating bis) when priority is Period and
        # priority has relevance. This ensures bis only include bi that can represent
        # the full period.
        force_strict = self.gpp.priority is Priority.PERIOD and (
            self.gpp.anchor is Anchor.WORKBACK or self.gpp.ds_interval is None
        )
        # pylint: disable=too-many-try-statements
        try:  # get bis that can represent full period
            with self._strict_priority_as(force_strict or strict):
                bis = self._bis
        except errors.PricesIntradayUnavailableError as err:
            if strict or not str(err).startswith("Data is unavailable"):
                # strict or prices unavailable as start and end either both to left or
                # both to right of period over which intraday data is available
                raise
            priority = Priority.PERIOD
            if self.gpp.priority is Priority.END:
                try:
                    bis = self._bis_end
                except errors.PricesIntradayUnavailableError:
                    bis = self._bis_any
                else:
                    priority = Priority.END
            else:
                bis = self._bis_any
            bi = self._get_bi_from_bis(bis, priority=priority)
        else:  # ...from which get bi that can best represent period end, if relevant
            try:
                bis_end = self._bis_end
            except errors.PricesIntradayUnavailableError:
                priority = Priority.PERIOD
            else:
                priority = Priority.END
            bi = self._get_bi_from_bis(bis, priority=priority)
            if self.gpp.priority is Priority.END and priority is Priority.END:
                # if `priority` not END then data not available for end
                accurate_end_bis = self._bis_end_most_accurate
                if bi not in accurate_end_bis:
                    # if end could be represented more accurately by a bi that cannot
                    # represent the full period...
                    if strict:
                        raise errors.LastIndiceInaccurateError(
                            self, bis, accurate_end_bis
                        )
                    else:
                        bi = self._get_bi_from_bis(bis_end)

        # get table for bi
        with self._strict_priority_as(False):
            drg = self.gpp.drg_intraday
            drg.interval = bi
            start, end = drg.daterange_tight[0]

        table = self._get_bi_table(bi, (start, end))
        return table, bi

    def _downsample_bi_table(self, df: pd.DataFrame, bi: intervals.BI) -> pd.DataFrame:
        """Downsample a base interval table to requested ds_interval.

        Only for prices to be anchored on "open".
        """
        ds_interval = self.gpp.ds_interval
        start, end = df.pt.first_ts, df.pt.last_ts + ds_interval
        ignore_breaks = self._ignore_breaks(bi)
        assert isinstance(ds_interval, (BI, TDInterval))
        raise_overlaps, curtail, utc = True, True, False
        target_index = self.cc.trading_index(
            ds_interval,
            start,
            end,
            ignore_breaks,
            raise_overlaps,
            curtail,
            utc,
        )
        bi_index = df.index.left.tz_convert(None)
        target_indices = pd.cut(bi_index.to_list(), target_index)
        target_indices = target_indices.remove_unused_categories()
        agg_f = helpers.agg_funcs(df)
        df = df.groupby(target_indices, observed=False).agg(agg_f)
        df.index = pd.IntervalIndex(df.index)  # convert from CategoricalIndex
        df = helpers.volume_to_na(df)
        df.index = pdutils.interval_index_new_tz(df.index, UTC)
        if df.pt.interval is None:
            # Overlapping indices of a calendar-specific trading trading were curtailed.
            warnings.warn(errors.IntervalIrregularWarning())
        return df

    def _get_table_intraday(self) -> pd.DataFrame:
        """Return intraday price table for `get` and passed parameters."""
        # pylint: disable=too-many-locals, too-complex, too-many-branches
        df_bi, bi = self._get_bi_table_intraday()
        ds_interval = self.gpp.ds_interval
        anchor = self.gpp.anchor

        need_to_downsample = bi != ds_interval and ds_interval is not None

        if anchor is Anchor.OPEN and (
            self.gpp.openend == OpenEnd.SHORTEN or need_to_downsample
        ):
            # Define once here rather than each of the various parts that called.
            drg = self.gpp.drg_intraday
            drg.interval = bi
            daterange = drg.daterange

        if not need_to_downsample:
            df = df_bi
        else:  # downsample
            removed_last_indice = False  # NOTE: only for later temporary assert
            assert ds_interval is not None
            pdfreq = ds_interval.as_pdfreq
            if anchor is Anchor.OPEN:
                # There are two downsample methods for an intraday table anchored
                # to open. Tries the preferred option first, and if that's not viable
                # goes with the other.
                try:
                    # quicker but requires no calendar conflicts at ds_interval, i.e.
                    # wont work for calendars that have no conflicts at the bi although
                    # do at the ds_interval. This applies to conflicts between calendars
                    # and also within a single calendar where the ds_interval results in
                    # an indice at the end of a (sub)session overlapping with next
                    # indice at the start of the following (sub)session.
                    df = self._downsample_bi_table(df_bi, bi)
                except (
                    errors.CompositeIndexConflict,
                    errors.CompositeIndexCalendarConflict,
                ):
                    # not ideal as for any detached calendar will maintain the origin
                    # for each session as the lead symbol's open rather than
                    # realigning to the detached calendar's origin. Only an issue
                    # if there are at least two calendars, there is a calendar conflict
                    # (hence have to use this method) and one calendar is both detached
                    # and unaligned (i.e. would ideally have its own origin for
                    # each session.)
                    df = df_bi.pt.downsample(
                        pdfreq, anchor.value, self.gpp.calendar, False, self.cc
                    )
                    # Will get missing rows between any detached calendar(s) and those
                    # calendars that overlap with the lead symbol's calendar.
                    missing_rows = df.isna().all(axis=1)
                    if missing_rows.any():
                        df = df.drop(missing_rows[missing_rows].index)

                    # Covers at least the possibility that a non-lead symbol trades
                    # immediately following the evaluated period end and the last
                    # session of the lead calendar had a break which as a result of
                    # ignoring (will have been ignored as downsampling via pt) the
                    # indices are unaligned with the pm subsession start. In this case
                    # the last downsampled indice can have a left side before the
                    # evaluated period end and the right after it. As the bi table runs
                    # to the evaluated period end such a last ds indice should only
                    # reflect the period to the evaluated period end. If OpenEnd is
                    # "shorten then this will be handled later below, but if OpenEnd is
                    # "maintain" then any such inaccurate final indice is removed here.
                    if (
                        df_bi.pt.last_ts < df.pt.last_ts
                        and self.gpp.openend is OpenEnd.MAINTAIN
                    ):
                        excess_start = df_bi.pt.last_ts + helpers.ONE_MIN
                        excess_end = df.pt.last_ts
                        if any(
                            (
                                cal.is_trading_minute(excess_start, _parse=False)
                                or cal.is_trading_minute(excess_end, _parse=False)
                            )
                            for cal in self.calendars_unique
                        ):
                            # at least one calendar is open during the excess period
                            # that the downsampled df includes post the end of the bi
                            # df - i.e. symbols traded during this excess period but
                            # their prices are not reflected in the indice. Lose the
                            # excess indice, unless the last indice is a live indice
                            # (in which case excess justified as those excess prices
                            # don't exist yet!).
                            end = daterange[0][1]
                            last_indice_is_live = end >= helpers.now(bi) + bi
                            if not last_indice_is_live:
                                df = df[:-1]
                                removed_last_indice = True  # NOTE: for temporary assert
                                assert df_bi.pt.last_ts > df.pt.last_ts

            else:  # anchor is workback
                df = df_bi.pt.downsample(
                    pdfreq, anchor.value, self.gpp.calendar, False, self.cc
                )

            # TODO This whole clause is an assert...
            # TODO last raised March 22. Lose if hasn't raised by Jan 25 (i.e. one year
            # after changes for `PricesCSV`)
            if anchor is Anchor.OPEN and not removed_last_indice:
                # if removed last indice then already resolved excess rows.
                tables_end_align = (
                    df.pt.last_ts == df_bi.pt.last_ts
                    # Possible if to 'now' and right side of live index of downsampled
                    # table is > right side of live index of bi table...
                    or df.index[-1].left == df_bi.index[-1].left
                )
                accuracy_is_close = (
                    daterange[1].tz_convert(None) in drg.cal.closes.values
                )

                _, end = daterange[0]
                last_indice_is_live = end >= helpers.now(bi) + bi - self.gpp.delay

                shorten_to_close = (
                    accuracy_is_close
                    and self.gpp.openend == OpenEnd.SHORTEN
                    and drg.end_alignment == Alignment.BI
                )
                maintain_to_right_of_close = (
                    accuracy_is_close
                    and self.gpp.openend == OpenEnd.MAINTAIN
                    and drg.end_alignment == Alignment.FINAL
                )
                assert (
                    tables_end_align
                    or shorten_to_close
                    or maintain_to_right_of_close
                    or last_indice_is_live
                ), (
                    f"Assumed there were no excess rows, but there"
                    f" were, such that prices for last indice"
                    f" ({df.index[-1]}) are aggregated from a lesser number"
                    f" of base interval indices than would be necessary to"
                    f" fully reflect the indice."
                )

        # reindex last indice if otherwise unreflective of truth
        if anchor is Anchor.OPEN and self.gpp.openend == OpenEnd.SHORTEN:
            accuracy = daterange[1]
            last = df.index[-1]
            if accuracy < last.right and not drg.cal.is_trading_minute(accuracy):
                # if accuracy < last.right and not a trading minute then
                # the accuracy represents an unaligned close.
                new = pd.Interval(last.left, accuracy, last.closed)
                index = df.index[:-1].insert(len(df) - 1, new)
                df.index = index

        return df

    def _get_table_daily(self, force_ds_daily: bool = False) -> pd.DataFrame:
        """Get daily table for `get` and passed parameters.

        Parameters
        ----------
        force_ds_daily
            True: override gpp downsample interval with 'one day'.
        """
        if self.bi_daily is None:
            raise errors.PricesDailyIntervalError()

        drg = self.gpp.drg_daily_raw if force_ds_daily else self.gpp.drg_daily
        daterange = drg.daterange[0]
        df_bi = self._get_bi_table(intervals.BI_ONE_DAY, daterange)

        ds_interval = self.gpp.ds_interval
        if force_ds_daily:
            downsample = False
        else:
            assert ds_interval is not None
            downsample = not ds_interval.is_one_day

        if not downsample:
            df = df_bi
        else:
            assert ds_interval is not None
            calendar = self.gpp.calendar
            if ds_interval.is_daily:
                pdfreq = ds_interval.freq_value * calendar.day
                df = df_bi.pt.downsample(pdfreq, calendar)

                if df.pt.last_ts > df_bi.pt.last_ts + helpers.ONE_DAY:
                    # Do not suggest last indice includes prices for any sessions of
                    # other calendars that fall between last session of base table
                    # and next session lead calendar.
                    req_final_ts = df_bi.pt.last_ts + helpers.ONE_DAY
                    last_indice = pd.Interval(df.index[-1].left, req_final_ts, "left")
                    index_end = pd.IntervalIndex([last_indice])
                    index = df.index[:-1].union(index_end)
                    df.index = index
            else:  # downsample for monthly
                pdfreq = ds_interval.as_pdfreq
                df = df_bi.pt.downsample(
                    pdfreq, calendar, drop_incomplete_last_indice=False
                )
                if df.pt.first_ts < self.limits[intervals.BI_ONE_DAY][0]:
                    # This can happen if getting all data. As the Getter's .daterange
                    # can return start as None (at least as at April 22). Ideal would
                    # be for the Getter to set the daterange start value based on as if
                    # had receieved pp["start"] as the limit. However, assumes that for
                    # all providers it's possible to know the daily limit to which
                    # prices are available.
                    df = df[1:]
        return df

    def _get_daily_intraday_composite(
        self, table_intraday: pd.DataFrame
    ) -> pd.DataFrame:
        """Return composite of daily table with `table_intraday`.

        Table will start on session corresponding with period start.

        Parameters
        ----------
        table_intraday
            Intraday price table starting on the composite session open.
            Return will concantenate this table at the end of daily table
            that ends on the prior composite session.

        Raises
        ------
        errors.CompositePricesCalendarError
            If overlapping sessions preclude creating a composite table.
        """
        ts = table_intraday.index[0].left
        split_s = self.cc.opens[self.cc.opens == ts].index[0]

        if not self.has_single_calendar:
            previous_sessions = []
            for cal in self.calendars_unique:
                try:
                    previous_sessions.append(cal.previous_session(split_s))
                except xcals.errors.NotSessionError:
                    continue
            prev_sesh = max(previous_sessions)
            latest_prev_close = self.cc.session_close(prev_sesh)
            if latest_prev_close > ts:
                # ts is earlier than split_s open
                raise errors.CompositePricesCalendarError

        table_daily = self._get_table_daily(force_ds_daily=True)
        # up to and exclusive of split_s
        table_daily = table_daily[: split_s - helpers.ONE_DAY]
        table_daily = table_daily.tz_localize(UTC)
        table_daily.index = pd.IntervalIndex.from_arrays(
            table_daily.index, table_daily.index, "left"
        )
        table = pd.concat([table_daily, table_intraday])
        return table

    def _get_table_composite(self) -> pd.DataFrame:
        """Get composite table with max end accuracy for `get` parameters."""
        # pylint: disable=too-many-locals, too-many-statements, too-complex
        # will raise error.PricesIntradayUnavailableError if intraday data not
        # available to meet end.
        with self._strict_priority_as(False, Priority.END):
            table2 = self._get_table_intraday()

        def _get_next_table2(table2: pd.DataFrame) -> pd.DataFrame:
            """Return table2 for next lowest base interval."""
            failed_bi = table2.pt.interval
            bis = self._bis_available_end
            bis.remove(failed_bi)
            if not bis:
                raise errors.PricesIntradayUnavailableError(self)

            bi = self._get_bi_from_bis(bis, mptypes.Priority.END)
            with self._strict_priority_as(False):
                drg = self.gpp.drg_intraday
                drg.interval = bi
                daterange = drg.daterange_tight[0]
            return self._get_bi_table(bi, daterange)

        def _trim_table_to_last_open(
            table: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.Timestamp]:
            """Remove initial `table` rows that fall before last session open."""
            table_end = table.pt.last_ts
            last_session = self.cc.minute_to_sessions(table_end, "previous")[0]
            last_open = self.cc.session_open(last_session)
            return table[last_open:], last_open

        intraday_available = True
        # pylint: disable=too-many-try-statements
        try:
            with self._strict_priority_as(True, Priority.PERIOD) as saved_values:
                table1 = self._get_table_intraday()
        except errors.PricesIntradayUnavailableError:
            # intraday data not available to cover full period.
            # Use daily data for table1.
            if self.gpp.intraday_duration:
                raise
            # context manager won't have exited when error raised
            # pylint: disable=used-before-assignment
            self.gpp.strict, self.gpp.priority = saved_values
            intraday_available = False

        if not intraday_available:
            table2_, last_open = _trim_table_to_last_open(table2)
            if last_open in table2.index.left:
                return self._get_daily_intraday_composite(table2_)
            else:
                # daily/intraday edge case.
                # possible table2_ started later than day open if limit of
                # availability between day open and period end.
                table2 = _get_next_table2(table2)
                table2_, _ = _trim_table_to_last_open(table2)
                return self._get_daily_intraday_composite(table2_)

        # should cover 99% of cases that can be served with an intraday composite table
        try:
            return create_composite(
                (table1, table1.index[0]), (table2, table2.index[-1])
            )
        except ValueError as e:
            msg = "`first` table must preceed and partially overlap `second`."
            if e.args[0] != msg:
                raise

        table1_end = table1.pt.last_ts
        if self.limits[table2.pt.interval][0] > table1_end:
            # edge case 1
            # if period end falls within the first interval of a session (interval as
            # table1) although the availability of data at the table2 base interval is
            # such that the limit currently falls between the prior session close and
            # the session open then table1 will end on the prior close and table2 will
            # start on the following open. Perfectly reasonable to concatentate the
            # tables, although `create_composite` would fail, rightly, as the tables
            # do not overlap.
            table2_start = table2.pt.first_ts
            if table2_start.value in self.gpp.calendar.opens_nanos:
                table1_end = table1.pt.last_ts
                cal = self.gpp.calendar
                if cal.minute_to_trading_minute(table1_end, "next") == table2_start:
                    if not self.has_single_calendar:
                        # table 1 will end at end of calendar session, although want all
                        # prices associated with all calendars through to the start of
                        # table2
                        bi = table1.pt.interval
                        # add an indice to get all the data, then take the last indice
                        # off to leave only the data corresponding to other calendars.
                        end = table2.pt.first_ts + bi
                        table1 = self._get_bi_table(bi, (table1.pt.first_ts, end))[:-1]
                    return pd.concat([table1, table2])

            # edge case 2
            # possible that the limit of availability of the table2 base interval data
            # fell between right of the last table1 indice and the period end according
            # to the table2 indice. Table's don't overlap and there'll be a gap in the
            # data. Create table2 from the next lowest base interval.
            table2 = _get_next_table2(table2)
            if table2.pt.interval == table1.pt.interval:
                raise errors.PricesIntradayUnavailableError(self)
            return create_composite(
                (table1, table1.index[0]), (table2, table2.index[-1])
            )

        assert False, "Something should have returned or an error been raised!!"

    def _force_partial_indices(self, table: pd.DataFrame) -> pd.IntervalIndex:
        """Force parital indices to trading times.

        Tightens indices which include a non-trading period to reflect only
        the trading period. This effects indices that straddle:
            a calendar's session close where no calendar is open during the
            period following the close.

            a calendar's break end, where no calendar is open during the
            period prior to the break end. This circumstance is only
            possible where a calendar has breaks ignored for the `bi`.
        """
        cal = self.gpp.calendar
        start = cal.minute_to_session(table.pt.first_ts)
        end = cal.minute_to_session(table.pt.last_ts)
        nti = self.cc.non_trading_index(start, end, utc=False)
        # operations quicker when all tz-naive
        index = table.pt.set_tz(None).index
        union = index.union(nti, sort=False).sort_values()

        # boolean vector indicating indices that overlap with subsequent
        # non-trading period.
        bv = (union.right[:-1] > union.left[1:]) & union[:-1].isin(index)
        indices_to_force = union[:-1][bv]
        corresponding_nti_indices = union[1:][bv]
        replacement_indices = pd.IntervalIndex.from_arrays(
            indices_to_force.left, corresponding_nti_indices.left, table.index.closed
        )
        indices_to_stay = index.difference(indices_to_force)
        index = indices_to_stay.union(replacement_indices, sort=False).sort_values()

        if self._ignore_breaks_any or not self.has_single_calendar:
            # boolean vector indicating indices that overlap with prior
            # non-trading period.
            bv = (union.left[1:] < union.right[:-1]) & union[1:].isin(index)
            if bv.any():
                indices_to_force = union[1:][bv]
                corresponding_nti_indices = union[:-1][bv]
                replacement_indices = pd.IntervalIndex.from_arrays(
                    corresponding_nti_indices.right,
                    indices_to_force.right,
                    table.index.closed,
                )
                indices_to_stay = index.difference(indices_to_force)
                index = indices_to_stay.union(replacement_indices, sort=False)
                index = index.sort_values()

        index = pdutils.interval_index_new_tz(index, UTC)
        return index

    @staticmethod
    def _inferred_intraday_interval(
        cal: xcals.ExchangeCalendar, pp: mptypes.PP
    ) -> bool:
        """Query if period parmeters infer intraday data required.

        Infers intraday data if:
            - start or end passed as a time (as opposed to a date)
            - `start` passed and start to end represents a period in which
                there are five or less complete sessions. If `end` not
                passed then taken as 'now'.
            - duration defined in terms of minutes and/or hours
            - `days` is 5 or less.
        """
        minutes, hours, days, start, end = (
            pp["minutes"],
            pp["hours"],
            pp["days"],
            pp["start"],
            pp["end"],
        )

        for ts in [start, end]:
            if ts is not None and not helpers.is_date(ts):
                return True

        # neither start nor end is a time...

        if start is not None:
            if end is None:
                end = cal.minute_to_past_session(helpers.now())

            if start > (end - 5 * cal.day):
                return True

        if sum([minutes, hours]) or (0 < days <= 5):
            return True

        return False

    @contextlib.contextmanager
    def _strict_priority_as(
        self, strict: bool | None = None, priority: mptypes.Priority | None = None
    ):
        """Set strict and/or priority.

        During context, sets `self.gpp.strict` to `strict` and
        `self.gpp.priority` to `priority` (in both cases, only if passed).
        Reverts to previous values when context ends.

        Yields tuple with strict and priority values prior to entering
        context.
        """
        strict_saved = self.gpp.strict
        priority_saved = self.gpp.priority
        if strict is not None:
            self.gpp.strict = strict
        if priority is not None:
            self.gpp.priority = priority

        yield strict_saved, priority_saved

        if strict is not None:
            self.gpp.strict = strict_saved
        if priority is not None:
            self.gpp.priority = priority_saved

    @dataclasses.dataclass
    class GetPricesParams:
        """Store for `PricesBase.get` parameters."""

        prices: "PricesBase"
        pp_raw: mptypes.PP
        ds_interval: intervals.PTInterval | None
        lead_symbol: str
        anchor: mptypes.Anchor
        openend: mptypes.OpenEnd
        strict: bool
        priority: mptypes.Priority

        @property
        def calendar(self) -> xcals.ExchangeCalendar:
            """Calendar associated with lead symbol."""
            return self.prices.calendars[self.lead_symbol]

        @property
        def delay(self) -> pd.Timedelta:
            """Delay associated with lead symbol."""
            return self.prices.delays[self.lead_symbol]

        def pp(self, intraday: bool) -> mptypes.PP:
            """Return period parameters.

            Parameters
            ----------
            intraday
                True: return period parameters for intraday price data.
                False: return period parameters for daily price data.
            """
            pp_ = copy.copy(self.pp_raw)
            gregorian = self.ds_interval is not None and self.ds_interval.is_monthly
            mr_minute = (
                None
                if self.prices.live_prices
                else self.prices.latest_requestable_minute
            )
            mr_session = (
                None
                if self.prices.live_prices
                else self.prices.last_requestable_session_any
            )
            pp_["start"], pp_["end"] = parsing.parse_start_end(
                pp_["start"],
                pp_["end"],
                intraday,
                self.calendar,
                self.delay,
                self.strict,
                gregorian,
                mr_session,
                mr_minute,
            )
            return pp_

        @property
        def intraday_limit(self) -> Callable[[intervals.BI], pd.Timestamp]:
            """Callable to get earliest minute from which data can be requested."""

            # pylint: disable=protected-access
            def limit(bi: BI) -> pd.Timestamp:
                return self.prices.limit_intraday_bi_calendar(bi, self.calendar)

            return limit

        @property
        def intraday_limit_right(self) -> Callable[[intervals.BI], pd.Timestamp | None]:
            """Callable to get latest minute for which data can be requested.

            Callable will return None if can request prices through to 'now'.
            """

            def limit_right(bi: BI) -> pd.Timestamp | None:
                limit = self.prices.base_limits_right[bi]
                if limit is None:
                    return None
                elif (
                    limit.value in self.calendar.closes_nanos
                    or limit.value in self.calendar.break_starts_nanos
                ):
                    return limit
                return self.calendar.minute_to_trading_minute(limit, "previous")

            return limit_right

        @property
        def daily_limit(self) -> pd.Timestamp:
            """Earliest session for which daily data can be requested."""
            # pylint: disable=protected-access
            return self.prices._earliest_requestable_calendar_session(self.calendar)

        @property
        def daily_limit_right(self) -> pd.Timestamp | None:
            """Latest session for which daily data can be requested.

            None if can request prices through to 'now'.
            """
            return self.prices.base_limits_right[intervals.BI_ONE_DAY]

        def _drg(
            self,
            intraday: bool,
            limit: pd.Timestamp,
            limit_right: pd.Timestamp | None,
            ds_interval: intervals.PTInterval | None,
            **kwargs,
        ) -> dr.GetterIntraday | dr.GetterDaily:
            pp = self.pp(intraday)
            Cls = dr.GetterIntraday if intraday else dr.GetterDaily
            kwargs.setdefault("strict", self.strict)
            return Cls(
                self.calendar,
                limit=limit,
                pp=pp,
                ds_interval=ds_interval,
                limit_right=limit_right,
                **kwargs,
            )

        @property
        def _end_alignment(self) -> mptypes.Alignment:
            align_bi = self.anchor is Anchor.WORKBACK or self.openend is OpenEnd.SHORTEN
            return Alignment.BI if align_bi else Alignment.FINAL

        @property
        def _drg_intraday_params(self) -> dict:
            # pylint: disable=protected-access
            ignore_breaks = {
                interval: self.prices._ignore_breaks_cal(self.calendar, interval)
                for interval in self.prices.bis_intraday
            }
            return {
                "intraday": True,
                "limit": self.intraday_limit,
                "ds_interval": self.ds_interval,
                "limit_right": self.intraday_limit_right,
                "composite_calendar": self.prices.cc,
                "delay": self.delay,
                "anchor": self.anchor,
                "end_alignment": self._end_alignment,
                "ignore_breaks": ignore_breaks,
            }

        @property
        def drg_intraday(self) -> dr.GetterIntraday:
            """Intraday drg for stored parameters."""
            drg = self._drg(**self._drg_intraday_params)
            assert isinstance(drg, dr.GetterIntraday)
            return drg

        @property
        def drg_intraday_no_limit(self) -> dr.GetterIntraday:
            """Intraday drg for stored parameters without restrictions.

            Strict is False.
            Left limit bound only by calendar.
            """
            kwargs = self._drg_intraday_params
            # pylint: disable=protected-access
            kwargs["limit"] = self.prices._calendars_latest_first_minute
            kwargs["limit_right"] = None
            kwargs["strict"] = False
            drg = self._drg(**kwargs)
            assert isinstance(drg, dr.GetterIntraday)
            return drg

        @property
        def drg_daily(self) -> dr.GetterDaily:
            """Daily drg for stored parameters."""
            drg = self._drg(
                False, self.daily_limit, self.daily_limit_right, self.ds_interval
            )
            assert isinstance(drg, dr.GetterDaily)
            return drg

        @property
        def drg_daily_raw(self) -> dr.GetterDaily:
            """Daily drg for 'one day' downsample interval.

            All other params as stored.
            """
            drg = self._drg(
                False, self.daily_limit, self.daily_limit_right, intervals.ONE_DAY
            )
            assert isinstance(drg, dr.GetterDaily)
            return drg

        @property
        def intraday_duration(self) -> bool:
            """Query if `pp_raw` define duration in minutes and/or hours."""
            return bool(sum([self.pp_raw["hours"], self.pp_raw["minutes"]]))

        @property
        def duration(self) -> bool:
            """Query if `pp_raw` define a duration."""
            duration_params = [
                self.pp_raw["minutes"],
                self.pp_raw["hours"],
                self.pp_raw["days"],
                self.pp_raw["weeks"],
                self.pp_raw["months"],
                self.pp_raw["years"],
            ]
            return bool(sum(duration_params))

        @property
        def request_earliest_available_data(self) -> bool:
            """Query if params represent request for earliest available data."""
            return self.pp_raw["start"] is None and not self.duration

        @property
        def request_all_available_data(self) -> bool:
            """Query if params represent request for all available data."""
            return (
                self.pp_raw["start"] is None
                and self.pp_raw["end"] is None
                and not self.duration
            )

    @parse
    def get(
        self,
        interval: Annotated[
            str | pd.Timedelta | datetime.timedelta | None,
            Parser(intervals.parse_interval, parse_none=False),
        ] = None,
        start: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
        ] = None,
        end: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
        ] = None,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
        weeks: int = 0,
        months: int = 0,
        years: int = 0,
        add_a_row: bool = False,
        lead_symbol: Annotated[str | None, Parser(parsing.lead_symbol)] = None,
        tzin: Annotated[
            str | ZoneInfo | None,
            Parser(parsing.to_prices_timezone, parse_none=False),
        ] = None,
        anchor: Literal["workback", "open"] = "open",
        openend: Literal["maintain", "shorten"] = "maintain",
        priority: Literal["period", "end"] = "end",
        strict: bool = True,
        composite: bool = False,
        force: bool = False,
        tzout: Annotated[
            str | ZoneInfo | None,
            Parser(parsing.to_prices_timezone, parse_none=False),
        ] = None,
        fill: Literal["ffill", "bfill", "both"] | None = None,
        include: mptypes.Symbols | None = None,
        exclude: mptypes.Symbols | None = None,
        side: Literal["left", "right"] | None = None,
        close_only: bool = False,
        lose_single_symbol: bool = False,
    ) -> pd.DataFrame:
        """Get price data.

        Quickuse examples:

            p = PricesYahoo('AMZN GOOG AZN.L')
            p.get('5min', days=5) last five trading days at five minute
                intervals.
            p.get('10T', weeks=2) last two calendar weeks at ten minute
                intervals.
            p.get(days=5) last five trading days at inferred interval.
            p.get('1d', months=6) daily data for last six months.
            p.get('1h', months=3, end='2021-03-17 15:00') three months at
                hourly intervals to defined datetime (timezone of 'end'
                assumed as most common timezone of all symbols).

        The period over which to get prices can be defined as:

            Either the period between `start` and `end` (end defaults to
            'now' if only `start` is passed).

            Or a duration bound by either `start` or `end`. The duration
            can be defined in terms of any of:
                trading time: pass any combination of `minutes` and
                    `hours`. For a symbol that trades 8 hours a day,
                    hours=24 will return 4 days of data, not 1 day.

                trading sessions: pass `days` (a session is the trading
                    period associated with a specific date).

                calendar time: pass any combination of `weeks`, `months`
                    and `years`.

            It is not possible to combine duration parameters that define
            duration in different terms. For example, the following are
            invalid combinations of period parameters:
                `minutes` and `days`
                `hours` and `days`
                `days` and `months`

        Examples of periods defined by valid combinations of parameters:
            (start='2021-04-15 12:30', end='2021-11-17 16:30') from `start`
                through `end`.
            (start='2021-04-15 12:30') from `start` through to most recent
                available datetime.
            (end='2021-11-17 16:30') from earliest available datetime
                through `end`.
            (hours=2, minutes=30, end='2021-11-17 16:30') the two and a
                half trading hours to 16.30 on 17th November 2021.
            (hours=2, minutes=30, start='2021-04-14 12:30') the two and a
                half trading hours from 12.30 on 14th April 2021.
            (hours=12) the 12 trading hours to the most recent available
                datetime. For many instruments this data will cross more
                than one session.
            (days=20) the 20 consecutive trading days immediately
                preceeding the most recent available datetime.
            (months=3, weeks=2) the 3 months and 2 weeks preceeding most
                recent available datetime.
            () from earliest available datetime through most recent
                available datetime, i.e. all available data.

        Data will NEVER be returned for any date or datetime that falls
        outside of the requested or evaluated `start` / `end` bounds. If
        available data does not align with the bounds then the returned
        data will start and/or end within the bounds, never outside of
        them.
            If `start` or `end` is unaligned and combined with duration
            parameters then the other endpoint is evaluated from the
            algined start / end, not from `start`/`end` as passed.

        Parameters
        ----------
        - Period Parameters -
        These parameters define the period over which to get prices.

        start : pd.Timestamp | str | datetime.datetime | int | float | None
        default: earliest available datetime (only if `start` required)
            The first date or minute of the period for which to get
            prices.

            Defining `start` as a date:
                - `start` should be passed as a date, with no time
                    component (or have time component defined as 00:00).
                - If passing as a pd.Timestamp, it should be tz-naive.
                - if `start` does not represent an actual session the
                    period will start at the start of the closest session
                    that follows `start`.

            Defining `start` as a minute:
                - `start` should be defined with a time component.
                - to defne as midnight, pass a tz-aware pd.Timestamp.
                - the timezone of the minute will be assumed as the
                    timestamp's timezone, if passed as a tz-aware
                    pd.Timestamp, or otherwise as `tzin`.

        end : pd.Timestamp | str | datetime.datetime | int | float | None
        default: most recent available datetime (only if `end` required)
            The last date or minute of the period for which to get
            prices.

            Defining `end` as a date:
                - `end` should be passed as a date, with no time
                    component (or have time component defined as 00:00).
                - If passing as a pd.Timestamp, it should be tz-naive.
                - if `end` does not represent an actual session the period
                    will end at the end of the closest session that
                    preceeds `end`.

            Defining `end` as a minute:
                - `end` should be defined with a time component.
                - to defne as midnight, pass a tz-aware pd.Timestamp.
                - the timezone of the minute will be assumed as the
                    timestamp's timezone, if passed as a tz-aware
                    pd.Timestamp, or otherwise as `tzin`.

            Note: If `end` parses to a datetime later than 'now' then
            period end will be assumed as 'now'.

        minutes : int, default: 0
            Period duration in minutes. Can be combined with `hours`.

        hours : int, default: 0
            Period duration in hours. Can be combined with `minutes`.

        days : int, default: 0
            Period duration in trading days. Can not be combined with any
            other duration parameter.

        weeks : int, default: 0
            Period duration in calendar weeks. Can be combined with
            `months` and `years`.

        weeks : int, default: 0
            Period duration in calendar months. Can be combined with
            `weeks` and `years`.

        years : int, default: 0
            Period duration in calendar years. Can be combined with
            `weeks` and `months`.

        tzin : str | BaseTzinfo | None,
        default: timezone of any `lead_symbol`, otherwise `self.default_tz`
            Timezone of any input to `start` and `end` that represents a
            minute (as opposed to a session).

            Can be passed as a timezone defined as a `str` that's valid
            input to `zoneinfo.ZoneInfo`, for example 'UTC' or
            'US/Eastern`, or as an instance of `zoneinfo.ZoneInfo`.

            Can alternatively be passed as any symbol of `self.symbols`
            to define as timezone associated with that symbol, for example
            "GOOG".

        add_a_row : bool, default: False
            True:
                Include indice immediately prior to evaluated start.

            This option is useful if evaluating a change over the period.
            For example, if evaluating change over 1 day then will require
            2 days of data (if basing change on difference between close
            prices).

        See period.ipynb tutorial for further explanation and examples of
        period parameters.

        - Parameters related to index -

        interval : str | timedelta | pd.Timedelta | None,
        default: None (interval inferred)
            Time interval to be represented by each price row.

            Pass as either:
                pd.Timedelta: components as either:
                    - minutes and/or hours
                    - days

                timedelta: defined from kwargs passed as either:
                    - minutes and/or hours
                    - days
                    (or equivalent of, i.e seconds=120 is valid although
                    seconds=121 is not.)

                str: comprising:
                    value:
                        one or more digits.
                    unit:
                        "min", "MIN", "T" or "t" for minutes
                        "h" or "H" for hours
                        "d" or "D' for days
                        'm' or "M" for months

            Examples:
                thirty minutes:
                    "30min", "30T"
                    pd.Timedelta(30, "min"), pd.Timedelta(minutes=30)
                    timedelta(minutes=30)

                three hours:
                    "3h", "3H"
                    pd.Timedelta(3, "h"), pd.Timedelta(hours=3)
                    timedelta(hours=3)

                one day:
                    "1d", "1D"
                    pd.Timedelta(1, "D"), pd.Timedelta(days=1)
                    timedelta(days=1), timedelta(1)

                two months:
                    "2m", "2M"

                others:
                    "90min", "90MIN" - ninety minutes
                    "5D", "5d" - five days
                    "40D", "40d" - forty days
                    "1M", "1m" - one month
                    pd.Timedelta(hours=3, minutes=30) - three and a half
                        hours
                    pd.timedelta(hours=1, minutes=20) - one hour and twenty
                        minutes

            Intervals representing months must be defined as a string.

            Value for any unit must not be higher than the following:
            limits:
                minutes: 1320
                hours: 22
                days: 250
                months: 36

            If `interval` is not passed then a suitable interval will be
            inferred from period parameters and available data.

            See intervals.ipynb tutorial for further explanation and
            examples.

        lead_symbol : str | None,
        default: symbol associated with most common calendar
            A symbol associated with the calendar that should be used to
            evaluate the period over which prices are returned.

            See periods.ipynb tutorial for further explanation and
            examples of passing lead_symbol.

        anchor : Literal["open", "workback"], default: "open"
            Ignored if interval is (or inferred as) daily or higher.

            Determines the origin (i.e. anchor) against which indices
            should be evaluated.

            "open":
                Group indices by session and anchor the left side of an
                indice on each session open (where session open evaluted as
                for the `lead_symbol`).

                Intervals will all have length `interval`. Only
                circumstance to the contrary is if this would result in the
                last indice of a (sub)session overlapping with the first
                indice of the next (sub)session. In this case a Warning is
                raised to advise that some indices have been curtailed to
                avoid overlapping.

                If `end` infers 'now' and any calendar is open then 'now'
                will fall within the last indice. This last indice will be
                'live' and will update on any further request.

                (Sub)sessions' closes will only be represented by the right
                side of an indice if the interval is a factor of the
                (sub)session duration. Otherwise, the last indice that
                covers part of the (sub)session will have a right side that
                falls after the (sub)session close. Hence, only part of the
                interval covered by this last (sub)session indice will
                represent a trading period (at least as evaluated against
                the `lead_symbol` calendar). NB The `force` option provides
                for curtailing such indices to (sub)session closes.

                If symbols are assoicated with more than one calendar then
                indices may continue after a (sub)session close and/or
                before a (sub)session open (where session evaluated against
                the lead-).

                See 'See Also' section for methods to interrogate nature of
                indices.

            "workback":
                Anchors right side of final interval on the evaluted period
                end and works backwards. Indices evaluated such that each
                has the same interval in terms of trading minutes (where
                trading minutes comprise the union of all trading minutes
                of all calendars).

                Note: If `strict` is False then "workback" can result
                in price data being returned over a significantly
                lesser period than if anchor were "open" (albeit with a
                potentially more accurate last indice). This
                possibility is most notable at higher intervals. See the
                data_availability.ipynb tutorial for examples.

            See anchor.ipynb tutorial for further explanation and examples.

        openend : Literal["maintain", "shorten"], default: "maintain"
            Only relevant if anchor is 'open', otherwise ignored.

            Determines how the final indice should be defined if `end`
            evaluates to a session close (as evaluated against
            `lead_symbol` calendar) which does not align with the indices:

                "maintain" (default) (maintain the interval):
                    The final indice will have length `interval`.

                    Considering the period between the session close and
                    the right of the indice that contains the session
                    close:
                        If no symbol trades during this period then the
                        session close, such that the right side of the
                        final indice will be the indice that contains the
                        final indice will be defined to the right of the
                        session close.

                        If any symbol trades during this period then the
                        final indice will be the latest indice with a
                        right side that preceeds the session close.
                        Consequently the right of the final indice will be
                        to the left of the session close and prices at the
                        session close will not be included.

                    Note: The final indice may still be shortened if
                    `force` is True.

                "shorten":
                    Define final indice shorter than `interval` in order
                    that right side of final indice aligns with session
                    close.

                    The final interval will only be defined in this way
                    if either:
                        No symbol trades during the part of the final
                        indice that falls after the session close.

                        Data is available to create the table by
                        downsampling data of a base interval that aligns
                        with the session close.

                    If it is not possible to define a shorter indice then
                    the final indice will be defined as for "maintain".

            NOTE: In no event will the final indice include prices
            registered after the evaluated period end.

            See anchor.ipynb tutorial for further explanation and examples.

        priority : Literal["period", "end"], default: "end"
            Only relevant if `anchor` is "workback" or `interval` is None.
            Ignored if `composite` is True.

            What should be prioritised when it is possible to only return
            prices EITHER for the full requested period OR with a final
            indice that aligns exactly with the period end.

                "period":
                    Prioritise period length at the expense of the final
                    indice expressing the period end with lesser accuracy
                    than would be possible with a smaller interval.

                "end" (default):
                    Prioritise the accuracy with which the final indice can
                    express the period end at the expense of returning
                    prices for only the later part of the requested period.

                    NB In this case prices will only be returned if
                    `strict` is False (if `strict` is True then a
                    `errors.LastIndiceInaccurateError` will be raised).

                    NB If price data is not available at the end of the
                    requested period then passing 'end' will have no effect
                    and price data will be returned based on maximising
                    period length.

            See data_availability.ipynb tutorial for further explanation
            and example usage.

        strict : bool, default: True
            (forced to True if `composite` True)

            What to do if data is only available over part of the
            requested period.
                True (default): raise an error (subclass of
                errors.PricesUnavailableError).

                False: return prices for the part of the period for which
                data is available.

            See data_availability.ipynb tutorial for example usage.

        composite : bool, default: False
            Only available if `anchor` is 'open' and `interval` is None.

            Should a composite table be returned if this would represent
            the period end with greater accuracy.

                True:
                    Return a composite price table if this allows for the
                    period end to be represented with a greater accuracy
                    than a table with a regular interval.

                    Composite table will comprise two intervals, a higher
                    interval for which price data is available over the
                    full requested period and a lower interval that can
                    express the period end with the greatest possible
                    accuracy.

                    ValueError will be raised if `interval` is not None
                    or `anchor` is not "open".

                False (default):
                    Do not return a composite table.

            See data_availability.ipynb tutorial for example usage.

        force : bool, default: False
            Only available if anchor is 'open'.
            Not relevant if interval is daily or higher.

            True:
                Force indices to only cover periods during which at least
                one symbol trades.

                If the left side of the first indice of a session is
                earlier that the earliest session open of any calendar
                then the indice will be curtailed with the left side being
                forced forward to the earliest session open.

                If the right side of the last indice of a session is
                later that the latest session close of any calendar then
                the indice will be curtailed with the right side being
                forced back to the latest session close.

                Any indice that partially covers a session break will
                similarly by curtailed if no other symbol trades during the
                period that conincides with the break. Indices that include
                the morning subsession close will have their right side
                forced back to that close whilst indices that include the
                afternoon subsession open will have their left side forced
                forward to that open.

            Notes:

                Any indice that is forced will be shorter than `interval`
                and the table interval will become irregular.

                ValueError will be raised if `force` True and `anchor` is
                passed as "workback".

                See 'See Also' section for methods to interrogate trading
                status of indices.

            See anchor.ipynb tutorial for further explanation and examples
            of using the `force` option.

         - Post-processing options (formatting and tidying) -

        tzout : str | BaseTzinfo | None,
        default: as `tzin` if `interval` intraday, otherwise None
            Timezone to set index to.

            If interval daily or higher:

                Can only accept "utc", `zoneinfo.ZoneInfo("UTC")` or (for
                tz-naive dates) None.

            If interval intraday:

                Can be passed as a timezone defined as a `str` that's valid
                input to `zoneinfo.ZoneInfo`, for example 'UTC' or
                'US/Eastern`, or as an instance returned by
                `zoneinfo.ZoneInfo`.

                Can alternatively be passed as any symbol of `self.symbols`
                to define as timezone associated with that symbol, for
                example "GOOG".

            Note: If table is returned as a composite comprising both daily
            and intraday data then `tzout` will be forced to "UTC". In this
            case it should be noted that prices over the daily part of the
            table reflect session prices, not prices between successive
            midnight UTC.

        fill : Literal["ffill", "bfill", "both"] | None, default: None
            Fill missing values where a symbol's calendar is not open
            during the interval covered by an indice.

                "ffill": fill 'open', 'high', 'low' and 'open' values with
                closest prior close value.

                "bfill": fill 'open', 'high', 'low' and 'open' values with
                closest subsequent open value.

                "both": first fill as "ffill" and then fill any missing
                values at the start of the table with "bfill".

                None: (default) do not fill.

        include : list[str] | str | None
            Symbol or symbols to include. All other symbols will be
            excluded. If passed, do not pass `exclude`.

        exclude : list[str] | str | None
            Symbol or symbols to exclude. All other symbols will be
            included. If passed, do not pass `include`.

        side : Literal['left', 'right'] | None, default: None
            Ignored if interval is (or inferred as) daily.

            Determines index:

                None (default):
                    Index with IntervalIndex.

                "left":
                    Index with DatetimeIndex representing the left side of
                    each interval.

                "right":
                    Index with DatetimeIndex representing the right side of
                    each interval.

            Note: If passed as "left" or "right" then returned DataFrame
            will not have access to the .pt accessor.

        close_only : bool, default: False

            True:
                Return only close price columns. All other columns will be
                removed. Columns will be indexed by symbol only.
                Returned DataFrame will not have access to the .pt
                accessor.

            False (default):
                Return 'open', 'high', 'low', 'close' and 'volume' columns.

        lose_single_symbol : bool, default: False
            Ignored if `close_only` is True.

            True:
                If prices are for a single symbol then will lose symbol
                level from columns MultiIndex. Columns will be labelled
                instead with simple pd.Index.

        Raises
        ------
        `ValueError`
            If an invalid combination of parameters are passed.

            This includes any of the following invalid combinations of
            period parameters:
                `end` and `start` and a duration parameter.
                `minutes` and/or `hours` and any other duration parameter.
                `days` and any other duration parameter.

        `errors.PricesUnavailableError`
            If price data is not available for the passed parameters then a
            subclass of `errors.PricesUnavailableError` will be raised
            which describes the specific cause.

                errors.PricesDateRangeEmpty:
                        Requested period contains no trading session/minutes.

                errors.PricesIntradayUnavailableError:
                    Intraday price data is not available over the full
                    requested period for the passed (or inferred) interval.

                errors.LastIndiceInaccurateError:
                    Intraday price data is not available over the full
                    requested period at a sufficiently low interval to
                    represent the period end with the greatest possible
                    accuracy.

                    Can only be raised if `strict` is True and `priority`
                    is "end".

                errors.PricesUnavailableFromSourceError:
                    Price data not returned from source over a period for
                    which prices would be expected to be available.

                errors.PricesUnavailableIntervalDurationError:
                    `interval` is greater than a period duration defined in
                    terms of 'minutes' and/or 'hours'.

                errors.PricesUnavailableIntervalPeriodError:
                    An intraday or daily `interval` is longer than the
                    evaluated period.

                    Note: This error can be raised if `anchor` is "open"
                    and `start` and `end` are values that cover a period
                    greater than the interval although the indice bounds
                    they evaluate to do not. For example, if a session
                    opens at 09.00, `start` is 09.01 and `end` is 09.09
                    then a call for data at a "5min" `interval` will
                    evalute the period bounds as from 09.05 through 09.05.

                errors.PricesUnavailableDOIntervalPeriodError:
                    Monthly `interval` is longer than the evaluated period.

                errors.PricesUnavailableInferredIntervalError
                    `interval` is None and inferred as intraday although
                    the request can only be met with daily data.

                errors.PricesUnvailableDurationConflict:
                    Interval is daily although duration specified in hours
                    and/or minutes.

                errors.StartNotEarlierThanEnd:
                    `start` passed as a date or minute that is later than
                    the passed `end`.

                errors.EndTooEarlyError:
                    `end` is or resolves to a session/minute before the
                    first session/minute for which prices are available
                    at any interval.

                    Note: Call `limits` for limits by base interval.

                    Note: errors.EndOutOfBoundsError will be raised if
                    session/minute is earlier than the first session/minute
                    of the associated calendar.

                errors.StartTooEarlyError:
                    `start` is or resolves to a session/minute before the
                    first session/minute for which prices are available at
                    any interval.

                    Note: Call `limits` for limits by base interval.

                    Note: errors.StartOutOfBoundsError will be raised if
                    session/minute is earlier than the first session/minute
                    of the associated calendar.

                errors.EndOutOfBoundsError:
                    `end` is or resolves to a session/minute earlier than
                    the associated calendar's first session/minute.

                errors.StartOutOfBoundsError:
                    `start` is or resolves to a session/minute earlier than
                    the associated calendar's first session/minute.

                errors.StartTooLateError:
                    `start` is or resolves to a session/minute later than
                    the most recent session/minute for which prices are
                    available.

        Warnings
        --------
        A subclass of `errors.PricesWarning` will be raised to offer
        advices in the event an operation has been undertaken that could
        result in prices differenting from what user may have reasonably
        assumed:

            errors.PricesMissingWarning:
                Prices not returned from source for at least one session
                for which prices were expected to be available at a
                specific base interval. Warning message advises of those
                sessions for which data is missing.

            errors.IntervalIrregularWarning:
                Interval irregular as a result of the last indice of one or
                more sessions being curtailed so as not to overlap with the
                first indice of the following session.

                This warning can not be raised when all symbols are
                associated with the same calendar.

            errors.DuplicateIndexWarning:
                Indices were removed from data received from source as a
                result of duplications. Warning message includes advices of
                removed indices.

        Returns
        -------
        pd.DataFrame

            Return prices as a pd.DataFrame with a .pt accessor:

                index : pd.IntervalIndex | pd.DatetimeIndex

                    If `interval` passed or infered as intraday:
                        pd.IntervalIndex with each row covering prices for
                        time interval defined by the indice (closed left).

                    if `interval` passed or infered as daily:
                        pd.DatetimeIndex with each row covering prices for
                        session indicated by indice.

                    if `interval` greater than one day:
                        pd.IntervalIndex with each row covering prices over
                        sessions defined by indice (closed left such that
                        prices exclusive of session defined as indices'
                        right side).

                columns : pd.MultiIndex
                    level 0: symbol
                    level 1: ['open', 'high', 'low', 'close', 'volume']

            - pd.DataFrame.pt PricesTable accessor -
            The .pt accessor offers price-related functionality via access
            to the methods of one of the following classes:
                `pt.PTDaily`: if price data is daily (i.e. by session).

                `pt.PTIntraday`: if price data is intraday.

                `pt.PTMultipleSessions`: if price data has an interval
                higher than one day.

                `pt.PTDailyIntradayComposite`: if price table is a
                composite where first part of table has a one day interval
                and final part has an intraday interval.

            Note: .pt accessor will not be available if:
                `side` not None and `interval` not daily (or inferred as
                daily).
                `close_only` passed as True.

            See following 'See Also' section for example usage.
            See pt_accessor.ipynb tutorial for examples of all .pt methods.

        See Also
        --------
        Tutorials are vailable that offer comprehensive explanation and
        example usage of the `get` method. See the project home page.

        The following .pt methods can be used to interrogate whether indices
        cover trading periods, non-trading periods or partial trading
        periods (where an exchange traded during only part of the period
        covered by the indice):
            .indices_trading_status()
            .indices_all_trading()
            .indices_trading()
            .indices_non_trading()
            .indices_partial_trading()
            .indices_partial_trading_info()

        The following .pt methods can be used to interrogate the number of
        trading minutes associated with indices (`pt.PTIntraday` only):
            indices_trading_minutes()
            indices_trading_minutes_values()
            trading_minutes_interval()
            indices_have_regular_trading_minutes()

        The following .pt properties can be used to interrogate the length
        of indice intervals (`pt.PTIntraday` only):
            indices_length
            by_indice_length

        Example usage:
            prices = PricesYahoo('GOOG AZN.L', lead_calendar='GOOG')
            table = prices.get(days=5)
            table.pt.indices_non_trading(calendar=prices.calendars['GOOG'])
            table.pt.indices_partial_trading(
                calendar=prices.calendars['AZN.L']
            )

        See 'Notes' section of PricesBase.__doc__ for under-the-bonnet
        notes explaining how price data is served.
        """
        # pylint: disable=too-complex, too-many-arguments, too-many-locals
        # pylint: disable=too-many-branches, too-many-statements, missing-param-doc
        # pylint: disable=differing-type-doc

        if TYPE_CHECKING:
            assert start is None or isinstance(start, pd.Timestamp)
            assert end is None or isinstance(start, pd.Timestamp)
            assert isinstance(lead_symbol, str)
            assert tzin is None or isinstance(tzin, ZoneInfo)
            assert tzout is None or isinstance(tzout, ZoneInfo)

        anchor_ = Anchor.WORKBACK if anchor.lower() == "workback" else Anchor.OPEN
        openend_ = OpenEnd.SHORTEN if openend.lower() == "shorten" else OpenEnd.MAINTAIN
        priority_ = Priority.PERIOD if priority.lower() == "period" else Priority.END

        if force and anchor_ is Anchor.WORKBACK:
            raise ValueError("Cannot force close when anchor is 'workback'.")

        if composite:
            if anchor_ is Anchor.WORKBACK:
                raise ValueError(
                    "Cannot create a composite table when anchor is 'workback'."
                )
            elif interval is not None:
                msg = (
                    "Cannot pass an interval for a composite table, although"
                    f" receieved interval as {interval}."
                )
                raise ValueError(msg)

        if tzin is None:
            tzin = self.timezones[lead_symbol]

        if start is not None:
            start = parsing.parse_timestamp(start, tzin)

        if end is not None:
            end = parsing.parse_timestamp(end, tzin)

        pp: mptypes.PP = {
            "minutes": minutes,
            "hours": hours,
            "days": days,
            "weeks": weeks,
            "months": months,
            "years": years,
            "start": start,
            "end": end,
            "add_a_row": add_a_row,
        }
        parsing.verify_period_parameters(pp)

        cal = self.calendars[lead_symbol]

        interval_: intervals.PTInterval | None
        if interval is None and not self._inferred_intraday_interval(cal, pp):
            interval_ = intervals.ONE_DAY
        else:
            if TYPE_CHECKING:
                assert interval is None or isinstance(
                    interval, (TDInterval, intervals.DOInterval, BI)
                )
            interval_ = interval

        self._gpp = self.GetPricesParams(
            self, pp, interval_, lead_symbol, anchor_, openend_, strict, priority_
        )

        table = None
        if interval_ is not None and not interval_.is_intraday:
            # get a daily table
            if self.gpp.intraday_duration:
                raise errors.PricesUnvailableDurationConflict()
            table = self._get_table_daily()
        elif not self.bis_intraday:
            if interval_ is not None:
                raise errors.PricesIntradayIntervalError()
            table = self._get_table_daily(force_ds_daily=True)
        else:
            # get an intraday table
            # if composite, only interested in a non-composite table if end
            # represented with max posible accuracy and can serve full period.
            with self._strict_priority_as(
                True if composite else strict, Priority.END if composite else priority_
            ):
                try:
                    table = self._get_table_intraday()
                except errors.PricesIntradayUnavailableError as e:
                    orig_err = e
                    if interval_ is not None:
                        raise
                    if not composite:
                        if isinstance(e, errors.LastIndiceInaccurateError):
                            # Interval None, can only offer EITHER data over less than
                            # full period and most accurate end OR data over full period
                            # but not most accurate end. Composite not wanted. Can't
                            # meet request.
                            assert priority_ is Priority.END and strict is True
                            raise
                        elif priority_ is Priority.END:
                            # Want maximum end accuracy but not a composite table.
                            try:
                                bis_acc = self._bis_end_most_accurate
                            except errors.PricesIntradayUnavailableError:
                                # intraday prices not available at end of period,
                                # try to fulfill from daily (which now represents
                                # greatest end accuracy)
                                pass
                            else:
                                assert bis_acc
                                # the period end can be represented with intraday data
                                # although period start can only be met with daily data
                                # (if period start could be met with intraday data then
                                # would have raised `LastIndiceInaccurateError`).
                                # Daily/intraday composite not wanted. Can't meet
                                # request.
                                raise errors.LastIndiceInaccurateError(
                                    self, [], bis_acc
                                ) from None

            if table is None and composite:
                try:
                    table = self._get_table_composite()
                except errors.PricesIntradayUnavailableError:
                    # intraday prices unavailable, serve request from daily if available
                    pass
                except errors.PricesDailyIntervalError:
                    raise orig_err from None

            if table is None:
                # interval inferred, intraday table unavailable at a single
                # interval to fulfil request, composite either not wanted or
                # not available, so try to serve from daily.
                try:
                    table = self._get_table_daily(force_ds_daily=True)
                except errors.PricesUnavailableError:
                    raise orig_err from None
                else:
                    if self._inferred_intraday_interval(cal, pp):
                        # daily available but parameters suggest wasn't wanted
                        raise errors.PricesUnavailableInferredIntervalError(
                            pp
                        ) from None

        # TODO Lose assertion Jan 25 if hasn't raised following Jan 24 changes
        # to accommodate CSVs.
        assert table is not None and not table.empty

        if force and table.pt.is_intraday and anchor_ is Anchor.OPEN:
            table.index = self._force_partial_indices(table)

        if table.pt.is_intraday:
            tzout_: ZoneInfo | Literal[False] = tzin if tzout is None else tzout
        elif tzout is not UTC:
            # if tzout tz aware or None
            tzout_ = False  # output as default tz
        else:
            tzout_ = tzout

        return table.pt.operate(
            tzout_,
            fill,
            include,
            exclude,
            side=side,
            close_only=close_only,
            lose_single_symbol=lose_single_symbol,
        )

    def request_all_prices(self) -> dict[BI, list[pd.Interval]]:
        """Request all available prices at all base intervals.

        Returns
        -------
        dict
            key:
                Base interval.
            value:
                Date range over which data now stored locally.
        """
        for bi in self.bis:
            try:
                self.get(bi)
            except errors.PricesIntradayUnavailableError:
                limit = TDInterval.T10
                if not self.has_single_calendar and bi > limit:
                    # indices may be unaligned at bi. Would not expect this if bi <= T10
                    continue
                raise
        return self._pdata_ranges

    @parse
    def session_prices(
        self,
        session: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_datetimestamp, parse_none=False),
        ] = None,
        stack: bool = False,
    ) -> pd.DataFrame:
        """Return prices for specific session.

        Parameters
        ----------
        session:
            pd.Timestamp | str | datetime.datetime | datetime.date
            | int | float | None
        default: most recent available session
            Session to return prices for.

            Must be passd as a date, with no time component (or have time
            component defined as 00:00). If passsed as a pd.Timestamp,
            must be tz-naive.

            `session` must represent a session of at least one calendar
            of `self.calendars_unique`.

        stack: bool, default: False
            Stack symbols to separate rows. Same as calling `.pt.stacked`
            on the default return.

        Returns
        -------
        pd.DataFrame
            If `stack` False (default):
                index: pd.DatatimeIndex
                    Single indice expressing session that prices relate to.

                columns: pd.MultiIndex
                    level 0: symbol
                    level 1: 'open', 'high', 'low', 'close', 'volume'

            If `stack` True:
                index: pd.MultiIndex
                    level 0: DatetimeIndex
                        Single indice expressing session that prices relate to.

                    level 1: pd.Index
                        symbols.

                columns: pd.Index
                    'open', 'close', 'high' 'low' 'volume'.

        See Also
        --------
        `close_at`: Prices at at end of a specific 'day'.
        `price_at`: Prices as at a specific minute.

        See `specific_query_methods.ipynb` tutorial for example usage.
        """
        # pylint: disable=missing-param-doc, differing-type-doc
        if TYPE_CHECKING:
            assert session is None or isinstance(session, pd.Timestamp)

        if self.bi_daily is None:
            raise errors.MethodUnavailableNoDailyInterval("session_prices")

        mr_session = self.last_requestable_session_any
        if session is None:
            table = self._get_bi_table(self.bi_daily, (mr_session, mr_session))
            return table.pt.stacked if stack else table

        first_session = self.earliest_requestable_session
        parsing.verify_date_not_oob(session, first_session, mr_session, "session")

        if not any(cal.is_session(session) for cal in self.calendars_unique):
            msg = f"{session} is not a session of any associated calendar."
            raise ValueError(msg)

        table = self._get_bi_table(self.bi_daily, (session, session))
        return table.pt.stacked if stack else table

    def _date_to_session(
        self,
        date: pd.Timestamp,
        extreme: Literal["earliest", "latest"],
        direction: Literal["previous", "next"],
    ) -> pd.Timestamp:
        """Convert date to a session an associated calendars."""
        f = min if extreme == "earliest" else max
        session = f([c.date_to_session(date, direction) for c in self.calendars_unique])
        return session

    @parse
    def close_at(
        self,
        date: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_datetimestamp, parse_none=False),
        ] = None,
    ) -> pd.DataFrame:
        """Return most recent end-of-day prices as at a specific date.

        Parameters
        ----------
        date : pd.Timestamp, str | datetime.datetime | datetime.date
        | int | float | None
        default: most recent date
            Date for which to return most recent end-of-day prices.

            Must be passd as a date, with no time component (or have time
            component defined as 00:00). If passsed as a pd.Timestamp,
            must be tz-naive.

            If date is a live session then prices will represent the most
            recent price available and will update on further calls.

        Returns
        -------
        pd.DataFrame
            index: DatatimeIndex
                Single indice expressing date that close prices relate to.

            columns: Index
                symbol

        See Also
        --------
        `session_prices`: Prices for a session.
        `price_at`: Prices as at a specific minute.

        See `specific_query_methods.ipynb` tutorial for example usage.
        """
        # pylint: disable=missing-param-doc, differing-type-doc
        if TYPE_CHECKING:
            assert date is None or isinstance(date, pd.Timestamp)

        if self.bi_daily is None:
            raise errors.MethodUnavailableNoDailyInterval("close_at")

        mr_session = self.last_requestable_session_any
        if date is None:
            date = mr_session
        else:
            first_session = self.earliest_requestable_session
            parsing.verify_date_not_oob(date, first_session, mr_session)

        end_sesh = self._date_to_session(date, "latest", "previous")
        start_sesh = self._date_to_session(date, "earliest", "previous")
        start_sesh = min([start_sesh, self.last_requestable_session_all])
        table = self._get_bi_table(self.bi_daily, (start_sesh, end_sesh))
        return table.pt.naive.pt.close_at(end_sesh)

    def _price_at_rng(
        self, bis: list[BI], minute: pd.Timestamp
    ) -> dict[BI, tuple[pd.Timestamp, pd.Timestamp]]:
        """Return request range to serve `price_at` at `minute`.

        Returns
        -------
        dict
            key:
                bi (of `bis`).

            value:
                Request range that will provide for a price to be evaluated
                at `minute` for all symbols. For example, if all calendars
                are open mon-fri except one that covers the weekend, and
                minute for `price_at` is sun pm, will need to include at
                least last indice of fri session to be able to fil
                forwards price to sun pm.
        """
        # pylint: disable=too-many-locals
        live_prices = self.live_prices
        delay = pd.Timedelta(0)  # `price_at` ignores delays
        rss = {}
        for bi in bis:
            limit_s, limit_e = self.limits[bi]
            if TYPE_CHECKING:
                assert limit_s is not None
                assert limit_e is not None
            earliest_minute = self._minute_to_earliest_previous_trading_minute(limit_s)
            if minute < earliest_minute:
                continue
            if not live_prices and minute > limit_e:
                continue
            first: pd.Timestamp | None = None
            last: pd.Timestamp | None = None
            limit_right = None if live_prices else limit_e
            for cal in self.calendars_unique:
                ignore_breaks = self._ignore_breaks_cal(cal, bi)
                drg = dr.GetterIntraday(
                    cal,
                    self.cc,
                    delay,
                    earliest_minute,
                    ignore_breaks,
                    None,
                    bi,
                    limit_right=limit_right,
                )
                start, _ = drg.get_end(minute)
                try:
                    end = drg.get_start(minute)
                except errors.StartTooLateError:
                    # one full bi of data not available over required request range
                    if first is None or start < first:
                        first = start
                    continue
                if first is None or start < first:
                    first = start
                if last is None or end > last:
                    last = end
            if last is None:
                last = minute
            assert first is not None  # for mypy
            # need at least one indice of data.
            # NB the +/- bi could be doing from_, to_ = last, first
            from_, to_ = first - bi, last + bi
            to_ = last + bi if limit_right is None else min(limit_right, last + bi)
            assert isinstance(from_, pd.Timestamp)  # for mypy
            rss[bi] = (from_, to_)
        return rss

    def _price_at_most_accurate(
        self, bis: list[BI], minute: pd.Timestamp
    ) -> dict[BI, pd.Timestamp]:
        """Return closest minute to `minute` for which price available by bi.

        Return will be `minute` or earlier.
        """
        live_prices = self.live_prices
        delay = pd.Timedelta(0)  # `price_at` ignores delays
        ma_tss = {}
        for bi in bis:
            limit_s, limit_e = self.limits[bi]
            assert limit_s is not None
            earliest_minute = self._minute_to_earliest_previous_trading_minute(limit_s)
            limit_right = (
                self._minute_to_latest_next_trading_minute(limit_e)
                if live_prices
                else None
            )
            closest: pd.Timestamp | None = None
            for cal in self.calendars_unique:
                ignore_breaks = self._ignore_breaks_cal(cal, bi)
                drg = dr.GetterIntraday(
                    cal,
                    self.cc,
                    delay,
                    earliest_minute,
                    ignore_breaks,
                    None,
                    bi,
                    limit_right=limit_right,
                )
                cal_closest, cal_closest_acc = drg.get_end(minute)
                if cal.is_trading_minute(minute):  # NB querying `minute` not closest
                    advance_close = (
                        cal_closest_acc.value in cal.closes_nanos
                        and cal_closest_acc.value not in cal.opens_nanos  # not 24h
                    )
                    if advance_close or cal_closest_acc.value in cal.break_starts_nanos:
                        # if `minute` is open or in first indice of a (sub)session then
                        # `cal_closest_acc` will be the prior sub(session) close. Roll
                        # forwards to open so that different overlapping calendars can
                        # fairly compare accuracy.
                        cal_closest = cal.minute_to_trading_minute(cal_closest, "next")
                if closest is None or cal_closest > closest:
                    closest = cal_closest
            assert closest is not None  # for mypy
            ma_tss[bi] = closest
        return ma_tss

    def _price_at_from_daily(
        self, minute: pd.Timestamp | None, tz: ZoneInfo
    ) -> pd.DataFrame:
        """Serve call for `_price_at` from daily prices table.

        `minute` will be assumed as 'now' if receieved as None.
        """
        if self.bi_daily is None:
            # In theory this should never raise as if there's no daily interval
            # then the limits will be based on available intraday data and an
            # error should already have raised based on `minute` lying outside of
            # these limits. However, there's at least a small gap to the right of
            # `limit_intraday` which can't be served from intraday data and hence
            # the code path ends up here.
            # NB there's no test for this raising given that it's very edge and
            # considered to be have negligible consequences - raises this error
            # instead of a 'TooEarly' or 'TooLate' one.
            raise errors.PriceAtUnavailableDailyIntervalError(minute)

        now = helpers.now()
        if minute is not None and minute < now - self.min_delay:
            set_indice_to_now = False
        else:
            set_indice_to_now = self.cc.is_open_on_minute(now, ignore_breaks=True)

        minute = now if minute is None else minute

        cc = self.cc
        session = cc.minute_to_sessions(minute, "previous")[-1]
        sessions = [session]
        itr_limit = 5

        for _ in range(itr_limit):
            opens = [
                c.session_open(session) for c in cc.calendars if c.is_session(session)
            ]
            closes = [
                c.session_close(session) for c in cc.calendars if c.is_session(session)
            ]
            times = sorted(
                [tm for tm in itertools.chain(opens, closes) if minute >= tm],
                reverse=True,
            )

            indice = None
            for tm in times:
                overlaps = False
                for c in cc.calendars:
                    if not c.is_session(session):
                        continue
                    if c.opens[session] < tm < c.closes[session]:
                        overlaps = True
                        break
                if not overlaps:
                    indice = tm
                    break

            if indice is not None:
                break

            session = cc.previous_session(session)
            sessions.append(session)

        if not indice:
            # Considered unlikely that this would ever raise, but it could
            # if the sessions of the underlying calendars continuously overlap.
            # NB There's no test for this raising.
            raise errors.PriceAtUnavailableError(minute, itr_limit)

        not_represented = set(self.calendars_unique)
        j = 0
        while not_represented:
            j += 1
            # first iteration takes session as defined prior to breaking prior loop
            session = cc.previous_session(session)
            sessions.append(session)
            for c in list(not_represented):
                if c.is_session(session):
                    not_represented.remove(c)
            if j == 5:
                # Considered even more unlikely that would raise here, but
                # not impossible if the sessions of the underlying calendars
                # start to continuously overlap as work backwards.
                # NB There's no test for this raising.
                raise errors.PriceAtUnavailableError(minute, itr_limit + j)

        table = self._get_bi_table(self.bi_daily, (sessions[-1], sessions[0]))
        d: dict[str, float] = {}  # key: symbol, value: price_at
        for s in self.symbols:
            c = self.calendars[s]
            sdf = table[s].dropna()
            if sdf.empty:
                d[s] = np.nan
                continue
            v = None
            if set_indice_to_now:
                v = sdf.iloc[-1].close
            else:
                for session in sessions:
                    if session < self.limit_daily:
                        raise errors.PriceAtUnavailableLimitError(
                            minute, self.limit_daily
                        )
                    if not c.is_session(session):
                        continue
                    if indice >= c.session_close(session):
                        v = sdf.loc[session, "close"]
                        break
                    if indice >= c.session_open(session):
                        v = sdf.loc[session, "open"]
                        break
            assert v is not None
            d[s] = v

        if set_indice_to_now:
            indice = minute

        indice = indice.tz_convert(tz) if indice is not pd.NaT else indice
        index = pd.DatetimeIndex([indice])
        df = pd.DataFrame(d, index=index)
        df.columns.name = "symbol"
        return df

    @parse
    def price_at(
        self,
        minute: Annotated[
            pd.Timestamp | str | datetime.datetime | int | float | None,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_timetimestamp, parse_none=False),
        ] = None,
        tz: Annotated[
            str | ZoneInfo | None,
            Parser(parsing.to_prices_timezone, parse_none=False),
        ] = None,
    ) -> pd.DataFrame:
        """Most recent price as at a minute or 'now'.

        Returns a single row DataFrame with one column for each symbol.
        The price given for each symbol is the most recent price available
        as at the start of the minute indicated by the indice. In turn the
        indice will be `minute` or the most recent minute prior to `minute`
        EITHER at which price data is available for any symbol and at least
        one of the underlying exchanges was open OR an underlying exchange
        closed.

        NOTE: For any symbol subject to a real-time price delay, if
        `minute` is None (default), or a minute within the period for which
        prices are currently delayed, then the price will not be the actual
        price as at the defined indice, but rather the delayed price
        available as at that time.

        NOTE: All data in the return will be sourced from EITHER intraday
        data OR daily data.

        Parameters
        ----------
        minute :
        pd.Timestamp | str | datetime.datetime | int | float | None,
        default: now
            Minute at which require price data.

            By default (None) will return the most recent available prices.

            Will raise ValueError if `minute` passed as a date. To
            request prices at a minute representing midnight pass as a
            timezone aware pd.Timestamp.

        tz : str | ZoneInfo | None, default: `default_tz`
            Timezone of `minute` (if `minute` otherwise timezone naive) and
            for returned index. Can be passed as:

                ZoneInfo:
                    Any instance returned by `zoneinfo.ZoneInfo`.

                str:
                    - valid input to `zoneinfo.ZoneInfo`, for example "UTC"
                    or "US/Eastern".
                    - any symbol of `symbols`. For example, pass "GOOG" to
                    define timezone as timezone associated with that
                    symbol.

            If `minute` is tz-aware then `tz` will NOT override the
            timezone of `the `minute` although `tz` will be used as the
            output timezone.

        See Also
        --------
        `close_at`: Prices as at end of a day.
        `session_prices`: Prices for a session.

        See `specific_query_methods.ipynb` tutorial for example usage.
        """
        # pylint: disable=missing-param-doc, differing-type-doc, differing-param-doc
        # pylint: disable=too-complex, too-many-locals, too-many-branches
        # pylint: disable=too-many-statements
        if TYPE_CHECKING:
            assert minute is None or isinstance(minute, pd.Timestamp)
            assert tz is None or isinstance(tz, ZoneInfo)

        if tz is None:
            tz = self.tz_default

        if minute is None:
            if not self.live_prices:
                raise errors.PriceAtUnavailableLivePricesError()
            elif self.bi_daily is not None:
                return self._price_at_from_daily(minute, tz)
        else:
            l_limit = self.earliest_requestable_minute
            r_limit = self.latest_requestable_minute
            minute = parsing.parse_timestamp(minute, tz)
            parsing.verify_time_not_oob(minute, l_limit, r_limit)

            if minute < self.limit_intraday():
                return self._price_at_from_daily(minute, tz)
            # serve from daily any timestamp within delay, if daily available
            if self.bi_daily is not None and minute > helpers.now() - self.min_delay:
                return self._price_at_from_daily(minute, tz)

        # get bis for which indices are not misaligned
        start_sesh = self._minute_to_session(minute, "earliest", "previous")
        end_sesh = self._minute_to_session(minute, "latest", "previous")
        bis_synced = []
        for bi in self.bis_intraday:
            if self._indices_aligned[bi][slice(start_sesh, end_sesh)].all():
                bis_synced.append(bi)

        minute_received = minute
        minute_advanced = False
        if not self.cc.is_open_on_minute(minute):
            # only useful if `minute_` is between a (sub)session close and right side of
            # an unaligned final indice (inclusive of close). Advancing non-trading
            # minute will include the unanligned indice and hence get price_at the
            # close rather than as at end of prior indice.
            adv = self.cc.minute_to_trading_minute(minute, "next") - helpers.ONE_MIN
            assert isinstance(adv, pd.Timestamp)
            minute = adv
            minute_advanced = True

        rngs = self._price_at_rng(bis_synced, minute)
        # of those, bis for which prices available
        bis = []
        for bi, rng in rngs.items():
            pdata = self._pdata[bi]
            if pdata.available_range(rng):
                bis.append(bi)

        if not bis:
            # shouldn't be able to get to here if daily data not available
            return self._price_at_from_daily(minute, tz)

        ma_tss = self._price_at_most_accurate(bis, minute)
        srs = pd.Series(ma_tss)
        diff = minute - srs
        ma_bis_srs = diff[diff == min(diff)]
        ma_bis = [intervals.to_ptinterval(td) for td in ma_bis_srs.index]

        # of those, those that have stored data
        bis_stored = []
        for bi in ma_bis:
            pdata = self._pdata[bi]
            if pdata.requested_range(rngs[bi]):
                bis_stored.append(bi)

        bi = bis_stored[-1] if bis_stored else ma_bis[-1]
        table = self._get_bi_table(bi, rngs[bi])

        minute_ma = ma_tss[bi]
        if minute_advanced and self._ignore_breaks_any:
            # where breaks are ignored and minute falls in break would otherwise return
            # 'open' of indice that immediately follows last indice of the am session
            # (rather than close of that last am subsession indice). The minus one_min
            # ensures that last indice is last indice of am subsession (which will have
            # right side as `minute_ma`), thereby precluding knowledge of next indice).
            table = table.loc[: minute_ma - helpers.ONE_MIN]
        table_pa = table.pt.operate(fill="ffill").pt.price_at(minute_ma)
        indice = table_pa.index[0]
        if (
            bi > TDInterval.T5
            and indice.value not in self.cc.closes_nanos
            and not self.cc.is_open_on_minute(indice)
        ):
            if indice < minute_received and self.cc.is_open_on_minute(minute_received):
                # indice is left side of an indice that is unaligned with a
                # subsession open.
                rolled = self.cc.minute_to_trading_minute(indice, "next")
                table_pa.index = pd.DatetimeIndex([rolled])
            else:
                rolled = self.cc.minute_to_trading_minute(indice, "previous")
                if rolled < indice:
                    # indice was right side of an indice that is unaligned
                    # with (sub)session close. Can roll back to last trading minute + 1
                    table_pa.index = pd.DatetimeIndex([rolled + helpers.ONE_MIN])
        table_pa.index = table_pa.index.tz_convert(tz)
        return table_pa

    @parse
    def price_range(
        self,
        start: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
        ] = None,
        end: Annotated[
            pd.Timestamp | str | datetime.datetime | datetime.date | int | float | None,
            Coerce(pd.Timestamp),
        ] = None,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
        weeks: int = 0,
        months: int = 0,
        years: int = 0,
        lead_symbol: Annotated[str | None, Parser(parsing.lead_symbol)] = None,
        tzin: Annotated[
            str | ZoneInfo | None,
            Parser(parsing.to_prices_timezone, parse_none=False),
        ] = None,
        strict: bool = True,
        tzout: Annotated[
            str | ZoneInfo | None,
            Parser(parsing.to_prices_timezone, parse_none=False),
        ] = None,
        include: mptypes.Symbols | None = None,
        exclude: mptypes.Symbols | None = None,
        stack: bool = False,
        underlying: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Return OHLCV data for a period.

        Returns the following for each symbol:
            'open' - price at start of the period.
            'high' - highest price registered during the period.
            'low' - lowest price registered during the period.
            'close' - price at end of the period.
            'volume' - total volume registered over the period.

        Period defined from parameters in the same was as for the `get`
        method when `anchor` is open. See `get.__doc__`.

        Parameters
        ----------
        Method parameters as for `get`, except:

        tzout : str | BaseTzinfo | None, default: as `tzin`
            Timezone of period as expressed by the index (or level 0 of).

            Defined in same way as `tzout` parameter of `get`.

            Index expressing period will always be tz-aware. If `tzout` is
            not passed then by default period will be expressed with
            timezone as that which `tzin` evalutes to (see `get.__doc__`).

        stack: bool  (default: False)
            False: (default) Return single-row DataFrame with 'open',
            'high', 'low', 'close' and 'volume' columns for every symbol.

            True: Return multi-row DataFrame with each row representing
            one symbol and columns as 'open', 'high', 'low', 'close' and
            'volume'.

        underlying: bool  (default: False)
            Return 2-tuple with second item as underlying Dataframe of
            prices over the period. NB this DataFrame is equivalent to
            calling `get` with the passed parameters, 'anchor' as "open" and
            'composite' as True.

        Returns
        -------
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]

            If `stack` False and `underlying` False (default): pd.DataFrame
                Single-row pd.DataFrame with 'open', 'high' 'low', 'close'
                and 'volume' columns for each symbol.

                index: pd.IntervalIndex
                    pd.Interval describing period over which price data
                    corresponds. Get timezone of indice by calling
                    `df.pt.tz`.

                columns: pd.MultiIndex.
                    level 0: pd.Index
                        symbols.

                    level 1: pd.Index
                        'open', 'close', 'high' 'low' 'volume'.

            If `stack` is True and underlying False: pd.DataFrame
                Multi-row pd.DataFrame with each row representing a symbol.

                index: pd.MultiIndex
                    level 0: IntervalIndex
                        pd.Interval describing period over which price data
                        corresponds. Get timezone of indice by calling
                        `df.index.levels[0].left.tz`.

                    level 1: pd.Index
                        symbols.

                columns: pd.Index
                    'open', 'close', 'high' 'low' 'volume'.

            If underlying is True: tuple[pd.DataFrame, pd.DataFrame]
                [0] As above

                [1] Underlying Dataframe from which range data evaluated.
                NB this DataFrame is equivalent to calling `get` with the
                passed parameters, 'anchor' as "open" and 'composite' as
                True.

        See Also
        --------
        See `specific_query_methods.ipynb` tutorial for example usage.
        """
        # pylint: disable=missing-param-doc, too-many-arguments, too-many-locals
        if TYPE_CHECKING:
            assert start is None or isinstance(start, pd.Timestamp)
            assert end is None or isinstance(start, pd.Timestamp)
            assert isinstance(lead_symbol, str)
            assert tzin is None or isinstance(tzin, ZoneInfo)
            assert tzout is None or isinstance(tzout, ZoneInfo)

        interval = None
        add_a_row = False
        force = False
        anchor, openend, priority, composite = "open", "shorten", "end", True
        fill = None

        df = self.get(
            interval,
            start,
            end,
            minutes,
            hours,
            days,
            weeks,
            months,
            years,
            add_a_row,
            lead_symbol,
            tzin,
            anchor,
            openend,
            priority,
            strict,
            composite,
            force,
            tzout,
            fill,
            include,
            exclude,
        )

        if df.empty:
            return df

        # Aggregate (via group to allow use of 'first' and 'last' functions)
        group = np.array([0] * len(df))
        groups = df.groupby(by=group)
        res = groups.agg(helpers.agg_funcs(df))

        if tzout is None:
            if tzin is not None:
                tzout = tzin
            else:
                tzout = self.timezones[self.gpp.lead_symbol]

        # Define indice
        if df.pt.is_daily:
            left = self.cc.session_open(df.pt.first_ts)
            right = self.cc.session_close(df.pt.last_ts)
        else:
            left, right = df.pt.first_ts, df.pt.last_ts
            if isinstance(df.pt, pt.PTDailyIntradayComposite):
                first_session = helpers.to_tz_naive(left)
                left = self.cc.session_open(first_session)
        right = min(right, helpers.now())
        interval = pd.Interval(left.tz_convert(tzout), right.tz_convert(tzout))
        res.index = pd.IntervalIndex([interval])

        if stack:
            res = res.pt.stacked

        if underlying:
            return res, df
        return res

    @staticmethod
    def _remove_non_trading_indices(
        df: pd.DataFrame, cals: list[xcals.ExchangeCalendar]
    ) -> pd.DataFrame:
        """Remove indices that include no minutes of any of `cals`."""
        non_trading = df.pt.indices_non_trading(cals[0])
        for cal in cals[1:]:
            non_trading = non_trading.intersection(df.pt.indices_non_trading(cal))
        return df.drop(labels=non_trading)

    def _get_class_instance(self, symbols: list[str], **kwargs) -> "PricesBase":
        """Return an instance of the prices class with the same parameters as self.

        Notes
        -----
        If required, subclass should override or extend this method.
        """
        cals_all = {s: self.calendars[s] for s in symbols}
        delays_all = {s: self.delays[s].components.minutes for s in symbols}
        if self.lead_symbol_default in symbols:
            kwargs.setdefault("lead_symbol", self.lead_symbol_default)
        return type(self)(
            symbols=symbols, calendars=cals_all, delays=delays_all, **kwargs
        )

    def prices_for_symbols(self, symbols: Symbols) -> "PricesBase":
        """Return instance of prices class for one or more symbols.

        Populates instance with any pre-existing price data.

        Parameters
        ----------
        symbols
            Symbols to include to the new instance. Passed as class'
            'symbols' parameter.
        """
        # pylint: disable=protected-access
        symbols = helpers.symbols_to_list(symbols)
        difference = set(symbols).difference(set(self.symbols))
        if difference:
            msg = (
                "symbols must be a subset of Prices' symbols although"
                f" received the following symbols which are not:"
                f" {difference}.\nPrices symbols are {self.symbols}."
            )
            raise ValueError(msg)
        prices_obj = self._get_class_instance(symbols)
        cals = list(prices_obj.calendars_unique)
        fewer_cals = len(cals) < len(self.calendars_unique)
        for bi in self.bis:
            new_pdata = copy.deepcopy(self._pdata[bi])
            if new_pdata._table is not None:
                table = new_pdata._table[symbols].copy()
                if fewer_cals:
                    table = self._remove_non_trading_indices(table, cals)
                new_pdata._table = table
            prices_obj._pdata[bi] = new_pdata
        return prices_obj

    @parse
    def to_csv(
        self,
        path: Annotated[str | Path, Coerce(Path), Parser(parsing.verify_directory)],
        intervals: (
            str
            | pd.Timedelta
            | datetime.timedelta
            | list[str]
            | list[pd.Timedelta]
            | list[datetime.timedelta]
            | None
        ) = None,
        include: mptypes.Symbols | None = None,
        exclude: mptypes.Symbols | None = None,
        **kwargs,
    ) -> list[Path]:
        """Export price data to .csv file(s).

        Note: Exported price data can be retrieved with the default
        implementation of the `PricesCsv` class (requires that the
        exported data conforms with the requirements of the
        `PricesCsv` class, for example that prices are anchored on
        the 'open' and have an interval no higher than daily).

        Price data will be exported by symbol by interval, such that if
        data is requested for 3 intervals and 5 symbols then 15 .csv
        files will be created.

        .csv filenames will follow the format:
            <SYMBOL>_<INTERVAL>_<YYMMDD>_<YYMMDD>.csv
            For example:
                MSFT_5T_240122_240215.csv
            This file would hold '5T' (i.e. 5 minute) price data for
            the symbol MSFT covering the period from 2024-01-22 through
            2024-02-15. Note: for intraday intervals the dates will
            represent the earliest and latest sessions for which at least
            some price data is included.

        Parameters
        ----------
        path
            Directory to which .csv files should be written. This path
            must exist.

        intervals
            Intervals for which price data is to be exported. To define
            a single interval pass as for the 'interval` parameter of the
            `.get` method. To define multiple intervals pass as a list
            of one of the types that's acceptable input to the 'interval`
            parameter of the `.get` method.

            By default (None) .csv files are exported for all available
            base intervals.

        include : list[str] | str | None
            Symbol or symbols to include in export. All other symbols will
            be excluded. If passed, do not pass `exclude`.

            By default, if neither include nor exclude are passed then data
            will be exported for all symbols.

        exclude : list[str] | str | None
            Symbol or symbols to exclude from export. Data will be exported
            for all other symbols. If passed, do not pass `include`.

            By default, if neither exclude nor include are passed then data
            will be exported for all symbols.

        kwargs
            All other kwargs will be passed on to the `.get` method to
            define the period over which prices are to be exported. Can
            include other options, for example 'anchor', 'priority',
            'strict' etc.

            If no other kwargs are not passed then by default all available
            data will be exported for each requested symbol / interval.

        Returns
        -------
        paths
            List of Path objects to which data exported.
        """
        if TYPE_CHECKING:
            assert isinstance(path, Path)

        intervals_: list[PTInterval]
        if isinstance(intervals, list):
            intervals_ = [to_ptinterval(intrvl) for intrvl in intervals]
        elif intervals is None:
            intervals_ = self.bis
        else:
            intervals_ = [to_ptinterval(intervals)]

        if kwargs.get("lose_single_symbol", False):
            kwargs["lose_single_symbol"] = False

        dfs = {}
        for intrvl in intervals_:
            if kwargs:
                try:
                    df = self.get(intrvl, include=include, exclude=exclude, **kwargs)
                except Exception as err:
                    raise errors.PricesUnavailableForExport(intrvl, kwargs) from err
            else:
                try:
                    df = self.get(
                        intrvl, include=include, exclude=exclude, strict=False
                    )
                except Exception as err:
                    raise errors.PricesUnavailableForExport(intrvl) from err
            if intervals is None and helpers.ONE_MIN < intrvl < helpers.ONE_DAY:
                # Do not include if data was downsampled from a lower base interval
                # due to unalignment at `intrvl`, i.e. only export base data.
                data_ = self._pdata[intrvl]
                if not data_.requested_range((df.pt.first_ts, df.pt.last_ts)):
                    continue
            if include is not None or exclude is not None:
                df.columns = df.columns.remove_unused_levels()
            dfs[intrvl] = df

        def get_freq_str(intrvl: PTInterval) -> str:
            freq = intrvl.as_pdfreq
            if freq.endswith("min"):
                return freq.replace("min", "T")
            elif freq.endswith("h"):
                return freq.replace("h", "H")
            return freq

        store = {}
        for intrvl, df in dfs.items():
            freq = get_freq_str(intrvl)
            for symb in df.columns.levels[0]:
                sdf = df[symb].dropna(how="all")

                if intrvl.is_intraday:
                    cal = self.calendars[symb]
                    start = cal.minute_to_session(sdf.pt.first_ts, "next")
                    end = cal.minute_to_session(sdf.pt.last_ts, "previous")
                    sdf.index = sdf.index.left.tz_convert(None)
                else:
                    start = sdf.pt.first_ts
                    end = sdf.pt.last_ts
                start_ = start.strftime("%y%m%d")
                end_ = end.strftime("%y%m%d")

                sdf.index.name = "date"
                filename = f"{symb}_{freq}_{start_}_{end_}.csv"
                store[path / filename] = sdf

        for path_, df in store.items():
            df.to_csv(path_)

        return list(store.keys())
