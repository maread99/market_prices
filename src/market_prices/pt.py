"""Prices Table (PT) DataFrame extension (.pt accessor)."""

from __future__ import annotations

import abc
import collections
import datetime
import functools
import warnings
from typing import TYPE_CHECKING, Literal, Annotated
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from valimp import parse, Coerce, Parser

import market_prices.utils.calendar_utils as calutils
from market_prices import errors, helpers, intervals, mptypes, parsing
from market_prices.helpers import UTC
from market_prices.utils import general_utils as genutils
from market_prices.utils import pandas_utils as pdutils
from market_prices.utils.calendar_utils import CompositeCalendar
from market_prices.mptypes import Symbols

# pylint: disable=too-many-lines


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns of a _PT that has been altered.

    Parameters
    ----------
    df
        DataFrame to clean.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns.names = ("symbol", "")
        df.columns = df.columns.remove_unused_levels()
    return df


@pd.api.extensions.register_dataframe_accessor("pt")
class PT:
    """pandas Price Table accessor.

    Wrapper that returns an instance of a subclass of `_PT` specific to the
    nature of the DataFrame on which the .pt accessor is called.
    """

    # pylint: disable=too-few-public-methods

    def __new__(cls, df: pd.DataFrame):  # pylint: disable=missing-return-type-doc
        """Return new instance of PT class (as subclass of _PT)."""
        new_cls: type[_PT]

        index = df.index.left if isinstance(df.index, pd.IntervalIndex) else df.index
        if not isinstance(index, pd.DatetimeIndex):
            msg = (
                "To use PT accessor index must be of type pd.DatetimeIndex or"
                " pd.IntervalIndex with left and right sides as pd.DatetimeIndex,"
                f" although index is of type {type(index)}."
            )
            raise TypeError(msg)

        if df.empty:
            new_cls = PTIntraday
        elif not isinstance(df.index, pd.IntervalIndex):
            index_normalized = pdutils.index_is_normalized(df.index)
            if index_normalized and (
                df.index.freq is None or df.index.freq == df.index.freq.base
            ):
                new_cls = PTDaily
            elif not index_normalized:
                msg = (
                    "PT accessor not available where index is a pd.DatatimeIndex"
                    " with one or more indices that have a time component (Index must"
                    " be pd.IntervalIndex if any indice includes a time component)."
                )
                raise ValueError(msg)
            else:
                # MultSession table that has been forcefully reindexed to left or right
                assert df.index.freq is not None and df.index.freq != df.index.freq.base
                msg = (
                    "PT accessor not available where index DatatimeIndex"
                    " and frequency is greater than one day."
                )
                raise ValueError(msg)
        elif pdutils.index_is_normalized(df.index):
            new_cls = PTMultipleSessions
        elif df.index[0].left == df.index[0].right and pdutils.is_midnight(
            df.index[0].left
        ):
            new_cls = PTDailyIntradayComposite
        else:
            new_cls = PTIntraday
        self = new_cls.__new__(new_cls, df)  # type: ignore[call-arg]  # as required
        self.__init__(df)  # type: ignore[misc]  # can access __init__ directly
        return self


class _PT(metaclass=abc.ABCMeta):
    """Base for pandas accessor for a prices table."""

    # pylint: disable=too-many-public-methods

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices
        self._validate()

    def _validate(self):
        """Verify DataFrame has necessary attributes to utilise extension."""
        if self.has_symbols:
            cols = self.columns.levels[1]
        else:
            cols = self.columns
        if not all(c in helpers.AGG_FUNCS for c in cols):
            msg = (
                "To use PricesTable accessor columns must by in"
                f" {helpers.AGG_FUNCS.keys()} although columns evaluated as"
                f" {cols}."
            )
            raise KeyError(msg)

    def _verify_tz_awareness(self):
        if self.tz is None:
            msg = (
                "Index is timezone naive although operation requires"
                " timezone awareness."
            )
            raise ValueError(msg)

    def _not_implemented(self, method):
        msg = f"{method.__name__} is not implemented for {type(self)}."
        raise NotImplementedError(msg)

    @property
    def prices(self) -> pd.DataFrame:
        """Return prices table."""
        return self._prices

    # Index properties

    @property
    @abc.abstractmethod  # abstracted so each subclass knows actual type
    def index(self) -> pd.DatetimeIndex | pd.IntervalIndex:
        """Return index of prices table."""

    @property
    def columns(self) -> pd.Index:
        """Return columns of prices table."""
        return self.prices.columns

    @property
    def symbols(self) -> list[str] | None:
        """Return price table's symbols.

        None if table does not have a 'symbol' columns level.
        """
        try:
            return self.columns.get_level_values("symbol").unique().to_list()
        except KeyError:
            return None

    @property
    def has_symbols(self) -> bool:
        """Query if columns has a 'symbols' level."""
        return helpers.has_symbols(self.prices)

    @property
    def first_ts(self) -> pd.Timestamp:
        """Return first timestamp represented in table."""
        return self.index[0]

    @property
    def last_ts(self) -> pd.Timestamp:
        """Return last timestamp represented in table."""
        return self.index[-1]

    @property
    @abc.abstractmethod
    def is_intraday(self) -> bool:
        """Query if interval is less than daily."""

    @property
    @abc.abstractmethod
    def is_daily(self) -> bool:
        """Query if interval is daily."""

    def _new_index(self, index: pd.Index) -> pd.DataFrame:
        """Return `self.prices` indexed with `index`."""
        df = self.prices.copy()
        if isinstance(index, pd.IntervalIndex):
            index.left.freq = self.freq
            index.right.freq = self.freq
        else:
            index.freq = self.freq
        df.index = index
        return df

    @property
    @abc.abstractmethod
    def _naive_index(self):
        """Index as naive timestamps."""

    @property
    def naive(self) -> pd.DataFrame:
        """Price table indexed with naive dates.

        Note: Price table methods may not work with a naive index.
        """
        return self._new_index(self._naive_index)

    @property
    @abc.abstractmethod
    def _utc_index(self) -> pd.DatetimeIndex | pd.IntervalIndex:
        """`self.index` as utc."""

    @property
    def utc(self) -> pd.DataFrame:
        """Convert index to utc."""
        return self._new_index(self._utc_index)

    @property
    @abc.abstractmethod
    def _tz(self) -> ZoneInfo | None:
        """Timezone of index."""

    @property
    def tz(self) -> ZoneInfo | None:
        """Timezone of index."""
        return self._tz

    # Index operations

    def _set_tz_non_local(self, tz: ZoneInfo | None) -> pd.DataFrame:
        """Set tz to None or utc."""
        if tz == self._tz:
            return self.prices

        if self.is_daily:
            assert isinstance(self.index, pd.DatetimeIndex)
            index = self.index.tz_localize(tz)
        elif tz == UTC:
            index = pdutils.interval_index_new_tz(self.index, tz)
        else:
            index = pdutils.interval_index_new_tz(self.index, None)
        return self._new_index(index)

    def _set_tz(self, tz: ZoneInfo | None) -> pd.DataFrame:
        """Convert index to a timezone.

        Subclass should overide if provides for setting tz to anything
            other than only naive or UTC.
        """
        return self._set_tz_non_local(tz)

    def set_tz(self, tz: str | ZoneInfo):
        """Convert index timezone.

        Parameters
        ----------
        tz
            Timezone to set index to.
        """
        # pylint: disable=missing-param-doc
        # subclass to override if implemented
        _ = tz
        msg = (
            f"set_tz is not implemented for {type(self)}. Index of this"
            f" class can be only timezone naive or have timezone as 'UTC'."
            f" Use .pt.utc and .pt.naive methods."
        )
        raise NotImplementedError(msg)

    def convert_to_table_tz(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Convert a given timestamp to table timezone.

        Parameters
        ----------
        ts
            Timestamp to convert.
        """
        if ts.tz == self.tz:
            return ts
        elif ts.tz is None:
            return ts.tz_localize(self.tz)
        else:
            return ts.tz_convert(self.tz)

    @property
    def freq(self) -> pd.offsets.BaseOffset | None:
        """Any frequency declared by index."""
        if isinstance(self.index, pd.IntervalIndex):
            l_freq = self.index.left.freq
            r_freq = self.index.right.freq
            if l_freq != r_freq:
                return None
            else:
                freq = l_freq
        else:
            freq = self.index.freq
        return freq

    @property
    def _index_length(self) -> intervals.TDInterval | None:
        """Index length if length regular, otherwise None."""
        lengths = set(self.index.length)
        if len(lengths) != 1:
            return None
        else:
            return intervals.TDInterval(lengths.pop())

    @property
    def _interval(self) -> intervals.PTInterval | None:
        freq = self.freq
        if freq is not None:
            value, unit_ = helpers.extract_freq_parts(freq.freqstr)
            if isinstance(freq, pd.offsets.CustomBusinessDay):
                unit = "D"
            elif unit_ == "MS":
                unit = "M"
            else:
                unit = unit_
            return intervals.to_ptinterval(str(value) + unit)
        else:
            return self._index_length

    @property
    def interval(self) -> intervals.PTInterval | None:
        """Time delta represented by each indice.

        None if not regular or interval cannot be ascertained.
        """
        return self._interval

    @property
    def has_regular_interval(self) -> bool:
        """Query if every indice is known to represent same period."""
        return self.interval is not None

    # Reindexing

    @abc.abstractmethod
    def _compatible_sessions(
        self, index: pd.IntervalIndex, calendar: xcals.ExchangeCalendar
    ) -> pd.Series:
        """Query if `index` indices are compatible with table indices.

        Result offered by-session.

        Parameters
        ----------
        index
            Index against which to evaluate compatiblity.

        calendar
            Calendar against which `index` was evaluated.
        """

    @abc.abstractmethod
    def _compatible_index(self, index: pd.IntervalIndex) -> bool:
        """Query if `index` compatible with table index."""

    @parse
    def get_trading_index(
        self,
        calendar: xcals.ExchangeCalendar,
        closed: Literal["left", "right"] | None = "left",
        force=False,
        ignore_breaks=False,
    ) -> pd.DatetimeIndex | pd.IntervalIndex:
        """Calendar-based trading index over period covered by table.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate the trading index.

        closed
            On which side should the intervals of the trading index be
            closed.

        force : default: False
            True: force right side of last indice of each session to the
            session close. If session has a break, also force right side of
            last indice of am subsession to am subsession close.

        ignore_breaks : default: False
            (ignored if table interval is daily)
            (irrelevant if no session has a break)

            Defines whether trading index should respect session breaks.

            If False, treat sessions with breaks as comprising independent
            morning and afternoon subsessions.

            If True, treat all sessions as continuous, ignoring any
            breaks.
        """
        # pylint: disable=missing-param-doc
        if not self.has_regular_interval:
            raise ValueError(
                "`get_trading_index` requires price table to have a regular interval."
            )

        interval = self.interval
        assert interval is not None

        return calendar.trading_index(
            start=self.first_ts,
            end=self.last_ts,
            period=interval.as_pdfreq,
            intervals=True,
            closed=closed,
            force=force,
            ignore_breaks=ignore_breaks,
        )

    def _check_index_compatible(
        self, index: pd.IntervalIndex, calendar: xcals.ExchangeCalendar
    ):
        """Verify that an index is compatible with table's index.

        Parameters
        ----------
        index
            Index against which to evaluate compatiblity.

        calendar
            Calendar against which `index` was evaluated.

        Raises
        ------
        Raises `errors.IndexConflictError` if any table indice is conflicts
        with an indice of `index`.
        """
        if not self._compatible_index(index):
            compat_sessions = self._compatible_sessions(index, calendar)
            non_compat_sessions = compat_sessions[~compat_sessions]
            raise errors.IndexConflictError(calendar, non_compat_sessions.index)

    @parse
    def reindex_to_calendar(
        self,
        calendar: xcals.ExchangeCalendar,
        force: bool = False,
        ignore_breaks: bool = True,
        fill: Literal["ffill", "bfill", "both"] | None = None,
    ) -> pd.DataFrame:
        """Reindex prices table against a given calendar.

        If `self.index` is tz-naive, will be assumed to represent UTC.

        Parameters
        ----------
        calendar
            Calendar against which to reindex table.

        force : default: False
            True: force right side of last indice of each session to the
            session close. If session has a break, also force right side of
            last indice of am subsession to am subsession close.

        ignore_breaks : default: False
            (ignored if table interval is daily)
            (irrelevant if no session has a break)

            Defines whether trading index should respect session breaks.

            If False, treat sessions with breaks as comprising independent
            morning and afternoon subsessions.

            If True, treat all sessions as continuous, ignoring any
            breaks.

        fill : default: "ffill"
            Fill missing values for any non-trading indices that result
            from reindexing:
                "ffill": fill with closest prior value.
                "bfill": fill with closest subsequent value.
                "both": first forwardfill, then backfill. Has effect of
                    filling any initial missing values that would not
                    otherwise be filled with 'ffill' alone.
        """
        # pylint: disable=missing-param-doc
        if not self.has_regular_interval:
            raise ValueError(
                "`reindex_to_calendar` requires price table to have a regular interval."
            )

        df = self.prices if self.is_daily else self.utc
        # Assumes table index has not forced closes, as if were to then interval would
        # not be regular.
        # NOTE assumes table index based on ignoring breaks.
        new_index = df.pt.get_trading_index(calendar, force=False, ignore_breaks=True)
        # pylint: disable=protected-access
        df.pt._check_index_compatible(new_index, calendar)

        if force or not ignore_breaks:
            new_index = df.pt.get_trading_index(
                calendar, force=force, ignore_breaks=ignore_breaks
            )

        if df.pt.is_intraday:
            df = df.pt.indexed_left.reindex(new_index.left)
            df.index = new_index
        else:
            df = df.pt.prices.reindex(new_index)

        if fill is not None:
            df = df.pt.fillna(fill)

        df = df.pt._set_tz(self.tz)  # pylint: disable=protected-access
        return df

    # Indices queries and subsets

    @abc.abstractmethod
    def indices_trading_status(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Query indices trading/non-trading status.

        Query if indices represent trading periods, non-trading periods or
        both.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.

        Returns
        -------
        pd.Series:
            Index: as price table index.

            Values:
                True: indice represents a trading session or a period
                during which there are no non-trading dates/times.

                False: indice represents a non-trading date or a period
                during which there are no trading sessions/times.

                nan: indice represents both trading and non-trading periods
                (NB not possible for PTDaily).
        """

    @parse
    def indices_trading(
        self, calendar: xcals.ExchangeCalendar
    ) -> pd.DatetimeIndex | pd.IntervalIndex:
        """Return trading indices only.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.

        Returns
        -------
        pd.DatetimeIndex | pd.IntervalIndex
            pd.DatetimeIndex:
                If the price table interval is daily.
                Index comprised of dates that represent trading sessions.
            pd.IntervalIndex:
                If the price table interval is not daily.
                Index comprised of indices that represent periods during
                which there are no non-trading dates/times.

            Partial trading indices are not included in the return.

        See Also
        --------
        indices_all_trading
        indices_non_trading
        indices_partial_trading
        """
        its = self.indices_trading_status(calendar)
        # Can't use its.all() as nan equates to True
        return its[~(its.isna() | its.eq(False))].index

    @parse
    def indices_non_trading(
        self, calendar: xcals.ExchangeCalendar
    ) -> pd.DatetimeIndex | pd.IntervalIndex:
        """Return non-trading indices only.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.

        Returns
        -------
        pd.DatetimeIndex | pd.IntervalIndex
            pd.DatetimeIndex:
                If the price table interval is daily.
                Index comprised of dates that do not represent sessions.
            pd.IntervalIndex:
                If the price table interval is not daily.
                Index comprised of indices that represent periods during
                which there are no trading sessions/times.

            Partial trading indices are not included in the return.

        See Also
        --------
        indices_trading
        indices_partial_trading
        """
        its = self.indices_trading_status(calendar)
        return its[its.eq(False)].index

    @parse
    def indices_partial_trading(
        self, calendar: xcals.ExchangeCalendar
    ) -> pd.DatetimeIndex | pd.IntervalIndex:
        """Return only partial-trading indices.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.

        Returns
        -------
        pd.DatetimeIndex | pd.IntervalIndex
            pd.DatetimeIndex:
                If the price table interval is daily.
                Always empty as a date can only represent either a session
                or a non-trading date.
            pd.IntervalIndex:
                If the price table interval is not daily.
                Index comprised of indices that represent periods during
                which there are both trading sessions/times and non-trading
                dates/times.

        See Also
        --------
        indices_trading
        indices_non_trading
        indices_partial_trading_info
        """
        its = self.indices_trading_status(calendar)
        return its[its.isna()].index

    @parse
    def indices_all_trading(self, calendar: xcals.ExchangeCalendar) -> bool:
        """Query if all indices represent trading indices.

        Will return False if any indice represents a non-trading date/time
        or a period that includes a non-trading date/time.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.
        """
        its = self.indices_trading_status(calendar)
        # can't use tis.all() as nan equates to True
        return not (its.isna().any() or its.eq(False).any())

    @staticmethod
    def _partial_non_trading(
        calendar: xcals.ExchangeCalendar, partial_indice: pd.Interval
    ) -> pd.IntervalIndex:
        """Return non-trading periods of a partial trading indice."""
        # pylint: disable=too-many-locals
        start = partial_indice.left
        end = partial_indice.right
        trading_mins = calendar.minutes_in_range(start, end)

        # remove opens and break_starts from `trading_mins` because want to
        # include these values in `not_t` as they mark the edges of the non
        # trading periods (close and break ends already excluded from trading
        # minutes as calendar has side "left"). NB leaves them if they are the
        # first `trading_mins' to avoid considering this minute as an isolated
        # minute long non trading period.
        if start != trading_mins[0]:
            for nanos in [calendar.first_minutes_nanos, calendar.break_ends_nanos]:
                arr = np.intersect1d(nanos, trading_mins.values.astype("int64"))
                if arr.size > 0:
                    trading_mins = trading_mins.drop(pd.DatetimeIndex(arr, tz=UTC))

        all_mins = pd.date_range(start, end, freq="1min")
        non_t = all_mins.difference(trading_mins)
        bv_discontinuous = non_t[1:] != (non_t[:-1] + helpers.ONE_MIN)
        side = "left"
        if not bv_discontinuous.any():
            interval = pd.Interval(non_t[0], non_t[-1], side)
            return pd.IntervalIndex([interval])
        else:
            ends = non_t[:-1][bv_discontinuous]
            starts = non_t[1:][bv_discontinuous]
            first = [pd.Interval(non_t[0], ends[0], side)]
            last = [pd.Interval(starts[-1], non_t[-1], side)]
            middle = []
            for s, e in zip(starts[:-1], ends[1:]):
                middle.append(pd.Interval(s, e, side))
            return pd.IntervalIndex(first + middle + last)

    @parse
    def indices_partial_trading_info(
        self, calendar: xcals.ExchangeCalendar
    ) -> dict[pd.IntervalIndex, pd.IntervalIndex]:
        """Return information on partial trading indices.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading periods/sessions.

        Returns
        -------
        dict[pd.IntervalIndex, pd.IntervalIndex]
            key:
                Partial trading indice
            value:
                Period or periods of indice that are not trading periods.
        """
        partial_indices = self.indices_partial_trading(calendar)
        d = {}
        for p_indice in partial_indices:
            d[p_indice] = self._partial_non_trading(calendar, p_indice)
        return d

    # Query table data.

    @abc.abstractmethod
    def get_subset_from_indices(  # pylint: disable=missing-param-doc
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Get subset of table between given indices.

        Parameters
        ----------
        start : default: first table indice
            Start indice.

        end : default: last table indice
            End indice.
        """

    @abc.abstractmethod
    def price_at(self, ts: mptypes.TimeTimestamp) -> pd.DataFrame:
        """Return most recent price as at a given timestamp.

        Parameters
        ----------
        ts
            Timestamp at which to return most recent price.
        """

    @abc.abstractmethod
    def session_prices(self, session: mptypes.DateTimestamp) -> pd.DataFrame:
        """Return OHLCV for a given session.

        Parameters
        ----------
        session
            Session for which to return OHLCV.
        """

    @abc.abstractmethod
    def close_at(self, date: mptypes.DateTimestamp) -> pd.DataFrame:
        """Return price as at end of a given day.

        Parameters
        ----------
        date
            Date for which to return prices as at day end.
        """

    # Table properties.

    @property
    def _notna_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return bounds that exclude first / last rows with missing values."""
        cols = [col for col in self.columns if col[1] != "volume"]
        df = self.prices[cols]
        df_notna = df[df.notna().all(axis=1)]
        if df_notna.empty:
            return None
        return df_notna.pt.first_ts, df_notna.pt.last_ts

    @property
    def _empty_table(self) -> pd.DataFrame:
        """Return empty copy of price table."""
        return pd.DataFrame(columns=self.columns)

    @property
    def data_for_all_start(self) -> pd.DataFrame:
        """Table excluding any initial rows for which price data missing.

        Rows will be excluded if the value of any price column is missing.

        Property useful to compare price data of various symbols with
        different price history availability - all initial indices where
        prices are not available for all instruments will be excluded.
        """
        bounds = self._notna_bounds
        if bounds is None:
            return self._empty_table
        return self.get_subset_from_indices(bounds[0], None)

    @property
    def data_for_all_end(self) -> pd.DataFrame:
        """Table excluding any final rows for which price data unavailable.

        Rows will be excluded if the value of any price column is missing.
        """
        bounds = self._notna_bounds
        if bounds is None:
            return self._empty_table
        return self.get_subset_from_indices(None, bounds[1])

    @property
    def data_for_all(self) -> pd.DataFrame:
        """Table excluding initial and final rows with missing price data."""
        bounds = self._notna_bounds
        if bounds is None:
            return self._empty_table
        return self.get_subset_from_indices(*bounds)

    # Table operations

    def fillna(self, method: Literal["ffill", "bfill", "both"]) -> pd.DataFrame:
        """Return copy with filled missing values for non-trading indices.

        Missing values will result when a symbol is included to a table
        with an index that includes indices beyond those corresponding
        with the symbol's calendar. For example, if:
            Have multiple symbols with different trading hours.
            Reindexing to a different calendar.

        Parameters
        ----------
        method
            How to fill missing values.
                "ffill": fill with closest prior value.

                "bfill": fill with closest subsequent value.

                "both": first forward fill, then back fill. Has effect
                    of filling any initial missing values that would not
                    otherwise be filled with "ffill" alone.
        """
        df = self.prices.copy()
        if df.notna().all(axis=None):
            return df

        def fill(s: str | None):
            close_key = (s, "close") if s is not None else "close"
            open_key = (s, "open") if s is not None else "open"
            volume_key = (s, "volume") if s is not None else "volume"

            closes_missing = bv = df[close_key].isna()
            if closes_missing.all() or not closes_missing.any():
                return
            if method != "bfill":
                df.loc[:, close_key] = df[close_key].ffill()
                df.loc[bv, open_key] = df.loc[bv, close_key]
                bv = df[close_key].isna()
            if method != "ffill":
                df.loc[:, open_key] = df[open_key].bfill()
                df.loc[bv, close_key] = df.loc[bv, open_key]

            closes_still_missing = df[close_key].isna()
            closes_filled = closes_missing & ~closes_still_missing
            df.loc[closes_filled, volume_key] = 0
            cols = ["high", "low"]
            if s is not None:
                cols = [(s, col) for col in cols]  # type: ignore[misc]
            if not df.loc[closes_filled, cols].empty:
                df.loc[closes_filled, cols] = df.loc[closes_filled, close_key]

        if self.has_symbols:
            assert self.symbols is not None
            for s in self.symbols:
                fill(s)
        else:
            fill(None)

        return df

    @parse
    def operate(
        self,
        tz: Literal[False] | str | ZoneInfo | None = False,
        fill: Literal["ffill", "bfill", "both"] | None = None,
        include: Symbols | None = None,
        exclude: Symbols | None = None,
        data_for_all: bool = False,
        data_for_all_start: bool = False,
        data_for_all_end: bool = False,
        side: Literal["left", "right"] | None = None,
        close_only: bool = False,
        lose_single_symbol: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Undertake common table operation(s).

        Note: some operations are also provided via the `get` method of
        Prices classes.

        Note: depending on nature of requested operation(s), returned
        DataFrame may not conform with requirements necessary to use to
        the .pt accessor.

        Parameters
        ----------
        tz : Literal[False] | str | ZoneInfo | None,
        default: False (no change)
            For tables with an intraday interval:
                timezone to set the index to. Only available if index is
                timezone aware. If passed as string should be a valid
                arg for `zoneinfo.ZoneInfo`.

            For tables with an interval that is daily or higher (including
            tables that are a composite of daily and intraday price data):
                "UTC" or `zoneinfo.ZoneInfo("UTC")`: UTC
                None: tz-naive index.

        fill : Literal['ffill', 'bfill', 'both'] | None, default: None
            Fill missing values.
                "ffill": fill with closest prior value.
                "bfill": fill with closest subsequent value.
                "both": first forwardfill, then backfill. Has effect of
                    filling any initial missing values that would not
                    otherwise be filled with "ffill" alone.

        include : list[str] | str | None, default: include all symbols
            Symbols to include. All other symbols will be excluded. If
                passed, do not pass `exclude`.

            Ignored if table does not have symbols.

        exclude : list[str] | str | None, default: exclude no symbols
            Symbols to exclude. All other symbols will be included. If
                passed, do not pass `include`.

            Ignored if table does not have symbols.

        data_for_all : bool, default: False
            True: exclude any initial and final rows for which price
            data is not available for all included symbol(s).

        data_for_all_start : bool, default: False
            True: exclude any initial rows for which price data is not
            available for all included symbol(s).

        data_for_all_end : bool, default: False
            True: exclude any final rows for which price data is not
            available for all included symbol(s).

        side : Literal['left', 'right', None], default: leave index as is
            (ignored if index is not pd.IntervalIndex)
            Index table with a pd.DatetimeIndex that represents either the
            "left" or "right" side of each interval. Returned DataFrame
            will not have access to .pt accessor.

        close_only : bool, default: False
            True: return only close prices. Columns will be indexed by
                symbol only (simple pd.Index). Notes: Retruned DataFrame
                will not have access to .pt accessor.
            False: return 'open', 'high', 'low', 'close' and 'volume'
                columns for each symbol.

        lose_single_symbol : bool, default: False
            (ignored if `close_only` is True)
            True: If prices are for a single symbol then will lose symbol
                level from columns MultiIndex. Columns will be instead
                labelled with simple pd.Index.
        """
        # pylint: disable=missing-param-doc, too-complex, differing-type-doc
        # pylint: disable=too-many-branches, too-many-arguments
        if exclude is not None and include is not None:
            raise ValueError(
                "Pass only `exclude` or `include`, not both.\n`exclude`"
                f" received as {exclude}.\n`include` received as {include}."
            )

        prices = self.prices.copy()
        if include is not None and self.has_symbols:
            prices = prices[helpers.symbols_to_list(include)]
        elif exclude is not None and self.has_symbols:
            assert self.symbols is not None
            if set(exclude) == set(self.symbols):
                raise ValueError("Cannot exclude all symbols.")
            exclude = helpers.symbols_to_list(exclude)
            prices = prices[prices.columns.levels[0].difference(exclude)]

        if data_for_all or data_for_all_start:
            prices = prices.pt.data_for_all_start
        if data_for_all or data_for_all_end:
            prices = prices.pt.data_for_all_end

        if fill is not None:
            prices = prices.pt.fillna(fill)

        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        if tz or tz is None:
            if self.is_intraday:
                if tz is None:
                    prices = prices.pt.naive
                else:
                    self._verify_tz_awareness()
                    prices = prices.pt._set_tz(tz)
            else:  # pylint: disable=else-if-used
                if tz not in (None, UTC):
                    raise ValueError(
                        f"`tz` for class {type(self)} can only be UTC or timezone naive"
                        f" (None), not {tz}."
                    )
                elif tz is None:
                    prices = prices.pt.naive
                else:
                    prices = prices.pt.utc

        if close_only:
            prices = prices[[c for c in prices.columns if "close" in c]]
            prices.columns = prices.columns.droplevel(1)
        elif lose_single_symbol:
            clean_index = prices.columns.remove_unused_levels()
            if len(clean_index.levels[0]) == 1:
                prices.columns = prices.columns.droplevel()

        if side is not None and isinstance(self.index, pd.IntervalIndex):
            prices.index = getattr(prices.index, side)

        return prices

    @property
    def stacked(self) -> pd.DataFrame:
        """Stack symbols of a single-row price table to separate rows.

        Returns
        -------
        df.DataFrame
            index: pd.MultiIndex
                level 0: IntervalIndex
                    Index of received `df`.

                level 1: pd.Index
                    symbols.

            columns: pd.Index
                'open', 'close', 'high' 'low' 'volume'.

        Raises
        ------
        ValueError
            If price table has more than one row.
        """
        num_rows = len(self.index)
        if num_rows != 1:
            raise ValueError(
                "Only price tables with a single row can be stacked (price table"
                f" has {num_rows} rows)."
            )
        df = self.prices.stack(level=0, future_stack=True)
        assert isinstance(df, pd.DataFrame)
        df.sort_index(axis=0, level=0, ascending=True, inplace=True)
        return helpers.order_cols(df)

    @abc.abstractmethod
    def downsample(self, pdfreq: str | pd.offsets.BaseOffset) -> pd.DataFrame:
        """Return table downsampled to a given pandas frequency.

        Parameters
        ----------
        pdfreq
            Pandas frequency to which to downsample prices table.
        """


class PTDaily(_PT):
    """.pt accessor for prices table with daily interval."""

    @property
    def index(self) -> pd.DatetimeIndex:
        """Return index of prices table."""
        return self.prices.index

    @property
    def is_intraday(self) -> bool:
        """Query if interval is less than daily."""
        return False

    @property
    def is_daily(self) -> bool:
        """Query if interval is daily."""
        return True

    @property
    def _naive_index(self) -> pd.DatetimeIndex:
        return self.prices.index.tz_localize(None)

    @property
    def _utc_index(self) -> pd.DatetimeIndex:
        return self.prices.index.tz_localize(None).tz_localize(UTC)

    @property
    def _tz(self) -> ZoneInfo | None:
        """Timezone of index."""
        return self.index.tz

    @property
    def _interval(self) -> intervals.TDInterval:
        return intervals.ONE_DAY

    def get_subset_from_indices(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Get subset of table between given indices.

        Parameters
        ----------
        start : default: first indice
            First subset session.

        end : default: last indice
            Last subset session.
        """
        # pylint: disable=missing-param-doc
        start = self.first_ts if start is None else start
        end = self.last_ts if end is None else end
        start = self.convert_to_table_tz(start)
        end = self.convert_to_table_tz(end)

        if start not in self.index:
            msg = f"`start` ({start}) is not an indice."
            raise ValueError(msg)
        if end not in self.index:
            msg = f"`end` ({end}) is not an indice."
            raise ValueError(msg)
        return self.prices[start:end]  # type: ignore[misc]  # can slice with Timestamp

    def _compatible_sessions(self, *_, **__):
        self._not_implemented(self._compatible_sessions)

    def _compatible_index(self, *_, **__) -> bool:
        """Query if table index compatible with an index of daily frequency."""
        return True

    @functools.cache
    def indices_trading_status(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Query indices trading/non-trading status.

        Query if indices represent trading sessions or non-trading dates.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions.

        Returns
        -------
        pd.Series:
            Index: as price table index.

            Values:
                True: indice represents a trading session.
                False: indice represents a non-trading date.
        """
        sessions = calendar.sessions
        return pd.Series(self.index.isin(sessions), index=self.index)

    def price_at(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            "price_at is not implemented for daily price interval. Use"
            " `close_at` or `session_prices`."
        )
        raise NotImplementedError(msg)

    @parse
    def session_prices(
        self,
        session: Annotated[
            pd.Timestamp | str | datetime.datetime | int | float,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_datetimestamp),
        ],
    ) -> pd.DataFrame:
        """Return OHLCV prices for a given session.

        Parameters
        ----------
        session : pd.Timestamp | str | datetime.datetime | int | float
            Session for which require prices. Must not include time
            component. If passsed as a pd.Timestamp must be tz-naive.

        Returns
        -------
        DataFrame
            index: pd.DatatimeIndex
                Single indice of session that OHLCV data relates to.
            columns: pd.MultiIndex
                level 0: symbol
                level 1: ['open', 'high', 'low', 'close', 'volume']
        """
        if TYPE_CHECKING:
            assert isinstance(session, pd.Timestamp)
        parsing.verify_date_not_oob(session, self.first_ts, self.last_ts, "session")
        if self.index.tz is not None:
            session = session.tz_localize(UTC)
        if session not in self.index:
            raise ValueError(f"`session` {session} is not present in the table.")
        return self.prices.loc[[session]]

    @parse
    def close_at(
        self,
        date: Annotated[
            pd.Timestamp | str | datetime.datetime | int | float,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_datetimestamp),
        ],
    ) -> pd.DataFrame:
        """Return price as at end of a given day.

        For symbols where `date` represents a trading session, price will
        be session close. For all other symbols price will be as previous
        close (or NaN if prices table starts later than the previous
        close).

        Parameters
        ----------
        date : pd.Timestamp | str | datetime.datetime | int | float
            Date for which require end-of-day prices. Must not include time
            component. If passsed as a pd.Timestamp must be tz-naive.

        Returns
        -------
        DataFrame
            index: pd.DatatimeIndex
                Single indice of date that close prices relate to.
            columns: Index
                symbol.
        """
        if TYPE_CHECKING:
            assert isinstance(date, pd.Timestamp)
        parsing.verify_date_not_oob(date, self.first_ts, self.last_ts)
        if self.index.tz is not None:
            date = date.tz_localize(UTC)
        prices = self.operate(fill="ffill", close_only=True)
        i = prices.index.get_indexer([date], "ffill")[0]
        return prices.iloc[[i]]

    # Downsampling

    def _downsample_cbdays(
        self,
        pdfreq: pd.offsets.CustomBusinessDay,
        calendar: xcals.ExchangeCalendar,
    ) -> pd.DataFrame:
        """Downsample to a frequency defined in CustomBusinessDay."""
        error_start = (
            "To downsample to a frequency defined in terms of CustomBusinessDay"
            " `calendar` must be passed as an instance of"
            " `exchange_calendars.ExchangeCalendar`"
        )

        advices = (
            "\nNB. Downsampling will downsample to a frequency defined in"
            " CustomBusinessDay when either `pdfreq` is passed as a"
            " CustomBusinessDay (or multiple of) or when the table has a"
            ' CustomBusinessDay frequency and `pdfreq` is passed with unit "d".'
        )

        if not isinstance(calendar, xcals.ExchangeCalendar):
            raise TypeError(error_start + f" although received {calendar}." + advices)

        if calendar.day != pdfreq.base:
            raise ValueError(
                error_start + " which has a `calendar.day` attribute equal to the"
                " base CustomBusinessDay being downsampled to. Received calendar as"
                f" {calendar}." + advices
            )

        df = self.prices

        # Passing 'end' to the origin argument of DataFrame.resample has no
        # effect when passing rule as a CustomBusinessDay.
        origin = "start"

        # To ensure every indice, INCLUDING the last indice, comprises x
        # CustomBusinessDays it's necessary to curtail the start of the dataframe
        # to remove any indices that would otherwise result in the last indice
        # being comprised of less than the required number of CustomBuisnessDays.
        sessions = calendar.sessions_in_range(df.pt.first_ts, df.pt.last_ts)
        excess_sessions = len(sessions) % pdfreq.n

        # first row of dataframe to be resampled has to be a `calendar` session,
        # to the contrary initial rows (labeled earlier than a calendar session)
        # would aggregated and included as the first row of the downsampled table.
        # This first row would be labeled with a left side as the calendar's last
        # session prior to the table start. This row would not be accurate as it
        # would not include data for that session.
        first_session = calendar.date_to_session(self.first_ts, "next")
        start_session = calendar.session_offset(first_session, excess_sessions)
        df = df[start_session:]

        resampled = helpers.resample(df, pdfreq, origin=origin)
        resampled.index = pdutils.get_interval_index(resampled.index, pdfreq)
        resampled.index.left.freq = pdfreq
        resampled.index.right.freq = pdfreq
        return resampled

    def _downsample_days(self, pdfreq: str | pd.offsets.BaseOffset) -> pd.DataFrame:
        """Downsample to a frequency with unit "d"."""
        df = self.prices
        pd_offset = pd.tseries.frequencies.to_offset(pdfreq)

        # `origin` should reflect left side of last indice
        origin = df.pt.last_ts - pd_offset + helpers.ONE_DAY
        resampled = helpers.resample(df, pdfreq, origin=origin)
        resampled.index = pdutils.get_interval_index(resampled.index, pdfreq)

        drop_labels = resampled.index[(resampled.index.left < df.pt.first_ts)]
        if not drop_labels.empty:
            resampled.drop(drop_labels, inplace=True)

        return resampled

    def _downsample_months(
        self,
        pdfreq: str | pd.offsets.BaseOffset,
        calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar | None,
        drop_incomplete_last_indice: bool,
    ) -> pd.DataFrame:
        """Downsample to a frequency with unit that represents a period > daily.

        Example frequencies, "1MS", "3MS", "2QS".
        """
        df = self.prices
        pd_offset = pd.tseries.frequencies.to_offset(pdfreq)
        assert pd_offset is not None

        start_table = df.pt.first_ts
        if calendar is None:
            start_ds = pd_offset.rollforward(start_table)
            df = df[start_ds:]
        else:
            start_ds = pd_offset.rollback(start_table)
            if start_ds < start_table:
                pre_table_sessions = calendar.sessions_in_range(
                    start_ds, start_table - helpers.ONE_DAY
                )
                if not pre_table_sessions.empty:
                    start_ds = pd_offset.rollforward(start_table)
                    df = df[start_ds:]
        resampled = helpers.resample(df, pdfreq, origin="start", nominal_start=start_ds)
        resampled.index = pdutils.get_interval_index(resampled.index, pdfreq)

        if drop_incomplete_last_indice:
            index = resampled.index
            right_limit = df.pt.last_ts + helpers.ONE_DAY
            drop_labels = index[(index.right > right_limit)]
            if not drop_labels.empty:
                resampled.drop(drop_labels, inplace=True)

        return resampled

    @parse
    def downsample(  # pylint: disable=arguments-differ
        self,
        pdfreq: str | pd.offsets.BaseOffset,
        calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar | None = None,
        drop_incomplete_last_indice: bool = True,
    ) -> pd.DataFrame:
        """Return table downsampled to a given pandas frequency.

        Initial rows of the table may be ommitted from the downsampled data
        in order to ensure each row accurately reflects `pdfreq` and that
        table's last row is fully represented in the downsampled data.

        Parameters
        ----------
        pdfreq
            Downsample frequency as CustomBusinessDay or any valid input to
            pd.tseries.frequencies.to_offset. Examples:
                '2d', '3d', '5d', '10d', 'MS', 'QS'
                pd.offsets.Day(5), pd.offsets.Day(15)
                3 * calendar.day (where calendar is an instance of
                    xcals.ExchangeCalendar and hence calendar.day is an
                    instance of CustomBusinessDay).

            If `pdfreq` has unit 'd', for example "5d", and table has a
            frequency in CustomBusinessDay then will assume required
            frequency is the table's CustomBusinessDay frequency.

            If `pdfreq` defined in terms of or assumed as CustomBusinessDay
            then `calendar` must be passed as a calendar with
            `calendar.day` as the CustomBusinessDay frequency.

            If `pdfreq` has a unit that represents a period greater than
            daily, e.g. "1MS", "5MS", "1QS", then the first indice of the
            downsampled data will be determined by `calendar`:
                In all cases, the left side of the first indice of the
                downsampled data will be the table's first indice if and
                only if that indice coincides with `pdfreq`. For example,
                left side of first indice of downsampled data will be
                "2021-01-01" if table starts "2021-01-01". Otherwise:

                If `calendar` is not passed then the left side of the first
                indice of the downsampled data will be the table's first
                indice rolled forward to the next coincidence with
                `pdfreq`. For example, left side of first indice of
                downsampled data will be "2021-02-01" if table starts,
                for example, "2021-01-04".

                If `calendar` is passed then the left side of the first
                indice of the downsampled data will be the table's first
                indice rolled back to the prior coincidence with `pdfreq`
                IF there are no `calendar` sessions between the table's
                first indice and this rolled back date. Otherwise, as if
                `calendar` not passed, will be table's first indice rolled
                forward to the next coincidence with `pdfreq`. For example,
                if table starts "2021-04-01" then left side of first indice
                of donwsampled data will be "2021-01-01" only if there are
                no calendar sessions between "2021-01-01" through
                "2021-01-03", otherwise will be "2021-02-01".

        calendar
            Calendar against which to evaluate downsampled index.

            If `pdfreq` is defined in terms of CustomBusinessDay then
            `calendar` must be passed as an `xcals.ExchangeCalendar`.

            If `pdfreq` is defined in terms of days:
                If table frequency is a CustomBusinessDay then `calendar`
                must be passed as a calendar with `calendar.day` as the
                table frequency.

                `calendar` otherwise ignored.

            If `pdfreq` is defined in terms of a unit longer than daily
            (for example  "1MS", "5MS", "1QS") then passing `calendar`
            is optional with effect as described under `pdfreq`. In this
            case calendar can be passed as either an
            `xcals.ExchangeCalendar` or `calutils.CompositeCalendar`.

        drop_incomplete_last_indice
            Only relevant if `pdfreq` has a unit that represents a period
            greater than daily, e.g. "1MS", "5MS", "1QS".

            True: drop any last indice that has a right side beyond the
            table's last indice.

            False: include any last indice which has a right side beyond
            the tables last indice.
        """
        if isinstance(pdfreq, pd.offsets.CustomBusinessDay):
            return self._downsample_cbdays(pdfreq, calendar)

        try:
            offset = pd.tseries.frequencies.to_offset(pdfreq)
        except (ValueError, KeyError):
            msg = (
                f"Received `pdfreq` as {pdfreq} although must be either of"
                " type pd.offsets.CustomBusinessDay or acceptable input to"
                " pd.tseries.frequencies.to_offset that describes a"
                ' frequency greater than one day. For example "2d", "5d"'
                ' "QS" etc.'
            )
            raise ValueError(msg) from None
        else:
            assert offset is not None
            freqstr: str = offset.freqstr

        value, unit = helpers.extract_freq_parts(freqstr)
        if unit.lower() == "d":
            if isinstance(self.freq, pd.offsets.CustomBusinessDay):
                pdfreq = self.freq.base * value
                return self._downsample_cbdays(pdfreq, calendar)
            else:
                return self._downsample_days(pdfreq)

        invalid_units = ["h", "min", "MIN", "s", "L", "ms", "U", "us", "N", "ns"]
        ext = ["t", "T", "H", "S"]  # for pandas pre 2.2 compatibility
        if unit in invalid_units + ext:
            raise ValueError(
                "Cannot downsample to a `pdfreq` with a unit more precise than 'd'."
            )

        return self._downsample_months(freqstr, calendar, drop_incomplete_last_indice)


class _PTIntervalIndex(_PT):  # pylint: disable=abstract-method  # imp'd in subclasses
    """Base extension for prices table indexed with a pd.IntervalIndex."""

    @property
    def index(self) -> pd.IntervalIndex:
        """Return index of prices table."""
        return self.prices.index

    @property
    def first_ts(self) -> pd.Timestamp:
        """Return first timestamp represented in table."""
        return self.index[0].left

    @property
    def last_ts(self) -> pd.Timestamp:
        """Return last timestamp represented in table."""
        return self.index[-1].right

    @property
    def is_daily(self) -> bool:
        """Query if interval is daily."""
        return False

    @property
    def _naive_index(self) -> pd.IntervalIndex:
        index = self.index
        return pd.IntervalIndex.from_arrays(
            index.left.tz_localize(None),
            index.right.tz_localize(None),
            closed=index.closed,
        )

    @property
    def _utc_index(self) -> pd.IntervalIndex:
        ii = self.index
        indexes = []
        for index in (ii.left, ii.right):
            if index.tz is None:
                indexes.append(index.tz_localize(UTC))
            elif index.tz == UTC:
                indexes.append(index)
            else:
                indexes.append(index.tz_convert(UTC))
        return pd.IntervalIndex.from_arrays(indexes[0], indexes[1], ii.closed)

    @property
    def _tz(self) -> ZoneInfo | None:
        if self.index.left.tz == self.index.right.tz:
            return self.index.left.tz
        else:
            return None

    def get_subset_from_indices(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Get subset of table between given indices.

        Parameters
        ----------
        start : default: left side of first indice
            Left side of first indice required subset.

        end : default: right side of last indice
            Right side of or inside last indice of required subset.

        Raises
        ------
        ValueError
            If `start` is not the left side of an indice or if `right` is
            not the right side of or within and indice.
        """
        # pylint: disable=missing-param-doc
        start = self.first_ts if start is None else start
        end = self.last_ts if end is None else end
        start = self.convert_to_table_tz(start)
        end = self.convert_to_table_tz(end)

        if start not in self.index.left:
            msg = f"`start` ({start}) is not the left side of an indice."
            raise ValueError(msg)

        if pd.Interval(start, start, "left") in self.index:
            # daily composite tables indexed [session, session) such that when session
            # is start this 'start indice' will be excluded (although the interval is
            # closed "left", pandas will not consider any value to fall within an
            # interval of zero length).
            start -= helpers.ONE_SEC
            # typing note: datetime/timestamp tomato/tomarto

        if not (end in self.index.right or self.index.contains(end).any()):
            msg = (
                f"`end` ({end}) is not the right side of or contained within"
                " an indice."
            )
            raise ValueError(msg)
        rng = pd.Interval(start, end, "left")
        bv = self.index.overlaps(rng)
        return self.prices[bv]


class PTIntraday(_PTIntervalIndex):
    """.pt accessor for prices table with intraday interval.

    Notes
    -----
    Class also serves as base class for prices tables with interval of
    multiple days (which share much of the implementation concerned with
    indexing against a pd.IntervalIndex).
    """

    @property
    def indexed_left(self) -> pd.DataFrame:
        """Price table indexed as left side of intervals."""
        index = self.index.left.rename("left")
        return pd.DataFrame(self.prices.values, index, columns=self.columns)

    @property
    def indexed_right(self) -> pd.DataFrame:
        """Price table indexed as right side of intervals."""
        index = self.index.right.rename("right")
        return pd.DataFrame(self.prices.values, index, columns=self.columns)

    @property
    def is_intraday(self) -> bool:
        """Query if interval is less than daily."""
        return True

    def _tz_index(self, tz: str | ZoneInfo) -> pd.IntervalIndex:
        """Return index with tz as `tz`."""
        return pdutils.interval_index_new_tz(self.index, tz)

    def _set_tz(self, tz: str | ZoneInfo) -> pd.DataFrame:  # type: ignore[override]
        """Convert index to a given timezone."""
        # typing note: extends super args to provide for tz to be passed as str.
        return self._new_index(self._tz_index(tz))

    def set_tz(self, tz: str | ZoneInfo) -> pd.DataFrame:
        """Set index timezone.

        Parameters
        ----------
        tz
            Timezone to set index to.
        """
        return self._set_tz(tz)

    # PRICE OPERATIONS

    # Mappings to sessions

    @parse
    def sessions(
        self,
        calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
        direction: Literal["previous", "next"] | None = "previous",
    ) -> pd.Series:
        """Return pd.Series mapping indices sessions.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate sessions. If `calendar` is
            a `calutils.CompositeCalendar` and the left of an indice is a
            minute of two overlapping sessions then will be assigned the
            earlier session.

        direction
            In event the left of an indice does not represent a trading
            minute:
                "previous": assign previous session
                "next": assign next session
                None: assign pd.NaT (Not a Time)
            Note: 'open' anchored prices will have been evaluated by
            assigning any out of hours indices to the previous session.
        """

        def _get_session(indice) -> pd.Timestamp:
            try:  # pylint: disable=too-many-try-statements
                if isinstance(calendar, xcals.ExchangeCalendar):
                    direction_ = "none" if direction is None else direction
                    s = calendar.minute_to_session(indice, direction=direction_)
                else:
                    s = calendar.minute_to_sessions(indice, direction=direction)[0]
            except ValueError:
                s = pd.NaT
            return s

        srs = self.index.left.to_series(index=self.index)
        srs = srs.apply(_get_session)
        srs.name = "session"
        return srs

    @parse
    def session_column(
        self,
        calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
        direction: Literal["previous", "next", None] = "previous",
    ) -> pd.DataFrame:
        """Return table with extra column mapping indices to sessions.

        Note: Does not make changes to the prices table in place, rather
        returns a copy of DataFrame on which accessor called.

        Parameters
        ----------
        As `sessions`
        """
        # pylint: disable=missing-param-doc
        srs = self.sessions(calendar, direction)
        if self.has_symbols:
            srs.name = (srs.name, srs.name)
        df = self.prices.copy()
        return pd.concat([srs, df], axis=1)

    def _compatible_sessions(
        self, index: pd.IntervalIndex, calendar: xcals.ExchangeCalendar
    ) -> pd.Series:
        """Query if `index` indices are compatible with table indices.

        Result offered by-session.

        Parameters
        ----------
        index
            Index against which to evaluate compatiblity.

        calendar
            Calendar against which `index` was evaluated.
        """
        # pylint: disable=too-many-locals, too-many-statements
        start = calendar.minute_to_session(index[0].left)
        end = calendar.minute_to_session(index[-1].right - helpers.ONE_MIN, "previous")
        slc = slice(start, end)
        opens = calendar.opens[slc].astype("int64")
        closes = calendar.closes[slc].astype("int64")
        sessions = opens.index

        index_union = self.utc.pt.index.union(index, sort=False).sort_values()
        nano_index = index_union.left.asi8

        srs = pd.Series(True, index=sessions)
        for session, open_, close in zip(sessions, opens, closes):
            bv = (nano_index >= open_) & (nano_index < close)
            srs[session] = index_union[bv].is_non_overlapping_monotonic
        return srs

    def _compatible_index(self, index: pd.IntervalIndex) -> bool:
        """Query if `index` compatible with table index."""
        if self.interval == helpers.ONE_MIN:  # shortcut
            return True
        assert index.left.tz is UTC
        df = self.prices if self.is_daily else self.utc
        index_union = df.pt.index.union(index, sort=False).sort_values()
        return index_union.is_non_overlapping_monotonic

    # indices_trading_status

    @staticmethod
    def _is_trading_period(
        indice: pd.Interval, cal: xcals.ExchangeCalendar
    ) -> bool | float:
        """Query if indice represents a trading period. np.nan if partial."""
        start = indice.left
        end = indice.right - helpers.ONE_MIN
        trading_minutes = len(cal.minutes_in_range(start, end))
        interval_minutes = indice.length.total_seconds() / 60
        if trading_minutes == interval_minutes:
            return True
        elif trading_minutes > 0:
            return np.nan
        else:
            return False

    @functools.cache
    @parse
    def indices_trading_status(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Query indices trading/non-trading status.

        Query if indices represent trading periods, non-trading periods or
        both.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading times.

        Returns
        -------
        pd.Series:
            Index: as price table index.

            Values:
                True: indice represents a period during which there are no
                non-trading times.

                False: indice represents a period during which there are no
                trading times.

                nan: indice represents both trading and non-trading
                periods.
        """
        srs = self.index.to_series()
        return srs.apply(self._is_trading_period, args=[calendar])

    # trading minutes

    @parse
    def indices_trading_minutes(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Return number of trading minutes that comprise each indice.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading minutes.
        """
        partial_indices = self.indices_partial_trading(calendar)
        non_trading_indices = self.indices_non_trading(calendar)

        trading_mins = []
        for i in self.index:
            if i in non_trading_indices:
                mins = 0
            elif i in partial_indices:
                start, end = i.left, i.right
                mins = len(calendar.minutes_in_range(start, end - helpers.ONE_SEC))
            else:
                mins = int(i.length.total_seconds() / 60)
            trading_mins.append(mins)
        return pd.Series(trading_mins, index=self.index, name="trading_mins")

    def indices_trading_minutes_values(
        self, calendar: xcals.ExchangeCalendar
    ) -> np.ndarray:
        """Retrun set of trading minutes that indices can be comprised of.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading minutes.
        """
        arr = self.indices_trading_minutes(calendar).unique()
        assert isinstance(arr, np.ndarray)
        return arr

    def trading_minutes_interval(
        self, calendar: xcals.ExchangeCalendar
    ) -> intervals.TDInterval | None:
        """Return price table interval as trading minutes of each indice.

        Returns None if all indices do not represent the same number of
        trading minutes.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading minutes.
        """
        trading_mins = self.indices_trading_minutes_values(calendar)
        if len(trading_mins) != 1:
            return None
        else:
            mins = int(trading_mins[0])
            return intervals.TDInterval(pd.Timedelta(mins, "min"))

    def indices_have_regular_trading_minutes(
        self, calendar: xcals.ExchangeCalendar
    ) -> bool:
        """Query if all indices comprise same number of trading minutes.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading minutes.
        """
        return self.trading_minutes_interval(calendar) is not None

    # indice length properties

    @property
    def indices_length(self) -> pd.DataFrame:
        """Return indices length count."""
        return self.index.length.value_counts()

    @property
    def by_indice_length(
        self,
    ) -> collections.abc.Iterable[tuple[pd.Timedelta, pd.DataFrame]]:
        """Return generator of indices grouped by length.

        Ordered from least common to most common indice length.

        Example:
            it = prices.pt.by_indice_length()
            td, df = next(it)
            td # Timedelta of least common indice length
            df # DataFrame of indices of least common indice length
            td, df = next(it)
            td # Timedelta of next least common indice length
            df # DataFrame of indices of next least common indice length
            etc...through to most common indice length.

        Yields
        ------
        Iterable[tuple[pd.Timedelta, DataFrame]
            [0]: indice length
            [1]: DataFrame of indices with this length.
        """
        vcs = self.indices_length
        for i in range(len(vcs) - 1, -1, -1):
            td = vcs.index[i]
            df = self.prices[self.index.length == td]
            yield td, df  # type: ignore[misc]  # return as required.

    # price_at and dependencies

    def _get_indice(self, ts: pd.Timestamp) -> pd.Interval | None:
        """Return indice that contains `ts`, or None if not in any."""
        indices = self.index[self.index.contains(ts)]
        if not any(indices):
            return None
        else:
            assert len(indices) == 1
            return indices[0]

    def _get_indice_loc(self, ts: pd.Timestamp) -> int | None:
        """Return index of indice that includes `ts`, or None if not in any."""
        indice = self._get_indice(ts)
        if indice is None:
            return None
        else:
            return self.index.get_loc(indice)  # type: ignore[return-value]  # as req'd

    def _get_loc(
        self, ts: pd.Timestamp, method: Literal["ffill", "bfill"] = "ffill"
    ) -> int:
        """Return index for `ts`.

        Parameters
        ----------
        method
            If `ts` is not included in any indice then return index for:
                "ffill": closest prior indice.
                "bfill": closest subsequent indice.
        """
        i = self._get_indice_loc(ts)
        if i is None:
            i = self.index.left.get_indexer([ts], method=method)[0]
        return i

    def _get_row(self, ts: pd.Timestamp) -> pd.DataFrame:
        """Return DataFrame with the single indice that contains `ts`.

        If `ts` is not included in any indice then returns closest prior
        indice.
        """
        i = self._get_loc(ts, method="ffill")
        return self.prices.iloc[[i]]

    @parse
    def price_at(  # pylint: disable=arguments-differ
        self,
        ts: Annotated[
            pd.Timestamp | str | datetime.datetime | int | float,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_timetimestamp),
        ],
        tz: Annotated[
            ZoneInfo | str | None,
            Parser(parsing.to_timezone, parse_none=False),
        ] = None,
    ) -> pd.DataFrame:
        """Return most recent price as at a given timestamp.

        Parameters
        ----------
        ts
            Timestamp as at which to return most recent available price.
            Will raise ValueError if `ts` represents a session. To request
            prices at midnight pass as a tz-aware `pd.Timestamp`.

        tz : default: table's tz
            Timezone of `ts` and to use for returned index.
            If `ts` is tz-aware then `tz` will NOT override `ts` timezone,
            although will be used to define the index of the returned
            table.

        See Also
        --------
        close_at
        session_prices
        """
        # pylint: disable=missing-param-doc
        if TYPE_CHECKING:
            assert isinstance(ts, pd.Timestamp)
            assert tz is None or isinstance(tz, ZoneInfo)

        if tz is None:
            self._verify_tz_awareness()
            tz = self.tz

        ts = parsing.parse_timestamp(ts, tz)
        parsing.verify_time_not_oob(
            ts, self.first_ts.astimezone(UTC), self.last_ts.astimezone(UTC)
        )
        df = self.utc.pt.fillna("ffill")
        row = df.pt._get_row(ts)  # pylint: disable=protected-access
        side = "left" if row.index.contains(ts) else "right"
        column = "open" if side == "left" else "close"
        if self.has_symbols:
            if TYPE_CHECKING:
                assert self.symbols is not None
            columns: list[tuple[str, str]] | list[str]
            columns = [(s, column) for s in self.symbols]
        else:
            columns = [column]
        res = row[columns]
        ts = getattr(row.index, side)
        if TYPE_CHECKING:
            assert isinstance(ts, pd.Timestamp)
        res.index = ts if tz is None else ts.tz_convert(tz)
        res.columns = (
            res.columns.droplevel(1) if self.has_symbols else pd.Index(["price"])
        )
        return res

    # Base methods that are not implemented.

    def close_at(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            "`close_at` is not implemented for intraday price intervals. Use"
            " `price_at`."
        )
        raise NotImplementedError(msg)

    def session_prices(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            "`session_prices` is not implemented for intraday price"
            " intervals. Use `price_at`."
        )
        raise NotImplementedError(msg)

    # Downsampling

    @staticmethod
    def _consolidate_resampled_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate duplicate indices.

        Consolidates duplicate indices resulting from final indice of a
        session having same indice as first indice of next session.
        """
        dups = df.index[df.index.duplicated()]
        new_rows = []
        for dup in dups:
            dup_df = df.loc[dup]
            # Aggregate (via group to allow use of 'first' and 'last' functions)
            group = np.array([0] * len(dup_df))
            groups = dup_df.groupby(by=group)
            res = groups.agg(helpers.agg_funcs(dup_df))
            res.index = dup_df.index[:1]
            new_rows.append(res)
        new_rows_df = pd.concat(new_rows)
        new_rows_df = helpers.volume_to_na(new_rows_df)
        drop_bv = df.index.duplicated(False)
        no_dups = df.drop(df.index[drop_bv])
        df = pd.concat([no_dups, new_rows_df])
        df = df.sort_values("left")
        return df

    def _downsample_open(
        self,
        pdfreq: mptypes.PandasFrequency,
        cal: xcals.ExchangeCalendar,
        curtail_end: bool = True,
        cc: CompositeCalendar | None = None,
    ) -> pd.DataFrame:
        """Return downsampled table anchored to sessions' opens.

        Parameters as `self.downsample`.
        """
        # pylint: disable=too-complex, too-many-locals, too-many-statements
        offset = pdfreq.as_offset
        df = self.utc.pt.indexed_left

        # clean first session
        first_session = cal.minute_to_session(self.first_ts, "next")
        open_fs = cal.session_open(first_session)
        # lose rows prior to calendar's first session
        df = df[df.index >= open_fs]
        if df.empty:
            raise errors.PricesUnavailableIntervalResampleError(offset, self)

        # if table starts later than open, lose any initial surplus rows
        deficit = (df.index[0] - open_fs) % offset
        if deficit:
            surplus = offset - deficit
            surplus_rows = surplus // self.interval
            df = df.iloc[surplus_rows:]
            if df.empty:
                raise errors.PricesUnavailableIntervalResampleError(offset, self)

        # clean last session
        # remove final rows to prevent last indice comprising incomplete data
        if curtail_end:
            last_session = cal.minute_to_session(self.last_ts, "previous")
            open_ls = cal.session_open(last_session)
            surplus = (self.utc.pt.last_ts - open_ls) % offset
            if surplus:
                surplus_rows = surplus // self.interval
                df = df[:-surplus_rows]
                if df.empty:
                    raise errors.PricesUnavailableIntervalResampleError(offset, self)

        def _groupby_session(indice) -> pd.Timestamp:
            return cal.minute_to_session(indice, "previous", _parse=False)

        def _groupby_composite_session(indice) -> pd.Timestamp:
            assert cc is not None
            sessions = cc.minute_to_sessions(indice, _parse=False)
            if len(sessions) == 1:
                return sessions[0]
            else:
                # session overlap, assign to calendar session
                session = cal.minute_to_session(indice, "previous", _parse=False)
                return session

        f = _groupby_session if cc is None else _groupby_composite_session
        grouped = df.groupby(by=f)

        r_dfs = []
        for group in grouped:
            g_df = group[1]
            # (at least) pandas 2.2.0 fix if not supporting these versions
            # then can directly pass group[0] to cal.session_open
            # See https://github.com/pandas-dev/pandas/issues/57192
            session = group[0] if group[0].tz is None else group[0].tz_convert(None)
            try:
                origin = cal.session_open(session)
            except xcals.errors.NotSessionError:
                origin = "start"
            r_dfs.append(helpers.resample(g_df, rule=offset, origin=origin))

        resampled = pd.concat(r_dfs)
        if resampled.index.has_duplicates:
            resampled = self._consolidate_resampled_duplicates(resampled)

        index = pdutils.get_interval_index(resampled.index, offset)
        if not index.is_overlapping:
            resampled.index = index
            return resampled.pt.set_tz(self.tz)

        # index is overlapping

        # curtail first indice of each session to earliest session open
        for group, r_df in zip(grouped, r_dfs):
            open_ = group[1].index[0]
            idx = pdutils.get_interval_index(r_df.index, offset)
            first_indice = pd.Interval(open_, idx[0].right, "left")
            r_df.index = pd.IntervalIndex([first_indice]).union(idx[1:])
        resampled = pd.concat(r_dfs)

        if resampled.index.is_overlapping:
            # curtail right of last indice of each session to left of first
            # indice of next session
            resampled.index = pdutils.make_non_overlapping(resampled.index)

        warnings.warn(errors.IntervalIrregularWarning())
        return resampled.pt.set_tz(self.tz)

    def _downsample_workback(self, interval: pd.Timedelta) -> pd.DataFrame:
        """Return table downsampled to `interval` and anchored to last indice."""
        assert self.interval is not None
        table_interval = self.interval
        # Resample each n rows, avoids introducing non-trading times and
        # allows a row to encompass prices from contiguous trading sessions
        num_rows = interval // table_interval
        excess_rows = len(self.prices) % num_rows
        df = self.prices[excess_rows:].copy()
        agg_functions = helpers.agg_funcs(df)
        name_l: tuple[str, str] | str = ("l", "l") if self.has_symbols else "l"
        name_r: tuple[str, str] | str = ("r", "r") if self.has_symbols else "r"
        df[name_l], df[name_r] = df.index.left, df.index.right
        df.reset_index(drop=True, inplace=True)
        groups = df.groupby(df.index // num_rows)
        agg_functions[name_l], agg_functions[name_r] = "first", "last"
        res = groups.agg(agg_functions)
        res.index = pd.IntervalIndex.from_arrays(res[name_l], res[name_r], "left")
        for col in [name_l, name_r]:
            del res[col]
        res = clean_columns(res)
        res = helpers.volume_to_na(res)
        return res

    @parse
    def downsample(
        # pylint: disable=arguments-differ
        self,
        pdfreq: Annotated[str, Coerce(mptypes.PandasFrequency)],
        anchor: Literal["workback", "open"] = "workback",
        calendar: xcals.ExchangeCalendar | None = None,
        curtail_end: bool = False,
        composite_calendar: CompositeCalendar | None = None,
    ) -> pd.DataFrame:
        """Return table downsampled to a frequency defined in minutes/hours.

        Method only available if table has a regular interval (as
        determined by .pt.has_regular_interval).

        The downsampled data will include any indices that represent
        either non-trading or partial trading periods. The following
        PTIntraday methods can be used to evaluate such periods:
            indices_all_trading()
            indices_trading()
            indices_non_trading()
            indices_partial_trading()
            indices_partial_trading_info()

        Parameters
        ----------
        pdfreq : str
            Resample frequency as valid str input to
            `pd.tseries.frequencies.to_offset` with unit in
            ["min", "h"]. Examples:
                "15min", "30min", "1h", '4h', '12h'

            `pdfreq` must represent an interval higher than the
            table interval.

            Table interval (`self.interval`) must be a factor of the
            interval represented by `pdfreq`.

        anchor : Literal["workback", "open"], default: "workback"
            "workback": Anchor on last indice and work backwards. Indices
                evaluated such that each maintains the interval in terms of
                trading minutes (but not necessarily minutes of any
                particular calendar).

            "open": Group by session and anchor on each session open.
                Sessions evaluated from `calendar`. Each session open will
                be represented by the left side of an indice. A session's
                close will only be represented by the right side of an
                indice in the event that downsample interval is a factor
                of total session hours (or of total pm subsession hours
                if the session has a break). Otherwise, when evaluated
                against `calendar`, one indice per session will represent
                a partial trading indice. Any data falling within a break
                or between a session close and the following session open
                will be consolidated to non-trading indices (when evaluate
                against `calendar`).

                When there is no data immediately following a session close
                although there is subsequently data prior to the next
                session open, indices in between will be included and
                filled with missing values (session's evalutated against
                `calendar`). The introduction of these 'missing values'
                indices can be avoided by passing a `composite_calendar`.

                Indices will be introduced during any session break. If
                there is no data available for these indices they will be
                represented with rows of missing values.

        calendar : xcals.ExchangeCalendar, optional
            (ignored if `anchor` is "workback")
            Calendar against which to evaluate sessions' open.

        curtail_end : bool, default: False
            (ignored if `anchor` is "workback")
            True: Curtail final rows of base table in order that
                downsampled indice is evaluated from a full interval of
                data, i.e. if the resampled interval is 60 minutes then the
                final interval will be evaluated from 60 mintues of base
                data, albeit at the cost that any excess rows at the
                end of the base table will be excluded from the downsampled
                data.
            False: Include all final rows in the resampled data. This may
                result in the final indice being evaluted from fewer
                minutes of data than the interval length, e.g. if the
                table interval is 60 minutes then the final indice of
                the resampled table may have been evaluated from only, for
                example, the first 20 minutes of data.

        composite_calendar : CompositeCalendar, optional
            (ignored if `anchor` is "workback")
            Only useful if price table symbols are not all associated with
            the same calendar.

            If passed then the session each indice belongs to will be
            evaluated against `composite_calendar`. This has the advantage
            of avoiding the introduction of non-trading indices when there
            are symbols that open (for a given session) prior to the open
            evaluated from `calendar`.

            Note: origin for each session will continue to be determined by
            `calendar`

        Raises
        ------
        ValueError
            If table does not have a regular interval.

            If `pdfreq`:
                represents an interval lower than the table
                interval.

                does not have a valid unit.

                does not represent an interval that is a factor of
                the table interval.

        TypeError
            If `anchor` is open and `calendar` is not passed as an
            instance of `exchange_calendars.ExchangeCalendar`.

        See Also
        --------
        reindex_to_calendar
        """
        # pylint: disable=too-many-arguments, missing-param-doc
        if TYPE_CHECKING:
            assert isinstance(pdfreq, mptypes.PandasFrequency)

        if not self.has_regular_interval:
            raise ValueError(
                "Cannot downsample a table for which a regular interval"
                " cannot be ascertained."
            )
        assert self.interval is not None

        anchor_ = mptypes.Anchor.OPEN if anchor == "open" else mptypes.Anchor.WORKBACK

        if anchor_ is mptypes.Anchor.OPEN and calendar is None:
            raise TypeError(
                'If anchor "open" then `calendar` must be passed as an'
                " instance of xcals.ExchangeCalendar."
            )

        unit = genutils.remove_digits(pdfreq)
        # for pandas pre 2.2. compatibility
        if unit == "T":
            unit = "min"
        if unit == "H":
            unit = "h"

        valid_units = ["min", "h"]
        if unit not in valid_units:
            raise ValueError(
                f"The unit of `pdfreq` must be in {valid_units} although received"
                f" `pdfreq` as {pdfreq}."
            )

        value = int(genutils.remove_nondigits(pdfreq))
        minutes = value if unit == "min" else 60 * value
        interval = pd.Timedelta(minutes, "min")

        table_interval = self.interval
        if table_interval > interval:
            raise ValueError(
                "Downsampled interval must be higher than table interval,"
                f" although downsample interval evaluated as {interval}"
                f" whilst table interval is {table_interval}."
            )

        if interval % table_interval:
            raise ValueError(
                "Table interval must be a factor of downsample interval,"
                f" although downsampled interval evaluated as {interval}"
                f" whilst table interval is {table_interval}."
            )

        if anchor_ is mptypes.Anchor.WORKBACK:
            return self._downsample_workback(interval)
        else:
            return self._downsample_open(
                pdfreq, calendar, curtail_end, composite_calendar
            )


class _PTIntervalIndexNotIntraday(_PTIntervalIndex):  # pylint: disable=abstract-method
    """Base extension for non-intraday PT indexed with pd.IntervalIndex."""

    # pylint note: abstract methods implemented in subclasses

    @property
    def is_intraday(self) -> bool:
        """Query if interval is less than daily."""
        return False

    def _compatible_sessions(self, *_, **__):
        self._not_implemented(self._compatible_sessions)

    def _compatible_index(self, *_, **__):
        self._not_implemented(self._compatible_index)

    def get_trading_index(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        self._not_implemented(self.get_trading_index)

    def reindex_to_calendar(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        self._not_implemented(self.reindex_to_calendar)


class PTMultipleSessions(_PTIntervalIndexNotIntraday):
    """.pt accessor for prices table with interval greater than daily."""

    @staticmethod
    def _are_trading_sessions(
        indice: pd.Interval,
        cal: xcals.ExchangeCalendar,
    ) -> bool | float:
        """Query if all days are tradings sessions. NaN if partial."""
        start = indice.left
        end = indice.right - helpers.ONE_DAY
        sessions = cal.trading_index(start, end, "1D")
        dates = pd.date_range(start, end)
        if sessions.empty:
            return False
        elif len(dates) == len(sessions) and (dates == sessions).all():
            return True
        else:
            return np.nan

    @functools.cache
    @parse
    def indices_trading_status(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Query indices trading/non-trading status.

        Query if indices represent trading periods, non-trading periods or
        both.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions.

        Returns
        -------
        pd.Series:
            Index: as price table index.

            Values:
                True: indice represents a period during which all dates are
                sessions.

                False: indice represents a period during which there are no
                trading sessions.

                nan: indice represents both sessions and non-trading dates.
        """
        srs = self.index.to_series()
        return srs.apply(self._are_trading_sessions, args=[calendar])

    @parse
    def indices_partial_trading_info(
        self, calendar: xcals.ExchangeCalendar
    ) -> dict[pd.IntervalIndex, pd.DatetimeIndex]:
        """Return info on partial trading indices.

        Returns information on indices that cover both sessions and
        non-trading dates.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate partial trading indices.

        Returns
        -------
        dict[pd.IntervalIndex, pd.DatetimeIndex]
            key: partial trading indice
            item: date or dates of indice that are not calendar sessions.
        """
        cal = calendar
        partial_indices = self.indices_partial_trading(cal)
        d = {}
        for indice in partial_indices:
            start, end = indice.left, indice.right - helpers.ONE_DAY
            sessions = cal.sessions_in_range(start, end)
            dates = pd.date_range(start, end, freq="D")
            d[indice] = dates.difference(sessions)
        return d

    def close_at(self, *_, **__) -> pd.DataFrame:
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            f"`close_at` not implemented for {type(self)} as table"
            f" interval too high to offer close prices for a specific date."
        )
        raise NotImplementedError(msg)

    def session_prices(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            f"`session_prices` not implemented for {type(self)} as table"
            f" interval too high to offer prices for a specific session."
        )
        raise NotImplementedError(msg)

    def price_at(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            f"`price_at` is not implemented for {type(self)} as intervals"
            f" are not intraday."
        )
        raise NotImplementedError(msg)

    def downsample(self, *_, **__) -> pd.DataFrame:
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        msg = (
            f"downsample is not implemented for {type(self)}."
            " Downsample from daily data."
        )
        raise NotImplementedError(msg)


class PTDailyIntradayComposite(_PTIntervalIndexNotIntraday):
    """.pt accessor for daily/intraday composite prices table."""

    @property
    def _interval(self) -> None:
        return None

    @property
    def daily_part(self) -> pd.DataFrame:
        """Part of composite table comprising daily intervals."""
        df = self.prices[self.index.left == self.index.right]
        df.index = df.index.left.tz_convert(None)
        return df

    @property
    def intraday_part(self) -> pd.DataFrame:
        """Part of composite table comprising intraday intervals."""
        df = self.prices[self.index.left != self.index.right]
        return df

    def indices_trading_status(self, calendar: xcals.ExchangeCalendar) -> pd.Series:
        """Query indices trading/non-trading status.

        Query if indices represent trading periods, non-trading periods or
        both.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate trading sessions/times.

        Returns
        -------
        pd.Series:
            Index: as price table index.

            Values:
                True: indice represents a trading session or a period
                during which there are no non-trading dates/times.

                False: indice represents a non-trading date or a period
                during which there are no trading sessions/times.

                nan: indice represents both trading and non-trading periods
                (NB not possible for the daily part of the table).
        """
        its_daily = self.daily_part.pt.indices_trading_status(calendar)
        its_intraday = self.intraday_part.pt.indices_trading_status(calendar)
        srs = pd.concat([its_daily, its_intraday])
        srs.index = self.index
        return srs

    @parse
    def price_at(
        self,
        ts: Annotated[
            pd.Timestamp | str | datetime.datetime | int | float,
            Coerce(pd.Timestamp),
            Parser(parsing.verify_timetimestamp),
        ],
    ) -> pd.DataFrame:
        """Most recent registered price as at a given timestamp.

        Note: Only available over part of index defined by intraday
        times (as opposed to part of index defined by sesions).

        Parameters
        ----------
        ts
            Time as at which to return most recent available price.

            Must be tz-naive or have tz as 'UTC'.

            Will raise ValueError if `ts` passed as a date. To request
            prices at midnight pass `ts` as a pd.Timestamp with tz set
            to 'UTC'.
        """
        df = self.utc.pt.intraday_part
        return df.pt.price_at(ts)

    def session_prices(self, session: mptypes.DateTimestamp) -> pd.DataFrame:
        """Return OHLCV for a given session.

        Note: only available over range of index defined by sessions
        (as opposed to intraday times).

        Parameters
        ----------
        session
            Session for which require prices. Must not include time
            component. If passsed as a pd.Timestamp must be tz-naive or
            have tz as 'UTC'.

        Returns
        -------
        DataFrame
            index: DatatimeIndex
                Single indice of session that OHLCV data relates to.

            columns: MultiIndex
                level 0: symbol
                level 1: ['open', 'high', 'low', 'close', 'volume']
        """
        df = self.daily_part
        return df.pt.session_prices(session)

    def close_at(self, date: mptypes.DateTimestamp) -> pd.DataFrame:
        """Return price as at close of a given day.

        For symbols where `date` represents a trading session, price will
        be as at session close. For all other symbols price is as at
        previous close (or np.nan if prices table starts later than the
        previous close).

        Note: only available over range of index defined by sessions
        (as opposed to intraday times).

        Parameters
        ----------
        date
            Date for which require end-of-day prices. Must not include time
            component. If passsed as a pd.Timestamp must be tz-naive or
            have tz as 'UTC'.

        Returns
        -------
        pd.DataFrame:
            index: DatatimeIndex
                Single indice of date that close prices relate to.

            columns: Index
                symbol.
        """
        df = self.daily_part
        return df.pt.close_at(date)

    def downsample(self, *_, **__):
        """Not implemented for this PT class."""
        # pylint: disable=missing-param-doc
        self._not_implemented(self.downsample)
