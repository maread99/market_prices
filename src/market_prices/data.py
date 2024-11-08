"""Class to administer price data.

Data: administrator of price data for a specific base interval.
"""

import math
from collections.abc import Callable, Sequence

import pandas as pd
from pandas import DataFrame

from market_prices import errors, helpers, intervals
from market_prices.utils import calendar_utils as calutils
from market_prices.mptypes import DateRangeReq

from .utils.pandas_utils import (
    interval_contains,
    remove_intervals_from_interval,
    timestamps_in_interval_of_intervals,
)


class Data:
    """Administrator of price data for a specific base interval.

    For a specific base interval:
        Requests data from source.
        Tracks requested data.
        Stores requested data.
        Serves price data.

    Where price data will not change (i.e. not a 'live' interval) data will
    only be requested from source for any difference between stored and
    requested data. Data for any non-live datetime is only requested from
    source once or never.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        request: Callable,
        cc: calutils.CompositeCalendar,
        bi: intervals.BI,
        delay: pd.Timedelta | None = None,
        left_limit: pd.Timestamp | pd.Timedelta | None = None,
        right_limit: pd.Timestamp | None = None,
        source_live: bool = True,
    ):
        """Construct instance.

        Parameters
        ----------
        request
            Callable that requests price data from a source.

            Callable should have signature (bi, start, end) where:

            bi: intervals.BI
                Base interval describing time delta covered by each row.

            start: pd.Timestamp | None
                Datetime from which data is being requested. None to
                request data from first datetime available.

            end: pd.Timestamp | None
                Datetime to which data is being requested. None to
                request data to mmost recent datetime available.

            Callable should return price data as a pandas DataFrame.

            Callable should raise `errors.PricesUnavailableFromSourceError`
            if prices are unavailable.

        cc
            Composite Calendar representing symbols for which prices will
            be requested.

        bi
            Base interval describing time delta covered by each row. Cannot
            be higher than one day.

        delay
            Maximum real-time price delay of any instrument being
            administered. Only required if `bi` is intraday.

        left_limit
            Earliest date for which data can be requested. None if no
            limit.

            If limit based on a period to 'now' pass period as a
            pd.Timedelta. For example if limit is the last thirty days,
            left_limit = pd.Timedelta(days=30).

            If `bi` is daily and passing limit as a pd.Timestamp then limit
            must be a date (tz-naive with time component as midnight).

        right_limit : default: 'now'
            Latest date for which data can be requested. None if no limit,
            in which case will be assumed as 'now'.

            If `bi` is daily then limit must be a date (no time component).

        source_live
            True: always re-request prices for any 'live indice' (i.e. any
            indice which is yet to conclude in real time) and and period
            through which any 'delay' applies even when such indices have
            been previously requested.

            False: never request prices for the same indice more than once.
        """
        # pylint: disable=too-many-arguments
        self._request_data_func = request
        self._source_live = source_live
        self.cc = cc
        assert bi <= intervals.ONE_DAY
        self._bi = bi
        if delay is None and bi.is_intraday:
            raise ValueError("`delay` must be passed if `bi` is intraday.")
        self._delay = delay
        self._table: DataFrame | None = None
        self._ranges: list[pd.Interval] = []
        self._ll = left_limit
        self._rl = right_limit

    @property
    def bi(self) -> intervals.BI:
        """Time delta represented by each table row."""
        return self._bi

    # Expose requested ranges and properties of

    @property
    def ranges(self) -> list[pd.Interval]:
        """Ranges over which data has been requested."""
        return self._ranges

    @property
    def ranges_index(self) -> pd.IntervalIndex:
        """Ranges over which data has been requested."""
        return pd.IntervalIndex(self.ranges)

    @property
    def leftmost_requested(self) -> pd.Timestamp | None:
        """Leftmost timestamp for which data has been requested."""
        if self.ranges:
            return self.ranges[0].left
        return None

    @property
    def rightmost_requested(self) -> pd.Timestamp | None:
        """Rightmost timestamp for which data has been requested."""
        if self.ranges:
            return self.ranges[-1].right
        return None

    # Expose limits

    @property
    def ll(self) -> pd.Timestamp | None:
        """Left limit from which data available from source.

        Only available if known or ascertained from requested data.
        """
        if isinstance(self._ll, pd.Timestamp) or self._ll is None:
            return self._ll
        elif self.bi.is_one_day:
            return helpers.now(self.bi) - self._ll
        else:
            # one minute added to provide processing margin
            return helpers.now(self.bi, side="right") + helpers.ONE_MIN - self._ll

    @property
    def rl(self) -> pd.Timestamp:
        """Right limit to which data from source available."""
        if self._rl is not None:
            return self._rl
        else:
            now = helpers.now(self.bi)
            # if intraday adds bi to provide for requesting 'live' interval
            return now if self.bi.is_one_day else now + self.bi

    def _set_ll(self, limit: pd.Timestamp):
        self._ll = limit

    def _set_rl(self, limit: pd.Timestamp):
        self._rl = limit

    # Query if data is available and/or has been requested

    @property
    def from_ll(self) -> bool:
        """Query if stored data starts at the left limit."""
        if self.ll is None or not self.ranges:
            return False
        else:
            assert self.leftmost_requested is not None
            return self.leftmost_requested <= self.ll

    @property
    def to_rl(self) -> bool:
        """Query if stored data ends at right limit."""
        if self._rl is None or not self.ranges:
            return False
        else:
            assert self.rightmost_requested is not None
            return self.rightmost_requested >= self.rl

    @property
    def has_all_data(self) -> bool:
        """Query if stored data represents all available data."""
        return len(self.ranges) == 1 and self.from_ll and self.to_rl

    @property
    def has_requested_data(self) -> bool:
        """Query if a request has been made for data."""
        return bool(self.ranges)

    @property
    def has_data(self) -> bool:
        """Query if have any stored data."""
        return self._table is not None

    def available(
        self, timestamps: pd.Timestamp | Sequence[pd.Timestamp]
    ) -> bool | None:
        """Query if data is available from source for given timestamp(s).

        Parameters
        ----------
        timestamps
            Timestamp(s) to query.

        Returns
        -------
        bool | None
            False: known that at least one timestamp is not available from
                source.
            None: unknown if one or more timestamp available from source.
            True: known that all timestamps available from source.
        """
        tss = [timestamps] if not isinstance(timestamps, Sequence) else timestamps
        availability = []
        leftmost = self.leftmost_requested
        for ts in tss:
            if self.ll is not None:
                avail = self.ll <= ts <= self.rl
            elif ts > self.rl:
                avail = False
            elif leftmost is not None and ts >= leftmost:
                avail = True
            else:
                avail = None

            if avail is False:  # pylint: disable=compare-to-zero  # avail can be None
                return False
            else:
                availability.append(avail)

        return True if all(availability) else None

    def available_range(
        self, daterange: DateRangeReq, push: bool = False
    ) -> bool | None:
        """Query if data is available from source over a date range.

        Parameters
        ----------
        daterange
            Range of dates to query.

        push : default: False
            True: if unknown if one or both extremes available from source
            then will determine by requesting data. Return will accordingly
            be True or False.

        Returns
        -------
        bool | None
            False:  known that either extreme is not available from source.
            None: unknown if one or both extremes available from source.
            True: known that both extremes available from source.
        """
        # pylint: disable=missing-param-doc
        if daterange[0] is not None:
            avail = self.available((daterange[0], daterange[1]))
        else:
            avail = self.available(daterange[1])

        if avail is None and push:
            self.get_table(daterange)
            avail = self.available_range(daterange, push=False)
            assert avail is not None

        return avail

    def available_any(self, daterange: DateRangeReq) -> bool | None:
        """Query if data is available for any timestamp within a range.

        Parameters
        ----------
        daterange
            Range over which to query.

        Returns
        -------
        bool | None
            False: known that no timestamp within range is available from
                source.
            None: unknown if a timestamp within range is available from
                source.
            True: known that data is available for at least one timestamp
                within range.

        """
        left, right = daterange
        if left > self.rl:
            return False
        if self.ll is not None:
            if left is None:
                return right >= self.ll
            else:
                return (left <= self.rl) and (right >= self.ll)

        leftmost = self.leftmost_requested
        if leftmost is None:
            return None

        if left is None and right >= leftmost:
            return True
        elif (left <= self.rl) and (right >= leftmost):
            return True
        return None

    def requested(self, timestamps: pd.Timestamp | Sequence[pd.Timestamp]) -> bool:
        """Query if uninterrupted data has been requested.

        Query if uninterrupted data has been requested over all of given
        `timestamps`.

        Note: the end of a previously requested range will be considered as
        requested, although the minute that follows will not have been
        requested. For example, if "2021-08-17 13:22" is included to
        `timestamps` and represents the end of a requested range then will
        return True (assuming all other timstamps fall in that same range)
        although the actual minute 13:22 through 13:23 will not have been
        requested.

        Parameters
        ----------
        timestamps
            Timestamp(s) to query.
        """
        if not self.has_data:
            return False
        else:
            return timestamps_in_interval_of_intervals(timestamps, self.ranges_index)

    def _require_limits(self, daterange: DateRangeReq) -> tuple[bool, bool]:
        """Query if `daterange` includes the left/right limits."""
        start = daterange[0]
        ll = self.ll
        req_ll = start is None or (ll is not None and start <= ll)
        end = daterange[1]
        req_rl = end >= self.rl
        return req_ll, req_rl

    def requested_range(self, daterange: DateRangeReq) -> bool:
        """Query if requested all available data over a date range.

        Parameters
        ----------
        daterange
            Daterange to query.
        """
        req_ll, req_rl = self._require_limits(daterange)
        if req_ll and req_rl:
            return self.has_all_data
        elif req_ll:
            return self.from_ll and daterange[1] in self.ranges[0]
        elif req_rl:
            return self.to_rl and daterange[0] in self.ranges[-1]
        else:
            assert daterange[0] is not None
            return timestamps_in_interval_of_intervals(
                (daterange[0], daterange[1]), self.ranges_index
            )

    def available_and_requested(
        self, timestamps: pd.Timestamp | Sequence[pd.Timestamp]
    ) -> bool:
        """Query if uninterrupted data available and requested.

        Queries if uninterrupted data available and requested for all given
        `timestamps`.

        Note: the end of a previously requested range will be considered as
        requested, although the minute that follows will not have been
        requested. For example, if "2021-08-17 13:22" is included to
        `timestamps` and represents the end of a requested range then will
        return True (assuming all other timstamps fall in that same range)
        although the actual minute 13:22 through 13:23 will not have been
        requested.

        Parameters
        ----------
        timestamps
            Timestamp(s) to query.
        """
        available = self.available(timestamps)
        if not available:
            return False
        else:
            return self.requested(timestamps)

    # Get date range to request

    def _consolidate_overlapping(
        self, dateranges: list[DateRangeReq]
    ) -> list[DateRangeReq]:
        """Consolidate overlapping dateranges."""
        if len(dateranges) == 1:
            return dateranges
        for i, dr in enumerate(dateranges):
            if dr == dateranges[-1]:
                break
            if dateranges[i][1] >= dateranges[i + 1][0]:  # type: ignore[operator]
                # ignore mypy error unsupported operand type >= Timestamp and None,
                # only first daterange of `dateranges` can have None at index 0, and
                # this will not be compared.
                dateranges[i] = (dateranges[i][0], dateranges[i + 1][1])
                del dateranges[i + 1]
                return self._consolidate_overlapping(dateranges)
        return dateranges

    def _request_dates(self, daterange: DateRangeReq) -> list[DateRangeReq]:
        """Return date ranges over which to request data.

        Returns date range over which to request data in order to have
        requested all available data over `daterange`.
        """
        # pylint: disable=too-many-branches,too-complex
        if not self.ranges:
            start = daterange[0]
            if start is None or (self.ll is not None and start < self.ll):
                start = self.ll
            end = daterange[1]
            if end > self.rl:
                end = self.rl
            return [(start, end)]

        req_ll, req_rl = self._require_limits(daterange)

        rr = [daterange[0], daterange[1]]  # request_range
        leftmost, rightmost = self.leftmost_requested, self.rightmost_requested
        assert leftmost is not None and rightmost is not None

        # If extremes lie in a already requested interval, exclude
        # dates of that interval already requested. 'Tighten the range'.
        for interval in self.ranges:
            if not req_ll and rr[0] in interval:
                rr[0] = interval.right
            if not req_rl and rr[1] in interval:
                rr[1] = interval.left

        if req_ll and self.ll is not None:
            rr[0] = self.ll
            if leftmost is not None:
                rr[0] = min([self.ll, leftmost])
        if req_rl:
            rr[1] = self.rl
            if rightmost is not None:
                rr[1] = max([self.rl, rightmost])

        # remove any dates in ranges that have already been requested
        start = rr[0] if rr[0] is not None else leftmost
        assert rr[1] is not None
        end = rr[1]
        if not (start >= rightmost or end <= leftmost):
            interval = pd.Interval(start, end, closed="both")
            if interval_contains(interval, self.ranges_index).any():
                ranges = remove_intervals_from_interval(interval, self.ranges_index)
                rds = [(i.left, i.right) for i in ranges]  # request dates
            else:
                rds = [tuple(rr)]  # type: ignore[list-item]
        else:
            rds = [tuple(rr)]  # type: ignore[list-item]

        # if limit needed and not covered, add daterange to cover
        if rr[0] is None and (not rds or rds[0][0] is not None):
            rds = [(None, self.ranges[0].left)] + rds

        return self._consolidate_overlapping(rds)

    # Update knowledge of ranges requested.

    def _update(self, daterange: DateRangeReq, dates: pd.IntervalIndex):
        """Update object to reflect having requested data over `daterange`.

        Parameters
        ----------
        daterange
            Range of dates for which data requested from source.

        dates
            Dates of data returned by source.
        """
        # pylint: disable=too-many-branches,too-complex

        # Do not register daterange if no data was returned for a daterange
        # that required a limit (for example if difference between left
        # limit and end of daterange covered only non-trading times) and
        # the daterange does not overlap with any previously requested
        # interval nearest to the limit. This avoids object seeing that limit
        # has been requested and thereby assuming that data is available up
        # to the limit.
        if dates.empty:
            rqr_ll, rqr_rl = self._require_limits(daterange)
            if not rqr_ll and not rqr_rl:
                pass
            elif not self.ranges:
                return
            assert self.leftmost_requested is not None
            if rqr_ll and daterange[1] < self.leftmost_requested:
                return
            assert self.rightmost_requested is not None
            if (
                daterange[0] is not None
                and rqr_rl
                and daterange[0] > self.rightmost_requested
            ):
                return

        # Update limits
        if self.ll is None and not dates.empty:
            first_dt = dates[0].left if not self.bi.is_one_day else dates[0]
            if daterange[0] is None or daterange[0] < first_dt:
                self._set_ll(first_dt)

        # set start and end
        end = daterange[1]

        if daterange[0] is not None:
            start = daterange[0]
        elif not dates.empty:
            start = dates[0].left
        else:
            # start None but got no data back, i.e. end before unknown limit
            start = end

        # Update ranges
        dates_interval: pd.Interval | None = pd.Interval(start, end, closed="both")
        new_ranges = []
        if not self.ranges:
            assert isinstance(dates_interval, pd.Interval)
            new_ranges.append(dates_interval)
        else:
            for rng in self.ranges:
                if dates_interval is None:
                    new_ranges.append(rng)
                    continue
                if rng.overlaps(dates_interval) or (
                    self.bi.is_one_day
                    and (
                        rng.left - helpers.ONE_DAY == dates_interval.right
                        or rng.right + helpers.ONE_DAY == dates_interval.left
                    )
                ):
                    left = min(rng.left, dates_interval.left)
                    right = max(rng.right, dates_interval.right)
                    dates_interval = pd.Interval(left, right, closed="both")
                elif dates_interval.right < rng.left:
                    new_ranges.append(dates_interval)
                    new_ranges.append(rng)
                    dates_interval = None
                else:
                    new_ranges.append(rng)
            if dates_interval is not None:
                new_ranges += [dates_interval]
        self._ranges = new_ranges

    # Request data from source

    def _request_data(
        self, daterange: DateRangeReq
    ) -> tuple[list[DateRangeReq], list[DataFrame]]:
        """Request data to cover daterange."""
        req_dates = self._request_dates(daterange)
        dfs = []
        for rd in req_dates:
            try:
                df = self._request_data_func(self.bi, rd[0], rd[1])
            except errors.PricesUnavailableFromSourceError:
                intrvl = pd.Interval(*rd, closed="left")
                start_session = self.cc.minute_to_sessions(intrvl.left, "previous")[-1]
                end_session = self.cc.minute_to_sessions(intrvl.right, "next")[0]
                if intrvl not in self.cc.non_trading_index(start_session, end_session):
                    raise
                df = DataFrame()
            dfs.append(df)
        return req_dates, dfs

    # Add new data to table.

    def _drop_stored_labels(self, dfs: list[DataFrame]):
        """Drop from existing table any label in `dfs`."""
        assert self._table is not None
        index = dfs[0].index
        for df in dfs[1:]:
            index = index.union(df.index)
        s = self._table.index.to_series()
        bv = s.apply(lambda x: x in index)
        if bv.any():
            self._table.drop(self._table.index[bv], inplace=True)

    @staticmethod
    def _drop_duplicate_labels(dfs: list[DataFrame]) -> list[DataFrame]:
        """Drop duplicate labels in each df of `dfs`."""
        new_dfs = []
        for df in dfs:
            bv = df.index.duplicated()
            if bv.any():
                df.drop(df.index[bv], inplace=True)
            new_dfs.append(df)
        return new_dfs

    def _concat_tables(self, dfs: list[DataFrame]) -> DataFrame:
        assert self._table is not None
        table = pd.concat([self._table] + dfs)
        table.sort_index(inplace=True)
        return table

    def _index_is_clean(self, index: pd.Index) -> bool:
        """Query if index is sorted, does not overlap and has no duplicates."""
        if self.bi.is_one_day:
            return index.is_monotonic_increasing & index.is_unique
        else:
            return index.is_non_overlapping_monotonic

    def _add_prices(self, dfs: list[DataFrame]):
        """Add price data to table."""
        if self._table is None:
            assert len(dfs) == 1
            table = dfs[0]
        else:
            table = self._concat_tables(dfs)
            if not self._index_is_clean(table.index):
                self._drop_stored_labels(dfs)
                table = self._concat_tables(dfs)
            if not self._index_is_clean(table.index):
                dfs = self._drop_duplicate_labels(dfs)
                table = self._concat_tables(dfs)
        assert self._index_is_clean(table.index)
        self._table = table

    def _requested_dates_adjust(self, dr: DateRangeReq) -> DateRangeReq | None:
        """Adjust request daterange for any delay or live interval.

        Adjusts request daterange to exclude:
            - datetimes that would not be available for every symbol as a
                result of real-time price delay.
            - any live interval.
        """
        end = dr[1]
        if self.bi.is_one_day and end == helpers.now(self.bi):
            if self._delay is None:
                margin = 1
            else:
                delay_in_days = self._delay.total_seconds() / (24 * 60 * 60)
                margin = max(1, math.ceil(delay_in_days))
            end -= helpers.ONE_DAY * margin
        elif self.bi.is_intraday:
            assert self._delay is not None
            # ten minutes to cover provider delays in publishing data.
            rerequest_from = helpers.now() - self._delay - pd.Timedelta(10, "min")
            if end < rerequest_from:
                return dr
            excess = end - rerequest_from
            n = math.ceil(excess / self.bi)
            end -= n * self.bi
        if dr[0] is not None and end < dr[0]:
            return None
        else:
            return (dr[0], end)

    def _extend_table(self, daterange: DateRangeReq):
        """Extend table to cover `daterange`."""
        req_dates, dfs = self._request_data(daterange)
        if dfs and not all(df.empty for df in dfs):
            self._add_prices(dfs)
        for rd, df in zip(req_dates, dfs):
            if self._source_live:
                rd = self._requested_dates_adjust(rd)
            if rd is None:
                continue
            self._update(rd, df.index)

    # Get table or part of

    def _get_table_part(self, daterange: DateRangeReq) -> DataFrame | None:
        if self._table is None:
            return None

        start, end = daterange
        if start is None:
            start = self._table.index[0]
            if self.bi.is_intraday:
                start = start.left

        if self.bi.is_one_day:
            index = self._table.index
            bv = (index >= start) & (index <= end)
        else:
            left, right = self._table.index.left, self._table.index.right
            bv = (left >= start) & (right <= end)
        return self._table[bv]

    def get_table(self, daterange: DateRangeReq | None = None) -> DataFrame | None:
        """Return price table covering at least a minimum range of dates.

        Parameters
        ----------
        daterange
            Date range that table should cover.

            If None will return price table 'as is', including any gaps
            over periods for which data has not been requested.

        Returns
        -------
        pd.DataFrame | None
            Requested table as pd.DataFrame.
            Returns None if price data not available over `daterange`.
        """
        if daterange is None:
            return self._table
        if not self.requested_range(daterange):
            self._extend_table(daterange)
        return self._get_table_part(daterange)

    def __repr__(self) -> str:
        return str(self.ranges)
