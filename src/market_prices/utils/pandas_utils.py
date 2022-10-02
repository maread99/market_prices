"""Utility functions and classes for pandas library."""

from __future__ import annotations

import warnings
from collections import abc
from contextlib import contextmanager
from typing import Any, List, Literal, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pydantic
from pytz.tzinfo import BaseTzInfo

from market_prices import mptypes


def pdfreq_to_offset(pdfreq: str) -> pd.offsets.BaseOffset:  # type: ignore[name-defined]
    """Pandas frequency string to a pandas offset.

    Parameters
    ----------
    pdfreq
        pandas frequency string to convert. For example, '15min',
            '15T', '3H'.

    Examples
    --------
    >>> pdfreq_to_offset('22T')
    <22 * Minutes>
    >>> pdfreq_to_offset('3H')
    <3 * Hours>
    >>> pdfreq_to_offset('3MS')
    <3 * MonthBegins>
    """
    return pd.tseries.frequencies.to_offset(pdfreq)


def is_midnight(timestamp: pd.Timestamp) -> bool:
    """Query if a timestamp has all time components as zero.

    Parameters
    ----------
    timestamp
        Timestamp to query.

    Examples
    --------
    >>> is_midnight(pd.Timestamp('2021-06-01 15:00'))
    False
    >>> is_midnight(pd.Timestamp('2021-06-01 00:00'))
    True
    >>> is_midnight(pd.Timestamp('2021-06-01'))
    True
    """
    return timestamp == timestamp.normalize()


def tuple_to_interval(
    tup: tuple, closed: Literal["right", "left", "both", "neither"] = "both"
) -> pd.Interval:
    """Convert 2-tuple to pandas interval.

    Parameters
    ----------
    tup
        2-tuple where first item represents left side of interval and
            second item represents right side.

    closed
        As `closed` parameter of `pd.Interval`.

    Examples
    --------
    >>> tuple_to_interval(
    ...     (pd.Timestamp('2021-04-22 12:00'), pd.Timestamp('2021-04-26 08:30')),
    ...     closed="left",
    ... )
    Interval('2021-04-22 12:00:00', '2021-04-26 08:30:00', closed='left')
    """
    return pd.Interval(tup[0], tup[1], closed=closed)


def timestamps_in_interval(
    timestamps: pd.DatetimeIndex, interval: pd.Interval
) -> pd.Series:
    """Query if multiple timestamps are in an interval.

    Parameters
    ----------
    timestamps
       Timestamps to query if in `interval`.

    interval
        Interval to query if timestamps lie inside of.

    Returns
    -------
    pd.Series
        Series indexed with `timestamps` and values as bool indicating if
        indice lies in `interval`.

    Examples
    --------
    >>> interval = pd.Interval(
    ...     pd.Timestamp('2021-03-15 12:00'),
    ...     pd.Timestamp('2021-03-20 18:00'),
    ...     closed="left"
    ... )
    >>> timestamps = pd.DatetimeIndex(
    ...     [
    ...         pd.Timestamp("2021-03-13 23:00"),
    ...         pd.Timestamp("2021-03-15 12:00"),
    ...         pd.Timestamp("2021-03-17 13:00"),
    ...         pd.Timestamp("2021-03-20 17:59"),
    ...         pd.Timestamp("2021-03-20 18:00"),
    ...         pd.Timestamp("2021-03-22 09:00"),
    ...     ]
    ... )
    >>> timestamps_in_interval(timestamps, interval)
    2021-03-13 23:00:00    False
    2021-03-15 12:00:00     True
    2021-03-17 13:00:00     True
    2021-03-20 17:59:00     True
    2021-03-20 18:00:00    False
    2021-03-22 09:00:00    False
    dtype: bool
    """
    ser = timestamps.to_series()
    ser = ser.apply(lambda x: x in interval)
    assert isinstance(ser, pd.Series)
    return ser


def timestamps_in_interval_of_intervals(
    timestamps: pd.Timestamp | abc.Sequence[pd.Timestamp] | pd.DatetimeIndex,
    intervals: pd.IntervalIndex,
) -> bool:
    # pylint: disable=line-too-long
    """Query if given timestamps are all in an interval of an interval index.

    Parameters
    ----------
    timestamps
        Timestamps to query if in an interval of `intervals`.

    intervals
        Intervals to query if all `timestamps` lie in one interval of.

    Returns
    -------
    bool
        True if all `timestamps` are in at least one interval of the interval
        index.

    Examples
    --------
    >>> timestamps = pd.DatetimeIndex(
    ...     [
    ...         pd.Timestamp('2021-03-12 14:00'),
    ...         pd.Timestamp('2021-03-13 23:00'),
    ...         pd.Timestamp('2021-03-15 12:00'),
    ...         pd.Timestamp('2021-03-16 13:00'),
    ...     ]
    ... )
    >>> start, end = pd.Timestamp('2021-03-02'), pd.Timestamp('2021-03-31')
    >>> closed = "both"
    >>> intervals = pd.interval_range(start, end, freq="3D", closed=closed)
    >>> timestamps_in_interval_of_intervals(timestamps, intervals)
    False

    >>> intervals = pd.interval_range(start, end, freq="10D", closed=closed)
    >>> intervals
    IntervalIndex([[2021-03-02, 2021-03-12], [2021-03-12, 2021-03-22]], dtype='interval[datetime64[ns], both]')
    >>> timestamps_in_interval_of_intervals(timestamps, intervals)
    True
    """
    timestamps = [timestamps] if isinstance(timestamps, pd.Timestamp) else timestamps
    ser = intervals.to_series()
    bv = ser.apply(lambda x: all({ts in x for ts in timestamps}))
    return any(bv)


def make_non_overlapping(
    index: pd.IntervalIndex,
    fully_overlapped: Literal["remove", "keep", None] = "keep",
) -> pd.IntervalIndex:
    """Make pd.IntervalIndex ascending non_overlapping.

    Where, post-sorting, an indice overlaps following indice, right side is
    curatiled to left side of following indice.

    Parameters
    ----------
    index
        pd.IntervalIndex to make non-overlapping. Cannot be closed on
        "both" sides.

    fully_overlapped : default: "keep"
        How to treat fully overlapped indices:
            "remove" - remove any indice that is fully overlapped by an
            earlier indice.

            "keep" - keep any indice that is fully overlapped by an earlier
            indice (right of earlier indice will be curtailed to left
            of overlapped indice).

            None - raise ValueError if any indice is fully overlapped by an
            earlier indice.

    Raises
    ------
    ValueError
        If `fully_overlapped` is None and any indice is fully overlapped by
        another.

        If `index` is overlapping and closed on "both" sides.

    Examples
    --------
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')

    >>> intervals = [
    ...     pd.Interval(left[0], left[0] + pd.Timedelta(45, 'min')),
    ...     pd.Interval(left[1], left[1] + pd.Timedelta(90, 'min')),
    ...     pd.Interval(left[2], left[2] + pd.Timedelta(45, 'min')),
    ... ]
    >>> interval_index = pd.IntervalIndex(intervals)
    >>> interval_index.to_series(range(3))
    0    (2021-05-01 12:00:00, 2021-05-01 12:45:00]
    1    (2021-05-01 13:00:00, 2021-05-01 14:30:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:45:00]
    dtype: interval
    >>> index = make_non_overlapping(interval_index)
    >>> index.to_series(range(3))
    0    (2021-05-01 12:00:00, 2021-05-01 12:45:00]
    1    (2021-05-01 13:00:00, 2021-05-01 14:00:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:45:00]
    dtype: interval

    Fully overlapped indices:
    >>> intervals = [
    ...     pd.Interval(left[0], left[0] + pd.Timedelta(45, 'min')),
    ...     pd.Interval(left[1], left[1] + pd.Timedelta(190, 'min')),
    ...     pd.Interval(left[2], left[2] + pd.Timedelta(45, 'min')),
    ...     pd.Interval(left[3], left[3] + pd.Timedelta(60, 'min')),
    ...     pd.Interval(left[4], left[4] + pd.Timedelta(60, 'min')),
    ... ]
    >>> interval_index = pd.IntervalIndex(intervals)
    >>> interval_index.to_series(range(5))
    0    (2021-05-01 12:00:00, 2021-05-01 12:45:00]
    1    (2021-05-01 13:00:00, 2021-05-01 16:10:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:45:00]
    3    (2021-05-01 15:00:00, 2021-05-01 16:00:00]
    4    (2021-05-01 16:00:00, 2021-05-01 17:00:00]
    dtype: interval
    >>> index = make_non_overlapping(
    ...     interval_index, fully_overlapped="remove"
    ... )
    >>> index.to_series(range(len(index)))
    0    (2021-05-01 12:00:00, 2021-05-01 12:45:00]
    1    (2021-05-01 13:00:00, 2021-05-01 16:00:00]
    2    (2021-05-01 16:00:00, 2021-05-01 17:00:00]
    dtype: interval
    >>> index = make_non_overlapping(
    ...     interval_index, fully_overlapped="keep"
    ... )
    >>> index.to_series(range(len(index)))
    0    (2021-05-01 12:00:00, 2021-05-01 12:45:00]
    1    (2021-05-01 13:00:00, 2021-05-01 14:00:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:45:00]
    3    (2021-05-01 15:00:00, 2021-05-01 16:00:00]
    4    (2021-05-01 16:00:00, 2021-05-01 17:00:00]
    dtype: interval
    """
    # pylint: disable=missing-param-doc
    index = index.sort_values()
    if not index.is_overlapping:
        return index
    if index.closed == "both":
        raise ValueError(
            "`index` to be made non-overlapping cannot be closed on 'both' sides."
        )

    try:
        tz = index.dtype.subtype.tz
    except AttributeError:
        tz = None
    else:
        index = interval_index_new_tz(index, None)  # type: ignore[arg-type]  # mptype

    # evaluate full_overlap_mask
    # as 'int64' to use expanding, as series to use expanding and shift
    right = pd.Index(index.right.view("int64")).to_series()
    next_right = right.shift(-1)
    max_right_to_date = right.expanding().max()
    # shift 1 to move from overlapping indice to overlapped indice
    # fill_value False as first indice cannot be overlapped
    full_overlap_mask = (next_right <= max_right_to_date).shift(1, fill_value=False)

    if full_overlap_mask.any():
        if fully_overlapped == "remove":
            index = index[~full_overlap_mask]
        elif fully_overlapped is None:
            msg = (
                "If `fully_overlapped` is None then index cannot contain an"
                " indice which fully overlaps another. The following indices"
                " are fully overlapped by an earlier indice:\n"
                f"{index[full_overlap_mask]}\nConsider passing"
                " `fully_overlapped` as 'keep' or 'remove'."
            )
            raise ValueError(msg)

    next_left = index.left.to_series().shift(-1)
    columns = dict(right=index.right, next_left=next_left)
    new_right = pd.DataFrame(columns, index=index.left).min(axis=1)
    ii = pd.IntervalIndex.from_arrays(index.left, new_right, closed=index.closed)
    if tz is not None:
        ii = interval_index_new_tz(ii, tz)
    return ii


def get_interval_index(
    left: pd.DatetimeIndex,
    offset: str | pd.offsets.BaseOffset,  # type: ignore[name-defined]
    closed="left",
    non_overlapping=False,
) -> pd.IntervalIndex:
    """Interval index with intervals defined as offset from an index.

    Parameters
    ----------
    left
        pd.DatetimeIndex describing representing left side of each
        interval.

    offset
        Duration of each interval, for example "15min", "15T", "3H" etc.

    closed : default: "left"
        Side on which to close intervals.

    non_overlapping : default: False
        If True, curtail any interval that would otherwise overlap with
        the following interval.

        Cannot pass as True if `closed` passed as "both" and would
        otherwise overlap or if any indice fully overlaps the subsequent
        indice. In either case will raise ValueError.

    Examples
    --------
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')
    >>> index = get_interval_index(left, '33T')
    >>> index.to_series(index=range(5))
    0    [2021-05-01 12:00:00, 2021-05-01 12:33:00)
    1    [2021-05-01 13:00:00, 2021-05-01 13:33:00)
    2    [2021-05-01 14:00:00, 2021-05-01 14:33:00)
    3    [2021-05-01 15:00:00, 2021-05-01 15:33:00)
    4    [2021-05-01 16:00:00, 2021-05-01 16:33:00)
    dtype: interval
    """
    # pylint: disable=missing-param-doc
    if isinstance(offset, str):
        offset = pdfreq_to_offset(offset)
    right = left.shift(freq=offset)
    right.freq = left.freq
    index = pd.IntervalIndex.from_arrays(left, right, closed=closed)
    if non_overlapping and index.is_overlapping:
        return make_non_overlapping(index)
    else:
        return index


def interval_of_intervals(
    intervals: pd.IntervalIndex,
    closed: Literal["left", "right", "both", "neither"] = "right",
) -> pd.Interval:
    """Return interval covered by a monotonic IntervalIndex.

    Parameters
    ----------
    intervals
        Monotonic pd.IntervalIndex for which require the encompassing
        interval.

    closed
        Side on which to close returned pd.Interval.

    Raises
    ------
    ValueError
        If `intervals` is not monotonic.

    Examples
    --------
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')
    >>> right = left + pd.Timedelta(30, 'T')
    >>> index = pd.IntervalIndex.from_arrays(left, right)
    >>> index.to_series(index=range(5))
    0    (2021-05-01 12:00:00, 2021-05-01 12:30:00]
    1    (2021-05-01 13:00:00, 2021-05-01 13:30:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:30:00]
    3    (2021-05-01 15:00:00, 2021-05-01 15:30:00]
    4    (2021-05-01 16:00:00, 2021-05-01 16:30:00]
    dtype: interval
    >>> interval_of_intervals(index)
    Interval('2021-05-01 12:00:00', '2021-05-01 16:30:00', closed='right')
    """
    if not intervals.is_monotonic_increasing:
        raise ValueError(f"`intervals` must be monotonic. Received as '{intervals}'.")
    return pd.Interval(intervals[0].left, intervals[-1].right, closed=closed)


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def interval_contains(interval: pd.Interval, intervals: pd.IntervalIndex) -> np.ndarray:
    """Query which intervals are contained within an interval.

    Parameters
    ----------
    interval
        Interval within which to query if `intervals` are contained.

    intervals
        Intervals to query if contained in `interval`.

    Returns
    -------
    np.ndarray
        Boolean vector. Values indicate if corresponding indice of
        `intervals` is contained in `interval`.

    Examples
    --------
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')
    >>> right = left + pd.Timedelta(30, 'T')
    >>> intervals = pd.IntervalIndex.from_arrays(left, right)
    >>> intervals.to_series(index=range(5))
    0    (2021-05-01 12:00:00, 2021-05-01 12:30:00]
    1    (2021-05-01 13:00:00, 2021-05-01 13:30:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:30:00]
    3    (2021-05-01 15:00:00, 2021-05-01 15:30:00]
    4    (2021-05-01 16:00:00, 2021-05-01 16:30:00]
    dtype: interval

    >>> interval = pd.Interval(
    ...     left = pd.Timestamp('2021-05-01 12:00'),
    ...     right = pd.Timestamp('2021-05-01 14:30'),
    ...     closed = "both"
    ... )
    >>> interval_contains(interval, intervals)
    array([ True,  True,  True, False, False])

    >>> interval = pd.Interval(interval.left, interval.right, closed="left")
    >>> interval_contains(interval, intervals)
    array([ True,  True, False, False, False])

    >>> interval = pd.Interval(interval.left, interval.right, closed="right")
    >>> interval_contains(interval, intervals)
    array([False,  True,  True, False, False])

    >>> interval = pd.Interval(interval.left, interval.right, closed="neither")
    >>> interval_contains(interval, intervals)
    array([False,  True, False, False, False])
    """
    if interval.closed == "left":
        left_cond = intervals.left >= interval.left
        right_cond = intervals.right < interval.right
    elif interval.closed == "right":
        left_cond = intervals.left > interval.left
        right_cond = intervals.right <= interval.right
    elif interval.closed == "both":
        left_cond = intervals.left >= interval.left
        right_cond = intervals.right <= interval.right
    else:
        left_cond = intervals.left > interval.left
        right_cond = intervals.right < interval.right
    return left_cond & right_cond


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def remove_intervals_from_interval(
    interval: pd.Interval, intervals: pd.IntervalIndex
) -> List[pd.Interval]:
    """Difference between an interval and some intervals.

    Parameters
    ----------
    interval
        Interval from which `intervals` to be subtracted

    intervals
        Intervals to be subtracted from `interval`. Must be monotonically
        increasing and not overlapping.

    Returns
    -------
    List[pd.Interval]
        List of intervals that remain after subtracting 'intervals' from
        'interval'.

    Raises
    ------
    ValueError
        If `intervals` not monotonically increasing or are overlapping.

    Examples
    --------
    >>> from pprint import pprint
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')
    >>> right = left + pd.Timedelta(30, 'T')
    >>> intervals = pd.IntervalIndex.from_arrays(left, right)
    >>> intervals.to_series(index=range(5))
    0    (2021-05-01 12:00:00, 2021-05-01 12:30:00]
    1    (2021-05-01 13:00:00, 2021-05-01 13:30:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:30:00]
    3    (2021-05-01 15:00:00, 2021-05-01 15:30:00]
    4    (2021-05-01 16:00:00, 2021-05-01 16:30:00]
    dtype: interval
    >>> interval = pd.Interval(
    ...     left = pd.Timestamp('2021-05-01 12:00'),
    ...     right = pd.Timestamp('2021-05-01 17:30'),
    ...     closed = "left"
    ... )
    >>> rtrn = remove_intervals_from_interval(interval, intervals)
    >>> pprint(rtrn)
    [Interval('2021-05-01 12:30:00', '2021-05-01 13:00:00', closed='left'),
     Interval('2021-05-01 13:30:00', '2021-05-01 14:00:00', closed='left'),
     Interval('2021-05-01 14:30:00', '2021-05-01 15:00:00', closed='left'),
     Interval('2021-05-01 15:30:00', '2021-05-01 16:00:00', closed='left'),
     Interval('2021-05-01 16:30:00', '2021-05-01 17:30:00', closed='left')]
    """
    if not intervals.is_monotonic_increasing:
        raise ValueError(
            "`intervals` must be monotonically increasing although receieved"
            f" '{intervals}'"
        )
    if intervals.is_overlapping:
        raise ValueError(
            f"`intervals` must not be overlapping although receieved '{intervals}'."
        )

    if len(intervals) == 1 and (
        interval.left == intervals[0].left and interval.right == intervals[0].right
    ):
        return []

    intervals_to_remove = intervals[intervals.overlaps(interval)].tolist()
    if not intervals_to_remove:
        return [interval]

    closed = interval.closed
    if intervals_to_remove[0].left <= interval.left:
        interval = pd.Interval(
            intervals_to_remove[0].right, interval.right, closed=closed
        )
        del intervals_to_remove[0]
        if not intervals_to_remove:
            return [interval]

    if intervals_to_remove[-1].right >= interval.right:
        interval = pd.Interval(
            interval.left, intervals_to_remove[-1].left, closed=closed
        )
        del intervals_to_remove[-1]
        if not intervals_to_remove:
            return [interval]

    diff = []
    for itr in intervals_to_remove:
        diff.append(pd.Interval(interval.left, itr.left, closed=closed))
        interval = pd.Interval(itr.right, interval.right, closed=closed)
    diff.append(interval)
    return diff


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def interval_index_new_tz(
    index: mptypes.IntervalDatetimeIndex, tz: Union[BaseTzInfo, str, None]
) -> pd.IntervalIndex:
    """Return pd.IntervalIndex with different timezone.

    Note: `index` is not changed in place.

    Parameters
    ----------
    index
        pd.IntervalIndex on which to base return. Must have left and right
        sides as pd.DatetimeIndex. If these pd.DatetimeIndex are non-naive
        return will convert indices to `tz`, if naive then indices will
        be localised to `tz`.

    tz
        Timezone for returned pd.IntervalIndex. Examples: "US/Eastern",
        "Europe/Paris", "UTC".

        Pass as None to return as timezone naive.

    Returns
    -------
    pd.IntervalIndex
        pd.IntervalIndex as `index` albeit with timezone as `tz`.

    Examples
    --------
    >>> left = pd.date_range(
    ...     '2021-05-01 12:00', periods=5, freq='1H', tz='US/Central'
    ... )
    >>> right = left + pd.Timedelta(30, 'T')
    >>> index = pd.IntervalIndex.from_arrays(left, right)
    >>> index.right.tz
    <DstTzInfo 'US/Central' LMT-1 day, 18:09:00 STD>
    >>> new_index = interval_index_new_tz(index, tz="UTC")
    >>> new_index.left.tz.zone == new_index.right.tz.zone == "UTC"
    True
    """
    indices = []
    for indx in [index.left, index.right]:
        try:
            indices.append(indx.tz_convert(tz))
        except TypeError:
            indices.append(indx.tz_localize(tz))
    return pd.IntervalIndex.from_arrays(indices[0], indices[1], closed=index.closed)


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def index_is_normalized(
    index: Union[pd.DatetimeIndex, mptypes.IntervalDatetimeIndex]
) -> bool:
    """Query if an index is normalized.

    Parameters
    ----------
    index
        Index to query

    Examples
    --------
    >>> index = pd.date_range('2021-05-01 12:00', periods=5, freq='1H')
    >>> index_is_normalized(index)
    False
    >>> index = pd.date_range('2021-05-01', periods=5, freq='1D')
    >>> index_is_normalized(index)
    True

    >>> index = pd.interval_range(
    ...     pd.Timestamp('2021-05-01 12:00'), periods=5, freq='1H'
    ... )
    >>> index_is_normalized(index)
    False

    >>> index = pd.interval_range(
    ...     pd.Timestamp('2021-05-01'), periods=5, freq='1D'
    ... )
    >>> index_is_normalized(index)
    True
    """
    if isinstance(index, pd.IntervalIndex):
        for indx in [index.left, index.right]:
            if not index_is_normalized(indx):
                return False
        return True
    else:
        return (index == index.normalize()).all()


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def indexes_union(indexes: List[pd.Index]) -> pd.Index:
    """Union multiple pd.Index objects.

    Parameters
    ----------
    indexes
        pd.Index objects to be joined. Note: All indexes must be of same
        dtype.

    Examples
    --------
    >>> index1 = pd.date_range('2021-05-01 12:20', periods=2, freq='1H')
    >>> index2 = pd.date_range('2021-05-02 17:10', periods=2, freq='22T')
    >>> index3 = pd.date_range('2021-05-03', periods=2, freq='1D')
    >>> indexes_union([index1, index2, index3])
    DatetimeIndex(['2021-05-01 12:20:00', '2021-05-01 13:20:00',
                   '2021-05-02 17:10:00', '2021-05-02 17:32:00',
                   '2021-05-03 00:00:00', '2021-05-04 00:00:00'],
                  dtype='datetime64[ns]', freq=None)
    """
    index = indexes[0]
    for indx in indexes[1:]:
        index = index.union(indx)
    return index


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def index_union(indexes: List[Union[pd.Index, Series, DataFrame]]) -> pd.Index:
    """Union indexes of multiple indexes, Series and/or DataFrame.

    Parameters
    ----------
    indexes
        List of pd.Index and/or pd.Series and/or pd.DataFrames with indexes
        to be joined. All indexes must be of same dtype.

    Examples
    --------
    >>> index1 = pd.date_range('2021-05-01 12:20', periods=2, freq='1H')
    >>> index2 = pd.date_range('2021-05-02 17:10', periods=2, freq='22T')
    >>> index3 = pd.date_range('2021-05-03', periods=2, freq='1D')
    >>> ser = pd.Series(range(2), index=index2)
    >>> df = pd.DataFrame({'col_int': range(2)}, index=index3)
    >>> index_union([index1, ser, df])
    DatetimeIndex(['2021-05-01 12:20:00', '2021-05-01 13:20:00',
                   '2021-05-02 17:10:00', '2021-05-02 17:32:00',
                   '2021-05-03 00:00:00', '2021-05-04 00:00:00'],
                  dtype='datetime64[ns]', freq=None)
    """
    indexes_ = [obj if isinstance(obj, pd.Index) else obj.index for obj in indexes]
    return indexes_union(indexes_)


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def rebase_to_row(
    data: Union[DataFrame, Series], row: int = 0, value: int = 100
) -> pd.DataFrame:
    """Rebase a pandas object to a given row.

    If `data` is a pd.DataFrame, each column will be rebased independently.

    Parameters
    ----------
    data
        pandas object to rebase.

    row
        index of row against which to rebase data.

    value
        Base value for each cell of `row`. All other rows will be rebased
        relative to this base value.

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> rebase_to_row(df, 2, 100)
       open_      close_
    0    0.0   71.428571
    1   50.0   85.714286
    2  100.0  100.000000
    3  150.0  114.285714
    4  200.0  128.571429
    """
    return data / data.iloc[row] * value


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def rebase_to_cell(
    data: Union[DataFrame, Series],
    row: Any,
    col: str,
    value: int = 100,
) -> pd.DataFrame:
    """Rebase a pandas object to a given base value for a given cell.

    Parameters
    ----------
    data
        pandas object to rebase.

    row
        Label of row containing cell to have base value.

    col
        Label of column containing cell to have base value.

    value
        Base value. Cell at `row`, `column` will have this value with all
        other cells rebased relative to this base value.

    Example
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> rebase_to_cell(df, 3, 'close_', 100)
       open_  close_
    0    0.0    62.5
    1   12.5    75.0
    2   25.0    87.5
    3   37.5   100.0
    4   50.0   112.5
    """
    return data / data.loc[row, col] * value


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def tolists(df: pd.DataFrame) -> List[List[Any]]:
    """Convert pd.DataFrame to list of lists.

    Each each inner list represents one column of the DataFrame.

    Parameters
    ----------
    df
        DataFrame to convert to list of lists.

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> tolists(df)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    return [col.values.tolist() for name, col in df.items()]


def missing_values_pct(df: DataFrame, axis: int = 1) -> tuple[float, float]:
    """Query percentage of a dataFrame's rows/columns with missing values.

    Parameters
    ----------
    df
        Dataframe to query.

    axis : default: 1
        Along which axis to query. 1 to query by row, 0 to query by column.

    Return
    ------
    tuple of (float, float)
        [0]: Percentage of rows/columns with at least one value missing.
        [1]: Percentage of rows/columns with all values missing.

    See Also
    --------
    missing_values_summary

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df.loc[1:2, 'open_'] = np.NAN
    >>> df.loc[2:3,'close_'] = np.NAN
    >>> df
       open_  close_
    0    0.0     5.0
    1    NaN     6.0
    2    NaN     NaN
    3    3.0     NaN
    4    4.0     9.0

    >>> missing_values_pct(df)
    (0.6, 0.2)

    >>> missing_values_pct(df, axis=0)
    (1.0, 0.0)
    """
    # pylint: disable=missing-param-doc
    num = df.shape[int(not axis)]
    num_nan_any = df.isna().any(axis=axis).sum()
    num_nan_all = df.isna().all(axis=axis).sum()
    pct_nan_any = num_nan_any / num
    pct_nan_all = num_nan_all / num
    return (pct_nan_any, pct_nan_all)


def missing_values_summary(df: DataFrame, axis: int = 1, precision: int = 2) -> str:
    """Return missing values summary.

    Summary of percentage of rows or columns missing values.

    Parameters
    ----------
    df
        Dataframe to query.

    axis : default: 1
        Along which axis to query. 1 to query by row, 0 to query by column.

    precision : default: 2
        Number of decimal places with which to report percentages.

    See Also
    --------
    missing_values_pct

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df.loc[1:2, 'open_'] = np.NAN
    >>> df.loc[2:3,'close_'] = np.NAN
    >>> df
       open_  close_
    0    0.0     5.0
    1    NaN     6.0
    2    NaN     NaN
    3    3.0     NaN
    4    4.0     9.0

    >>> summary = missing_values_summary(df)
    >>> print(summary[:-1]) # ignore new line
    5 rows of which:
        60.00% have a missing value
        20.00% are missing all values
    """
    # pylint: disable=missing-param-doc
    num = df.shape[int(not axis)]
    pct_nan_any, pct_nan_all = missing_values_pct(df, axis=axis)
    rows_cols = "rows" if axis == 1 else "columns"
    msg = (
        f"{num} {rows_cols} of which:\n"
        f"    {pct_nan_any:.{precision}%} have a missing value\n"
        f"    {pct_nan_all:.{precision}%} are missing all values\n"
    )
    return msg


def most_common(values: abc.Sequence[Any] | pd.Series) -> Any:
    """Return most common value in a sequence of values.

    Parameters
    ----------
    values
        Values from which to evaluate most common value. Must be pd.Series
        or valid input to pd.Series.

    Returns
    -------
    Any
        Most common value in `values`. If maximum incidence shared by more
        than one value, returns value that occurs first in the sequence.

    Examples
    --------
    >>> most_common(['foo', 'bar', 'foo', 'bar', 'spam', 'spam', 'foo'])
    'foo'
    >>> most_common([2, 1, 3, 3, 4, 2, 2, 2, 1, 1, 2, 2, 2])
    2

    Where two values share maximum incidence, first to appear is returned.
    >>> most_common(['foo', 'bar', 'bar', 'bar', 'foo', 'foo'])
    'foo'
    """
    ser = pd.Series(values)
    vcs = ser.value_counts()
    max_vcs = vcs[vcs == vcs.max()]
    if len(max_vcs) == 1:
        return max_vcs.index[0]
    else:
        return next((v for v in values if v in max_vcs.index))


@contextmanager
def supress_error(error: str):
    """Context manager to supress specific pandas error.

    Parameters
    ----------
    error
        Error class to be surpressed.

    Examples
    --------
    >>> with supress_error("PerformanceWarning"):
    ...     # code that would otherwise raise PerformanceWarning
    ...     pass
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=getattr(pd.errors, error))
        yield
