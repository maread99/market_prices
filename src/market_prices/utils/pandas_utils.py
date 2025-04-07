"""Utility functions and classes for pandas library."""

import warnings
from collections import abc
from contextlib import contextmanager
from typing import Any, Literal, Annotated
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from valimp import parse, Parser


def pdfreq_to_offset(pdfreq: str) -> pd.offsets.BaseOffset:
    """Pandas frequency string to a pandas offset.

    Parameters
    ----------
    pdfreq
        pandas frequency string to convert. For example, '15min', '3h'.

    Examples
    --------
    >>> pdfreq_to_offset('22min')
    <22 * Minutes>
    >>> pdfreq_to_offset('3h')
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
    >>> # ignore first part, for testing purposes only...
    >>> import pytest, pandas
    >>> v = pandas.__version__
    >>> if (
    ...     (v.count(".") == 1 and float(v) < 2.2)
    ...     or (
    ...         v.count(".") > 1
    ...         and float(v[:v.index(".", v.index(".") + 1)]) < 2.2
    ...     )
    ... ):
    ...     pytest.skip("printed return only valid from pandas 2.2")
    >>> #
    >>> # example from here...
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
    IntervalIndex([[2021-03-02 00:00:00, 2021-03-12 00:00:00], [2021-03-12 00:00:00, 2021-03-22 00:00:00]], dtype='interval[datetime64[ns], both]')
    >>> timestamps_in_interval_of_intervals(timestamps, intervals)
    True
    """
    # NOTE Can lose doctest skip when pandas support is >= 2.2
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
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='h')

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
        index = interval_index_new_tz(index, None)

    # evaluate full_overlap_mask
    # as 'int64' to use expanding, as series to use expanding and shift
    right = pd.Index(index.right.astype("int64")).to_series()
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
    offset: str | pd.offsets.BaseOffset,
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
        Duration of each interval, for example "15min", "3h" etc.

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
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1h')
    >>> index = get_interval_index(left, '33min')
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


@parse
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
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='1h')
    >>> right = left + pd.Timedelta(30, 'min')
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


@parse
def remove_intervals_from_interval(
    interval: pd.Interval, intervals: pd.IntervalIndex
) -> list[pd.Interval]:
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
    list[pd.Interval]
        List of intervals that remain after subtracting 'intervals' from
        'interval'.

    Raises
    ------
    ValueError
        If `intervals` not monotonically increasing or are overlapping.

    Examples
    --------
    >>> # ignore first part, for testing purposes only...
    >>> import pytest, pandas
    >>> v = pandas.__version__
    >>> if (
    ...     (v.count(".") == 1 and float(v) < 2.2)
    ...     or (
    ...         v.count(".") > 1
    ...         and float(v[:v.index(".", v.index(".") + 1)]) < 2.2
    ...     )
    ... ):
    ...     pytest.skip("printed return only valid from pandas 2.2")
    >>> #
    >>> # example from here...
    >>> from pprint import pprint
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='h')
    >>> right = left + pd.Timedelta(30, 'min')
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
    [Interval(2021-05-01 12:30:00, 2021-05-01 13:00:00, closed='left'),
     Interval(2021-05-01 13:30:00, 2021-05-01 14:00:00, closed='left'),
     Interval(2021-05-01 14:30:00, 2021-05-01 15:00:00, closed='left'),
     Interval(2021-05-01 15:30:00, 2021-05-01 16:00:00, closed='left'),
     Interval(2021-05-01 16:30:00, 2021-05-01 17:30:00, closed='left')]
    """
    # NOTE Can lose doctest skip when pandas support is >= 2.2
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


# valimp Parser function
def verify_interval_datetime_index(
    name: str, obj: pd.DatetimeIndex | pd.IntervalIndex, _
) -> pd.DatetimeIndex | pd.IntervalIndex:
    """Verify pd.IntervalIndex has both sides as pd.DatetimeIndex."""
    if isinstance(obj, pd.IntervalIndex) and not isinstance(obj.left, pd.DatetimeIndex):
        raise ValueError(
            f"'{name}' can only take a pd.IntervalIndex that has each side"
            " as type pd.DatetimeIndex, although received with left side"
            f" as type '{type(obj.left)}'."
        )
    return obj


@parse
def interval_index_new_tz(
    index: Annotated[pd.IntervalIndex, Parser(verify_interval_datetime_index)],
    tz: ZoneInfo | str | None,
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
    >>> tz = ZoneInfo("US/Central")
    >>> left = pd.date_range(
    ...     '2021-05-01 12:00', periods=5, freq='h', tz=tz
    ... )
    >>> right = left + pd.Timedelta(30, 'min')
    >>> index = pd.IntervalIndex.from_arrays(left, right)
    >>> index.right.tz
    zoneinfo.ZoneInfo(key='US/Central')
    >>> new_index = interval_index_new_tz(index, tz=ZoneInfo("UTC"))
    >>> new_index.left.tz.key == new_index.right.tz.key == "UTC"
    True
    """
    indices = []
    for indx in [index.left, index.right]:
        try:
            indices.append(indx.tz_convert(tz))
        except TypeError:
            indices.append(indx.tz_localize(tz))
    return pd.IntervalIndex.from_arrays(indices[0], indices[1], closed=index.closed)


@parse
def index_is_normalized(
    index: Annotated[
        pd.DatetimeIndex | pd.IntervalIndex,
        Parser(verify_interval_datetime_index),
    ],
) -> bool:
    """Query if an index is normalized.

    Parameters
    ----------
    index
        Index to query

    Examples
    --------
    >>> index = pd.date_range('2021-05-01 12:00', periods=5, freq='h')
    >>> index_is_normalized(index)
    np.False_
    >>> index = pd.date_range('2021-05-01', periods=5, freq='1D')
    >>> index_is_normalized(index)
    np.True_

    >>> index = pd.interval_range(
    ...     pd.Timestamp('2021-05-01 12:00'), periods=5, freq='h'
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


@parse
def indexes_union(indexes: list[pd.Index]) -> pd.Index:
    """Union multiple pd.Index objects.

    Parameters
    ----------
    indexes
        pd.Index objects to be joined. Note: All indexes must be of same
        dtype.

    Examples
    --------
    >>> index1 = pd.date_range('2021-05-01 12:20', periods=2, freq='1h')
    >>> index2 = pd.date_range('2021-05-02 17:10', periods=2, freq='22min')
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


@parse
def index_union(indexes: list[pd.Index | pd.Series | pd.DataFrame]) -> pd.Index:
    """Union indexes of multiple indexes, Series and/or DataFrame.

    Parameters
    ----------
    indexes
        List of pd.Index and/or pd.Series and/or pd.DataFrames with indexes
        to be joined. All indexes must be of same dtype.

    Examples
    --------
    >>> index1 = pd.date_range('2021-05-01 12:20', periods=2, freq='1h')
    >>> index2 = pd.date_range('2021-05-02 17:10', periods=2, freq='22min')
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
    np.int64(2)

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
