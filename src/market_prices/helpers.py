"""Helper functions with usage specific to `market_prices`."""

from __future__ import annotations

import re
import sys
from typing import Literal, Optional

import pandas as pd
import numpy as np
import pydantic
import pytz

from market_prices import intervals, mptypes
from market_prices.utils import general_utils as genutils
from market_prices.utils import pandas_utils as pdutils

if "pytest" in sys.modules:
    import pytest  # noqa: F401  # pylint: disable=unused-import  # used within doctest

ONE_DAY: pd.Timedelta = pd.Timedelta(1, "D")
ONE_MIN: pd.Timedelta = pd.Timedelta(1, "T")
ONE_SEC: pd.Timedelta = pd.Timedelta(1, "S")


def symbols_to_list(symbols: mptypes.Symbols) -> list[str]:
    """Convert Symbols to a list of strings.

    Parameters
    ----------
    symbols
        Symbols to be converted.

    Examples
    --------
    >>> symbols_to_list('AMZN')
    ['AMZN']
    >>> symbols_to_list('amzn aapl msft')
    ['AMZN', 'AAPL', 'MSFT']
    >>> symbols_to_list('AMZN, AAPL, MSFT')
    ['AMZN', 'AAPL', 'MSFT']
    >>> symbols_to_list(['amzn', 'aapl', 'msft'])
    ['AMZN', 'AAPL', 'MSFT']
    """
    if not isinstance(symbols, list):
        symbols = re.findall(r"[\w\-.=^]+", symbols)
    return [i.upper() for i in symbols]


def is_date(ts: pd.Timestamp) -> bool:
    """Query if a timestamp represents a date.

    `ts` considered to represent a date if tz-naive and has time component
    as 00:00.

    Parameters
    ----------
    ts
        Timestamp to query.

    Returns
    -------
    bool
        Boolean that indicates if `ts` represents a date.
    """
    return pdutils.is_midnight(ts) and ts.tz is None


def fts(ts: pd.Timestamp) -> str:
    """Format a given timestamp.

    Parameters
    ----------
    ts
        Timestamp to format. If `ts` represents a time, `ts` should be
        tz-aware.

    Returns
    -------
    str
        If timestamp represents a:
            date - returns str formatted as "%Y-%m-%d"
            time - returns st formatted as "%Y-%m-%d %H:%M %Z"
    """
    fmt = "%Y-%m-%d" if is_date(ts) else "%Y-%m-%d %H:%M %Z"
    return ts.strftime(fmt)


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a copy of a given timestamp with timezone set to UTC.

    If `ts` is tz-aware will convert `ts` to UTC.
    If `ts` is tz-naive will localize as UTC.

    Parameters
    ----------
    ts
        Timestamp to return a copy of with timezone set to UTC.
    """
    try:
        return ts.tz_convert(pytz.UTC)
    except TypeError:
        return ts.tz_localize(pytz.UTC)


def to_tz_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone naive copy of a given timestamp.

    Converts `ts` to tz-naive in terms of utc.

    Parameters
    ----------
    ts
        Timestamp to return a timezone-naive copy of.
    """
    if ts.tz is None:
        return ts  # type: ignore[unreachable]  # 'tis very reachable
    return ts.tz_convert(pytz.UTC).tz_convert(None)


def now(
    interval: intervals.PTInterval | None = None,
    side: Literal["left", "right"] = "left",
) -> pd.Timestamp:
    """Date or time 'now'.

    Parameters
    ----------
    interval : default: treat as intraday
        Interval for which require 'now'.

        A daily or higher interval will return 'now' as a date (time
        as 00:00 and tz-naive) based on the UTC time now.

        An intraday interval will return 'now' as a time (UTC, no
        component more accurate than 'minute').

    side
        Side of current minute/day.

        If returning a time then 'left'/'right' will round current
        minute down/up to nearest minute.

        If returning a date then 'left'/'right' will return 'today'
        or 'tomorrow' respectively, with 'today' based on current
        UTC time.
    """
    # pylint: disable=missing-param-doc
    now_ = pd.Timestamp.now(tz=pytz.UTC)
    if interval is not None and not interval.is_intraday:
        now_ = now_.tz_convert(None)
        res = "D"
    else:
        res = "T"
    return now_.ceil(res) if side == "right" else now_.floor(res)


# intervals / frequencies


def extract_freq_parts(freq: str) -> tuple[int, str]:
    """Extract value and unit parts of an interval/frequency.

    Assumes input previously verified as of type str.

    Does not verify that either the value or unit represent valid values.

    Parameters
    ----------
    freq
        Pandas frequency, for example "5D", "30T", "4H", "33MS".

    Raises
    ------
    ValueError
        If `freq` is not defined as any of:
            One or more consecutive non-digits (value will be assumed as
            1).

            One or more consecutive digits followed by one or more
            consecutive non-digits.

    Returns
    -------
    tuple[int, str]
        [0]: value
        [1]: unit

    Examples
    --------
    >>> extract_freq_parts("22T")
    (22, 'T')
    >>> extract_freq_parts("1m")
    (1, 'm')
    >>> extract_freq_parts("D")
    (1, 'D')
    >>> extract_freq_parts("33ms")
    (33, 'ms')
    >>> extract_freq_parts("33MS")
    (33, 'MS')
    >>> for invalid_input in ["2s2", "h2", "t2t", "3"]:
    ...     with pytest.raises(ValueError):
    ...         extract_freq_parts(invalid_input)
    >>> print("exceptions raised!")
    exceptions raised!
    """

    def freq_format_error_msg() -> str:
        return (
            f'interval/frequency received as "{freq}", although must be defined'
            " as one or more consecutive digits followed by one or more"
            ' consecutive letters. Valid examples: "30min", "3h", "1d", "3m".'
        )

    unit = genutils.remove_digits(freq)
    if unit == freq:
        freq = "1" + freq
    if freq[-len(unit) :] != unit or len(unit) == len(freq):
        raise ValueError(freq_format_error_msg())
    value = genutils.remove_nondigits(freq)
    if freq[: len(value)] != value or len(value) == len(freq) or value == "0":
        raise ValueError(freq_format_error_msg())
    return int(value), unit


# Price table helpers


def has_symbols(df: pd.DataFrame) -> bool:
    """Query if the columns of a DataFrame have a "symbol" index level.

    Parameters
    ----------
    df
        DataFrame to query.
    """
    return isinstance(df.columns, pd.MultiIndex) and df.columns.names[0] == "symbol"


AGG_FUNCS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def order_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Order colums of a DataFrame to order of a prices table.

    Parameters
    ----------
    df
        DataFrame to be ordered. .columns attribute must be of type
        pd.Index, not pd.MultiIndex.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with colums ordered as a prices table.
    """
    try:
        return df[list(AGG_FUNCS)]
    except KeyError:
        if isinstance(df.columns, pd.MultiIndex):
            msg = "Columns of `df` must be indexed with pd.Index, not a pd.MultiIndex."
            raise ValueError(msg) from None
        raise


def agg_funcs(df: pd.DataFrame) -> dict[str | tuple[str, str], str]:
    """Return mapping of column labels to aggregation function names.

    Parameters
    ----------
    df
        Prices table (pd.DataFrame with access to .pt accessor).

    Returns
    -------
    dict[str | tuple[str, str]], str]:
        key: str | tuple[str, str]
            Column label. Where `df` has symbols, keys will be 2-tuples
            with the first element describing the symbol, e.g.
            ('AMZN', 'open').

        value: str
            Name of aggregation function to be used to aggregate column.
    """
    if has_symbols(df):
        return {c: AGG_FUNCS[c[1]] for c in df}
    else:
        return {c: AGG_FUNCS[c] for c in df}


def volume_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of DataFrame with missing volumes set to nan.

    Sets volumes to nan if corresponding closes are nan.

    Can be used post resampling where `sum` used as the aggregate function
    on volume columns (which has effect of summing missing values to 0).

    Parameters
    ----------
    df
        pd.DataFrame for which missing volumes to be set to np.nan. NB
        A copy of `df` will be operated on and returned (`df` will remain
        unchanged).
    """
    df = df.copy()
    bv: pd.Series
    if has_symbols(df):
        for s in df.columns.remove_unused_levels().levels[0]:
            bv = df[(s, "close")].isna()  # type: ignore[assignment]  # is a Series
            df.loc[bv, (s, "volume")] = np.nan
    else:
        bv = df["close"].isna()
        df.loc[bv, "volume"] = np.nan
    return df


def resample(
    resample_me: pd.DataFrame | pd.core.groupby.groupby.GroupBy,
    rule: pd.offsets.BaseOffset,  # type: ignore[name-defined]  # is defined
    data: pd.DataFrame | None = None,
    origin: str = "start",
) -> pd.DataFrame:
    """Resample ohlcv data to a pandas rule.

    Parameters
    ----------
    resample_me
        Pandas object to be resampled. Object must have .resample method.

    rule
        Pandas offset to which data to be resampled.

    data
        If resample_me is not a DataFrame (but, for example, a GroupBy
        object), pass `data` as underlying DataFrame (for example, on which
        GroupBy).

    origin
        As `pd.DataFrame.resample` method.
    """
    if isinstance(resample_me, pd.DataFrame):
        resample_me = resample_me.copy()
        data = resample_me
    else:
        assert data is not None
        data = data.copy()

    columns_ = data.columns
    if has_symbols(data):
        data.columns = [t[1] + "_" + t[0] for t in data.columns.to_numpy()]
        agg_f = {
            k: AGG_FUNCS[c] for k, c in zip(data.columns, columns_.get_level_values(1))
        }
    else:
        agg_f = agg_funcs(data)

    resampler = resample_me.resample(rule, closed="left", label="left", origin=origin)
    resampled = resampler.agg(agg_f)
    resampled.columns = columns_
    resampled = volume_to_na(resampled)
    return resampled


# Arguments as strings

# TODO Method has no internal clients. If find an internal or external
# client for it then need to review, add a test and make method public.
@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def _period_string(
    minutes=0,
    hours=0,
    days=0,
    weeks=0,
    months=0,
    years=0,
    end: Optional[mptypes.Timestamp] = None,
    start: Optional[mptypes.Timestamp] = None,
    **kwargs,
) -> str:
    """Return string description of period defined with period parameters.

    See prices module docstring for notes on period parameters.

    If no parameters passed returns 'all'.

    NB string will assume `start` and `end` as the values passed. If these
    are the same as values passed to `end` and `start` parameters of
    prices.get() then the values in the returned string may not necessarily
    align with the start and end dates of the actual returned data.

    NB excess kwargs pass silently.

    Notes
    -----
    `period_string` has no internal clients within `market_prices`. Made
    available as a helper to clients.
    """
    # pylint: disable=too-complex, too-many-arguments
    is_period = bool(sum([minutes, hours, days, weeks, months, years]))

    if end is None and start is None and not is_period:
        return "all"

    end_str = fts(end) if end is not None else None  # type: ignore[arg-type]  # mptype
    start_str = fts(start) if start is not None else None  # type: ignore[arg-type]

    if end_str is not None and start_str is not None:
        return f"{start_str} to {end_str}"

    if start_str is not None and end_str is None and not is_period:
        return f"since {start_str}"

    if sum([minutes, hours]) > 0:
        duration = ""
        if minutes:
            duration += f"{minutes}min"
        if hours:
            if duration:
                duration += " "
            duration += f"{hours}h"
    elif days > 0:
        duration = f"{days}td"
    else:
        mapping = {"y": years, "mo": months, "w": weeks}
        duration = "".join([f"{v}{s}" for s, v in mapping.items() if v > 0])

    if end_str is None and start_str is None:
        return duration
    elif end_str is not None:
        return f"{duration} to {end_str}"
    else:
        return f"{duration} from {start_str}"


# TODO Method has no internal clients. If find an internal or external
# client for it then need to review, add a test and make method public.
def _range_string(df: pd.DataFrame, close=False, shand=False) -> str:
    """Return string describing range of dates covered by interval index.

    Minutes and Hours ommited if timestamp is midnight.

    Parameters
    ----------
    df: DataFrame
        DataFrame with rows indexed with pd.IntervalIndex.

    close: bool (default: False)
        True to evaluate range between close timestamps of first and last
        indices. If False will evaluate range from open of first indice
        to close of last indice.

    sh: bool (default: False)
        True to return shorthand string, where components of end timestamp
        that are the same as the corresponding component of the start
        timestamp are omitted.

    Notes
    -----
    `range_string` has no internal clients within `market_prices`. Made
    available as a helper to clients.
    """
    # TODO following needs revising in light of availabiliy of `fts` and
    # `is_date` methods
    index = df.index
    full_f_str = "%Y-%m-%d %H:%M"
    day_f_str = "%Y-%m-%d"
    start = index[0].left if not close else index[0].right
    end = index[-1].right
    start_f_str = day_f_str if pdutils.is_midnight(start) else full_f_str
    f_str = full_f_str

    if shand:
        to_replace = ["%Y-", "%m-", "%d", "%H:"]
        for i, attr in enumerate(["year", "month", "day", "hour"]):
            if getattr(start, attr) == getattr(end, attr):
                f_str = f_str.replace(to_replace[i], "")
            else:
                break

    end_f_str = f_str.replace(" %H:%M", "") if pdutils.is_midnight(end) else f_str
    return start.strftime(start_f_str) + " to " + end.strftime(end_f_str)
