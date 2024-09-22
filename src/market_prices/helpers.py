"""Helper functions with usage specific to `market_prices`."""

from __future__ import annotations

import re
import sys
from typing import Literal, TYPE_CHECKING
import zoneinfo

import pandas as pd
import numpy as np

from market_prices import mptypes
from market_prices.utils import general_utils as genutils
from market_prices.utils import pandas_utils as pdutils

if TYPE_CHECKING:
    from market_prices import intervals

if "pytest" in sys.modules:
    import pytest  # noqa: F401  # pylint: disable=unused-import  # used within doctest

UTC = zoneinfo.ZoneInfo("UTC")

ONE_DAY: pd.Timedelta = pd.Timedelta(1, "D")
ONE_MIN: pd.Timedelta = pd.Timedelta(1, "min")
ONE_SEC: pd.Timedelta = pd.Timedelta(1, "s")


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
        return ts.tz_convert(UTC)
    except TypeError:
        return ts.tz_localize(UTC)


def to_tz_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone naive copy of a given timestamp.

    Converts `ts` to tz-naive in terms of utc.

    Parameters
    ----------
    ts
        Timestamp to return a timezone-naive copy of.
    """
    if ts.tz is None:
        return ts
    return ts.tz_convert(UTC).tz_convert(None)


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
    now_ = pd.Timestamp.now(tz=UTC)
    if interval is not None and not interval.is_intraday:
        now_ = now_.tz_convert(None)
        res = "D"
    else:
        res = "min"
    return now_.ceil(res) if side == "right" else now_.floor(res)


# intervals / frequencies


def extract_freq_parts(freq: str) -> tuple[int, str]:
    """Extract value and unit parts of an interval/frequency.

    Assumes input previously verified as of type str.

    Does not verify that either the value or unit represent valid values.

    Parameters
    ----------
    freq
        Pandas frequency, for example "5D", "30min", "4h", "33MS".

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
    >>> extract_freq_parts("22min")
    (22, 'min')
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
            bv = df[(s, "close")].isna()
            df.loc[bv, (s, "volume")] = np.nan
    else:
        bv = df["close"].isna()
        df.loc[bv, "volume"] = np.nan
    return df


def resample(
    resample_me: pd.DataFrame | pd.core.groupby.groupby.GroupBy,
    rule: pd.offsets.BaseOffset | str,
    data: pd.DataFrame | None = None,
    origin: str = "start",
    nominal_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Resample ohlcv data to a pandas rule.

    Parameters
    ----------
    resample_me
        Pandas object to be resampled. Object must have .resample method.

    rule
        Pandas frequency or offset to which data to be resampled.

    data
        If resample_me is not a DataFrame (but, for example, a GroupBy
        object), pass `data` as underlying DataFrame (for example, on which
        GroupBy).

    origin
        As `pd.DataFrame.resample` method.

    nominal_start
        The earliest date prior to the first index of `resample_me` on and
        subsequent to which there are no trading sessions until the first
        index of `resample_me`.

        Only useful when `rule` describes a frequency greater than daily
        and there are no sessions between the first index and the date to
        which that first index would be rolled back to conicide with the
        nearest occurrence of 'rule'.
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
