"""Get prices from local csv files."""

from __future__ import annotations

import os
import typing
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from market_prices import intervals
from market_prices.intervals import TDInterval
from market_prices.utils.general_utils import remove_digits

ERROR_MALFORMED_FREQ = ValueError("Possible malformed frequency")
ERROR_DAILY_FREQ = ValueError("Daily frequency cannot have value greater than one")
FREQ_UNITS = {"MIN": "minutes", "T": "minutes", "H": "hours", "D": "days"}


def _get_csv_interval(s: str) -> TDInterval | None | ValueError:
    """Evaluate any interval represented in part of a filename.

    Returns
    -------
    interval
        `intervals.PTInterval` if able to evaluate frequency.

        None if the string does not represent a frequency.

        ValueError if identify a possibly malformed frequency. NB ValueError
        is retruned, not raised.

        ValueError if a daily frequency is identified with a value greater
        than one.

        NOTE: ValueErrors are returned, not raised.
    """
    s = s.upper()
    if s in FREQ_UNITS:
        return intervals.to_ptinterval(s)
    for freq in FREQ_UNITS:
        interval: intervals.TDInterval | timedelta = timedelta(0)
        if s.startswith(freq):
            value = s[len(freq) :]
            if not value.isdigit():
                if len(value) > 2:
                    return None
                return ERROR_MALFORMED_FREQ
            interval = intervals.to_ptinterval(value + freq)
        elif s.endswith(freq):
            value = s[: -len(freq)]
            if not value.isdigit():
                if len(value) > 2:
                    return None
                return ERROR_MALFORMED_FREQ
            interval = intervals.to_ptinterval(value + freq)
        if not interval:
            continue
        if typing.TYPE_CHECKING:
            assert isinstance(interval, intervals.TDInterval)
        if interval.is_daily and interval.days != 1:
            return ERROR_DAILY_FREQ
        return interval
    if remove_digits(s) in FREQ_UNITS:
        return ERROR_MALFORMED_FREQ  # frequency unit in middle of digits
    return None


def _get_symbol_from_filename(name: str, symbols: list[str]) -> str | None:
    """Return symbol indentified in filename, or None if indiscernible.

    Parameters
    ----------
    name
        Name of csv file, excluding any extension.

    symbols
        List of valid symbols.
    """
    parts = name.split("_")
    symbol = ""
    for part in parts:
        if part not in symbols:
            continue
        if symbol and symbol != part:
            # at least two different symbols in filename
            return None
        symbol = part
    return symbol if symbol else None


def _get_interval_from_filename(name: str) -> TDInterval | None:
    """Evaluate interval from filename, or None if indiscernible.

    Parameters
    ----------
    name
        Name of csv file, EXCLUDING any extension.
    """
    parts = name.split("_")
    intrvl = timedelta(0)
    for part in parts:
        if (intrvl_ := _get_csv_interval(part)) is None:
            continue
        if intrvl_ in [ERROR_MALFORMED_FREQ, ERROR_DAILY_FREQ]:
            return None
        if intrvl and intrvl != intrvl_:
            # at least two different valid frequencies in filename
            return None
        intrvl = intrvl_
    if intrvl:
        return intrvl
    return TDInterval.D1  # assume data is daily


def _raise_duplicate_csv_error(symbol: str, interval: TDInterval, p1: Path, p2: Path):
    raise ValueError(
        f"At least two paths have been found with data for symbol '{symbol}' over"
        f" interval '{interval}'. Paths are:"
        f"\n{p1}\n{p2}\nThere must be a single .csv file per symbol per interval."
    )


def get_csv_paths(
    root_dir: str, symbols: list[str]
) -> dict[str, dict[TDInterval, Path]]:
    """Get paths, by symbol, to valid csv fies.

    Parameters
    ----------
    root_dir
        Directory containing .csv files and/or a hierarchy of
        subdirectories containing .csv files.

    symbols
        Symbols for which to return paths to corresponding csv files.
    """
    paths: dict[str, dict[TDInterval, Path]] = defaultdict(dict)
    # NOTE from min python 3.12 can use .walk on the pathlib.Path instance
    for dir_, subdirs, files in os.walk(root_dir):
        path_dir = Path(dir_)
        for file in files:
            path = Path(path_dir / file)
            if path.suffix != ".csv":
                continue

            stem = path.stem
            parts = stem.split("_")

            if len(parts) == 1:
                # not split
                if stem not in symbols:
                    continue
                # filename is simply symbol
                symb = stem
                interval = TDInterval.D1
                if symb in paths and interval in paths[symb]:
                    _raise_duplicate_csv_error(
                        symb, interval, paths[symb][interval], path
                    )
                paths[symb][interval] = path
                continue

            if (symbol := _get_symbol_from_filename(stem, symbols)) is None:
                continue
            if (intrvl := _get_interval_from_filename(stem)) is None:
                continue

            if symbol in paths and intrvl in paths[symbol]:
                _raise_duplicate_csv_error(symbol, intrvl, paths[symbol][intrvl], path)
            paths[symbol][intrvl] = path

    return paths
