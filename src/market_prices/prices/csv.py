"""Get prices from local csv files."""

from __future__ import annotations

import copy
import os
import pprint
import traceback
import typing
from typing import Any, Literal, Annotated
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from exchange_calendars import ExchangeCalendar
import numpy as np
import pandas as pd
from valimp import parse, Coerce, Parser

from market_prices import helpers, intervals, mptypes, parsing
from market_prices.errors import MarketPricesError, PricesWarning, PricesMissingWarning
from market_prices.helpers import UTC
from market_prices.intervals import TDInterval
from market_prices.prices import base
from market_prices.utils.general_utils import remove_digits


ERROR_MALFORMED_INTRVL = ValueError("Possible malformed interval")
ERROR_DAILY_INTRVL = ValueError("Daily interval cannot have value greater than one")
INTRVL_UNITS = {"MIN": "minutes", "T": "minutes", "H": "hours", "D": "days"}

CSV_READ_DFLT_KWARGS: dict[str, Any] = {
    "header": 0,
    "usecols": lambda x: x.lower()
    in ["date", "open", "high", "low", "close", "volume"],
    "index_col": "date",
    "parse_dates": ["date"],
}

CSV_READ_DTYPE_VALUE: dict[str, str] = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
}

# IDENTIFY CSV FILES


def _get_csv_interval(s: str) -> TDInterval | None | ValueError:
    """Evaluate any interval represented in part of a filename.

    Returns
    -------
    interval
        `intervals.PTInterval` if able to evaluate interval.

        None if the string does not represent an interval.

        ValueError if identify a possibly malformed interval. NB ValueError
        is returned, not raised.

        ValueError if a daily interval is identified with a value greater
        than one.

        NOTE: ValueErrors are returned, not raised.
    """
    s = s.upper()
    if s in INTRVL_UNITS:
        return intervals.to_ptinterval(s)
    for unit in INTRVL_UNITS:
        interval: intervals.TDInterval | timedelta = timedelta(0)
        if s.startswith(unit):
            value = s[len(unit) :]
            if not value.isdigit():
                if len(value) > 2:
                    return None
                return ERROR_MALFORMED_INTRVL
            interval = intervals.to_ptinterval(value + unit)
        elif s.endswith(unit):
            value = s[: -len(unit)]
            if not value.isdigit():
                if len(value) > 2:
                    return None
                return ERROR_MALFORMED_INTRVL
            interval = intervals.to_ptinterval(value + unit)
        if not interval:
            continue
        if typing.TYPE_CHECKING:
            assert isinstance(interval, intervals.TDInterval)
        if interval.is_daily and interval.days != 1:
            return ERROR_DAILY_INTRVL
        return interval
    if remove_digits(s) in INTRVL_UNITS:
        return ERROR_MALFORMED_INTRVL  # interval unit in middle of digits
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
    errors = False
    for part in parts:
        if (intrvl_ := _get_csv_interval(part)) is None:
            continue
        if intrvl_ is ERROR_DAILY_INTRVL:
            return None
        if intrvl_ is ERROR_MALFORMED_INTRVL:
            errors = True
            continue
        if intrvl and intrvl != intrvl_:
            # at least two different valid intervals in filename
            return None
        intrvl = intrvl_
    if intrvl:
        return intrvl
    if errors:
        return None
    return TDInterval.D1  # assume data is daily


def _raise_duplicate_csv_error(symbol: str, interval: TDInterval, p1: Path, p2: Path):
    raise ValueError(
        f"At least two paths have been found with data for symbol '{symbol}' over"
        f" interval '{interval}'. Paths are:"
        f"\n{p1}\n{p2}\nThere must be a single .csv file per symbol per interval."
        " See the 'path' parameter section of help(PricesCsv) for advices on"
        " how csv files should be named."
    )


def get_csv_paths(
    root: str | Path, symbols: list[str]
) -> dict[str, dict[TDInterval, Path]]:
    """Get paths, by symbol, to valid csv fies.

    Parameters
    ----------
    root
        Directory containing .csv files and/or a hierarchy of
        subdirectories containing .csv files.

    symbols
        Symbols for which to return paths to corresponding csv files.
    """
    paths: dict[str, dict[TDInterval, Path]] = defaultdict(dict)
    # NOTE from min python 3.12 can use .walk on the pathlib.Path instance
    for dir_, subdirs, files in os.walk(root):
        path_dir = Path(dir_)
        for file in files:
            path = Path(path_dir / file).relative_to(root)
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


# PARSE CSV FILES


def _get_csv_read_kwargs(client_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Get kwargs for `pandas.csv_read`."""
    client_kwargs_ = {} if client_kwargs is None else client_kwargs
    kwargs = CSV_READ_DFLT_KWARGS | client_kwargs_
    kwargs["dtype"] = CSV_READ_DTYPE_VALUE
    return kwargs


# errors


class CsvError(MarketPricesError):
    """Error related to csv containing price data."""


class CsvPathsError(CsvError):
    """No csv files were found at any interval for at least one symbol.

    Parameters
    ----------
    root
        Path to root directory holding csv files.

    missing_symbols
        Symbols for no csv files were found at any interval.
    """

    def __init__(
        self,
        root: Path,
        missing_symbols: set[str] | list[str],
        paths: dict[str, dict[TDInterval, Path]],
    ):
        missing_symbols = list(missing_symbols)
        missing_symbols.sort()
        if not paths:
            msg_end = ""
        else:
            msg_end = f" Only the following files were found:\n{pprint.pformat(paths, indent=2)}"
        self._msg = (
            "No csv files for were found at any interval for symbols"
            f" '{missing_symbols}'. Searched files in and under the directory {root}."
            " See the 'path' parameter section of help(PricesCsv) for advices on"
            " how csv files should be named."
        ) + msg_end


class CsvPathsIntervalsError(CsvError):
    """No csv files were found for any interval that's represented by all symbols.

    Parameters
    ----------
    symbols
        Symbols to include to PricesCsv instance.

    paths
        Paths that were found for `symbols`.

    warnings_
        Warnings that occurred when removing intervals that were not
        available to all symbols.
    """

    def __init__(
        self,
        symbols: list[str],
        paths: dict[str, dict[TDInterval, Path]],
        warnings_: list[PricesCsvIntervalUnavailableWarning],
    ):
        self._msg = (
            f"The following paths were found for symbols '{symbols}'"
            " although no interval is available to all symbols:\n"
            f"{pprint.pformat(paths, indent=2)}"
            "\n\nThe following warnings occurred when evaluating available intervals:"
            f"\n{compile_error_msgs(warnings_, False)}"
            f"\n\nSee the 'path' parameter and 'Notes' sections of help(PricesCsv) for"
            " advices on how csv files should be named and formatted and for use of"
            " the `read_csv_kwargs` parameter."
        )


class CsvParsingError(CsvError):
    """Error when parsing a csv file to a pd.DataFrame.

    Parameters
    ----------
    path
        path representing csv file.

    source
        Original source error.
    """

    _msg = "Unable to parse csv file."

    def __init__(self, path: Path, *, source: Exception | None = None):
        self.path = path
        self.source = source

    def __str__(self) -> str:
        return (
            f"Unable to create dataframe from csv file at '{self.path}'"
            f" due to the following error:\n\t{type(self)} {self._msg}"
        )

    def str_inc_orig(self, verbose: bool = False) -> str:
        """Str representing error which includes any source error message.

        Parameters
        ----------
        verbose
            True to include full original traceback. False (default) to
            include only the original error message.
        """
        msg = str(self)
        if self.source is None:
            return msg

        if not verbose:
            msg += (
                "\nThe source error's message was:"
                f"\n\t{type(self.source)}: {str(self.source)}"
            )
            return msg

        try:
            raise self.source
        except Exception:
            msg += f"\nThe source error's traceback was:\n{traceback.format_exc()}"[:-2]
        return msg


class CsvReadError(CsvParsingError):
    """Error reading a csv file."""

    _msg = "`pd.read_csv` raises error."


class CsvIndexError(CsvParsingError):
    """Error parsing index from csv file."""

    _msg = "Index does not parse as a `pd.DatetimeIndex`."


class CsvVolDtypeError(CsvParsingError):
    """Error parsing index from csv file."""

    _msg = "'volume' column will not convert to 'float64' dtype."


class CsvDataframeEmptyError(CsvParsingError):
    """Error parsing data from csv file."""

    _msg = "No price data parsed from csv file."


class CsvHighLowError(CsvParsingError):
    """High values lower than corresponding low values."""

    _msg = (
        "At least one row has a high value that is lower than the"
        " corresponding low value."
    )


class CsvIntervalError(CsvParsingError):
    """Error with interval implied by csv file content."""

    _msg = "Date indices do not reflect the expected interval."


class CsvIncongruentValuesError(CsvParsingError):
    """Parsed high and/or low values incongruent with open and/or close values.

    Parameters
    ----------
    path
        Path representing csv file.

    thres
        Threshold for raising error in terms of percentage of incongruent
        rows.
    """

    def __init__(self, path: Path, thres: float):
        self.thres = thres
        super().__init__(path)
        self._msg = (
            "Parsed high and/or low values are incongrument with open"
            f" and/or close values (threshold {self.thres * 100}% of rows)."
        )


class CsvReindexedDataframeEmptyError(CsvError):
    """No data remaining after reindexing against evaluated index."""

    def __init__(self, symbol: str, interval: TDInterval, calendar: ExchangeCalendar):
        self._msg = (
            f"For symbol '{symbol} with interval '{interval}' no indice aligned with"
            f" index evaluated from calendar '{calendar.name}'."
        )


class CsvNoDataError(CsvError):
    """Data does not parse for any symbol at any interval."""

    def __init__(
        self,
        symbols: list[str],
        errors: list[MarketPricesError | PricesWarning],
        verbose: bool,
    ):
        self._msg = (
            f"For symbols '{symbols}' it was not possible to create a price table for"
            " any interval from csv files. The following errors and warnings"
            " occurred during parsing:"
            f"\n{compile_error_msgs(errors, verbose)}"
            "\n\nSee the 'path' parameter and 'Notes' sections of help(PricesCsv) for"
            " advices on how csv files should be named and formatted and for use of"
            " the `read_csv_kwargs` parameter."
        )


# warnings


class PricesCsvWarning(PricesWarning):
    """User warning advising of issues with price data sourced from csv files."""

    _msg = "CSV Prices Warning"


class PricesCsvIntervalUnavailableWarning(PricesCsvWarning):
    """Price data not available at an interval for all symbols."""

    def __init__(self, interval: TDInterval, symbols: set[str] | list[str]):
        symbols = list(symbols)
        symbols.sort()
        self._msg = (
            f"Prices are not available at base interval {interval} as (aligned) data"
            f" was not found at this interval for symbols '{symbols}'."
        )


class PricesCsvIgnoredUnalignedIndicesWarning(PricesCsvWarning):
    """Ingnored indices in csv file that were unaligned with evaluated index."""

    def __init__(self, indices: pd.DatetimeIndex, interval: TDInterval, symbol: str):
        self._msg = (
            f"For symbol '{symbol}' at base interval {interval} the csv file included"
            " the following indices that are not aligned with the evaluated index and:"
            f" have therefore been ignored:\n{indices}"
        )


class PricesCsvParsingConsolidatedWarning(PricesCsvWarning):
    """Consolidated errors and warnings that occurred during parsing."""

    def __init__(
        self,
        errors: list[MarketPricesError | PricesWarning],
        verbose: bool,
    ):
        self._msg = (
            "Price data has been found for all symbols at a least one interval,"
            " however, you may find that not all the expected price data is available."
            " See the `limits` property for available base intervals and the limits"
            " between which price data is available at each of these intervals."
            " See the `csv_paths` property for paths to all csv files that were found"
            " for the requested symbols. See the 'path' parameter and 'Notes' sections"
            " of help(PricesCsv) for advices on how csv files should be named and"
            " formatted and for use of the `read_csv_kwargs` parameter."
            "\n\nThe following errors and/or warnings occurred during parsing:"
            f"\n{compile_error_msgs(errors, verbose)}"
        )


def compile_error_msgs(
    errors: list[MarketPricesError | PricesWarning], verbose: bool
) -> str:
    """Compile a list of error and/or warning messages into a string.

    Parameters
    ----------
    verbose
        Inlcude full traceback of any source errors.
    """
    msgs = []
    for i, error in enumerate(errors):
        try:
            msg = error.str_inc_orig(verbose)
        except AttributeError:
            msg = str(error)
        msgs.append(f"{i}) {msg}")
    return "\n" + "\n\n".join(msgs)


def _remove_unavailable_intervals_from_paths(
    paths: dict[str, dict[TDInterval, Path]],
) -> tuple[
    dict[str, dict[TDInterval, Path]], list[PricesCsvIntervalUnavailableWarning]
]:
    """Removes from `paths` those intervals not available to all `symbols`.

    Returns paths, as amended, and warnings advising of unavailable intervals.

    Parameters
    ----------
    paths
        Paths to csv files, by symbol by interval. ALL symbols must be
        represented (the possibility that csv files are not available at
        any interval for a symbol should be handled before calling this
        method).
    """
    symbols = list(paths)
    intervals = []
    for symbol in symbols:
        if symbol in paths:
            intervals.extend(list(paths[symbol]))

    unique_intervals = set(intervals)

    intervals_to_remove = []
    for interval in unique_intervals:
        if not all(interval in paths[symbol] for symbol in symbols):
            intervals_to_remove.append(interval)

    warnings_ = []
    for interval in intervals_to_remove:
        missing_symbols = []
        for symbol in symbols:
            if interval in paths[symbol]:
                del paths[symbol][interval]
            else:
                missing_symbols.append(symbol)
        warnings_.append(PricesCsvIntervalUnavailableWarning(interval, missing_symbols))

    for symbol in list(paths):
        if not paths[symbol]:
            del paths[symbol]

    return paths, warnings_


def check_adjust_high_low(
    df: pd.DataFrame, thres: float, path: Path
) -> pd.DataFrame | CsvIncongruentValuesError | CsvHighLowError:
    """Check high and low columns and adjust values if required.

    Returns a CsvParsingHighLowError if any row has a high value that is
    lower than the corresponding low value.

    Returns a CsvParsingIncongruentValuesError if the number of rows with
    incongruent high and/or low values, relative to open and/or close
    values, are over a given threshold.

    If number of rows with incongruent values is lower than the threshold
    then returns a copy of the `df` adjusted to ensure congruence.

    Parameters
    ----------
    df
        OHLC DataFrame to be checked.

    thres
        Threshold for number of incongruent rows, as factor of total rows.
        For example, pass as 0.1 to reject a dataframe if more than 10% of
        rows are incongruent.

        NOTE it is not possible to set the threshold for rows where the
        high value is lower than the low value. If the dataframe includes
        such a row an error is always included and no adjustment made.

    path
        Path to csv file represented by `df`.
    """

    if (df.high < df.low).any():
        return CsvHighLowError(path)

    df_copy = df.copy()
    df_copy = base.adjust_high_low(df_copy)
    num_changed_rows = sum(df.ne(df_copy).any(axis=1))
    if num_changed_rows / len(df) > thres:
        return CsvIncongruentValuesError(path, thres)

    return df_copy


def parse_csv(
    root: Path,
    path: Path,
    kwargs: dict[str, Any],
    interval: TDInterval,
    ohlc_thres: float,
) -> pd.DataFrame | list[CsvParsingError]:
    """Parse a csv file to a pd.DataFrame.

    Parameters
    ----------
    root
        Root directory containing csv files and any hierarchy of
        subdirectories containing csv files.

    path
        Path to csv file, relative to `root`.

    kwargs
        kawrgs to pass to `pd.read_csv`.

    interval
        Interval of price data to be parsed.

    ohlc_thres
        Threshold of percentage of rows that can be adjusted to make
        ohlc data congruent rather than rejecting csv file.

    Returns
    -------
    result
        If successful, returns `pd.DataFrame`. Otherwise returns list of
        `CsvParsingError` detailing cause(s) of failure to parse.
    """
    try:
        df = pd.read_csv(root / path, **kwargs)
    except Exception as err:
        return [CsvReadError(path, source=err)]

    df.columns = df.columns.str.lower()
    df.index.name = "date"
    df.dropna(how="all", axis=0, inplace=True)

    errors = []
    # set type of any volume column
    if "volume" in df.columns:
        try:
            df["volume"] = df["volume"].astype(np.float64)
        except Exception as err:
            errors.append(CsvVolDtypeError(path, source=err))
    else:
        df["volume"] = np.nan

    # verify index
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(CsvIndexError(path))
        return errors

    # verify / adjust price data
    if df.empty:
        errors.append(CsvDataframeEmptyError(path))
    else:
        rtrn = check_adjust_high_low(df, ohlc_thres, path)
        if isinstance(rtrn, pd.DataFrame):
            df = rtrn
        else:
            errors.append(rtrn)

    # set tz as required
    if interval.is_daily:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        if not (df.index.normalize() == df.index).all():
            errors.append(CsvIntervalError(path))
    else:  # is intraday
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        elif df.index.tz is not UTC:
            df.index = df.index.tz_convert(UTC)
        if not ((df.index[1:] - df.index[:-1]) == interval).any():
            errors.append(CsvIntervalError(path))

    if errors:
        return errors
    df = helpers.order_cols(df)
    df.sort_index(inplace=True)
    return df


def parse_csvs(
    root: Path,
    paths: dict[str, dict[TDInterval, Path]],
    kwargs: dict[str, Any],
    ohlc_thres: float,
) -> tuple[
    dict[TDInterval, dict[str, pd.DataFrame]],
    list[CsvParsingError | PricesCsvWarning],
]:
    """Parse csv files at given paths.

    Parameters
    ----------
    root : Path
        Root directory containing csv files.

    paths : dict[str, dict[TDInterval, Path]]
        key: symbol
        value: dict[TDInterval, Path]
            key Interval
            value pathlib.Path to csv file containing prices for symbol at interval.

    kwargs
        Keyword arguments to pass to `pd.read_csv`.

    ohlc_thres
        Threshold to reject incongrument ohlc data, in terms of percentage
        of rows.

    Returns
    -------
    2-tuple
        [0]
            dict[TDInterval, dict[str, pd.DataFrame]]
                key: Interval
                value: dict[str, pd.DataFrame]
                    key: symbol
                    value: DataFrame with csv price info
        [1] list[CsvParsingError, PricesCsvWarning]
            List of errors and warnings that occurred when parsing csv files.
    """
    dfs: dict[TDInterval, dict[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]]
    dfs = defaultdict(dict)
    errors: list[CsvParsingError | PricesCsvWarning] = []
    for symb, d in paths.items():
        for interval, path in d.items():
            rtrn = parse_csv(root, path, kwargs, interval, ohlc_thres)
            if isinstance(rtrn, pd.DataFrame):
                dfs[interval][symb] = rtrn
                continue
            errors.extend(rtrn)

    symbols = set(paths.keys())
    to_remove: list[TDInterval] = []
    for interval, d in dfs.items():
        if len(d) < len(symbols):
            missing_symbols = symbols.difference(set(d))
            errors.extend(
                [PricesCsvIntervalUnavailableWarning(interval, missing_symbols)]
            )
            to_remove.append(interval)
    for interval in to_remove:
        del dfs[interval]
    return dfs, errors


def _get_limits_from_parsed(
    parsed: dict[TDInterval, dict[str, pd.DataFrame]],
) -> tuple[dict[TDInterval, pd.Timestamp], dict[TDInterval, pd.Timestamp]]:
    """Get left and right limits by interval from parsed price data.

    Parameters
    ----------
    parsed
        Dictionary as first element returned by `parse_csvs`.

    Returns
    -------
    limits : 2-tuple
        leftmost and rightmost timestamps of data availability for any symbol,
        by interval. For intraday intervals the right limit will be the right
        side of the last indice represented.
    """
    lls: dict[TDInterval, pd.Timestamp] = {}
    rls: dict[TDInterval, pd.Timestamp] = {}
    for intrvl, d in parsed.items():
        ll: None | pd.Timestamp = None
        rl: None | pd.Timestamp = None
        for df in d.values():
            ll_, rl_ = df.index[0], df.index[-1]
            if intrvl.is_intraday:
                rl_ += intrvl
            if ll is None or ll_ < ll:
                ll = ll_
            if rl is None or rl_ > rl:
                rl = rl_
        lls[intrvl] = ll
        rls[intrvl] = rl
    return lls, rls


class PricesCsv(base.PricesBase):
    """Retrieve and serve historic price data sourced from local csv files.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to directory containing .csv files and/or a hierarchy of
        subdirectories containing .csv files. Files and folders should
        conform with requirements detailed here and to the 'Notes' section.

        The constructor will search for .csv files in this directory and
        all directories under it. All files without the .csv extension will
        be ignored.

        Each csv file must contain data for a single symbol and for a
        single interval. The symbol and interval should be included within
        the filename and separated from each other and/or any other parts
        of the filename with a '_' separator. The following are examples
        of valid filenames:
            MSFT_5T.csv
            5T_MSFT.csv
            whatever_MSFT_5T.csv
            MSFT_5T_whatever.csv
            whatever_MSFT_5T_whatever.csv
            whatever_MSFT_whatever_5T_whatever.csv
            whatever_whatever_5T_whatever_MSFT_whatever.csv

        The interval part expresses the duration of the period corresonding
        with each row of data. The interval comprises two parts, a unit and
        a value. Valid units are:
            MIN - to describe mintues
            T - to describe mintues
            H - to describe hours
            D - to describe hours
        Units are not case-sensitive, for example T, t, Min, MIN and mIN
        are all valid units.

        The interval value defines the mulitple of units, for example '5T'
        defines the interval as 5 minutes. If the value is omitted then it
        will be assumed as 1, for example 'MSFT_T.csv' will be assumed to
        contain 1 minute data for MSFT.

        The value for daily data cannot be higher than 1, i.e. there is no
        support for weekly or monthly data. (Whilst the `get` method
        supports requests for weekly or monthly data, the request is
        fulfilled by resampling daily data.)

        The interval can optionally be omitted for daily data, for example
        all of 'MSFT.csv', 'MSFT_1D' and 'MSFT_D' will be assumed as
        containing daily data for MSFT.

        The following are all examples of valid filenames:
            5T_MSFT.csv
            other_5t_MSFT_123.csv
            MSFT_t_231116 MSFT.csv (assumed as minute data)
            MSFT_something_else.csv (assumed as daily data)

        Any files containing malformed intervals will be ignored.

        The following are examples of invalid filenames that will result in
        the file being ignored:
            MSFT_p5T_else.csv (malformed interval)
            MSFT_5T_15T,csv (ambiguous interval)
            MSFT_5T_TSLA.csv (two symbols)
            MSFT_2D.csv (if interval unit is day then value cannot be
                greater than one)
            MSFT.txt (not a .csv file)

        The `csv_paths` property shows all the csv files that have been
        included, by symbol by interval.

    symbols : str | list[str]
        Symbols for which require price data. For example:
            'AMZN'
            'FB AAPL AMZN NFLX GOOG MSFT'
            ['FB', 'AAPL', 'AMZN']

    calendars :
        mptypes.Calendar |
        list[myptypes.Calendar] |
        dict[str, mytypes.Calendar]

        Calendar(s) defining trading times and timezones for `symbols`.

        A single calendar representing all `symbols` can be passed as
        an mptype.Calendar, specifically any of:
            Instance of a subclass of
            `exchange_calendars.ExchangeCalendar`. Calendar 'side' must
            be "left".

            `str` of ISO Code of an exchange for which the
            `exchange_calendars` package maintains a calendar. See
            https://github.com/gerrymanoim/exchange_calendars#calendars
            or call market_prices.get_exchange_info`. For example:
                calendars="XLON",

            `str` of any other calendar name supported by
            `exchange_calendars`, as returned by
            `exchange_calendars.get_calendar_names`

        Multiple calendars, each representing one or more symbols, can
        be passed as any of:
            List of mptypes.Calendar (i.e. defined as for a single
            calendar). List should have same length as `symbols` with each
            element relating to the symbol at the corresponding index.

            Dictionary mapping each symbol with a calendar.
                key: str
                    symbol.
                value: mptypes.Calendar (i.e. as for a single calendar)
                    Calendar corresponding with symbol.

                For example:
                    calendars = {"MSFT": "XNYS", "AZN.L": "XLON"}

        Each Calendar should have a first session no later than the first
        session from which prices are available for any symbol
        corresponding with that calendar.

    lead_symbol : str
        Symbol with calendar that should be used as the default calendar to
        evaluate period from period parameters. If not passed default
        calendar will be defined as the most common calendar (and if there
        is no single most common calendar then the calendar associated
        with the first symbol passed that's associated with one of the most
        common calendars).

    read_csv_kwargs : Optional[dict[str, Any]]
        Keyword argumnets to pass to `pandas.read_csv` to parse a csv file
        to a pandas DataFrame. See the 'Notes' section for how a csv file
        can be formatted such that it parses under the default
        implementation.

        market_prices requires that the DataFrame parses with:
            index as a `pd.DatetimeIndex` named 'date'.

            columns labelled 'open', 'high', 'low', 'close' and optionally
            'volume', each with dtype "float64".

        If the following kwargs are not included to `read_csv_kwargs` then
        by default they will be passed to `pandas.read_csv` with the
        following values:
            "header": 0,
            "usecols": lambda x: x.lower() in [
                "date", "open", "high", "low", "close", "volume"
            ],
            "index_col": "date",
            "parse_dates": ["date"],

        See help(pandas.read_csv) for all available kwargs.

        Note that the following arguments will always be passed by
        market_prices to `pandas.read_csv` with the following values (these
        values cannot be overriden by `read_csv_kwargs`):
            "filepath_or_buffer": <csv file path>
            "dtype": {
                'open': "float64",
                'high': "float64",
                'low': "float64",
                'close': "float64",
            }

        EXAMPLE USAGE
        If in the csv files the:
            date column is labelled 'timestamp'
            close column is labelled 'price'
            volume column is labelled 'vol'
        Then the `names` kwarg can be used to override the labels that
        would otherwise be assigned to each column. If the columns in the
        csv file were ordered 'timestamp', 'price', 'low', 'high', 'open',
        'vol' then `read_csv_kwargs` could be passed as:
            read_csv_kwargs = {
                "names": ['date', 'close', 'low', 'high', 'open', 'volume'],
            }
        This would override the names as defined in the csv file's first
        row with the required values. Note that all references to column
        names in other kwargs, such as 'usecols' and 'dtype', will now
        refer to the overridden names (as required), not the names as
        defined in the csv files.

    ohlc_thres : float, default: 0.08
        Threshold to reject incongruent ohlc data, in terms of maximum
        percentage of incongrument rows to permit. For example, pass as 0.1
        to reject data if more than 10% of rows exhibit incongruent data.

        If the number of incongruent rows are below the threshold then
        adjustements will be made to force congruence.

        A row of data will be considered incongruent if any of:
            close is higher than high
                within threshold, high will be forced to close
            close is lower than low
                within threshold, low will be forced to close
            open is lower than low
                within threshold, open will be forced to low
            open is higher than high
                within threshold, open will be forced to high

        Note: Data will always be rejected if any row has a high value
        lower than the low value. No provision is made to permit this
        circumstance.

    pm_subsession_origin : Literal["open", "break_end"], default: "open"
        How to evaluate indices of sessions that include a break. (The
        'Notes' covers how, in order to offer a complete data set, prices
        are reindexed against an index evaluated in accordance with the
        corresponding calendar. This parameter determines the basis on
        which that index is evaluated for sessions that have a break.)

        'open' - evaluate all indices for a session based on the session
        open. If the session open and pm subsession open are not aligned
        then indices will be included through the break (i.e. treat as if
        the session did not have a break).

        'break_end' - evaluate indices for the am subsession based on the
        session open and indices for the pm subsession based on the
        pm subsession open (i.e. based on the break end). No indices will
        be included that would fall during the break.

    verbose : bool, default: False
        Within error and warning messages concerning the parsing of csv
        files, include the full traceback of any underlying errors.

    Notes
    -----
    By default csv files should have headers in the first line that include
    'date', 'open', 'high', 'low', 'close' and optionally 'volume'. Each
    further line should represent a single period starting on the value in
    'date' column and lasting a period corresponding with the interval
    declared in the filename. It is NOT necessary for every period to be
    represented (it's common for data sources to exclude intraday data for
    periods during which a symbol did not register a trade). The price data
    will be reindexed against expected indices as evaluated from the
    corresponding calendar of `calendars`.

    For daily price data values in the 'date' column should represent a
    date, for example '2023-11-16'.

    For intraday price data the values in the 'date' column should express
    either:
        a date and UTC time, for example '2023-11-16 15:30' or
        '2023-11-16 15:30:00'

        a date, local time and GMT offset for that local time, for example
        '2023-11-16 12:53:00-04:00'

    If the csv does not confrom with the above then the `read_csv_kwargs`
    parameter can be passed to define paramters to pass to
    `pandas.read_csv_kwargs` in order to parse the file as required.

    -- Alignment of Intraday 'date' values --
    Intraday 'date' values can define the time of a (sub)session open
    (according to the corresopnding calendar) and any time thereafter which
    is aligned with the interval, based on the (sub)session open, and which
    falls before the corresonding (sub)session close. See the
    `pm_subsession_origin` parameter for how to determine how indices are
    evaluated for sessions that include a break.

    Examples
    If a session opens at 10:00 and the interval is 15T then
    '2023-11-16 10:00' is a valid value and so is '2023-11-16 10:15',
    although '2023-11-16 10:10' is not as it does not align with the
    declared interval. All unaligned indices will be ignored.

    If the same session closes at 17:00 then the latest valid value for
    that session will be '2023-11-16 16:45'. '2023-11-16 17:00' is not
    valid as it would represent a period outside of trading hours, i.e. the
    15 minutes following the session close. All values that lie outside of
    regular trading hours will be ignored.
    """

    SOURCE_LIVE: bool = False

    @parse
    def __init__(
        self,
        path: Annotated[str | Path, Coerce(Path), Parser(parsing.verify_directory)],
        symbols: str | list[str],
        calendars: mptypes.Calendars,
        lead_symbol: str | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        ohlc_thres: float = 0.08,
        pm_subsession_origin: Literal["open", "break_end"] = "open",
        verbose: bool = False,
    ):
        if typing.TYPE_CHECKING:
            assert isinstance(path, Path)

        self._receieved_kwargs = dict(
            path=path,
            read_csv_kwargs=read_csv_kwargs,
            ohlc_thres=ohlc_thres,
            pm_subsession_origin=pm_subsession_origin,
        )  # for `prices_for_symbols`

        root = path
        self.PM_SUBSESSION_ORIGIN = pm_subsession_origin  # override class attr
        symbols_ = helpers.symbols_to_list(symbols)
        delays = {symb: pd.Timedelta(0) for symb in symbols_}

        csv_kwargs = _get_csv_read_kwargs(read_csv_kwargs)
        paths = self._csv_paths = get_csv_paths(root, symbols_)
        if missing_symbols := set(symbols_).difference(set(paths)):
            raise CsvPathsError(root, missing_symbols, paths)

        paths = copy.deepcopy(paths)
        paths, all_errors_warnings = _remove_unavailable_intervals_from_paths(paths)
        if not paths:
            raise CsvPathsIntervalsError(symbols_, self._csv_paths, all_errors_warnings)

        parsed_data, parsing_errors = parse_csvs(root, paths, csv_kwargs, ohlc_thres)
        all_errors_warnings.extend(parsing_errors)
        if not parsed_data:
            raise CsvNoDataError(symbols_, all_errors_warnings, verbose)

        lls, rls = _get_limits_from_parsed(parsed_data)
        bis_enum = intervals.create_base_intervals_enum(list(parsed_data))
        self._define_base_intervals(bis_enum)
        self._update_base_limits(lls)
        self._update_base_limits_right(rls)

        super().__init__(symbols_, calendars, lead_symbol, delays)

        self._tables, reindexing_warnings = self._compile_tables(parsed_data)
        all_errors_warnings.extend(reindexing_warnings)

        if not self._tables:
            raise CsvNoDataError(symbols_, all_errors_warnings, verbose)

        if all_errors_warnings:
            warnings.warn(
                PricesCsvParsingConsolidatedWarning(all_errors_warnings, verbose)
            )

        # request all available data for all intervals
        for intrvl, pdata in self._pdata.items():
            if intrvl not in self._tables:
                continue  # did not align when compiled
            dr = (None, self.limits[intrvl][1])
            pdata.get_table(dr)
        # delete tables to avoid copies
        del self._tables

    def _compile_tables(
        self, parsed_data: dict[TDInterval, dict[str, pd.DataFrame]]
    ) -> tuple[
        dict[TDInterval, pd.DataFrame],
        list[
            PricesMissingWarning,
            PricesCsvIgnoredUnalignedIndicesWarning,
            PricesCsvIntervalUnavailableWarning,
            CsvReindexedDataframeEmptyError,
        ],
    ]:
        """Compile data frames including all symbols by interval."""
        all_warnings = []
        dfs_by_intrvl: dict[TDInterval, pd.DataFrame] = {}
        intervals_to_remove: dict[TDInterval, list[str]] = defaultdict(list)
        for interval, symbol_dict in parsed_data.items():
            dfs = []
            for symbol, df_symb in symbol_dict.items():
                cal = self.calendars[symbol]
                df_symb_, warnings_ = self._reindex_df(df_symb, symbol, interval, cal)
                all_warnings.extend(warnings_)
                if df_symb_.isna().all(axis=None):
                    intervals_to_remove[interval].append(symbol)
                    continue
                dfs.append(df_symb_)
            if not dfs:
                continue
            df = pd.concat(dfs, axis=1)
            df.sort_index(inplace=True)
            df.columns = df.columns.set_names("symbol", level=0)
            dfs_by_intrvl[interval] = df

        for intrvl, missing_symbols in intervals_to_remove.items():
            all_warnings.extend(
                [PricesCsvIntervalUnavailableWarning(intrvl, missing_symbols)]
            )
            dfs_by_intrvl.pop(intrvl, None)

        return dfs_by_intrvl, all_warnings

    def _reindex_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: TDInterval,
        calendar: ExchangeCalendar,
    ) -> tuple[
        pd.DataFrame,
        list[CsvReindexedDataframeEmptyError]
        | list[PricesCsvIgnoredUnalignedIndicesWarning | PricesMissingWarning],
    ]:
        """Reindex dataframe for a single symbol"""
        source = "csv files"
        start, end = df.index[0], df.index[-1]
        if interval.is_intraday:
            end += interval
        index = self._get_trading_index(calendar, interval, start, end)
        reindex_index = index if interval.is_daily else index.left
        orig_index = df.index
        df = df.reindex(reindex_index)
        if df.isna().all(axis=None):  # no price data post reindexing
            error = CsvReindexedDataframeEmptyError(symbol, interval, calendar)
            return df, [error]
        all_warnings = []
        if not (diff := orig_index.difference(df.index)).empty:
            warning = PricesCsvIgnoredUnalignedIndicesWarning(diff, interval, symbol)
            all_warnings.append(warning)
        if interval.is_intraday:
            df, warnings_ = base.fill_reindexed(df, calendar, interval, symbol, source)
            df.index = index  # set index as interval index
        else:
            df, warnings_ = base.fill_reindexed_daily(
                df, calendar, df.index[0], pd.Timedelta(0), symbol, source
            )
        all_warnings.extend(warnings_)
        df.columns = base.get_columns_multiindex(symbol, df.columns)
        return df, all_warnings

    @property
    def csv_paths(self) -> dict[str, dict[TDInterval, Path]]:
        """Paths to all csv files that have contributed to data.

        Paths are provided by symbol, by interval.
        """
        return self._csv_paths

    def _request_data(
        self,
        interval: intervals.BI,
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
        """
        if not hasattr(self, "_tables"):
            raise NotImplementedError(
                "`PricesCsv._request_data` is not implemented outside of constructor"
            )
        return self._tables[interval]

    def _get_class_instance(self, symbols: list[str], **kwargs) -> "PricesCsv":
        """Return an instance of PricesCsv with same arguments as self.

        Notes
        -----
        If required, subclass should override or extend this method.
        """
        cals = {s: self.calendars[s] for s in symbols}
        if self.lead_symbol_default in symbols:
            kwargs.setdefault("lead_symbol", self.lead_symbol_default)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rtrn = type(self)(
                symbols=symbols, calendars=cals, **self._receieved_kwargs, **kwargs
            )

        return rtrn

    def prices_for_symbols(self, symbols: mptypes.Symbols) -> "PricesCsv":
        """Return instance of prices class for one or more symbols.

        Creates new instance for `symbols` with freshly retrieved price data.

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
        return self._get_class_instance(symbols)
