"""Tests for market_prices.prices.csv module."""

from __future__ import annotations

import copy
import re
from collections import abc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_prices.intervals import TDInterval
from market_prices.prices import csv as m
from market_prices.prices.csv import ERROR_DAILY_INTRVL, ERROR_MALFORMED_INTRVL

from .utils import (
    create_temp_file,
    create_temp_subdir,
    RESOURCES_PATH,
    get_resource_pbt,
)


@pytest.fixture
def csv_dir() -> abc.Iterator[Path]:
    """resources/csv directory"""
    path = RESOURCES_PATH / "csv"
    assert path.is_dir()
    yield path


@pytest.fixture
def symbols_fict() -> abc.Iterator[list[str]]:
    """Fictitious symbols, not found in any permanent test resource files."""
    yield ["SYMA", "SYMB", "SYMC"]


@pytest.fixture
def symbols() -> abc.Iterator[list[str]]:
    """Selection of symbols represented in resources csv files."""
    yield ["MSFT", "AZN.L", "9988.HK"]


@pytest.fixture
def calendars(symbols) -> abc.Iterator[dict[str, str]]:
    """Mapping of `symbols` to calendar names."""
    calendars = {
        "MSFT": "XNYS",
        "AZN.L": "XLON",
        "9988.HK": "XHKG",
    }
    assert set(symbols) == set(calendars)
    yield calendars


@pytest.fixture
def columns() -> abc.Iterator[list[str]]:
    yield ["open", "high", "low", "close", "volume"]


def test_contstants(columns):
    """Test module contants."""
    with pytest.raises(ValueError):
        raise m.ERROR_DAILY_INTRVL

    with pytest.raises(ValueError):
        raise m.ERROR_MALFORMED_INTRVL

    assert m.INTRVL_UNITS == {
        "MIN": "minutes",
        "T": "minutes",
        "H": "hours",
        "D": "days",
    }

    assert isinstance(m.CSV_READ_DFLT_KWARGS, dict)
    for col in ["date"] + columns:
        assert m.CSV_READ_DFLT_KWARGS["usecols"](col)
        assert m.CSV_READ_DFLT_KWARGS["usecols"](col.upper())
    assert not m.CSV_READ_DFLT_KWARGS["usecols"]("dates")

    assert isinstance(m.CSV_READ_DTYPE_VALUE, dict)


def test__get_csv_interval():
    inputs_outputs = [
        # valid inputs
        ("T", TDInterval.T1),
        ("MIN", TDInterval.T1),
        ("H", TDInterval.H1),
        ("D", TDInterval.D1),
        ("t", TDInterval.T1),
        ("min", TDInterval.T1),
        ("h", TDInterval.H1),
        ("d", TDInterval.D1),
        ("1d", TDInterval.D1),
        ("mIN", TDInterval.T1),
        ("T2", TDInterval.T2),
        ("2T", TDInterval.T2),
        ("MIN3", TDInterval.T3),
        ("3MIN", TDInterval.T3),
        ("h7", TDInterval.H7),
        ("7H", TDInterval.H7),
        # invalid input
        ("notvalid", None),
        ("NOTVALID", None),
        ("notvalidi", None),
        ("NOTVALIDI", None),
        ("n2d", ERROR_MALFORMED_INTRVL),
        ("nvd", ERROR_MALFORMED_INTRVL),
        ("an2d", None),
        ("nvld", None),
        ("2", None),
        ("123", None),
        ("1t2d3", None),
        ("TT", ERROR_MALFORMED_INTRVL),
        ("tt", ERROR_MALFORMED_INTRVL),
        ("TMin", ERROR_MALFORMED_INTRVL),
        ("Tmin", ERROR_MALFORMED_INTRVL),
        ("TD", ERROR_MALFORMED_INTRVL),
        ("Ts", ERROR_MALFORMED_INTRVL),
        ("ST", ERROR_MALFORMED_INTRVL),
        ("T1T", ERROR_MALFORMED_INTRVL),
        ("1D1", ERROR_MALFORMED_INTRVL),
        ("1daily1", None),
        ("D5", ERROR_DAILY_INTRVL),
        ("5d", ERROR_DAILY_INTRVL),
    ]
    for inp, out in inputs_outputs:
        assert m._get_csv_interval(inp) is out


def test__get_symbol_from_filename(symbols_fict: list[str]):
    in_out = [
        # valid
        ("SYMA_a12_3what_s4o_ever5", "SYMA"),
        ("a12_3what_s4o_SYMA_ever5", "SYMA"),
        ("a12_SYMA_s4o_SYMA_ever5", "SYMA"),
        # invalid
        ("a12_3what_s4o_SYM_ever5", None),
        ("SYM_a12_3what_s4o_ever5", None),
        ("SYMD_a12_3what_s4o_ever5", None),
        ("a12_SYMA_s4o_SYMB_ever5", None),
    ]

    for inp, out in in_out:
        assert m._get_symbol_from_filename(inp, symbols_fict) == out


def test__get_interval_from_filename():
    in_out = [
        # valid
        ("SYM_T", TDInterval.T1),
        ("SYM_2t_semething_else", TDInterval.T2),
        ("5T_SYM_5T", TDInterval.T5),
        ("some_15T_thing_SYM_else", TDInterval.T15),
        ("SYM_D", TDInterval.D1),
        ("2T_what_SYM_2t", TDInterval.T2),
        # assumed valid with no freq info here...
        ("what_SYM_ever", TDInterval.D1),
        ("SYM_nofreq", TDInterval.D1),
        ("SYM_nofreqd", TDInterval.D1),
        ("SYM_an2d", TDInterval.D1),
        ("SYM_nvld", TDInterval.D1),
        ("SYM_2", TDInterval.D1),
        ("SYM_123", TDInterval.D1),
        ("SYM_1t2d3", TDInterval.D1),
        ("SYM_1daily1", TDInterval.D1),
        # invalid, either no freq info or assumed freq info malformed
        ("o5T_SYM", None),
        ("SYM_5T5", None),
        ("SYM_5T5_whatever", None),
        ("T_SYM_2T", None),
        ("5T_what_SYM_15T", None),
        ("SYM_n2d", None),
        ("SYM_nvd", None),
        ("SYM_TT", None),
        ("SYM_tt", None),
        ("SYM_TMin", None),
        ("SYM_Tmin", None),
        ("SYM_TD", None),
        ("SYM_Ts", None),
        ("SYM_ST", None),
        ("SYM_T1T", None),
        ("SYM_1D1", None),
        ("SYM_D5", None),
        ("SYM_5d", None),
    ]
    for inp, out in in_out:
        assert m._get_interval_from_filename(inp) == out


def test_get_csv_paths_0(symbols_fict: list[str], temp_dir):
    """Test returns expected paths only."""
    # will not raise errors
    filenames = [
        # should include
        "SYMA_T.csv",
        "SYMA_2t_semething_else.csv",
        "5T_SYMA_5T.csv",
        "some_15T_thing_SYMA_else.csv",
        "SYMA_D.csv",
        "what_SYMB_ever.csv",
        # should ignore
        "SYMANOT_T.csv",
        "SYMANOT_2t_something_else.csv",
        "5T_NOTSYMA.csv",
        "o5T_SYMB.csv",
        "15T_SYMB_SYMC.csv",
        "T_SYMC_2T.csv",
        "Dal_SYMC.csv",
    ]
    for filename in filenames:
        create_temp_file(filename)

    temp_subdir = create_temp_subdir("temp_subdir")
    subdir_filenames = [
        # should include
        "SYMB_T.csv",
        "2T_what_SYMB_2t.csv",
        # should ignore
        "SYMBNOT_T.csv",
        "DD_SYMC.csv",
        "15T_SYMD_what_SYMD_what.csv",
        "5T_what_SYMB_15T.csv",
        "5T_SYMB_what_SYMC.csv",
    ]
    for filename in subdir_filenames:
        create_temp_file(filename, temp_subdir)

    paths = m.get_csv_paths(temp_dir, symbols_fict)

    def to_path(s: str) -> Path:
        return Path(s)

    def to_subdir_path(s: str) -> Path:
        return Path(temp_subdir.relative_to(temp_dir) / s)

    expected = {
        "SYMA": {
            TDInterval.T1: to_path("SYMA_T.csv"),
            TDInterval.T2: to_path("SYMA_2t_semething_else.csv"),
            TDInterval.T5: to_path("5T_SYMA_5T.csv"),
            TDInterval.T15: to_path("some_15T_thing_SYMA_else.csv"),
            TDInterval.D1: to_path("SYMA_D.csv"),
        },
        "SYMB": {
            TDInterval.T1: to_subdir_path("SYMB_T.csv"),
            TDInterval.T2: to_subdir_path("2T_what_SYMB_2t.csv"),
            TDInterval.D1: to_path("what_SYMB_ever.csv"),
        },
    }

    assert paths == expected


@pytest.fixture
def csv_dir_paths(csv_dir) -> abc.Iterator[dict[str, dict[TDInterval, Path]]]:
    """Paths to test csv files in csv directory, by symbol, by interval"""
    yield {
        "AZN.L": {
            TDInterval.D1: csv_dir / "AZN.L_D1.csv",
            TDInterval.T1: csv_dir / "AZN.L_T1.csv",
            TDInterval.T2: csv_dir / "AZN.L_T2.csv",
            TDInterval.T5: csv_dir / "AZN.L_T5.csv",
            TDInterval.H1: csv_dir / "f_AZN.L_H1_fails_on_vol_dtype.csv",
            TDInterval.T20: csv_dir / "f_AZN.L_T20_fails_on_read_csv.csv",
        },
        "9988.HK": {
            TDInterval.D1: csv_dir / "9988.HK_D1.csv",
            TDInterval.T1: csv_dir / "9988.HK_T1.csv",
            TDInterval.T2: csv_dir / "9988.HK_T2.csv",
            TDInterval.T5: csv_dir / "9988.HK_T5.csv",
            TDInterval.H1: csv_dir / "9988.HK_H1.csv",
            TDInterval.T20: csv_dir / "f_9988.HK_T20_fails_on_ohlc_data.csv",
        },
        "MSFT": {
            TDInterval.D1: csv_dir / "MSFT_D1.csv",
            TDInterval.T1: csv_dir / "MSFT_T1.csv",
            TDInterval.T2: csv_dir / "MSFT_T2.csv",
            TDInterval.T5: csv_dir / "MSFT_T5_with_added_indice.csv",
            TDInterval.H1: csv_dir / "f_MSFT_H1_fails_on_no_data.csv",
            TDInterval.T20: csv_dir / "f_MSFT_T20_fails_on_high_low.csv",
        },
    }


def test__remove_unavailable_intervals_from_paths(csv_dir_paths):
    del csv_dir_paths["AZN.L"][TDInterval.T5]
    del csv_dir_paths["9988.HK"][TDInterval.T20]

    paths, warnings_ = m._remove_unavailable_intervals_from_paths(csv_dir_paths)

    expected_intervals = {TDInterval.D1, TDInterval.T1, TDInterval.T2, TDInterval.H1}
    for symbol in list(paths):
        assert set(paths[symbol]) == expected_intervals

    assert len(warnings_) == 2
    match = (
        "Prices are not available at base interval 0:20:00 as data was not"
        " found at this interval for symbols '['9988.HK']'."
    )
    assert str(warnings_[0]) == match

    match = (
        "Prices are not available at base interval 0:05:00 as data was not"
        " found at this interval for symbols '['AZN.L']'."
    )
    assert str(warnings_[1]) == match


def test_get_csv_paths_1(csv_dir, symbols, csv_dir_paths):
    """Test returns expected paths from resources csv dir."""
    expected = copy.deepcopy(csv_dir_paths)
    for symb, intrvl_path in csv_dir_paths.items():
        for intrvl, path in intrvl_path.items():
            expected[symb][intrvl] = path.relative_to(csv_dir)
    paths = m.get_csv_paths(csv_dir, symbols)
    assert paths == expected


def test_get_csv_paths_error0(symbols_fict: list[str], temp_dir):
    """Test raises error on duplicate symbol/interval."""
    filenames = [
        # valid
        "SYMA_T.csv",
        "SYMA_2t_semething_else.csv",
        "5T_SYMA_5T.csv",
        # should error on repeating SYMA 1T
        "SYMA_T_SYMA.csv",
    ]
    for filename in filenames:
        create_temp_file(filename)

    match = re.escape(
        "At least two paths have been found with data for symbol 'SYMA' over"
        " interval '0:01:00'. Paths are:\n"
    )
    with pytest.raises(ValueError, match=match):
        m.get_csv_paths(temp_dir, symbols_fict)


def test_get_csv_paths_error1(symbols_fict: list[str], temp_dir):
    """Test raises error for name as single symbol when 1D interval already assigned."""
    filenames = [
        # valid
        "SYMA_T.csv",
        "SYMA_2t_semething_else.csv",
        "5T_SYMA_5T.csv",
        "what_SYMB_ever.csv",
    ]
    for filename in filenames:
        create_temp_file(filename)

    temp_subdir = create_temp_subdir("temp_subdir")
    subdir_filenames = [
        # valid
        "SYMB_T.csv",
        # should error on repeating STMB 1D
        "SYMB.csv",
    ]
    for filename in subdir_filenames:
        create_temp_file(filename, temp_subdir)

    match = re.escape(
        "At least two paths have been found with data for symbol 'SYMB' over"
        " interval '1 day, 0:00:00'. Paths are:\n"
    )
    with pytest.raises(ValueError, match=match):
        m.get_csv_paths(temp_dir, symbols_fict)


def test_check_adjust_high_low():
    f = m.check_adjust_high_low
    cols = pd.Index(["open", "high", "low", "close", "volume"])
    ohlcv = (
        [100.0, 103.0, 98.0, 103.4, 0],  # close higher than high
        [104.0, 109.0, 104.0, 107.0, 0],
        [106.0, 108.0, 104.0, 107.0, 0],
        [108.0, 112.0, 108.0, 112.0, 0],
    )
    index = pd.date_range(
        start=pd.Timestamp("2022-01-01"), freq="D", periods=len(ohlcv)
    )
    df = pd.DataFrame(ohlcv, index=index, columns=cols)

    rtrn = f(df, 0.3, "a path")

    # assert adjusted
    assert rtrn.ne(df).any(axis=None)
    pd.testing.assert_frame_equal(rtrn[1:], df[1:])

    rtrn = f(df, 0.2, "a path")
    assert isinstance(rtrn, m.CsvIncongruentValuesError)

    extra_row = pd.Series([110, 108, 109, 112, 0], index=df.columns)
    df.loc[pd.Timestamp("2022-01-05")] = extra_row

    rtrn = f(df, 0.2, "a path")
    assert isinstance(rtrn, m.CsvHighLowError)


@pytest.fixture
def csv_read_kwargs() -> abc.Iterator[dict[str, Any]]:
    """Default `read_csv_kwargs` value."""
    yield {
        "header": 0,
        "usecols": m.CSV_READ_DFLT_KWARGS["usecols"],
        "index_col": "date",
        "parse_dates": ["date"],
        "dtype": {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
        },
    }


def test__get_csv_read_kwargs(csv_read_kwargs):
    rtrn_naked = m._get_csv_read_kwargs(None)
    assert rtrn_naked == csv_read_kwargs

    names = ["date", "close", "low", "high", "open", "volume"]
    client_kwargs = {
        "header": 3,
        "names": names,
    }
    rtrn = m._get_csv_read_kwargs(client_kwargs)
    expected = {
        "names": names,
        "header": 3,
        "usecols": m.CSV_READ_DFLT_KWARGS["usecols"],
        "index_col": "date",
        "parse_dates": ["date"],
        "dtype": {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
        },
    }
    assert rtrn == expected
    # ensure returning a new dictionary
    assert rtrn is not expected
    assert rtrn is not m.CSV_READ_DFLT_KWARGS


def test_parse_csv_valid(csv_dir, csv_read_kwargs, utc):
    """Test valid inputs to parse_csv"""

    # test valid intraday input
    path = "MSFT_T5_with_added_indice.csv"
    interval = TDInterval.T5
    kwargs = csv_read_kwargs
    df = m.parse_csv(csv_dir, path, kwargs, interval, 0.0)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is utc
    # assert difference between most indices are as intervall
    assert ((df.index[1:] - df.index[:-1]) == interval).sum() > (len(df) * 0.9)
    assert df.columns.to_list() == ["open", "high", "low", "close", "volume"]
    for col in df:
        assert df[col].dtype == np.float64

    # test valid intraday input
    path = "MSFT_D1.csv"
    interval = TDInterval.D1
    df = m.parse_csv(csv_dir, path, kwargs, interval, 0.0)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert (df.index == df.index.normalize()).all()
    assert df.columns.to_list() == ["open", "high", "low", "close", "volume"]
    for col in df:
        assert df[col].dtype == np.float64


def test_parse_csv_invalid(csv_dir, csv_read_kwargs):
    """Test invalid inputs to parse_csv

    NOTE many of the failing resource files have T5 data which doesn't
    reflect their filename. Filenames ficticious to ensure rejected by
    other tests.
    """
    path = "f_AZN.L_T20_fails_on_read_csv.csv"
    kwargs = csv_read_kwargs
    interval = TDInterval.T5
    rtrn = m.parse_csv(csv_dir, path, kwargs, interval, 0.0)
    assert len(rtrn) == 1
    assert isinstance(rtrn[0], m.CsvReadError)

    path = "f_MSFT_H1_fails_on_no_data.csv"
    rtrn = m.parse_csv(csv_dir, path, kwargs, interval, 0.0)
    assert rtrn
    assert isinstance(rtrn[0], m.CsvDataframeEmptyError)

    path = "f_MSFT_T20_fails_on_high_low.csv"
    rtrn = m.parse_csv(csv_dir, path, kwargs, interval, 1.0)
    assert len(rtrn) == 1
    assert isinstance(rtrn[0], m.CsvHighLowError)

    path = "f_9988.HK_T20_fails_on_ohlc_data.csv"
    rtrn = m.parse_csv(csv_dir, path, kwargs, interval, 0.02)
    assert len(rtrn) == 1
    assert isinstance(rtrn[0], m.CsvIncongruentValuesError)

    path = "f_AZN.L_H1_fails_on_vol_dtype.csv"
    rtrn = m.parse_csv(csv_dir, path, kwargs, TDInterval.T5, 0.0)
    assert len(rtrn) == 1
    assert isinstance(rtrn[0], m.CsvVolDtypeError)

    rtrn = m.parse_csv(csv_dir, path, kwargs, TDInterval.T15, 0.0)
    assert len(rtrn) == 2
    assert isinstance(rtrn[0], m.CsvVolDtypeError)
    assert isinstance(rtrn[1], m.CsvIntervalError)

    rtrn = m.parse_csv(csv_dir, path, kwargs, TDInterval.D1, 0.0)
    assert len(rtrn) == 2
    assert isinstance(rtrn[0], m.CsvVolDtypeError)
    assert isinstance(rtrn[1], m.CsvIntervalError)

    kwargs["index_col"] = None

    rtrn = m.parse_csv(csv_dir, path, kwargs, interval, 0.0)
    assert len(rtrn) == 2
    assert isinstance(rtrn[0], m.CsvVolDtypeError)
    assert isinstance(rtrn[1], m.CsvIndexError)


def test_parse_csvs(csv_dir, csv_dir_paths, csv_read_kwargs, symbols):
    parsed_dict, errors = m.parse_csvs(csv_dir, csv_dir_paths, csv_read_kwargs, 0.02)
    for symb in symbols:
        # verify that H1 interval present although not in results
        # will be rejected as AZN.L H1 and MSFT H1 fail parsing
        assert TDInterval.H1 in csv_dir_paths[symb]
    expected_intervals = [TDInterval.T1, TDInterval.T2, TDInterval.T5, TDInterval.D1]
    assert set(parsed_dict.keys()) == set(expected_intervals)

    for interval in expected_intervals:
        intrvl_dict = parsed_dict[interval]
        assert set(intrvl_dict) == set(symbols)
        for symb in symbols:
            df = intrvl_dict[symb]
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 100

    expected_errors = (
        (m.CsvVolDtypeError, "f_AZN.L_H1_fails_on_vol_dtype.csv"),
        (m.CsvIntervalError, "f_AZN.L_H1_fails_on_vol_dtype.csv"),
        (m.CsvReadError, "f_AZN.L_T20_fails_on_read_csv.csv"),
        (m.CsvIncongruentValuesError, "f_9988.HK_T20_fails_on_ohlc_data.csv"),
        (m.CsvIntervalError, "f_9988.HK_T20_fails_on_ohlc_data.csv"),
        (m.CsvDataframeEmptyError, "f_MSFT_H1_fails_on_no_data.csv"),
        (m.CsvIntervalError, "f_MSFT_H1_fails_on_no_data.csv"),
        (m.CsvHighLowError, "f_MSFT_T20_fails_on_high_low.csv"),
        (m.CsvIntervalError, "f_MSFT_T20_fails_on_high_low.csv"),
    )

    for error, (type_, filename) in zip(errors, expected_errors):
        assert isinstance(error, type_)
        assert str(error.path).endswith(filename)

    assert len(errors) == len(expected_errors) + 1

    interval_error = errors[-1]
    assert isinstance(interval_error, m.PricesCsvIntervalUnavailableWarning)
    match = (
        "Prices are not available at base interval 1:00:00 as data was not found at"
        " this interval for symbols '['AZN.L', 'MSFT']'."
    )
    assert str(interval_error) == match


def test__get_limits_from_parsed():
    start = start_5t = pd.Timestamp("2023-11-21 09:00", tz="UTC")
    end = end_5t = pd.Timestamp("2023-11-27 16:30", tz="UTC")
    freq = "5T"
    delta = pd.Timedelta("10T")
    index_5t_0 = pd.date_range(start, end - delta, freq=freq)
    index_5t_1 = pd.date_range(start + delta, end - delta, freq=freq)
    index_5t_2 = pd.date_range(start + delta, end, freq=freq)

    start = start_1d = pd.Timestamp("2023-10-21")
    end = end_1d = pd.Timestamp("2023-11-21")
    freq = "1D"
    delta = pd.Timedelta("2D")
    index_1d_0 = pd.date_range(start, end - delta, freq=freq)
    index_1d_1 = pd.date_range(start + delta, end - delta, freq=freq)
    index_1d_2 = pd.date_range(start + delta, end, freq=freq)

    def get_df(index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame({"open": 1}, index=index)

    test_input = {
        TDInterval.T5: {
            "SYMA": get_df(index_5t_0),
            "SYMB": get_df(index_5t_1),
            "SYMC": get_df(index_5t_2),
        },
        TDInterval.D1: {
            "SYMA": get_df(index_1d_0),
            "SYMB": get_df(index_1d_1),
            "SYMC": get_df(index_1d_2),
        },
    }

    rtrn = m._get_limits_from_parsed(test_input)

    expected = (
        {TDInterval.T5: start_5t, TDInterval.D1: start_1d},
        {TDInterval.T5: end_5t + pd.Timedelta("5T"), TDInterval.D1: end_1d},
    )

    assert rtrn == expected


def test_raises_no_paths_error(csv_dir, symbols, calendars):
    symbols_ = ["FICTA", "FICTB"]
    calendars_ = {"FICTA": "XNYS", "FICTB": "XLON"}
    match = re.escape(
        "No csv files for were found at any interval for symbols"
        f" '['FICTA', 'FICTB']'. Searched files in and under the directory {csv_dir}."
        " See the 'path' parameter section of help(PricesCsv) for advices on"
        " how csv files should be named."
    )
    with pytest.raises(m.CsvPathsError, match=match):
        m.PricesCsv(csv_dir, symbols_, calendars_)

    symbols.append("FICTA")
    calendars["FICTA"] = "XNYS"
    match = re.escape(
        "No csv files for were found at any interval for symbols"
        f" '['FICTA']'. Searched files in and under the directory {csv_dir}."
        " See the 'path' parameter section of help(PricesCsv) for advices on"
        " how csv files should be named. Only the following files were found:"
    )
    with pytest.raises(m.CsvPathsError, match=match):
        m.PricesCsv(csv_dir, symbols, calendars)


def test_raises_csv_paths_intervals_error(csv_dir, symbols, calendars):
    symbols.extend(["RAND", "TSLA"])
    calendars["RAND"] = calendars["TSLA"] = "XNYS"

    match = re.escape(
        "The following warnings occurred when evaluating available intervals:\n\n0) Prices are not available at base interval 0:20:00 as data was not found at this interval for symbols '['RAND', 'TSLA']'.\n\n1) Prices are not available at base interval 0:01:00 as data was not found at this interval for symbols '['RAND', 'TSLA']'.\n\n2) Prices are not available at base interval 0:05:00 as data was not found at this interval for symbols '['TSLA']'.\n\n3) Prices are not available at base interval 1 day, 0:00:00 as data was not found at this interval for symbols '['RAND']'.\n\n4) Prices are not available at base interval 1:00:00 as data was not found at this interval for symbols '['RAND', 'TSLA']'.\n\n5) Prices are not available at base interval 0:02:00 as data was not found at this interval for symbols '['RAND', 'TSLA']'.\n\nSee the 'path' parameter and 'Notes' sections of help(PricesCsv) for advices on how csv files should be named and formatted and for use of the `read_csv_kwargs` parameter."
    )
    with pytest.raises(m.CsvPathsIntervalsError, match=match):
        m.PricesCsv(csv_dir, symbols, calendars)


def test_raises_csv_no_data_error_0(csv_dir, symbols, calendars):
    # verify raises before compiling table when no data
    symbols.append("RAND")
    calendars["RAND"] = "XNYS"

    match = re.escape(
        "For symbols '['MSFT', 'AZN.L', '9988.HK', 'RAND']' it was not possible to create a price table for any interval from csv files. The following errors and warnings occurred during parsing:\n\n0) Prices are not available at base interval 0:20:00 as data was not found at this interval for symbols '['RAND']'.\n\n1) Prices are not available at base interval 0:01:00 as data was not found at this interval for symbols '['RAND']'.\n\n2) Prices are not available at base interval 1 day, 0:00:00 as data was not found at this interval for symbols '['RAND']'.\n\n3) Prices are not available at base interval 1:00:00 as data was not found at this interval for symbols '['RAND']'.\n\n4) Prices are not available at base interval 0:02:00 as data was not found at this interval for symbols '['RAND']'.\n\n5) Unable to create dataframe from csv file at 'RAND_T5_fails_on_high_low.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvHighLowError'> At least one row has a high value that is lower than the corresponding low value.\n\n6) Prices are not available at base interval 0:05:00 as data was not found at this interval for symbols '['RAND']'.\n\nSee the 'path' parameter and 'Notes' sections of help(PricesCsv) for advices on how csv files should be named and formatted and for use of the `read_csv_kwargs` parameter."
    )
    with pytest.raises(m.CsvNoDataError, match=match):
        m.PricesCsv(csv_dir, symbols, calendars)


def test_raises_csv_no_data_error(csv_dir, symbols, calendars):
    # verify raises when no data after compiling tables
    symbols.append("MSFTEXTRA")
    # assign non-overlapping calendar so that all indices are rejected when compiling table
    calendars["MSFTEXTRA"] = "XHKG"

    match = re.escape(
        "For symbols '['MSFT', 'AZN.L', '9988.HK', 'MSFTEXTRA']' it was not possible to create a price table for any interval from csv files. The following errors and warnings occurred during parsing:\n\n0) Prices are not available at base interval 0:20:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\n1) Prices are not available at base interval 0:01:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\n2) Prices are not available at base interval 1 day, 0:00:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\n3) Prices are not available at base interval 1:00:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\n4) Prices are not available at base interval 0:02:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\n5) For symbol 'MSFT' at base interval 0:05:00 the csv file included the following indices that are not aligned with the evaluated index and: have therefore been ignored:\nDatetimeIndex(['2022-04-18 16:02:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)\n\n6) For symbol 'MSFTEXTRA with interval '0:05:00' no indice aligned with index evaluated from calendar 'XHKG'.\n\n7) Prices are not available at base interval 0:05:00 as data was not found at this interval for symbols '['MSFTEXTRA']'.\n\nSee the 'path' parameter and 'Notes' sections of help(PricesCsv) for advices on how csv files should be named and formatted and for use of the `read_csv_kwargs` parameter."
    )
    with pytest.raises(m.CsvNoDataError, match=match):
        m.PricesCsv(csv_dir, symbols, calendars)


def test_consolidated_warning(csv_dir, symbols, calendars):
    match = re.escape(
        "Price data has been found for all symbols at a least one interval, however, you may find that not all the expected price data is available. See the `limits` property for available base intervals and the limits between which price data is available at each of these intervals. See the `csv_paths` property for paths to all csv files that were found for the requested symbols. See the 'path' parameter and 'Notes' sections of help(PricesCsv) for advices on how csv files should be named and formatted and for use of the `read_csv_kwargs` parameter.\n\nThe following errors and/or warnings occurred during parsing:\n\n0) Unable to create dataframe from csv file at 'f_9988.HK_T20_fails_on_ohlc_data.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvIntervalError'> Date indices do not reflect the expected interval.\n\n1) Unable to create dataframe from csv file at 'f_AZN.L_H1_fails_on_vol_dtype.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvVolDtypeError'> 'volume' column will not convert to 'float64' dtype.\nThe source error's message was:\n\t<class 'ValueError'>: could not convert string to float: 'not a volume'\n\n2) Unable to create dataframe from csv file at 'f_AZN.L_H1_fails_on_vol_dtype.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvIntervalError'> Date indices do not reflect the expected interval.\n\n3) Unable to create dataframe from csv file at 'f_AZN.L_T20_fails_on_read_csv.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvReadError'> `pd.read_csv` raises error.\nThe source error's message was:\n\t<class 'ValueError'>: could not convert string to float: 'not_digits'\n\n4) Unable to create dataframe from csv file at 'f_MSFT_H1_fails_on_no_data.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvDataframeEmptyError'> No price data parsed from csv file.\n\n5) Unable to create dataframe from csv file at 'f_MSFT_H1_fails_on_no_data.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvIntervalError'> Date indices do not reflect the expected interval.\n\n6) Unable to create dataframe from csv file at 'f_MSFT_T20_fails_on_high_low.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvHighLowError'> At least one row has a high value that is lower than the corresponding low value.\n\n7) Unable to create dataframe from csv file at 'f_MSFT_T20_fails_on_high_low.csv' due to the following error:\n\t<class 'market_prices.prices.csv.CsvIntervalError'> Date indices do not reflect the expected interval.\n\n8) Prices are not available at base interval 1:00:00 as data was not found at this interval for symbols '['AZN.L', 'MSFT']'.\n\n9) For symbol 'MSFT' at base interval 0:05:00 the csv file included the following indices that are not aligned with the evaluated index and: have therefore been ignored:\nDatetimeIndex(['2022-04-18 16:02:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)"
    )
    with pytest.warns(m.PricesCsvParsingConsolidatedWarning, match=match) as warning_:
        m.PricesCsv(csv_dir, symbols, calendars)
    assert len(warning_) == 1
    warning = str(warning_[0].message)

    match = re.escape(
        "Price data has been found for all symbols at a least one interval, however,"
        " you may find that not all the expected price data is available. See the"
        " `limits` property for available base intervals and the limits between which"
        " price data is available at each of these intervals. See the `csv_paths`"
        " property for paths to all csv files that were found for the requested"
        " symbols. See the 'path' parameter and 'Notes' sections of help(PricesCsv)"
        " for advices on how csv files should be named and formatted and for use of the"
        " `read_csv_kwargs` parameter.\n\nThe following errors and/or warnings occurred"
        " during parsing:"
    )
    with pytest.warns(m.PricesCsvParsingConsolidatedWarning, match=match) as warning_v_:
        m.PricesCsv(csv_dir, symbols, calendars, verbose=True)
    assert len(warning_v_) == 1
    warning_v = str(warning_v_[0].message)

    # can't match full string as will include local paths wtihin the traceback.
    assert len(warning_v) > len(warning)  # verbose warning should be longer
    submatch = "The source error's traceback was:\nTraceback (most recent call last):"
    assert warning_v.count(submatch) == 2

    # just check that the first line of each of the errors is repeated
    expected_lines = [
        line
        for line in warning.split("\n")
        if len(line) > 2 and line[0].isdigit() and line[1] == ")"
    ]
    actual_lines = [
        line
        for line in warning_v.split("\n")
        if len(line) > 2 and line[0].isdigit() and line[1] == ")"
    ]
    assert len(expected_lines) == 10
    assert len(actual_lines) == 10
    for expected, actual in zip(expected_lines, actual_lines):
        assert actual == expected


def test_read_csv_kwargs(csv_dir):
    """Test can pass through `read_csv_kwargs`."""

    symbol, calendar = "MSFTALT", "XNYS"
    with pytest.raises(m.CsvNoDataError):
        m.PricesCsv(csv_dir, symbol, calendar)

    kwargs = {
        "names": ["date", "open", "high", "low", "close", "volume"],
    }
    m.PricesCsv(csv_dir, symbol, calendar, read_csv_kwargs=kwargs)


@pytest.fixture
def res_us_lon_hk() -> abc.Iterator[tuple[dict[str, pd.DataFrame], pd.Timestamp]]:
    """PricesBaseTst resource for single equity of three different exchanges.

    Resource for one equity of each of New York, London and Hong Kong.

    Prices tables created via `utils.save_resource_pbt` for prices
    instance:
        PricesYahoo("MSFT, AZN.L, 9988.HK")
    at:
        Timestamp('2022-06-17 15:57:09', tz=ZoneInfo("UTC"))
    """
    yield get_resource_pbt("us_lon_hk")


def test_tables(csv_dir, symbols, calendars, res_us_lon_hk):
    with pytest.warns(m.PricesCsvParsingConsolidatedWarning):
        prices = m.PricesCsv(csv_dir, symbols, calendars)

    for interval, pdata in prices._pdata.items():
        table = pdata._table
        res = res_us_lon_hk[0][interval.as_pdfreq[-1::-1]]  # just reversed freq str
        if interval.is_daily:
            expected = res.loc[table.index[0] : table.index[-1]].dropna(
                how="all", axis=0
            )
        else:
            expected = res.loc[table.index.left[0] : table.index.left[-1]]
        for symbol in symbols:
            pd.testing.assert_frame_equal(table[symbol], expected[symbol])


def test_prices_for_symbol(csv_dir, symbols, calendars):
    """Simple verification."""
    with pytest.warns(m.PricesCsvParsingConsolidatedWarning):
        prices = m.PricesCsv(csv_dir, symbols, calendars)

    assert_frame_equal = pd.testing.assert_frame_equal

    kwargs_daily = dict(days=20)
    daily_df = prices.get(**kwargs_daily)
    kwargs_intraday = dict(minutes=1111, end=prices.limit_right_intraday)
    intraday_df = prices.get(**kwargs_intraday)

    new = prices.prices_for_symbols("MSFT AZN.L")

    daily_df_new = new.get(**kwargs_daily)
    intraday_df_new = new.get(**kwargs_intraday)

    assert_frame_equal(daily_df.drop(columns="9988.HK", level=0), daily_df_new)
    expected = intraday_df.drop(columns="9988.HK", level=0)
    expected = expected.dropna(axis=0, how="all")
    assert_frame_equal(expected, intraday_df_new)
