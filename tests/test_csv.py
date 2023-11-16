"""Tests for market_prices.prices.csv module."""

from __future__ import annotations

import re
from collections import abc
from pathlib import Path

import pytest

from market_prices.intervals import TDInterval
from market_prices.prices import csv as m
from market_prices.prices.csv import ERROR_DAILY_FREQ, ERROR_MALFORMED_FREQ

from .utils import create_temp_file, create_temp_subdir


@pytest.fixture
def symbols() -> abc.Iterator[list[str]]:
    yield ["SYMA", "SYMB", "SYMC"]


def test_contstants():
    """Test module contants."""
    with pytest.raises(ValueError):
        raise m.ERROR_DAILY_FREQ

    with pytest.raises(ValueError):
        raise m.ERROR_MALFORMED_FREQ

    assert m.FREQ_UNITS == {"MIN": "minutes", "T": "minutes", "H": "hours", "D": "days"}


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
        ("n2d", ERROR_MALFORMED_FREQ),
        ("nvd", ERROR_MALFORMED_FREQ),
        ("an2d", None),
        ("nvld", None),
        ("2", None),
        ("123", None),
        ("1t2d3", None),
        ("TT", ERROR_MALFORMED_FREQ),
        ("tt", ERROR_MALFORMED_FREQ),
        ("TMin", ERROR_MALFORMED_FREQ),
        ("Tmin", ERROR_MALFORMED_FREQ),
        ("TD", ERROR_MALFORMED_FREQ),
        ("Ts", ERROR_MALFORMED_FREQ),
        ("ST", ERROR_MALFORMED_FREQ),
        ("T1T", ERROR_MALFORMED_FREQ),
        ("1D1", ERROR_MALFORMED_FREQ),
        ("1daily1", None),
        ("D5", ERROR_DAILY_FREQ),
        ("5d", ERROR_DAILY_FREQ),
    ]
    for inp, out in inputs_outputs:
        assert m._get_csv_interval(inp) is out


def test__get_symbol_from_filename(symbols: list[str]):
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
        assert m._get_symbol_from_filename(inp, symbols) == out


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


def test_get_csv_paths(symbols: list[str], temp_dir):
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

    paths = m.get_csv_paths(temp_dir, symbols)

    def to_path(s: str) -> Path:
        return Path(temp_dir / s)

    def to_subdir_path(s: str) -> Path:
        return Path(temp_subdir / s)

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


def test_get_csv_paths_error0(symbols: list[str], temp_dir):
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
        m.get_csv_paths(temp_dir, symbols)


def test_get_csv_paths_error1(symbols: list[str], temp_dir):
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
        m.get_csv_paths(temp_dir, symbols)
