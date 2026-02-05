"""Tests for market_prices.helpers module."""

from collections.abc import Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp as T
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

import market_prices.helpers as m
from market_prices import intervals
from market_prices.helpers import UTC

from .utils import get_resource


def test_constants():
    # Just to make sure they aren't inadvertently changed
    assert m.UTC is ZoneInfo("UTC")
    assert pd.Timedelta(1, "D") == m.ONE_DAY
    assert pd.Timedelta(1, "min") == m.ONE_MIN
    assert pd.Timedelta(1, "s") == m.ONE_SEC


def test_is_date(one_min):
    f = m.is_date

    assert f(T("2021-11-02"))
    assert f(T("2021-11-02 00:00"))
    assert f(T("2021-11-02 00:00:00.0000000"))
    assert not f(T("2021-11-02 00:00:00.000001"))
    assert not f(T("2021-11-01 23:59:00.999999"))
    assert not f(T("2021-11-02 12:00"))

    minutes = [
        T("2021-11-02", tz=UTC),
        T("2021-11-02", tz=ZoneInfo("US/Eastern")),
        T("2021-11-02", tz=UTC).tz_convert(ZoneInfo("US/Eastern")),
    ]
    for minute in minutes:
        assert not f(minute)
        assert not f(minute + one_min)


def test_fts():
    f = m.fts

    date = pd.Timestamp("2021-11-03")
    assert f(date) == "2021-11-03"

    time = pd.Timestamp("2021-11-03 12:44", tz=UTC)
    assert f(time) == "2021-11-03 12:44 UTC"

    midnight = pd.Timestamp("2021-11-03", tz=UTC)
    assert f(midnight) == "2021-11-03 00:00 UTC"


def test_is_utc():
    f = m.to_utc

    expected = T("2021-11-02", tz=UTC)
    assert f(T("2021-11-02", tz=UTC)) == expected
    assert f(T("2021-11-02")) == expected

    expected = T("2021-11-02 13:33", tz=UTC)
    assert f(T("2021-11-02 13:33")) == expected
    assert f(T("2021-11-02 09:33", tz=ZoneInfo("US/Eastern"))) == expected


def test_is_tz_naive():
    f = m.to_tz_naive
    expected = T("2021-11-02 15:30")
    assert f(T("2021-11-02 15:30")) == expected
    assert f(T("2021-11-02 15:30", tz=UTC)) == expected
    assert f(T("2021-11-02 11:30", tz=ZoneInfo("US/Eastern"))) == expected


def mock_now(mpatch, now: pd.Timestamp):
    """Use `mpatch` to mock pd.Timestamp.now to return `now`."""

    def mock_now_(*_, tz=None, **__) -> pd.Timestamp:
        return pd.Timestamp(now.tz_convert(None), tz=tz)

    mpatch.setattr("pandas.Timestamp.now", mock_now_)


def test_now(monkeypatch):
    """Test `now`.

    Tests at the limits of expected values.
    """
    f = m.now

    interval_intraday = intervals.TDInterval.T10
    intervals_daily = (intervals.TDInterval.D1, intervals.DOInterval.M1)

    # verify for intraday interval

    expected_left = pd.Timestamp("2022-05-01 14:32", tz=UTC)
    expected_right = pd.Timestamp("2022-05-01 14:33", tz=UTC)

    time_now = pd.Timestamp("2022-05-01 14:32", tz=UTC)
    mock_now(monkeypatch, time_now)
    assert f(interval_intraday) == expected_left
    assert f(interval_intraday, "left") == expected_left
    assert f(interval_intraday, "right") == expected_left

    time_now = pd.Timestamp("2022-05-01 14:32:01", tz=UTC)
    mock_now(monkeypatch, time_now)
    assert f(interval_intraday) == expected_left
    assert f(interval_intraday, "left") == expected_left
    assert f(interval_intraday, "right") == expected_right

    time_now = pd.Timestamp("2022-05-01 14:32:59", tz=UTC)
    mock_now(monkeypatch, time_now)
    assert f(interval_intraday) == expected_left
    assert f(interval_intraday, "left") == expected_left
    assert f(interval_intraday, "right") == expected_right

    # verify for daily intervals

    expected_left = pd.Timestamp("2022-05-01")
    expected_right = pd.Timestamp("2022-05-02")

    time_now = pd.Timestamp("2022-05-01", tz=UTC)
    mock_now(monkeypatch, time_now)
    for interval in intervals_daily:
        assert f(interval) == expected_left
        assert f(interval, "left") == expected_left
        assert f(interval, "right") == expected_left

    time_now = pd.Timestamp("2022-05-01 00:00:01", tz=UTC)
    mock_now(monkeypatch, time_now)
    for interval in intervals_daily:
        assert f(interval) == expected_left
        assert f(interval, "left") == expected_left
        assert f(interval, "right") == expected_right

    time_now = pd.Timestamp("2022-05-01 23:59:59", tz=UTC)
    mock_now(monkeypatch, time_now)
    for interval in intervals_daily:
        assert f(interval) == expected_left
        assert f(interval, "left") == expected_left
        assert f(interval, "right") == expected_right


@pytest.fixture
def intraday_pt() -> Iterator[pd.DataFrame]:
    """Intraday price table for test symbols.

    Recreate table with:
        See as defined on `test_pt` module.
    """
    yield get_resource("intraday_pt")


@pytest.fixture
def intraday_pt_ss() -> Iterator[pd.DataFrame]:
    """As intraday_pt for single symbol "MSFT"."""
    yield get_resource("intraday_pt_ss")


def test_has_symbols(intraday_pt, intraday_pt_ss):
    """Test `has_symbols`."""
    f = m.has_symbols

    df = intraday_pt
    assert f(df)

    # verify False when symbol level not called 'symbol'
    df_ = intraday_pt.copy()
    df_.columns.names = ["other_name", ""]
    assert not f(df_)

    # verify False when symbol level not included
    assert not f(intraday_pt_ss)


def test_agg_funcs_constant():
    """Verify integrity of `AGG_FUNCS` constant."""
    assert m.AGG_FUNCS == {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }


def test_order_cols(intraday_pt_ss, intraday_pt):
    """Test `order_cols`."""
    f = m.order_cols

    col_names = "volume", "low", "open", "close", "high"
    columns = {col_name: intraday_pt_ss[col_name] for col_name in col_names}
    df = pd.DataFrame(columns)

    expected_index = pd.Index(["open", "high", "low", "close", "volume"])
    # verify df does not have columns as required
    with pytest.raises(AssertionError):
        assert_index_equal(df.columns, expected_index)

    rtrn = f(df)
    assert_index_equal(rtrn.columns, expected_index)
    # verify column data as original
    for col in rtrn:
        assert_series_equal(rtrn[col], df[col])

    # verify raises error if not passed table that has columns index as pd.Index
    match = "Columns of `df` must be indexed with pd.Index, not a pd.MultiIndex."
    with pytest.raises(ValueError, match=match):
        f(intraday_pt)


def test_agg_funcs(intraday_pt, intraday_pt_ss):
    """Test `agg_funcs`."""
    f = m.agg_funcs

    expected = {
        ("MSFT", "open"): "first",
        ("MSFT", "high"): "max",
        ("MSFT", "low"): "min",
        ("MSFT", "close"): "last",
        ("MSFT", "volume"): "sum",
        ("AZN.L", "open"): "first",
        ("AZN.L", "high"): "max",
        ("AZN.L", "low"): "min",
        ("AZN.L", "close"): "last",
        ("AZN.L", "volume"): "sum",
        ("ES=F", "open"): "first",
        ("ES=F", "high"): "max",
        ("ES=F", "low"): "min",
        ("ES=F", "close"): "last",
        ("ES=F", "volume"): "sum",
    }
    assert f(intraday_pt) == expected

    expected = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    assert f(intraday_pt_ss) == expected


def test_volume_to_na(intraday_pt, intraday_pt_ss):
    """Test `agg_funcs`."""
    f = m.volume_to_na

    # verify for table with symbols
    assert intraday_pt.pt.has_symbols
    # set up frame with first 5 close values NaN for one symbol
    df = intraday_pt.iloc[:10].copy()
    bv = df.index.left < df.index[5].left
    df.loc[bv, ("MSFT", "close")] = np.nan

    rtrn = f(df)

    # make expected volume column
    vol_col_key = ("MSFT", "volume")
    vol_col = df[vol_col_key].copy()
    vol_col.iloc[:5] = np.nan

    assert_series_equal(rtrn[vol_col_key], vol_col)

    # verify rest of frame unchanged
    del rtrn[vol_col_key]
    del df[vol_col_key]
    assert_frame_equal(rtrn, df)

    # repeat for table without symbol level
    assert not intraday_pt_ss.pt.has_symbols
    # set up frame with first 5 close values NaN for one symbol
    df = intraday_pt_ss.iloc[:10].copy()
    bv = df.index.left < df.index[5].left
    df.loc[bv, "close"] = np.nan

    rtrn = f(df)

    # make expected volume column
    vol_col_key = "volume"
    vol_col = df[vol_col_key].copy()
    vol_col.iloc[:5] = np.nan

    assert_series_equal(rtrn[vol_col_key], vol_col)

    # verify rest of frame unchanged
    del rtrn[vol_col_key]
    del df[vol_col_key]
    assert_frame_equal(rtrn, df)


def test_resample(intraday_pt):
    """Test `resample`.

    Single simple test here. `resample` is more comprehensively tested via
    testing of clients.
    """
    f = m.resample
    symbols = intraday_pt.pt.symbols
    df = intraday_pt.pt.utc

    df = df.loc["2021-12-17":"2021-12-20"].copy()
    rtrn = f(df.pt.indexed_left, "1h")

    # create expected return
    groups = []
    for i in range(len(df) // 12):
        groups.extend([i] * 12)
    grouper = pd.Series(groups, index=df.pt.indexed_left.index)
    grouped = df.pt.indexed_left.groupby(grouper)
    cols = {}
    for s in symbols:
        key = (s, "high")
        cols[key] = grouped.max()[(key)]
        key = (s, "low")
        cols[key] = grouped.min()[(key)]
        key = (s, "open")
        cols[key] = grouped.first()[(key)]

        key = (s, "close")
        col_close = grouped.last()[(key)]
        cols[key] = col_close

        key = (s, "volume")
        col_volume = grouped.sum()[(key)]
        col_volume.loc[col_close.isna()] = np.nan
        cols[key] = col_volume

    expected_ooo = pd.DataFrame(cols)
    expected_ooo.index = df[::12].pt.indexed_left.index
    expected = expected_ooo[list(df.columns)]
    expected.columns.names = ("symbol", "")

    assert_frame_equal(rtrn.dropna(axis=0, how="all"), expected)

    # add rows with missing values to expected
    index = pd.date_range(
        start=df.pt.first_ts,
        end=df.pt.last_ts - pd.Timedelta(1, "h"),
        freq="1h",
        name="left",
    )
    expected = expected.reindex(index)
    assert_frame_equal(rtrn, expected)
