"""Tests for market_prices.utils.pandas_utils module."""

import re
from collections import abc
from typing import Annotated

import pandas as pd
import pytest
from valimp import Parser, parse

import market_prices.utils.pandas_utils as m
from market_prices.helpers import UTC


@pytest.fixture
def datetime_index_hourly_freq():
    yield pd.date_range("2021-05-01 12:00", periods=10, freq="1h")


@pytest.fixture
def interval():
    yield pd.Interval(
        pd.Timestamp("2021-05-01 12:00"), pd.Timestamp("2021-05-01 14:20")
    )


@pytest.fixture
def interval_index_non_overlapping(
    datetime_index_hourly_freq,
) -> abc.Iterator[pd.IntervalIndex]:
    """1H between indices. Interval 30min. 30min gap from right to next left."""
    right = datetime_index_hourly_freq + pd.Timedelta(30, "min")
    yield pd.IntervalIndex.from_arrays(datetime_index_hourly_freq, right)


@pytest.fixture
def interval_index_overlapping(
    interval_index_non_overlapping,
) -> abc.Iterator[pd.IntervalIndex]:
    """One indice of interval index fully overlaps following indice."""
    index = interval_index_non_overlapping
    i = 3
    new_indice = pd.Interval(index[i].left, index[i + 1].left + pd.Timedelta(5, "min"))
    index = index.insert(i, new_indice)
    index = index.drop(index[i + 1])
    assert index.is_monotonic_increasing
    assert index.is_overlapping
    yield index


@pytest.fixture
def interval_index_fully_overlapping(
    interval_index_non_overlapping,
) -> abc.Iterator[pd.IntervalIndex]:
    """One indice of interval index fully overlaps following indice."""
    index = interval_index_non_overlapping
    i = 3
    indice = pd.Interval(
        index[i].left - pd.Timedelta(5, "min"), index[i].right + pd.Timedelta(5, "min")
    )
    index = index.insert(i, indice)
    assert index.is_overlapping
    yield index


@pytest.fixture
def interval_index_not_monotonoic_increasing(
    interval_index_non_overlapping,
) -> abc.Iterator[pd.IntervalIndex]:
    index = interval_index_non_overlapping
    i = 3
    new_indice = pd.Interval(
        index[i].right + pd.Timedelta(5, "min"),
        index[i + 1].left - pd.Timedelta(5, "min"),
        closed=index.closed,
    )
    index = index.insert(i, new_indice)
    assert not index.is_monotonic_increasing
    yield index


def test_make_non_overlapping(
    interval_index_non_overlapping,
    interval_index_overlapping,
    interval_index_fully_overlapping,
):
    test_method = m.make_non_overlapping
    i = 4  # index of fully overlapped indice of interval_index_fully_overlapping

    # test doesn't alter non-overlapping
    rtrn = test_method(interval_index_non_overlapping)
    expected = interval_index_non_overlapping
    pd.testing.assert_index_equal(rtrn, expected)

    index = pd.IntervalIndex(interval_index_overlapping, closed="both")
    with pytest.raises(ValueError, match="cannot be closed on 'both'"):
        test_method(index)

    index = interval_index_fully_overlapping
    overlapping_indices = index[index == index[i]]
    error_msg = (
        "If `fully_overlapped` is None then index cannot contain an indice"
        " which fully overlaps another. The following indices are fully"
        f" overlapped by an earlier indice:\n{overlapping_indices}\nConsider"
        " passing `fully_overlapped` as 'keep' or 'remove'."
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        test_method(index, fully_overlapped=None)

    # test remove_fully_overlapped
    rtrn = test_method(index, fully_overlapped="remove")
    expected = index.drop(index[i])
    pd.testing.assert_index_equal(rtrn, expected)

    # test non tz_naive index
    index = m.interval_index_new_tz(index, UTC)
    rtrn = test_method(index, fully_overlapped="remove")
    expected = index.drop(index[i])
    pd.testing.assert_index_equal(rtrn, expected)

    # test keep_fully_overlapped
    rtrn = test_method(index, fully_overlapped="keep")
    pd.testing.assert_index_equal(rtrn.drop(rtrn[i - 1]), index.drop(index[i - 1]))
    assert rtrn[i - 1].right == index[i].left


def test_remove_intervals_from_interval_invalid_input(
    interval_index_overlapping, interval_index_not_monotonoic_increasing, interval
):
    with pytest.raises(ValueError, match="must not be overlapping"):
        m.remove_intervals_from_interval(interval, interval_index_overlapping)
    with pytest.raises(ValueError, match="must be monotonically increasing"):
        m.remove_intervals_from_interval(
            interval, interval_index_not_monotonoic_increasing
        )


# ------------------ tests for valimp.Parser functions --------------------


def test_verify_interval_datetime_index():
    @parse
    def mock_func(
        arg: Annotated[pd.IntervalIndex, Parser(m.verify_interval_datetime_index)],
    ) -> pd.IntervalIndex:
        return arg

    # verify valid input
    dti = pd.date_range("2021", periods=3, freq="MS")
    interval_index = pd.IntervalIndex.from_arrays(dti, dti)
    assert mock_func(interval_index) is interval_index

    # verify invalid input
    int_index = pd.Index([1, 2, 3])
    invalid_int_index = pd.IntervalIndex.from_arrays(int_index, int_index)
    match = re.escape(
        "'arg' can only take a pd.IntervalIndex that has each side"
        " as type pd.DatetimeIndex, although received with left side"
        " as type '<class 'pandas.Index'>'."
    )
    with pytest.raises(ValueError, match=match):
        mock_func(invalid_int_index)
