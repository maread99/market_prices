"""Tests for market_prices.utils.pandas_utils module."""

from __future__ import annotations
from collections import abc
import re

import pandas as pd
import pytest
import pytz

import market_prices.utils.pandas_utils as m

# pylint: disable=missing-function-docstring,redefined-outer-name,too-many-public-methods
# pylint: disable=missing-param-doc,missing-any-param-doc,too-many-locals
# pylint: disable=protected-access,unused-argument,no-self-use,too-many-arguments
#   missing-fuction-docstring - doc not required for all tests
#   protected-access not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments,too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name,no-self-use, missing-any-param-doc
#   unused-argument not compatible with pytest fixtures, caught by pylance anyway

# Any flake8 disabled violations handled via per-file-ignores on .flake8


@pytest.fixture
def datetime_index_hourly_freq():
    yield pd.date_range("2021-05-01 12:00", periods=10, freq="1H")


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
    right = datetime_index_hourly_freq + pd.Timedelta(30, "T")
    yield pd.IntervalIndex.from_arrays(datetime_index_hourly_freq, right)


@pytest.fixture
def interval_index_overlapping(
    interval_index_non_overlapping,
) -> abc.Iterator[pd.IntervalIndex]:
    """One indice of interval index fully overlaps following indice."""
    index = interval_index_non_overlapping
    i = 3
    new_indice = pd.Interval(index[i].left, index[i + 1].left + pd.Timedelta(5, "T"))
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
        index[i].left - pd.Timedelta(5, "T"), index[i].right + pd.Timedelta(5, "T")
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
        index[i].right + pd.Timedelta(5, "T"),
        index[i + 1].left - pd.Timedelta(5, "T"),
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
    index = m.interval_index_new_tz(index, pytz.UTC)
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
