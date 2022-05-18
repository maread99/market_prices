"""Tests for market_prices.mptypes module.

Tests mptypes for invalid input and expected return.
"""

from __future__ import annotations

from collections.abc import Callable
import re

import pandas as pd
import pydantic
import pytest
import pytz

from market_prices import mptypes


# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, no-self-use, unused-argument, invalid-name
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway.
#   invalid-name: names in tests not expected to strictly conform with snake_case.

# Any flake8 disabled violations handled via per-file-ignores on .flake8


def test_LeadSymbol():
    class MockCls:
        """Mock class to test mpytypes.LeadSymbol."""

        # pylint: disable=too-few-public-methods

        def _verify_lead_symbol(self, symbol: str):
            if symbol != "MSFT":
                raise ValueError(f"{symbol} not in symbols.")

        @pydantic.validate_arguments
        def mock_func(self, arg: mptypes.LeadSymbol) -> str:
            arg_ = str(arg)
            return arg_

    f = MockCls().mock_func

    # verify valid inpout
    s = "MSFT"
    assert f(s) is s

    # verify type other than str is invalid input
    obj = 3
    match = (
        "arg\n  LeadSymbol takes type <class 'str'> although receieved"
        f" <{obj}> of type <class 'int'>."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        f(3)

    # verify raises error if symbol not valid lead_symbol
    s = "RFT"
    match = f"arg\n  {s} not in symbols."
    with pytest.raises(pydantic.ValidationError, match=match):
        f("RFT")


def assert_valid_timezone(func: Callable, field: str):
    """Assert `func` arg takes input valid for pytz.timezone.

    Asserts valid input returns as would be returned by pytz.timezone.
    Verifies that invalid input for pytz.timezone raises an error.
    """
    # verify valid input
    assert func("UTC") == pytz.UTC
    expected = pytz.timezone("Europe/London")
    assert func("Europe/London") == expected
    assert func(expected) == expected

    # verify raises error if type invalid
    obj = 3
    match = re.escape(
        f"arg\n  {field} can take any type from [<class 'str'>, <class"
        f" 'pytz.tzinfo.BaseTzInfo'>] although receieved <{obj}> of type"
        " <class 'int'>."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        func(obj)


def test_Timezone():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.Timezone) -> pytz.BaseTzInfo:
        assert isinstance(arg, pytz.BaseTzInfo)
        return arg

    assert_valid_timezone(mock_func, "Timezone")


def test_PricesTimezone():
    tz = pytz.timezone("US/Eastern")

    class MockCls:
        """Mock class to test mpytypes.PricesTimezone."""

        @property
        def symbols(self) -> list[str]:
            return ["MSFT"]

        @property
        def timezones(self) -> dict:
            return {"MSFT": tz}

        @pydantic.validate_arguments
        def mock_func(self, arg: mptypes.PricesTimezone) -> pytz.BaseTzInfo:
            assert isinstance(arg, pytz.BaseTzInfo)
            return arg

    f = MockCls().mock_func

    # verify valid input
    assert_valid_timezone(f, "PricesTimezone")

    # verify can take a symbol
    assert f("MSFT") == tz
    # but not any symbol
    with pytest.raises(pytz.UnknownTimeZoneError):
        f("HEY")


def assert_date_input(func: Callable):
    """Assert `func` arg takes a date.

    Asserts `func` arg can take valid single input to pd.Timestamp
    that represents a date.

    Verifies that invalid input to pd.Timestamp raises an error.
    """
    # verify valid input
    expected = pd.Timestamp("2022-03-01")
    assert func("2022-03-01") == expected
    assert func("2022-03") == expected
    assert func(expected) == expected
    assert func(expected.value) == expected

    # verify input has to be valid input to a pd.Timestamp
    obj = [expected]
    match = re.escape(
        f"arg\n  Cannot convert input [{obj}] of type <class 'list'>" " to Timestamp"
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        func(obj)


def assert_time_input(func: Callable):
    """Assert `func` arg takes a time.

    Asserts `func` arg can take valid single input to pd.Timestamp
    that represents a time.

    Verifies that invalid input to pd.Timestamp raises an error.
    """
    # verify valid input
    expected = pd.Timestamp("2022-03-01 00:01")
    assert func("2022-03-01 00:01") == expected
    assert func(expected) == expected
    assert func(expected.value) == expected

    # verify input has to be valid input to a pd.Timestamp
    obj = [expected]
    match = re.escape(
        f"arg\n  Cannot convert input [{obj}] of type <class 'list'>" " to Timestamp"
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        func(obj)


def test_Timestamp():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.Timestamp) -> pd.Timestamp:
        arg_ = pd.Timestamp(arg)  # type: ignore[call-overload]
        return arg_

    assert_date_input(mock_func)
    assert_time_input(mock_func)


def test_TimeTimestamp():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.TimeTimestamp) -> pd.Timestamp:
        arg_ = pd.Timestamp(arg)  # type: ignore[call-overload]
        return arg_

    assert_time_input(mock_func)
    # verify input can be midnight if tz aware
    ts = pd.Timestamp("2022-03-01 00:00", tz=pytz.UTC)
    assert mock_func(ts) == ts

    # verify input can be timezone naive if not midnight
    ts = pd.Timestamp("2022-03-01 00:01")
    assert mock_func(ts) == ts

    # verify input cannot be midnight and timezone naive
    ts = pd.Timestamp("2022-03-01 00:00")
    match = re.escape(
        "arg\n  `arg` must have a time component or be tz-aware,"
        f" although receieved as {ts}. To define arg as midnight pass"
        " as a tz-aware pd.Timestamp. For prices as at a session's"
        " close use .close_at()."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(ts)


def test_DateTimestamp():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.DateTimestamp) -> pd.Timestamp:
        arg_ = pd.Timestamp(arg)  # type: ignore[call-overload]
        return arg_

    assert_date_input(mock_func)

    # verify input cannot be timezone aware
    expected = pd.Timestamp("2022-03-01")
    ts = expected.tz_localize(pytz.UTC)
    match = re.escape(f"arg\n  `arg` must be tz-naive, although receieved as {ts}")
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(ts)

    # verify input cannot have a time component
    obj = "2022-03-01 00:01"
    match = re.escape(
        "arg\n  `arg` can not have a time component, although receieved"
        f" as {pd.Timestamp(obj)}. For an intraday price use .price_at()."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(obj)


def test_PandasFrequency():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.PandasFrequency) -> mptypes.PandasFrequency:
        return arg

    # verify valid input
    freq = "3H"
    rtrn = mock_func(freq)
    assert rtrn == freq
    # verify mptype property
    offset = rtrn.as_offset  # pylint: disable=no-member
    assert offset == pd.tseries.frequencies.to_offset(freq)

    # verify invalid input
    invalid_freq = "4p"
    match = (
        "arg\n  PandasFrequency must be a pandas frequency although"
        f" received '{invalid_freq}'."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(invalid_freq)

    # verify invalid type raises error
    obj = 3
    match = (
        "arg\n  PandasFrequency takes type <class 'str'> although"
        f" receieved <{obj}> of type <class 'int'>."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(obj)


def test_IntervalDatetimeIndex():
    @pydantic.validate_arguments
    def mock_func(arg: mptypes.IntervalDatetimeIndex) -> pd.IntervalIndex:
        return arg

    # verify valid input
    dti = pd.date_range("2021", periods=3, freq="MS")
    interval_index = pd.IntervalIndex.from_arrays(dti, dti)
    assert mock_func(interval_index) is interval_index

    # verify invalid input
    int_index = pd.Index([1, 2, 3])
    invalid_int_index = pd.IntervalIndex.from_arrays(int_index, int_index)
    match = re.escape(
        "arg\n  Parameter must have each side as type pd.DatetimeIndex"
        f" although received left side as '{int_index}'."
    )
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(invalid_int_index)

    # verify invalid types raises error
    match = "Parameter must be passed as an instance of pd.IntervalIndex."
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func("bob")
    with pytest.raises(pydantic.ValidationError, match=match):
        mock_func(3)
