"""Tests for market_prices.mptypes module.

Tests mptypes for invalid input and expected return.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
import pytest
from valimp import parse, Coerce

from market_prices import mptypes as m

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


def test_pandasfreq():
    @parse
    def mock_func(
        arg: Annotated[str, Coerce(m.PandasFrequency)],
    ) -> m.PandasFrequency:
        return arg

    # verify valid input
    freq = "3h"
    rtrn = mock_func(freq)
    assert rtrn == freq
    # verify mptype property
    offset = rtrn.as_offset  # pylint: disable=no-member
    assert offset == pd.tseries.frequencies.to_offset(freq)

    # verify invalid input
    invalid_freq = "4p"
    match = (
        "PandasFrequency must be a pandas frequency although"
        f" received '{invalid_freq}'."
    )
    with pytest.raises(ValueError, match=match):
        mock_func(invalid_freq)
