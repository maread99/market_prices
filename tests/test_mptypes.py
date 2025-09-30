"""Tests for market_prices.mptypes module.

Tests mptypes for invalid input and expected return.
"""

from typing import Annotated

import pandas as pd
import pytest
from valimp import Coerce, parse

from market_prices import mptypes as m


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
    offset = rtrn.as_offset
    assert offset == pd.tseries.frequencies.to_offset(freq)

    # verify invalid input
    invalid_freq = "4p"
    match = (
        "PandasFrequency must be a pandas frequency although"
        f" received '{invalid_freq}'."
    )
    with pytest.raises(ValueError, match=match):
        mock_func(invalid_freq)
