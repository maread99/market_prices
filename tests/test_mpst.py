"""Test market prices hypothesis strategies.

Only complex strategies tested.
"""

import hypothesis as hyp
import pandas as pd
from hypothesis import strategies as hypst

from . import hypstrtgy as m

# pylint: disable=missing-any-param-doc, missing-function-docstring
# pylint: disable=protected-access, invalid-name


@hyp.given(data=hypst.data())
def test_calendar_start_end_sessions(calendars_with_answers, data):
    def checks(start, end, limit, min_dist, max_dist):
        si, ei = ans.sessions.get_loc(start), ans.sessions.get_loc(end)
        assert start >= limit[0]
        assert end <= limit[-1]
        if isinstance(min_dist, int):
            assert ei - si >= min_dist
        else:
            assert end - start >= min_dist
        if isinstance(max_dist, int):
            assert ei - si <= max_dist
        else:
            assert end - start <= max_dist

    cal, ans = calendars_with_answers
    s_limit = ans.sessions[123]
    e_limit = ans.sessions[-123]
    limit = s_limit, e_limit

    min_dist, max_dist = 10, pd.Timedelta(30, "D")
    start, end = data.draw(
        m.calendar_start_end_sessions(cal.name, limit, min_dist, max_dist)
    )
    checks(start, end, limit, min_dist, max_dist)

    min_dist, max_dist = pd.Timedelta(10, "D"), 30
    start, end = data.draw(
        m.calendar_start_end_sessions(cal.name, limit, min_dist, max_dist)
    )
    checks(start, end, limit, min_dist, max_dist)
