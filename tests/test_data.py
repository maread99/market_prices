"""Tests for market_prices.data module."""

from __future__ import annotations

from collections import abc

import exchange_calendars as xcals
import pandas as pd
import pytest

import market_prices.data as m
from market_prices import helpers, intervals, errors
from market_prices.helpers import UTC
from market_prices.intervals import TDInterval
from market_prices.utils import calendar_utils as calutils

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


class MockRequestAdmin:
    """Mock request_data callable and attributes to track requested dataranges."""

    def __init__(
        self,
        calendar: xcals.ExchangeCalendar,
        left_bound: pd.Timestamp | None = None,
        right_bound: pd.Timestamp | None = None,
    ):
        self.calendar = calendar
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.requested_drs: list = []
        self.last_request: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None

    def update_request_info(self, start: pd.Timestamp | None, end: pd.Timestamp | None):
        self.last_request = (start, end)
        self.requested_drs.append(self.last_request)

    def reset_last_request(self):
        self.last_request = None

    def reset(self):
        self.requested_drs.clear()
        self.reset_last_request()

    def mock_request_data(self) -> abc.Callable:
        """Return mock request_data callable."""

        def request_data(
            bi: intervals.BI, start: pd.Timestamp | None, end: pd.Timestamp | None
        ) -> pd.DataFrame:
            """'start' and 'end' must be None, a session or a trading minute."""
            start = start if start is not None else self.left_bound
            start = start if self.left_bound is None else max(start, self.left_bound)

            end = end if end is not None else self.right_bound
            end = end if self.right_bound is None else min(end, self.right_bound)

            start_is_session = start is not None and helpers.is_date(start)
            end_is_session = end is not None and helpers.is_date(end)
            if start is not None and end is not None:
                # verify start and date are both dates or both times
                assert start_is_session == end_is_session
            if start_is_session or end_is_session:
                index = self.calendar.trading_index(start, end, bi)
            else:
                index = self.calendar.trading_index(start, end, bi)
            df = pd.DataFrame(
                dict(open=2, high=4, low=1, close=3, volume=5), index=index
            )
            self.update_request_info(start, end)
            if df.empty:
                params = dict(start=start, end=end)
                raise errors.PricesUnavailableFromSourceError(params, None)
            return df

        return request_data


@pytest.fixture(scope="class")
def mr_admin(xlon_calendar_extended) -> abc.Iterator[MockRequestAdmin]:
    """Return MockRequestAdmin instance."""
    yield MockRequestAdmin(xlon_calendar_extended)


@pytest.fixture(scope="class")
def no_delay() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(0)


def get_delta(ts: pd.Timestamp) -> pd.Timedelta:
    return helpers.ONE_DAY if helpers.is_date(ts) else helpers.ONE_MIN


def assert_empty(data: m.Data):
    """Assert general properties represent no data having been retrieved."""
    assert not data.ranges
    assert data.ranges_index.empty
    assert data.leftmost_requested is None
    assert data.rightmost_requested is None

    assert not data.from_ll
    assert not data.to_rl
    assert not data.has_all_data
    assert not data.has_requested_data
    assert not data.has_data


def assert_has_data(
    data: m.Data,
    left_limit: pd.Timestamp,
    right_limit: pd.Timestamp,
    data_from: pd.Timestamp,
    data_to: pd.Timestamp,
):
    """Assert general properties represent data having been retrieved."""
    assert data.ranges
    assert not data.ranges_index.empty
    assert data.has_requested_data
    assert data.has_data

    assert data.leftmost_requested == max(data_from, left_limit)
    assert data.rightmost_requested == min(data_to, right_limit)

    from_ll = data_from <= left_limit
    to_rl = data_to >= right_limit
    assert data.from_ll is from_ll
    assert data.to_rl is to_rl
    assert data.has_all_data is (from_ll and to_rl)


def assert_ts_not_available(data: m.Data, tss: pd.Timestamp | list[pd.Timestamp]):
    """Test that data not available for each of one or more timestamps."""
    if isinstance(tss, pd.Timestamp):
        tss = [tss]
    for ts in tss:
        assert not data.available(ts)
        assert not data.available_and_requested(ts)
        assert not data.requested(ts)


def assert_rng_available_unknown(data: m.Data, start: pd.Timestamp, end: pd.Timestamp):
    """Test that unknown if data is available over a range."""
    delta = get_delta(start)
    tss = [start, start + delta, end - delta, end]
    for ts in tss:
        assert data.available(ts) is None
        assert not data.available_and_requested(ts)
        assert not data.requested(ts)

    rng = (tss[0], tss[-1])
    assert data.available_range(rng) is None
    assert data.available_any(rng) is None
    assert not data.requested_range(rng)


def assert_rng_available_not_requested(
    data: m.Data, start: pd.Timestamp, end: pd.Timestamp
):
    """Test that data available although has not been requested over a range."""
    delta = get_delta(start)
    tss = [start, start + delta, end - delta, end]
    assert data.available(tss)
    for ts in tss:
        assert not data.requested(ts)
        assert not data.available_and_requested(ts)

    rng = (tss[0], tss[-1])
    assert data.available_range(rng)
    assert data.available_any(rng)
    assert not data.requested_range(rng)


def assert_rng_available_requested(
    data: m.Data, start: pd.Timestamp, end: pd.Timestamp
):
    """Test that data available and has beeen requested over a range."""
    delta = get_delta(start)
    if start == end:
        tss = [start, end]
    else:
        tss = [start, start + delta, end - delta, end]
    for ts in tss:
        assert data.available(ts)
        assert data.available_and_requested(ts)
        assert data.requested(ts)

    rng = (tss[0], tss[-1])
    assert data.available_range(rng)
    assert data.available_any(rng)
    assert data.requested_range(rng)


def verify_available_any(
    data: m.Data,
    left_limit: pd.Timestamp,
    right_limit: pd.Timestamp,
):
    """Tests `available_any`."""
    delta = helpers.ONE_MIN
    assert data.available_any((left_limit, right_limit))
    loll = left_limit - delta
    rorl = right_limit + delta
    assert data.available_any((loll, left_limit))
    assert data.available_any((right_limit, rorl))
    assert not data.available_any((loll - delta, loll))
    assert not data.available_any((rorl, rorl + delta))

    delta = helpers.ONE_DAY
    loll = left_limit - delta
    rorl = right_limit + delta
    assert data.available_any((loll, left_limit))
    assert data.available_any((right_limit, rorl))
    assert not data.available_any((loll - delta, loll))
    assert not data.available_any((rorl, rorl + delta))


@pytest.mark.parametrize("bi", [TDInterval.T1, TDInterval.H1, TDInterval.D1])
def test_pre_requests(
    mr_admin,
    no_delay,
    bi,
    xlon_calendar_with_answers,
    today,
):
    """Test Data properties prior to making any data requests."""
    cal, ans = xlon_calendar_with_answers
    cc = calutils.CompositeCalendar([cal])

    def get_data(delay, left_limit=None, right_limit=None) -> m.Data:
        return m.Data(
            mr_admin.mock_request_data(), cc, bi, delay, left_limit, right_limit
        )

    # default args
    if bi.is_intraday:
        with pytest.raises(
            ValueError, match="`delay` must be passed if `bi` is intraday"
        ):
            m.Data(mr_admin.mock_request_data(), cc, bi)
        delay = no_delay
    else:
        delay = None

    data = get_data(delay=delay)

    assert_empty(data)
    assert data.cc is cc
    assert data.bi == bi

    pool: pd.Series | pd.DatetimeIndex
    if bi.is_intraday:
        pool = ans.first_minutes
        r_edge = pd.Timestamp.now(tz=UTC) + bi
        l_edge = pool.iloc[0]
    else:
        pool = ans.sessions
        r_edge = today
        l_edge = pool[0]

    def get_pool_value(idx: int) -> pd.Timestamp:
        return pool.iloc[idx] if isinstance(pool, pd.Series) else pool[idx]

    delta = get_delta(get_pool_value(0))
    assert data.ll is None
    assert data.rl == r_edge

    assert_rng_available_unknown(data, l_edge, r_edge)
    assert_ts_not_available(data, r_edge + delta)

    # define left_limit, right_limit as default
    left_limit = get_pool_value(-30)
    data = get_data(delay, left_limit)

    assert_empty(data)

    assert data.ll == left_limit
    assert data.rl == r_edge

    assert_rng_available_not_requested(data, left_limit, r_edge)
    assert_ts_not_available(data, [left_limit - delta, r_edge + delta])

    # define left_limit and right_limit
    right_limit = get_pool_value(-5)
    data = get_data(delay, left_limit, right_limit)

    assert_empty(data)

    assert data.ll == left_limit
    assert data.rl == right_limit

    assert_rng_available_not_requested(data, left_limit, right_limit)
    assert_ts_not_available(data, [left_limit - delta, right_limit + delta])
    verify_available_any(data, left_limit, right_limit)


def assert_table_matches(
    table: pd.DataFrame,
    daterange: tuple[pd.Timestamp, pd.Timestamp],
    df: pd.DataFrame,
):
    """Assert `table` matches expected part of `df` over `daterange`."""
    from_, to = daterange
    subset = df.loc[from_:to]
    if isinstance(df.index, pd.IntervalIndex):
        # discard only partially covered indices
        if from_ in subset.index[0] and subset.index[0].left != from_:
            subset = subset.iloc[1:]
        if to in subset.index[-1] and subset.index[-1].right != to:
            subset = subset.iloc[:-1]
    # freq not expected to be consistent for 1D (will be C only if table continuous)
    pd.testing.assert_frame_equal(table, subset, check_freq=False)


def assert_single_daterange(
    data: m.Data,
    daterange: tuple[pd.Timestamp, pd.Timestamp],
    left_limit: pd.Timestamp,
    right_limit: pd.Timestamp,
    mr_admin: MockRequestAdmin,
    df: pd.DataFrame,
):
    """Assert properties reflect a single daterange having been requested."""
    from_, to = daterange
    ranges = [pd.Interval(*daterange, closed="both")]
    assert data.ranges == ranges
    pd.testing.assert_index_equal(data.ranges_index, pd.IntervalIndex(ranges))
    assert_has_data(data, left_limit, right_limit, *daterange)
    assert_rng_available_requested(data, *daterange)
    delta = get_delta(daterange[0])
    if from_ > left_limit:
        assert_rng_available_not_requested(data, left_limit, from_ - delta)
    if to < right_limit:
        assert_rng_available_not_requested(data, to + delta, right_limit)

    # verify not requesting from source
    mr_admin.reset_last_request()
    table = data.get_table(daterange)
    assert table is not None
    assert_table_matches(table, daterange, df)
    assert mr_admin.last_request is None
    delta = delta if delta == helpers.ONE_DAY else pd.Timedelta(30, "min")
    from_ += delta
    to -= delta
    table = data.get_table((from_, to))
    assert table is not None
    assert_table_matches(table, (from_, to), df)
    assert mr_admin.last_request is None


class TestRanges:
    """Tests requested ranges are updated as expected."""

    @pytest.fixture(scope="class")
    def session_ends(
        self, xlon_calendar
    ) -> abc.Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        cal = xlon_calendar
        end_session = cal.session_offset(cal.last_session, -5)
        start_date = end_session - pd.DateOffset(weeks=4)
        start_session = cal.date_to_session(start_date, "next")
        yield start_session, end_session

    @pytest.fixture(scope="class")
    def limits(
        self, xlon_calendar, session_ends
    ) -> abc.Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        cal = xlon_calendar
        left_limit = cal.session_open(session_ends[0])
        right_limit = cal.session_close(session_ends[1])
        yield left_limit, right_limit

    @pytest.fixture(scope="class")
    def rngs(
        self, xlon_calendar, session_ends
    ) -> abc.Iterator[
        list[
            tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]
        ]
    ]:
        cal = xlon_calendar
        start_session, end_session = session_ends
        rngs = []

        # minutes are on-frequency for all tested intraday intervals (T1, H1). This is
        # reasonable as passed dateranges will always have been evaluated via
        # daterange module which returns dateranges as on-frequency timestamps.
        from_session = cal.session_offset(start_session, 3)
        from_minute = cal.session_minutes(from_session)[60]
        to_session = cal.session_offset(from_session, 2)
        to_minute = cal.session_minutes(to_session)[-90]
        rngs.append(((from_minute, to_minute), (from_session, to_session)))

        from_session = cal.session_offset(to_session, 3)
        from_minute = cal.session_open(from_session)
        to_session = cal.session_offset(from_session, 3)
        to_minute = cal.session_minutes(to_session)[-90]
        rngs.append(((from_minute, to_minute), (from_session, to_session)))

        from_session = cal.session_offset(end_session, -5)
        from_minute = cal.session_open(from_session) - pd.Timedelta(70, "min")
        to_session = cal.session_offset(from_session, 2)
        to_minute = cal.session_close(to_session) + pd.Timedelta(70, "min")
        rngs.append(((from_minute, to_minute), (from_session, to_session)))

        yield rngs

    @pytest.fixture(scope="class", params=range(3))
    def rngs_parameterized(
        self, request, rngs
    ) -> abc.Iterator[
        tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]
    ]:
        yield rngs[request.param]

    @pytest.fixture(scope="class")
    def cc(self, xlon_calendar) -> abc.Iterator[calutils.CompositeCalendar]:
        yield calutils.CompositeCalendar([xlon_calendar])

    @pytest.fixture(scope="class")
    def bis(self) -> abc.Iterator[list[TDInterval]]:
        yield [TDInterval.T1, TDInterval.H1, TDInterval.D1]

    @pytest.fixture(scope="class", params=range(3))
    def bis_parameterized(self, request, bis) -> abc.Iterator[TDInterval]:
        yield bis[request.param]

    @pytest.fixture(scope="class")
    def bis_intraday(self) -> abc.Iterator[list[TDInterval]]:
        yield [TDInterval.T1, TDInterval.H1]

    @pytest.fixture(scope="class", params=range(2))
    def bis_intraday_parameterized(
        self, request, bis_intraday
    ) -> abc.Iterator[TDInterval]:
        yield bis_intraday[request.param]

    @pytest.fixture(scope="class")
    def dfs(
        self, mr_admin, limits, session_ends, bis
    ) -> abc.Iterator[dict[TDInterval, pd.DataFrame]]:
        """DataFrames representing all data that could be requeseted, by bi."""
        mock_request = mr_admin.mock_request_data()
        d = {}
        for bi in bis:
            bounds = session_ends if bi.is_one_day else limits
            d[bi] = mock_request(bi, bounds[0], bounds[1] + bi)
        mr_admin.reset()
        yield d

    @pytest.fixture
    def data_df(
        self, mr_admin, no_delay, limits, session_ends, dfs, bis_parameterized, cc
    ) -> abc.Iterator[tuple[m.Data, pd.DataFrame]]:
        mr_admin.reset()
        bi = bis_parameterized
        bounds = session_ends if bi.is_one_day else limits
        data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *bounds)
        df = dfs[bi]
        yield data, df

    def test_each_range(
        self, rngs_parameterized, data_df, mr_admin, limits, session_ends
    ):
        data, df = data_df
        times, sessions = rngs_parameterized
        dr = sessions if data.bi.is_one_day else times
        table = data.get_table(dr)
        assert mr_admin.last_request == dr
        assert_table_matches(table, dr, df)
        bounds = session_ends if data.bi.is_one_day else limits
        assert_single_daterange(data, dr, *bounds, mr_admin, df)

    @pytest.fixture(scope="class")
    def thirty_mins(self) -> abc.Iterator[pd.Timedelta]:
        yield pd.Timedelta(30, "min")

    def test_extending_requested_range(
        self,
        rngs,
        mr_admin,
        no_delay,
        limits,
        dfs,
        xlon_calendar,
        cc,
        thirty_mins,
        bis_intraday_parameterized,
    ):
        cal = xlon_calendar
        dr, (from_session, to_session) = rngs[0]
        from_minute, to_minute = dr

        open_ = cal.session_open(from_session)
        before_open = open_ - thirty_mins
        in_prior_session = cal.previous_close(from_minute) - thirty_mins
        close = cal.session_close(to_session)
        after_close = close + thirty_mins
        in_next_session = cal.next_open(to_minute) + thirty_mins

        bi = bis_intraday_parameterized
        df = dfs[bi]

        def get_data() -> tuple[m.Data, pd.DataFrame]:
            mr_admin.reset()
            data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *limits)
            table = data.get_table(dr)
            mr_admin.reset()
            assert table is not None
            return data, table

        def get_table_and_assert(dr, requested_drs):
            data, table = get_data()
            table = data.get_table(dr)
            assert mr_admin.requested_drs == requested_drs
            assert_table_matches(table, dr, df)
            assert_single_daterange(data, dr, *limits, mr_admin, df)

        # left side
        # extend left side to session open
        get_table_and_assert((open_, to_minute), [(open_, from_minute)])

        # extend left side to before session open
        get_table_and_assert((before_open, to_minute), [(before_open, from_minute)])

        # extend left side into prior session
        requested_drs = [(in_prior_session, from_minute)]
        get_table_and_assert((in_prior_session, to_minute), requested_drs)

        # right side
        # extend right side to session close
        get_table_and_assert((from_minute, close), [(to_minute, close)])

        # extend right side to beyond session close
        get_table_and_assert((from_minute, after_close), [(to_minute, after_close)])

        # extend right side into next session
        requested_drs = [(to_minute, in_next_session)]
        get_table_and_assert((from_minute, in_next_session), requested_drs)

        # both sides
        # extend both sides to session extremes
        requested_drs = [(open_, from_minute), (to_minute, close)]
        get_table_and_assert((open_, close), requested_drs)

        # extend both sides to beyond session extremes
        requested_drs = [(before_open, from_minute), (to_minute, after_close)]
        get_table_and_assert((before_open, after_close), requested_drs)

        # extend both sides into adjoining sessions
        requested_drs = [(in_prior_session, from_minute), (to_minute, in_next_session)]
        get_table_and_assert((in_prior_session, in_next_session), requested_drs)

    def test_extending_requested_range_daily(
        self,
        rngs,
        mr_admin,
        no_delay,
        session_ends,
        dfs,
        xlon_calendar,
        cc,
    ):
        cal = xlon_calendar
        _, dr = rngs[0]
        from_session, to_session = dr
        prior_session = cal.previous_session(from_session)
        next_session = cal.next_session(to_session)

        bi = TDInterval.D1
        df = dfs[bi]

        def get_data() -> tuple[m.Data, pd.DataFrame]:
            mr_admin.reset()
            data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *session_ends)
            table = data.get_table(dr)
            mr_admin.reset()
            assert table is not None
            return data, table

        def get_table_and_assert(dr, requested_drs):
            data, table = get_data()
            table = data.get_table(dr)
            assert mr_admin.requested_drs == requested_drs
            assert_table_matches(table, dr, df)
            assert_single_daterange(data, dr, *session_ends, mr_admin, df)

        # extend left side
        get_table_and_assert(
            (prior_session, to_session), [(prior_session, from_session)]
        )

        # extend right side
        get_table_and_assert((from_session, next_session), [(to_session, next_session)])

        # extend both sides
        requested_drs = [(prior_session, from_session), (to_session, next_session)]
        get_table_and_assert((prior_session, next_session), requested_drs)

    def test_multiple_range(
        self, rngs, mr_admin, limits, session_ends, data_df, one_day, thirty_mins
    ):
        data, df = data_df
        bi_daily = data.bi.is_one_day
        left_limit, right_limit = session_ends if bi_daily else limits

        dr_0 = rngs[0][1] if bi_daily else rngs[0][0]
        dr_2 = rngs[-1][1] if bi_daily else rngs[-1][0]

        _ = data.get_table(dr_0)
        _ = data.get_table(dr_2)

        dr_0_start, dr_0_end = dr_0
        dr_2_start, dr_2_end = dr_2

        ranges = [pd.Interval(*dr_0, closed="both"), pd.Interval(*dr_2, closed="both")]
        assert data.ranges == ranges
        pd.testing.assert_index_equal(data.ranges_index, pd.IntervalIndex(ranges))

        assert data.has_requested_data
        assert data.has_data

        assert data.leftmost_requested == dr_0_start
        assert data.rightmost_requested == dr_2_end

        assert not data.from_ll
        assert not data.to_rl
        assert not data.has_all_data

        delta = get_delta(dr_0[0])
        assert_rng_available_not_requested(data, left_limit, dr_0_start - delta)
        assert_rng_available_requested(data, *dr_0)

        assert_rng_available_not_requested(data, dr_0_end + delta, dr_2_start - delta)
        assert_rng_available_requested(data, *dr_2)
        assert_rng_available_not_requested(data, dr_2_end + delta, right_limit)

        # verify not requesting from source
        for dr in [dr_0, dr_2]:
            mr_admin.reset_last_request()
            table = data.get_table(dr)
            assert_table_matches(table, dr, df)
            assert mr_admin.last_request is None
            from_, to = dr
            delta = delta if delta == one_day else thirty_mins
            from_ += delta
            to -= delta
            table = data.get_table((from_, to))
            assert_table_matches(table, (from_, to), df)
            assert mr_admin.last_request is None

    def test_combine_multiple_ranges(
        self,
        rngs,
        mr_admin,
        limits,
        session_ends,
        no_delay,
        dfs,
        bis_parameterized,
        xlon_calendar,
        thirty_mins,
        one_day,
    ):
        # pylint: disable=too-many-statements
        cal = xlon_calendar
        cc = calutils.CompositeCalendar([cal])
        bi = bis_parameterized
        bi_daily = bi.is_one_day
        df = dfs[bi]
        bounds = session_ends if bi_daily else limits
        left_bound, right_bound = bounds

        dr_0 = rngs[0][1] if bi_daily else rngs[0][0]
        dr_1 = rngs[1][1] if bi_daily else rngs[1][0]
        dr_2 = rngs[2][1] if bi_daily else rngs[2][0]

        dr_0_start, dr_0_end = dr_0
        dr_1_start, dr_1_end = dr_1
        dr_2_start, dr_2_end = dr_2

        if bi_daily:
            of_prev_session = cal.previous_session(dr_0_start)
            of_next_session = cal.next_session(dr_2_end)
        else:
            of_prev_session = cal.previous_close(dr_0_start) - thirty_mins
            of_next_session = cal.next_open(dr_2_end) + (thirty_mins * 3)

        for i in range(2):

            def get_data() -> m.Data:
                mr_admin.reset()
                data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *bounds)
                _ = data.get_table(dr_0)
                _ = data.get_table(dr_2)
                if i:  # pylint: disable=cell-var-from-loop
                    _ = data.get_table(dr_1)
                mr_admin.reset()
                return data

            def overlap_and_assert(dr_to_overlap, requested):
                dr = dr_to_overlap
                data = get_data()  # pylint: disable=cell-var-from-loop
                table = data.get_table(dr)
                assert mr_admin.requested_drs == requested
                assert_table_matches(table, dr, df)
                start = min(dr[0], dr_0_start)
                end = max(dr[1], dr_2_end)
                assert_single_daterange(data, (start, end), *bounds, mr_admin, df)

            if i:
                requested = [(dr_0_end, dr_1_start), (dr_1_end, dr_2_start)]
            else:
                requested = [(dr_0_end, dr_2_start)]

            # overlap at far edge
            overlap_and_assert((dr_0_start, dr_2_end), requested)

            # overlap beyond far edge
            requested_ = (
                [(of_prev_session, dr_0_start)]
                + requested
                + [(dr_2_end, of_next_session)]
            )
            overlap_and_assert((of_prev_session, of_next_session), requested_)

            # overlap at near edge
            overlap_and_assert((dr_0_end, dr_2_start), requested)

            # near edge non-overlapping
            data = get_data()
            delta = get_delta(dr_0_start)
            delta_ = delta * 2 if delta == one_day else delta
            dr = dr_0_end + delta_, dr_2_start - delta_
            table = data.get_table(dr)
            requested_ = [(dr[0], dr_1_start), (dr_1_end, dr[1])] if i else [dr]
            assert mr_admin.requested_drs == requested_
            assert_table_matches(table, dr, df)

            ranges = [
                pd.Interval(*dr_0, closed="both"),
                pd.Interval(*dr, closed="both"),
                pd.Interval(*dr_2, closed="both"),
            ]
            assert data.ranges == ranges

            pd.testing.assert_index_equal(data.ranges_index, pd.IntervalIndex(ranges))

            assert data.has_requested_data
            assert data.has_data

            assert data.leftmost_requested == dr_0_start
            assert data.rightmost_requested == dr_2_end

            assert not data.from_ll
            assert not data.to_rl
            assert not data.has_all_data

            assert_rng_available_not_requested(data, left_bound, dr_0_start - delta)
            assert_rng_available_not_requested(data, dr_2_end + delta, right_bound)

            # near edge non-overlapping daily
            # daily treated differently in that contiguous ranges joined
            if bi_daily:
                data = get_data()
                dr = dr_0_end + one_day, dr_2_start - one_day
                table = data.get_table(dr)
                requested_ = [(dr[0], dr_1_start), (dr_1_end, dr[1])] if i else [dr]
                assert mr_admin.requested_drs == requested_
                assert_table_matches(table, dr, df)
                daterange = (dr_0_start, dr_2_end)
                assert_single_daterange(data, daterange, *bounds, mr_admin, df)

    def test_to_defined_limits(
        self, bis_parameterized, cc, dfs, session_ends, limits, rngs, mr_admin, no_delay
    ):
        bi = bis_parameterized
        bi_daily = bi.is_one_day
        df = dfs[bi]
        bounds = session_ends if bi_daily else limits

        dr_0 = rngs[0]
        dr_0_start, dr_0_end = dr_0[1] if bi_daily else dr_0[0]
        dr_2 = rngs[-1]
        dr_2_start, dr_2_end = dr_2[1] if bi_daily else dr_2[0]

        def get_data() -> m.Data:
            mr_admin.reset()
            data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *bounds)
            return data

        left_bound, right_bound = bounds

        delta = get_delta(dr_0_start)

        froms = [left_bound - (5 * delta), left_bound, left_bound + (2 * delta)]
        to = dr_0_start

        def assert_table_start(from_, table):
            if bi == TDInterval.H1 and from_ == left_bound + (2 * delta):
                return
            table_start = table.index[0] if bi_daily else table.index[0].left
            assert table_start == max(from_, left_bound)

        # first request includes left limit
        for from_ in froms:
            data = get_data()
            dr = from_, to
            table = data.get_table(dr)
            assert_table_start(from_, table)
            assert_has_data(data, *bounds, *dr)

        # second request includes left limit
        for from_ in froms:
            data = get_data()
            rng = dr_0[1] if bi_daily else dr_0[0]
            _ = data.get_table(rng)
            dr = from_, to
            table = data.get_table(dr)
            assert_table_start(from_, table)
            assert_single_daterange(
                data, (max(from_, left_bound), dr_0_end), *bounds, mr_admin, df
            )

        from_ = dr_2_end
        tos = [right_bound + (5 * delta), right_bound, right_bound - (2 * delta)]

        def assert_table_end(to, table):
            if bi == TDInterval.H1:
                return
            table_end = table.index[-1] if bi_daily else table.index[-1].right
            assert table_end == min(to, right_bound)

        # first request includes right limit
        for to in tos:
            data = get_data()
            dr = from_, to
            table = data.get_table(dr)
            assert_table_end(to, table)
            assert_has_data(data, *bounds, *dr)

        # second request includes right limit
        for to in tos:
            data = get_data()
            rng = dr_2[1] if bi_daily else dr_2[0]
            _ = data.get_table(rng)
            dr = from_, to
            table = data.get_table(dr)
            assert_table_end(to, table)
            assert_single_daterange(
                data, (dr_2_start, min(to, right_bound)), *bounds, mr_admin, df
            )

    def test_to_defined_td_ll(
        self,
        xlon_calendar_extended,
        bis_parameterized,
        cc,
        dfs,
        session_ends,
        rngs,
        mr_admin,
        no_delay,
        today,
        one_min,
        monkeypatch,
    ):
        """Test to left limit defined as Timedelta."""
        cal = xlon_calendar_extended
        bi = bis_parameterized
        left_limit = today - session_ends[0]
        bi_daily = bi.is_one_day
        df = dfs[bi]

        dr_0 = rngs[0]
        dr_0_start, dr_0_end = dr_0[1] if bi_daily else dr_0[0]

        def get_data() -> m.Data:
            mr_admin.reset()
            data = m.Data(
                mr_admin.mock_request_data(), cc, bi, no_delay, left_limit=left_limit
            )
            return data

        def set_now(now: pd.Timestamp):
            monkeypatch.setattr(
                "pandas.Timestamp.now", lambda *_, tz=None, **__: now.tz_convert(tz)
            )

        # now, and hence left_bound, set to a time that provides for 1H bi
        # to fall on frequency, save for left_bound + (2*delta)
        now = cal.previous_open(pd.Timestamp.now()) + pd.Timedelta(59, "min")
        set_now(now)
        if bi_daily:
            left_bound = today - left_limit
        else:
            left_bound = pd.Timestamp.now(tz=UTC) - left_limit
            left_bound += one_min  # one_min provided for processing

        data = get_data()
        assert data.ll == left_bound

        def assert_table_start(from_, table):
            if bi == TDInterval.H1 and from_ == left_bound + (2 * delta):
                return
            table_start = table.index[0] if bi_daily else table.index[0].left
            assert table_start == max(from_, left_bound)

        delta = get_delta(dr_0_start)
        to = dr_0_start
        froms = [left_bound - (5 * delta), left_bound, left_bound + (2 * delta)]

        right_bound = today if bi_daily else now
        bounds = left_bound, right_bound
        # first request includes left limit
        for from_ in froms:
            data = get_data()
            dr = from_, to
            table = data.get_table(dr)
            assert_table_start(from_, table)
            assert_has_data(data, *bounds, *dr)

        # second request includes left limit
        for from_ in froms:
            data = get_data()
            rng = dr_0[1] if bi_daily else dr_0[0]
            table = data.get_table(rng)
            dr = from_, to
            table = data.get_table(dr)
            assert_table_start(from_, table)
            assert_single_daterange(
                data, (max(from_, left_bound), dr_0_end), *bounds, mr_admin, df
            )

    def test_to_find_ll(
        self,
        xlon_calendar_extended,
        cc,
        bis_parameterized,
        dfs,
        session_ends,
        limits,
        rngs,
        no_delay,
    ):
        cal = xlon_calendar_extended
        bi = bis_parameterized
        bi = TDInterval.H1
        bi_daily = bi.is_one_day
        df = dfs[bi]

        left_bound = session_ends[0] if bi_daily else limits[0]
        admin = MockRequestAdmin(cal, left_bound)
        request_data = admin.mock_request_data()

        dr_0 = rngs[0]
        dr_0_start, dr_0_end = dr_0[1] if bi_daily else dr_0[0]

        def get_data() -> m.Data:
            admin.reset()
            data = m.Data(request_data, cc, bi, no_delay)
            return data

        delta = get_delta(dr_0_start)

        data = get_data()
        dr = left_bound, dr_0_end
        _ = data.get_table(dr)
        assert data.ll is None

        data = get_data()
        if bi_daily:
            oob_left = cal.previous_session(left_bound).tz_convert(None)
        else:
            oob_left = cal.previous_minute(left_bound)
        dr = oob_left, dr_0_end
        table = data.get_table(dr)
        assert data.ll == left_bound

        table_start = table.index[0] if bi_daily else table.index[0].left
        assert table_start == left_bound
        assert_table_matches(table, dr, df)
        assert data.ranges == [pd.Interval(oob_left, dr_0_end, closed="both")]
        assert data.has_requested_data and data.has_data

        assert data.leftmost_requested == oob_left
        assert data.rightmost_requested == dr_0_end
        assert data.from_ll
        assert not data.to_rl
        assert not data.has_all_data

        assert data.requested_range(dr)
        assert not data.available_range(dr)
        assert data.available_range((left_bound, dr_0_end))

        # verify not requesting from source
        admin.reset_last_request()
        table = data.get_table(dr)
        assert_table_matches(table, dr, df)
        assert admin.last_request is None
        delta = delta if delta == helpers.ONE_DAY else pd.Timedelta(30, "min")
        from_ = dr[0] + delta
        to = dr[1] - delta
        table = data.get_table((from_, to))
        assert_table_matches(table, (from_, to), df)
        assert admin.last_request is None

    def test_to_now(
        self,
        xlon_calendar_extended,
        cc,
        bis_parameterized,
        session_ends,
        limits,
        rngs,
        no_delay,
        today,
        one_day,
        monkeypatch,
    ):
        # pylint: disable=too-complex, too-many-statements
        cal = xlon_calendar_extended
        bi = bis_parameterized
        bi_daily = bi.is_one_day

        last_session = helpers.to_tz_naive(cal.date_to_session(today, "previous"))
        start_session = helpers.to_tz_naive(cal.session_offset(last_session, -2))
        start = start_session if bi_daily else cal.session_open(start_session)
        end = last_session if bi_daily else cal.session_close(last_session)

        admin = MockRequestAdmin(cal)
        request_data = admin.mock_request_data()
        df = request_data(bi, start, end)
        del admin

        dr_2 = rngs[-1]
        dr_2_start, _ = dr_2[1] if bi_daily else dr_2[0]

        left_limit = session_ends[0] if bi_daily else limits[0]
        delta = get_delta(dr_2_start)

        def get_data(now, delay=no_delay) -> tuple[m.Data, MockRequestAdmin]:
            right_bound = now if bi_daily else now + bi
            admin = MockRequestAdmin(cal, right_bound=right_bound)
            request_data = admin.mock_request_data()
            data = m.Data(request_data, cc, bi, delay, left_limit)
            return data, admin

        def set_now(now: pd.Timestamp):
            monkeypatch.setattr(
                "pandas.Timestamp.now", lambda *_, tz=None, **__: now.tz_convert(tz)
            )

        if not bi_daily:
            now = cal.session_close(last_session) - pd.Timedelta(90, "min")
            data, _ = get_data(now)
            set_now(now)
            assert data.rl == now + bi
            for _ in range(18):
                now += pd.Timedelta(10, "min")
                set_now(now)
                assert data.rl == now + bi
                assert not data.to_rl
                assert data.available(now)
                assert data.available(now + bi)
                assert not data.available(now + bi + delta)
        else:
            now = cal.session_close(last_session)
            today = last_session
            data, _ = get_data(today)
            set_now(now)
            assert data.rl == today
            assert not data.to_rl
            assert data.available(today)
            assert not data.available(today + bi)

        def assertions(data, table, now, dr, admin, delay):
            # pylint: disable=too-many-statements
            bi = data.bi
            from_, to = dr
            # discount calculation reasonable only given the bis tested (1T, 1H, 1D)
            discount = max(bi, delay + pd.Timedelta(10, "min"))
            rightmost = now - discount

            table_dr = dr[0], min(dr[1], now + bi)
            assert_table_matches(table, table_dr, df)
            assert data.ranges
            assert not data.ranges_index.empty
            assert data.has_requested_data
            assert data.has_data
            assert data.leftmost_requested == from_
            assert data.rightmost_requested == rightmost
            assert not data.from_ll
            assert not data.to_rl
            assert not data.has_all_data

            ranges = [pd.Interval(from_, rightmost, closed="both")]
            assert data.ranges == ranges
            pd.testing.assert_index_equal(data.ranges_index, pd.IntervalIndex(ranges))

            assert_rng_available_requested(data, from_, rightmost)
            assert_rng_available_not_requested(data, left_limit, from_ - delta)

            if delta == one_day:
                # ensure today available but not requested
                assert data.available(now)
                for ts in [now, now + bi]:
                    assert not data.requested(ts)
                    assert not data.available_and_requested(ts)
                assert not data.available(today + bi)
                assert data.rl == now
                assert not data.to_rl
                assert data.available_range((from_, now))
                assert not data.requested_range((from_, now))

            else:
                # ensure live interval either side of now available but not requested
                start_live_interval = now - discount + delta
                tss = [start_live_interval, now + bi]
                assert data.available(tss)
                for ts in tss:
                    assert not data.requested(ts)
                    assert not data.available_and_requested(ts)

                rng = (tss[0], tss[-1])
                assert data.available_range(rng)
                assert not data.requested_range(rng)

                # ensure beyond live interval not available or requested
                beyond = now + bi + delta
                assert not data.available(beyond)
                assert not data.requested(beyond)
                assert not data.available_and_requested(beyond)
                assert not data.available_range((start_live_interval, beyond))
                assert not data.requested_range((start_live_interval, beyond))

                # verify repeating only requests live interval from source
                admin.reset_last_request()
                table = data.get_table(dr)
                assert_table_matches(table, table_dr, df)
                assert admin.last_request == (now - discount, min(to, now + bi))

                # verify not requesting from source when not requesting live interval
                admin.reset_last_request()
                dr_excluding_live_interval = (from_, now - discount)
                table = data.get_table(dr_excluding_live_interval)
                assert_table_matches(table, dr_excluding_live_interval, df)
                assert admin.last_request is None

        nows = [
            cal.session_close(cal.previous_session(last_session)),
            cal.session_open(last_session),
        ]
        if not bi_daily:
            nows.append(cal.session_open(last_session) + bi)
            nows.append(cal.session_close(last_session) - bi)

        for now in nows:
            set_now(now)

            dr = start, now + bi
            if bi_daily:
                now = now.normalize().tz_convert(None)
                dr = start, now

            for delay in [no_delay, pd.Timedelta(15, "min")]:
                # on right limit
                data, admin = get_data(now, delay)
                table = data.get_table(dr)
                assertions(data, table, now, dr, admin, delay)

                # beyond right limit
                if not bi_daily:
                    data, admin = get_data(now, delay)
                    dr = (start, end + (bi * 2))
                    table = data.get_table(dr)
                    assertions(data, table, now, dr, admin, delay)

    def test_request_non_trading_period(
        self,
        mr_admin,
        session_ends,
        xlon_calendar,
        cc,
        no_delay,
        limits,
        one_min,
        dfs,
    ):
        """Tests no change to ranges on requesting a non-trading period."""
        xlon = xlon_calendar
        bi = TDInterval.T1
        df = dfs[bi]

        # Get data for a session
        session, _ = session_ends
        session_open = xlon.session_open(session)
        session_close = xlon.session_close(session)
        data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *limits)
        session_dr = (session_open, session_close)
        table = data.get_table(session_dr)
        assert_rng_available_requested(data, *session_dr)
        assert_table_matches(table, session_dr, df)

        # Get data for next session
        next_session = helpers.to_tz_naive(xlon.next_session(session))
        next_session_open = xlon.session_open(next_session)
        next_session_close = xlon.session_close(next_session)
        next_session_dr = (next_session_open, next_session_close)
        table = data.get_table(next_session_dr)
        assert_rng_available_requested(data, *next_session_dr)
        assert_table_matches(table, next_session_dr, df)

        # assert period between sessions has not been requested
        assert_rng_available_not_requested(
            data, session_close + one_min, next_session_open - one_min
        )
        sessions_dr = (session_open, next_session_close)
        # assert period that will be requested limited to non_trading_period
        assert data._request_dates(sessions_dr) == [(session_close, next_session_open)]
        table = data.get_table(sessions_dr)
        assert_table_matches(table, sessions_dr, df)
        assert_rng_available_requested(data, *sessions_dr)

    def test_raises_PricesUnavailableFromSourceError(
        self, mr_admin, session_ends, xlon_calendar, cc, no_delay, limits
    ):
        """Test raises error if no prices returned and dr not a non-trading period."""
        # request prices over half hour period for hour bi
        xlon = xlon_calendar
        bi = TDInterval.H1
        data = m.Data(mr_admin.mock_request_data(), cc, bi, no_delay, *limits)

        session, _ = session_ends
        session_open = xlon.session_open(session)
        with pytest.raises(errors.PricesUnavailableFromSourceError):
            data.get_table((session_open, session_open + pd.Timedelta(30, "min")))
