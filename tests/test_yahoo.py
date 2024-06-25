"""Tests for market_prices.prices.yahoo module."""

from __future__ import annotations

from collections import abc
import functools
import itertools
import typing
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import yahooquery as yq

import market_prices.prices.yahoo as m
from market_prices import daterange, helpers, intervals, errors
from market_prices.helpers import UTC
from market_prices.support import tutorial_helpers as th
from market_prices.utils import calendar_utils as calutils
from .test_base_prices import (
    assertions_intraday_common,
    assertions_daily,
    assertions_intraday,
)

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

# pylint: disable=too-many-lines

# NOTE See ../docs/developers/testing.md...
# ...sessions that yahoo temporarily fails to return prices for if (seemingly)
# send a high frequency of requests for prices from the same IP address.
_flakylist = (
    pd.Timestamp("2024-05-28"),
    pd.Timestamp("2024-05-27"),
    pd.Timestamp("2024-01-21"),
    pd.Timestamp("2023-09-08"),
    pd.Timestamp("2023-09-01"),
    pd.Timestamp("2023-07-17"),
    pd.Timestamp("2023-04-23"),
    pd.Timestamp("2023-01-18"),
    pd.Timestamp("2023-01-17"),
    pd.Timestamp("2022-10-11"),
    pd.Timestamp("2022-06-01"),
    pd.Timestamp("2022-05-31"),
    pd.Timestamp("2022-05-30"),
    pd.Timestamp("2022-05-27"),
    pd.Timestamp("2022-05-26"),
    pd.Timestamp("2022-05-25"),
    pd.Timestamp("2022-05-24"),
    pd.Timestamp("2022-05-23"),
    pd.Timestamp("2022-05-10"),
    pd.Timestamp("2022-04-27"),
    pd.Timestamp("2022-04-22"),
    pd.Timestamp("2022-04-14"),
    pd.Timestamp("2022-04-01"),
    pd.Timestamp("2022-03-29"),
    pd.Timestamp("2022-03-28"),
    pd.Timestamp("2022-02-15"),
    pd.Timestamp("2022-02-07"),
    pd.Timestamp("2022-01-24"),
    pd.Timestamp("2022-01-14"),
    pd.Timestamp("2022-01-13"),
    pd.Timestamp("2022-01-12"),
    pd.Timestamp("2022-01-11"),
    pd.Timestamp("2021-10-18"),
    pd.Timestamp("2021-02-26"),
    pd.Timestamp("2020-06-25"),
    pd.Timestamp("2020-05-07"),
    pd.Timestamp("2012-05-28"),
)


def minute_in_flakylist(
    minute: pd.Timestamp | list[pd.Timestamp],
    calendars: list[xcals.ExchangeCalendar],
) -> bool:
    """Query if a given minute is a minute of a flakylisted session.

    Parameters
    ----------
    minute:
        Minute(s) to query.

    calendars
        Calendars against which to evaluate session. If `minute` is not a
        trading minute of a calendar then will return True if session
        preceeding minute is flakylisted.
    """
    minutes = minute if isinstance(minute, abc.Sequence) else [minute]
    for cal in calendars:
        for minute_ in minutes:
            session = cal.minute_to_session(minute_, "previous")
            if session in _flakylist:
                return True
    return False


def current_session_in_flakylist(calendars: list[xcals.ExchangeCalendar]) -> bool:
    """Query if current session is in the flakylist.

    Will return True if most recent session of any of `calendars` is
    flakylisted.

    Parameters
    ----------
    calendars
        Calendars against which to evaluate current session.
    """
    return minute_in_flakylist(helpers.now(), calendars)


class skip_if_fails_and_today_flakylisted:
    """Decorator to skip test if fails due to today being flakylisted.

    Skips test if test raises errors.PricesUnavailableFromSourceError and
    today is in the flakylist.

    Parameters
    ----------
    cal_names
        Names of calendars against which to evaluate 'today'. Test will be
        skipped if a defined exception is raised and the flakylist includes
        'today' as evaluated against any of these calendars.

    exceptions
        Additional exception types. In addition to
        `errors.PricesUnavailableFromSourceError` test will also be skipped
        if exception of any of these types is raised and 'today' is
        flakylisted.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        cal_names: list[str],
        exceptions: list[type[Exception]] | None = None,
    ):
        self.cals = [xcals.get_calendar(name) for name in cal_names]

        permitted_exceptions = [errors.PricesUnavailableFromSourceError]
        if exceptions is not None:
            permitted_exceptions += exceptions
        self.permitted_exceptions = tuple(permitted_exceptions)

    def __call__(self, f) -> abc.Callable:
        @functools.wraps(f)
        def wrapped_test(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except self.permitted_exceptions:
                if current_session_in_flakylist(self.cals):
                    pytest.skip(f"Skipping {f.__name__}: today in flakylist.")
                raise

        return wrapped_test


class skip_if_prices_unavailable_for_flakylisted_session:
    """Decorator to skip test if fails due to unavailable prices.

    Skips test if raises `errors.PricesUnavailableFromSourceError` for a
    period bounded on either side by a minute or date that corresponds with
    a flakylisted session.

    Parameters
    ----------
    cal_names
        Names of calendars against which to evaluate sessions corresponding
        with period bounds.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, cal_names: list[str]):
        cals = [xcals.get_calendar(name) for name in cal_names]
        self.cc = calutils.CompositeCalendar(cals)

    def _flaky_sessions(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[pd.Timestamp]:
        rtrn = []
        for bound in (start, end):
            if helpers.is_date(bound):
                if bound in _flakylist:
                    rtrn.append(bound)
            elif self.cc.minute_to_sessions(bound).isin(_flakylist).any():
                rtrn.append(bound)
        return rtrn

    def __call__(self, f) -> abc.Callable:
        @functools.wraps(f)
        def wrapped_test(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except errors.PricesUnavailableFromSourceError as err:
                sessions = self._flaky_sessions(err.params["start"], err.params["end"])
                if sessions:
                    pytest.skip(
                        f"Skipping {f.__name__}: prices unavailable for period bound"
                        f" with flakylisted session(s) {sessions}."
                    )
                raise

        return wrapped_test


# NOTE: Leave commented out. Uncomment to test decorator locally.
# @skip_if_prices_unavailable_for_flakylisted_session(["XLON"])
# def test_skip_if_prices_unavailable_for_flakylisted_session_decorator():
#     xlon = xcals.get_calendar("XLON", side="left")
#     params = {
#         'interval': '5m',
#         'start': xlon.session_open(_flakylist[2]),
#         'end': xlon.session_close(_flakylist[1]), # helpers.now() to test single bound
#     }
#     raise errors.PricesUnavailableFromSourceError(params, None)

# @skip_if_prices_unavailable_for_flakylisted_session(["XLON"])
# def test_skip_if_prices_unavailable_for_flakylisted_session_decorator2():
#     params = {
#         'interval': '5m',
#         'start': _flakylist[2],
#         'end': _flakylist[1],
#     }
#     raise errors.PricesUnavailableFromSourceError(params, None)


class DataUnavailableForTestError(Exception):
    """Base error class for unavailable test data."""

    def __str__(self) -> str:
        return getattr(self, "_msg", "Test Data unavailable.")

    def __unicode__(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()


def skip_if_data_unavailable(f: abc.Callable) -> abc.Callable:
    """Decorator to skip test if fails on `TestDataUnavailableError`."""

    @functools.wraps(f)
    def wrapped_test(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except DataUnavailableForTestError:
            pytest.skip(f"Skipping {f.__name__}: valid test inputs unavailable.")

    return wrapped_test


class ValidSessionUnavailableError(DataUnavailableForTestError):
    """No valid session available for requested restrictions.

    Parameters as for `get_valid_session`.
    """

    def __init__(self, session: pd.Timestamp, limit: pd.Timestamp):
        # pylint: disable=super-init-not-called
        self._msg = (
            f"There are no valid sessions from {session} and with limit as {limit}."
        )


def get_valid_session(
    session: pd.Timestamp,
    calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
    direction: typing.Literal["next", "previous"],
    limit: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return session that is not in flakylist.

    Returns `session` if `session` is not in flakylist, otherwise returns
    nearest session to `session`, in the direction of `direction`, that is
    not in flakylist. Sessions evaluated against `calendar`.

    Raises `ValidSessionUnavailableError` if session would be `limit` or 'beyond'.
    """
    session_received = session
    while session in _flakylist:
        # xcals 4.0 lose wrappers within clause
        if direction == "next":
            session = helpers.to_tz_naive(calendar.next_session(session))
            if limit is not None and session >= limit:
                raise ValidSessionUnavailableError(session_received, limit)
        else:
            session = helpers.to_tz_naive(calendar.previous_session(session))
            if limit is not None and session <= limit:
                raise ValidSessionUnavailableError(session_received, limit)
    return session


def get_valid_consecutive_sessions(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    calendar: xcals.ExchangeCalendar | calutils.CompositeCalendar,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return two valid consecutive sessions.

    Sessions will be sessions of a given `calendar` for which intraday data
    is available at a given `bi`.

    Raises `ValidSessionUnavailableError` if no such sessions available.
    """
    start, limit = th.get_sessions_range_for_bi(prices, bi)
    start = calendar.date_to_session(start, "next")
    start = get_valid_session(start, calendar, "next", limit)
    end = calendar.next_session(start)
    while end in _flakylist:
        start = get_valid_session(end, calendar, "next", limit)
        end = calendar.next_session(start)
        if end > limit:
            raise ValidSessionUnavailableError(start, limit)
    return start, end


class ValidSessionsUnavailableError(DataUnavailableForTestError):
    """Test data unavailable to 'get_valid_conforming_sessions'.

    There are an insufficient number of consecutive sessions of the
    requested lengths between the requested limits.

    Parameters
    ----------
    Parameters as receieved to `get_valid_conforming_sessions` except:

    start
        Start limit from which can evaluate valid sessions.

    end
        End limit to which can evaluate valid sessions.
    """

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calendars: list[xcals.ExchangeCalendar],
        session_length: list[pd.Timedelta],
        num_sessions: int,
    ):
        # pylint: disable=super-init-not-called
        self._msg = (
            f"{num_sessions} valid consecutive sessions of the requested lengths"
            f" are not available from {start} through {end}."
            f"\n`calendars` receieved as {calendars}."
            f"\n`session_length` receieved as {session_length}."
        )


def get_valid_conforming_sessions(
    prices: m.PricesYahoo,
    bi: intervals.BI,
    calendars: list[xcals.ExchangeCalendar],
    session_length: list[pd.Timedelta],
    num_sessions: int = 2,
) -> pd.DatetimeIndex:
    """Get conforming sessions for which prices available.

    Prices will be available for all sessions in return.

    Prices will be available at bi for at least one session (evaluated
    against CompositeCalendar) prior to first session.

    Raises `ValidSessionsUnavailableError` if there no sessions conform
    with the requirements.
    """
    # get sessions for which price data available at all base intervals
    session_start, session_end = th.get_sessions_range_for_bi(prices, bi)
    session_first = session_start

    # margin allowing for prices at bi to be available over at least one prior session
    session_start = prices.cc.next_session(session_start)
    session_start = get_valid_session(session_start, prices.cc, "next", session_end)

    try:
        sessions = th.get_conforming_sessions(
            calendars, session_length, session_start, session_end, num_sessions
        )
    except errors.TutorialDataUnavailableError:
        raise ValidSessionsUnavailableError(
            session_first, session_end, calendars, session_length, num_sessions
        ) from None
    while (
        any(sessions.isin(_flakylist))
        # make sure prices also available for session prior to conforming sessions
        or prices.cc.previous_session(sessions[0]) in _flakylist
    ):
        session_start = prices.cc.next_session(session_start)
        session_start = get_valid_session(session_start, prices.cc, "next", session_end)
        if session_start >= session_end:
            raise ValidSessionsUnavailableError(
                session_first, session_end, calendars, session_length, num_sessions
            )
        try:
            sessions = th.get_conforming_sessions(
                calendars, session_length, session_start, session_end, num_sessions
            )
        except errors.TutorialDataUnavailableError:
            continue
    return sessions


class TestConstructor:
    """Tests for attributes defined by PricesYahoo constructor.

    Only tests those attributes set by PricesYahoo's extension of the base
    constructor.
    """

    @pytest.fixture(scope="class")
    def symbols(self) -> abc.Iterator[str]:
        yield (
            "GOOG FTSEMIB.MI ^IBEX ^FTMC 9988.HK GBPEUR=X GC=F BTC-GBP CL=F"
            " ES=F ZB=F HG=F GEN.L QAN.AX CA.PA BAS.DE FER.MC SPY QQQ ARKG"
        )

    @pytest.fixture(scope="class")
    def calendars(self) -> abc.Iterator[dict[str, str]]:
        yield {
            "GOOG": "XNYS",
            "GEN.L": "XLON",
            "9988.HK": "XHKG",
            "QAN.AX": "XASX",
            "CA.PA": "XPAR",
            "BAS.DE": "XFRA",
            "FER.MC": "XMAD",
            "FTSEMIB.MI": "XMIL",
            "^IBEX": "XMAD",
            "^FTMC": "XLON",
            "GBPEUR=X": "24/5",
            "BTC-GBP": "24/7",
            "GC=F": "CMES",
            "CL=F": "us_futures",
            "ES=F": "CMES",
            "ZB=F": "CMES",
            "HG=F": "CMES",
            "SPY": "XNYS",  # ETF, yahoo exchange name is 'NYSEArca'
            "QQQ": "XNYS",  # ETF, yahoo exchange name is 'NasdaqGM'
            "ARKG": "XNYS",  # ETF, yahoo exchange name is 'BATS'
        }

    @pytest.fixture(scope="class")
    def delays(self) -> abc.Iterator[dict[str, int]]:
        yield {
            "GOOG": 0,
            "GEN.L": 15,
            "9988.HK": 15,
            "QAN.AX": 20,
            "CA.PA": 15,
            "BAS.DE": 15,
            "FER.MC": 15,
            "FTSEMIB.MI": 15,
            "^IBEX": 15,
            "^FTMC": 15,
            "GBPEUR=X": 0,
            "BTC-GBP": 0,
            "GC=F": 10,
            "CL=F": 10,
            "ES=F": 10,
            "ZB=F": 10,
            "HG=F": 10,
            "SPY": 0,
            "QQQ": 0,
            "ARKG": 0,
        }

    def test_invalid_symbol(self, symbols):
        invalid_symbol = "INVALIDSYMB"
        match = re.escape(
            "The following symbols are not recognised by the yahoo API:"
            f" {[invalid_symbol]}."
        )
        with pytest.raises(ValueError, match=match):
            _ = m.PricesYahoo(invalid_symbol)
        symbols_ = symbols.split()
        symbols_ = symbols_[:4] + [invalid_symbol] + symbols_[4:]
        with pytest.raises(ValueError, match=match):
            _ = m.PricesYahoo(symbols_)

    @pytest.fixture(scope="class")
    def prices(
        self, symbols, calendars, delays
    ) -> abc.Iterator[tuple[m.PricesYahoo, bool]]:
        api_working = True
        try:
            prices = m.PricesYahoo(symbols)
        except (m.YahooAPIError, ValueError):
            prices = m.PricesYahoo(symbols, calendars=calendars, delays=delays)
            api_working = False
        yield prices, api_working

    def test_ticker(self, prices, symbols):
        """Verify sets yq.Ticker."""
        prices, _ = prices
        assert isinstance(prices._ticker, yq.Ticker)
        assert set(prices._ticker.symbols) == set(helpers.symbols_to_list(symbols))

    def test_daily_limit(self, prices):
        """Verify constructor setting a daily base limit."""
        prices, api_working = prices
        if not api_working:
            pytest.skip("Yahoo API not working")
        assert isinstance(prices.base_limits[prices.BaseInterval.D1], pd.Timestamp)

    def test_calendars(self, prices, calendars):
        prices, _ = prices
        expected_calendars = calendars
        for k, cal in prices.calendars.items():
            assert isinstance(cal, xcals.ExchangeCalendar)
            assert expected_calendars[k] == cal.name

    def test_delays(self, prices, delays):
        prices, _ = prices
        expected_delays = delays
        for k, delay in prices.delays.items():
            assert isinstance(delay, pd.Timedelta)
            assert delay == pd.Timedelta(expected_delays[k], "min")

    def test_adj_close(self):
        """Verify `adj_close` parameter returns alternative close col.

        Verifies effect of `adj_close` for daily interval. Verifies
        parameter has no effect when interval is intraday.

        NOTE: Test verifies the difference (or not) between instances of
        PricesYahoo with `adj_close` parameter passed as False (default)
        and True. For each tested interval, test requests a single
        dataframe from yahooquery and then mocks the return from
        `PricesYahoo._ticker.history` to return this single source. The effect
        of `adj_close` is then tested against the return from
        `PricesYahoo._request_yahoo`. Why not just test the difference
        in the return from `_request_yahoo` without the mocking? Because
        the Yahoo API can return slightly different data for the same
        repeated call (see 'Tests for `PricesYahoo`'
        section of docs/developers/testing).
        """
        symbol, cal_name = "AZN.L", "XLON"
        prices = m.PricesYahoo(symbol, calendars=cal_name, delays=0)
        prices_adj = m.PricesYahoo(symbol, calendars=cal_name, delays=0, adj_close=True)

        # inputs for daily interval
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-12-31")
        interval, interval_yq = prices.bis.D1, "1d"

        # inputs for intraday interval
        end_id = pd.Timestamp.now(tz=UTC).floor("D") - pd.Timedelta(14, "D")
        start_id = end_id - pd.Timedelta(14, "D")
        interval_id, interval_yq_id = prices.bis.H1, "1h"

        # get return from yahooquery for each interval
        # get via prices method to go through locally implemented fixes.
        df_yq = prices._ticker.history(
            interval=interval_yq, start=start, end=end, adj_timezone=False
        )
        df_yq_id = prices._ticker.history(
            interval=interval_yq_id, start=start_id, end=end_id, adj_timezone=False
        )

        def mock_yq_history(prices: m.PricesYahoo, df: pd.DataFrame):
            prices._ticker.history = lambda *_, **__: df.copy()

        # test daily interval results in different "close" column
        mock_yq_history(prices, df_yq)
        mock_yq_history(prices_adj, df_yq)

        df = prices._request_yahoo(interval, start, end)
        df_adj = prices_adj._request_yahoo(interval, start, end)

        assert_series_equal(df["close"], df_yq["close"])
        assert_series_equal(df_adj["close"], df_yq["adjclose"], check_names=False)

        for col in df:
            if col == "close":
                assert (df[col] != df_adj[col]).any()
            else:
                assert_series_equal(df[col], df_adj[col])

        # test returns unchanged when interval intraday
        mock_yq_history(prices, df_yq_id)
        mock_yq_history(prices_adj, df_yq_id)

        df = prices._request_yahoo(interval_id, start_id, end_id)
        df_adj = prices_adj._request_yahoo(interval_id, start_id, end_id)

        assert_series_equal(df["close"], df_yq_id["close"])
        assert_series_equal(df_adj["close"], df_yq_id["close"])

        for col in df:
            assert_series_equal(df[col], df_adj[col])

    def test_api_down(self, monkeypatch):
        monkeypatch.setattr("yahooquery.Ticker.price", m.ERROR404)
        monkeypatch.setattr("yahooquery.Ticker.quote_type", m.ERROR404)

        msg = re.escape(
            "It appears that the Yahoo API endpoint 'price' is not currently"
            " available (via yahooquery). Pass the `calendars` parameter to manually"
            " assign calendars. See help(PricesYahoo) for documentation on the"
            " `calendars` parameter."
        )
        with pytest.raises(m.YahooAPIError, match=msg):
            m.PricesYahoo("MSFT")

        msg = re.escape(
            "NOTE: the Yahoo API is not currently available (via yahooquery) to"
            " determine the delay."
        )
        with pytest.raises(ValueError, match=msg):
            m.PricesYahoo("MSFT", calendars="XNYS")

        # verify loads
        m.PricesYahoo("MSFT", calendars="XNYS", delays=0)


@pytest.fixture
def session_length_xnys() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "h")


@pytest.fixture
def session_length_xhkg() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(6.5, "h")


@pytest.fixture
def session_length_xlon() -> abc.Iterator[pd.Timedelta]:
    yield pd.Timedelta(8.5, "h")


@pytest.fixture(scope="module")
def pricess() -> abc.Iterator[dict[str, m.PricesYahoo]]:
    yield dict(
        us=m.PricesYahoo(["MSFT"], calendars="XNYS", delays=0),
        with_break=m.PricesYahoo(["9988.HK"], calendars="XHKG", delays=0),
        us_lon=m.PricesYahoo(
            ["MSFT", "AZN.L"], calendars=["XNYS", "XLON"], delays=[0, 15]
        ),
        us_oz=m.PricesYahoo(
            ["MSFT", "QAN.AX"], calendars=["XNYS", "XASX"], delays=[0, 20]
        ),
        inc_245=m.PricesYahoo(
            ["AZN.L", "GBPEUR=X"], calendars=["XLON", "24/5"], delays=[15, 0]
        ),
        inc_247=m.PricesYahoo(
            ["AZN.L", "BTC-GBP"], calendars=["XLON", "24/7"], delays=[15, 0]
        ),
        only_247=m.PricesYahoo(["BTC-GBP"], calendars="24/7", delays=0),
    )


@pytest.fixture
def prices_us(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    key = "us"
    symbols = pricess[key].symbols
    calendars = pricess[key].calendars
    delays = pricess[key].delays
    yield m.PricesYahoo(symbols, calendars=calendars, delays=delays)


@pytest.fixture
def prices_us_lon(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    key = "us_lon"
    symbols = pricess[key].symbols
    calendars = pricess[key].calendars
    delays = pricess[key].delays
    yield m.PricesYahoo(symbols, calendars=calendars, delays=delays)


@pytest.fixture
def prices_us_lon_hkg() -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    symbols = "MSFT, AZN.L, 9988.HK"
    calendars = ["XNYS", "XLON", "XHKG"]
    delays = [0, 15, 15]
    yield m.PricesYahoo(symbols, lead_symbol="MSFT", calendars=calendars, delays=delays)


@pytest.fixture
def prices_with_break(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    key = "with_break"
    symbols = pricess[key].symbols
    calendars = pricess[key].calendars
    delays = pricess[key].delays
    yield m.PricesYahoo(symbols, calendars=calendars, delays=delays)


@pytest.fixture
def prices_only_247(pricess) -> abc.Iterator[m.PricesYahoo]:
    """Yield's unique copy for each test."""
    key = "only_247"
    symbols = pricess[key].symbols
    calendars = pricess[key].calendars
    delays = pricess[key].delays
    yield m.PricesYahoo(symbols, calendars=calendars, delays=delays)


class TestRequestDataDaily:
    """Verify implementation of abstract _request_data for daily interval."""

    def test_prices(self, pricess):
        """Verify standard assertions for all prices fixtures."""
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-12-31")
        for prices in pricess.values():
            interval = prices.BaseInterval.D1
            rtrn = prices._request_data(interval, start, end)
            assertions_daily(rtrn, prices, start, end)

    def test_data_unavailable_for_symbol(self, pricess):
        """Verify as expected when data not available for one or more symbols."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2022-01-02")
        rtrn = prices._request_data(interval, start, end)
        all_missing = []
        for s in prices.symbols:
            all_missing.append(rtrn[s].isna().all(axis=None))
        assert any(all_missing)
        assertions_daily(rtrn, prices, start, end)

    def test_start_none(self, pricess):
        """Verify as expected when start is None."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = None
        end = pd.Timestamp("2021-12-31")
        rtrn = prices._request_data(interval, start, end)
        base_limit = prices.base_limits[prices.BaseInterval.D1]
        if base_limit is not None:  # base_limit can be None if API not working
            assert rtrn.pt.first_ts == base_limit
        assert rtrn.pt.last_ts == end
        # assertions not expected to hold for dataframe that includes dates prior
        # to first date that any of the symbols first traded.
        start = rtrn["BTC-GBP"].first_valid_index()
        rtrn_ss = rtrn[start:]
        assertions_daily(rtrn_ss, prices, start, end)

    def test_live_indice(self, pricess):
        """Verify return with live indice as expected."""
        prices = pricess["inc_247"]
        interval = prices.BaseInterval.D1
        start = pd.Timestamp("2022-01-01")
        end = helpers.now(intervals.ONE_DAY)
        rtrn = prices._request_data(interval, start, end)
        assert rtrn.pt.first_ts == start
        assert rtrn.pt.last_ts == end
        assertions_daily(rtrn, prices, start, end)


def get_data_bounds(
    prices: m.PricesYahoo,
    interval: intervals.BI,
    limits: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], slice]:
    """Get start and end bounds for price data.

    Parameters
    ----------
    prices
        Prices instance against which to evaluate bounds.

    interval
        Base interval for which bounds required.

    limits : optional
        Define limits within which to get bounds. By default, will
        evaluate bounds for limits of data availability at `interval`.

    Returns
    -------
    tuple[]

        tuple[0]
            [0]: start bound. Open of session following session
            corresopnding to `limits`[0]. Session corresponding to
            `limits`[0] will be session of which `limits`[0] is a
            minute, or otherwise the next session.

            [1]: end bound. Open of session corresopnding to
            `limits`[1]. Session corresponding to `limits`[1] will be
            session of which `limits`[1] is a minute, or otherwise the
            previous session. Setting end bound to a session open
            ensure all the prior session is included (regardless of
            end alignment).

        tuple[1] : slice
            slice from start to end of sessions falling within bounds.
            Does not include session corresponding with `limits`[1].
            Can be used to index a composite calendar. slice dates are
            tz-naive.
    """
    ll, rl = prices.limits[interval] if limits is None else limits
    cc = prices.cc
    first_available_session = cc.minute_to_sessions(ll, "next")[0]
    start_session = cc.next_session(first_available_session)
    last_available_session = cc.minute_to_sessions(rl, "previous")[-1]
    end_session = cc.previous_session(last_available_session)
    slc = slice(start_session, end_session)
    start = cc.session_open(start_session)
    # set end as open of next session to ensure include all of end_session
    end = cc.session_open(last_available_session)
    return (start, end), slc


def expected_table_structure_us(
    prices: m.PricesYahoo,
    interval: intervals.BI | intervals.TDInterval,
    slc: slice,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], int, pd.Series]:
    """Return expected aspects of intraday price table of `prices`["us"].

    Parameters
    ----------
    prices
        `prices`["us"]

    interval
        Expected table base interval. Table will be assumed to maintain a
        regular interval.

    slc
        Slice of `prices`.cc.sessions that `table` is expected to cover.
        Table will be assumed to fully cover all sessions of this slice
        and only those sessions.

    Returns
    -------
    tuple
        [0] tuple[pd.Timestamp, pd.Timestamp]:
            First and last timestamps represented by table

        [1] int
            Expected table length.

        [2] pd.Series
            Expected last timestamp of each session.
    """
    cc = prices.cc
    sessions_durations = cc.closes[slc] - cc.opens[slc]
    sessions_rows = np.ceil(sessions_durations / interval)
    expected_num_rows = int(sessions_rows.sum())
    sessions_end = cc.opens[slc] + (interval.as_pdtd * sessions_rows)

    start = cc.opens[slc].iloc[0]
    end = sessions_end.iloc[-1]
    return (start, end), expected_num_rows, sessions_end


def assertions_intraday_us(
    prices: m.PricesYahoo,
    table: pd.DataFrame,
    interval: intervals.BI | intervals.TDInterval,
    bounds: tuple[pd.Timestamp, pd.Timestamp],
    expected_num_rows: int,
    slc: slice,
    sessions_end: pd.Series,
):
    """Assert intraday prices `table` as expected for `prices`["us"].

    Parameters
    ----------
    prices
        `prices`["us"]

    table
        table to be asserted as corresponding with `prices`["us"].

    interval
        Expected `table` base interval.

    bounds
        (first, last) timestamps covered by table.

    expected_num_rows
        Expected `table` length.

    slc
        Slice of `prices`.cc.sessions that `table` is expected to cover.
        Table will be assumed to fully cover all sessions of this slice
        and only those sessions.

    sessions_end
        Expected last indice of each session covered.
    """
    start, end = bounds
    assertions_intraday(table, interval, prices, start, end, expected_num_rows)

    cc = prices.cc
    assert cc.opens[slc].isin(table.index.left).all()
    assert not (cc.opens[slc] - interval).isin(table.index.left).any()
    assert sessions_end.isin(table.index.right).all()
    assert not (sessions_end + interval).isin(table.index.right).any()


class TestRequestDataIntraday:
    """Verify implementation of abstract _request_data for intraday intervals."""

    # TODO lose the last two filters when fixed on yahooquery, for example if
    # https://github.com/dpguthrie/yahooquery/pull/262 merged and inlcuded to a release
    @pytest.mark.filterwarnings("ignore:Prices from Yahoo are missing for:UserWarning")
    @pytest.mark.filterwarnings("ignore:A value is trying to be set on a:FutureWarning")
    @pytest.mark.filterwarnings("ignore:'S' is deprecated:FutureWarning")
    def test_request_from_left_limit(self, one_min):
        """Test data requests from left limit.

        The Yahoo API appears to exhibit flaky behaviour when requesting
        prices from the supposed left limit (can return an an empty
        dataframe). This test serves to ensure `PricesYahoo` always returns
        data when implicitely requesting prices from the left limit.

        Notes
        -----
        Doesn't use pricess fixture as require fresh copies for which no
        price data has been requested.
        """
        # verify for non-overlapping calendars
        prices = m.PricesYahoo(
            ["MSFT", "QAN.AX"], calendars=["XNYS", "XASX"], delays=[0, 20]
        )
        prices.request_all_prices()
        cal = prices.calendar_default
        for bi in prices.bis:
            table = prices._pdata[bi]._table
            assert table.pt.interval == bi
            if bi.is_intraday:
                ll = prices.limit_intraday_bi_calendar(bi, cal)
                assert ll - bi <= table.pt.first_ts <= ll + bi
                rl = prices.limits[bi][1]
                if not rl < table.pt.last_ts:
                    # <= bi.as_minutes as rl is 'now' + bi and that bi can all be
                    # trading minutes beyond the right of the last indice
                    # + one_min to provide for processing between getting calendar
                    # and evaluating minutes
                    num_mins = len(cal.minutes_in_range(table.pt.last_ts, rl))
                    assert num_mins <= bi.as_minutes + 1
            else:
                ll = prices.limit_daily
                assert table.pt.first_ts == prices.cc.date_to_session(ll, "next")
                session = cal.minute_to_session(pd.Timestamp.now(), "previous")
                assert table.pt.last_ts == session

        # verify for overlapping calendars
        prices = m.PricesYahoo(
            ["AZN.L", "BTC-GBP"],
            calendars=["XLON", "24/7"],
            delays=[15, 0],
            lead_symbol="BTC-GBP",
        )
        cal = prices.calendar_default
        for bi in prices.bis:
            table = prices.get(bi)
            assert table.pt.interval == bi
            if bi.is_intraday:
                ll = prices.limit_intraday_bi_calendar(bi, cal)
                assert ll - bi <= table.pt.first_ts <= ll + bi
                rl = prices.limits[bi][1]
                if not rl < table.pt.last_ts:
                    # <= bi.as_minutes as rl is 'now' + bi and that bi can all be
                    # trading minutes beyond the right of the last indice
                    # + one_min to provide for processing between getting calendar
                    # and evaluating minutes
                    num_mins = len(cal.minutes_in_range(table.pt.last_ts, rl))
                    assert num_mins <= bi.as_minutes + 1
            else:
                ll = prices.limit_daily
                assert table.pt.first_ts == prices.cc.date_to_session(ll, "next")
                rl = prices.limit_right_daily
                assert table.pt.last_ts == prices.cc.date_to_session(rl, "previous")

    def verify_limit_intraday_bi(self, prices: m.PricesYahoo):
        """Verifies `limit_intraday_bi` as overriden by `PricesYahoo`.

        Verifies method by way of verifying the number of trading minutes
        between the absolute left limit and the left limit as set by
        `PricesYahoo` (trading minutes as evaluated against the calendar
        that determines the left limit for `PricesYahoo`).
        """
        for bi in prices.bis_intraday:
            limit = prices.limit_intraday_bi(bi)
            len_mins = []
            for cal in prices.calendars_unique:
                len_mins.append(len(cal.minutes_in_range(prices.limits[bi][0], limit)))
            # NB the number of trading minutes will be less than 1 + bi during the
            # period between a session close and one bi prior to that close (unless
            # calendar 24h)
            assert (1 + bi.as_minutes) >= min(len_mins)

    def test_prices_us(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us"]
        self.verify_limit_intraday_bi(prices)
        for interval in prices.BaseInterval[:-1]:
            (start, end), slc = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            bounds, num_rows, sessions_end = expected_table_structure_us(
                prices, interval, slc
            )
            assertions_intraday_us(
                prices, df, interval, bounds, num_rows, slc, sessions_end
            )

    def test_prices_with_break(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["with_break"]
        self.verify_limit_intraday_bi(prices)
        cc = prices.cc
        symbols = prices.symbols
        symbol = symbols[0]
        cal = prices.calendars[symbol]

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)
            assertions_intraday_common(df, prices, interval)

            # verify index as expected
            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())

            exclude_break = prices._subsessions_synced(cal, interval)
            if exclude_break:
                break_durations = cal.break_ends[slc] - cal.break_starts[slc]
                break_rows = break_durations // interval
                expected_num_rows -= int(break_rows.sum())

            assert len(df) == expected_num_rows
            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            sessions_last_indice = cc.opens[slc] + (interval.as_pdtd * sessions_rows)
            assert sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

            subset = df[symbol].dropna()
            ignore_breaks = not exclude_break
            equiv = df[symbol].pt.reindex_to_calendar(cal, ignore_breaks=ignore_breaks)
            # Don't check dtype as a volume column dtype can be float if includes
            # missing values.
            assert_frame_equal(subset, equiv, check_freq=False, check_dtype=False)
            assert (subset.close <= subset.high).all()
            assert (subset.low <= subset.close).all()

    @staticmethod
    def get_expected_num_rows_us_lon(
        interval: intervals.TDInterval,
        cc: calutils.CompositeCalendar,
        slc: slice,
    ) -> tuple[int, pd.Series]:
        """Get expected number of rows of return for 'us_lon' prices fixture.

        Returns
        -------
        tuple(int, pd.Series)
            tuple[0] : int
                Expected number of rows.
            tuple[1] : pd.Series
                Expected number of rows per session, gross of any gaps, for
                any session, between close of xlon and subsequent xnys open.
        """
        sessions_durations = cc.closes[slc] - cc.opens[slc]
        sessions_rows_gross = np.ceil(sessions_durations / interval)

        xlon, xnys = cc.calendars[0], cc.calendars[1]
        if xlon.name != "XLON":
            xlon, xnys = xnys, xlon
        common_index = xlon.opens[slc].index.union(xnys.opens[slc].index)
        opens_xlon = xlon.opens[slc].reindex(common_index)
        closes_xlon = xlon.closes[slc].reindex(common_index)
        opens_xnys = xnys.opens[slc].reindex(common_index)
        closes_xnys = xnys.closes[slc].reindex(common_index)
        bv = closes_xlon < opens_xnys
        xlon_rows = np.ceil((closes_xlon[bv] - opens_xlon[bv]) / interval)
        xnys_rows = np.ceil((closes_xnys[bv] - opens_xnys[bv]) / interval)
        new_session_rows = xlon_rows + xnys_rows

        sessions_rows = sessions_rows_gross.copy()
        sessions_rows[bv.index[bv]] = new_session_rows

        expected_num_rows = int(sessions_rows.sum())
        return expected_num_rows, sessions_rows_gross

    def test_prices_us_lon(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us_lon"]
        self.verify_limit_intraday_bi(prices)
        cc = prices.cc

        # xnys and xlon indexes are out of phase for 1H
        for interval in prices.BaseInterval[:-2]:
            (start, end), slc = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            expected_num_rows, sessions_rows_gross = self.get_expected_num_rows_us_lon(
                interval, cc, slc
            )
            sessions_last_indice = cc.opens[slc] + (
                interval.as_pdtd * sessions_rows_gross
            )
            start = cc.opens[slc].iloc[0]
            end = sessions_last_indice.iloc[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            assert sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

    def test_prices_us_oz(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["us_oz"]
        self.verify_limit_intraday_bi(prices)
        cc = prices.cc
        symbols = prices.symbols

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)
            assertions_intraday_common(df, prices, interval)

            # verify index as expected
            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)

            cal0, cal1 = prices.calendars["QAN.AX"], prices.calendars["MSFT"]
            common_index = cal0.opens[slc].index.union(cal1.opens[slc].index)
            opens_0 = cal0.opens[slc].reindex(common_index)
            closes_0 = cal0.closes[slc].reindex(common_index)
            opens_1 = cal1.opens[slc].reindex(common_index)
            closes_1 = cal1.closes[slc].reindex(common_index)
            bv = closes_0 < opens_1
            cal0_rows = np.ceil((closes_0[bv] - opens_0[bv]) / interval)
            cal1_rows = np.ceil((closes_1[bv] - opens_1[bv]) / interval)
            new_session_rows = cal0_rows + cal1_rows

            sessions_rows_adj = sessions_rows.copy()
            sessions_rows_adj[bv.index[bv]] = new_session_rows

            expected_num_rows = int(sessions_rows_adj.sum())

            assert len(df) == expected_num_rows
            assert cc.opens[slc].isin(df.index.left).all()
            assert not (cc.opens[slc] - interval).isin(df.index.left).any()

            sessions_last_indice = cc.opens[slc] + (interval.as_pdtd * sessions_rows)
            sessions_last_indice.isin(df.index.right).all()
            assert not (sessions_last_indice + interval).isin(df.index.right).any()

            for s in symbols:
                if interval == prices.BaseInterval.H1:
                    # unable to reindex as cals open times are not synced at 1H.
                    continue
                subset = df[s].dropna()
                cal = prices.calendars[s]
                equiv = df[s].pt.reindex_to_calendar(cal)
                # Don't check dtype as a volume column dtype can be float if includes
                # missing values.
                assert_frame_equal(subset, equiv, check_freq=False, check_dtype=False)
                assert (subset.close <= subset.high).all()
                assert (subset.low <= subset.close).all()

    def test_prices_inc_245(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["inc_245"]
        self.verify_limit_intraday_bi(prices)
        cc = prices.cc

        for interval in prices.BaseInterval[:-1]:
            (start, end), slc = get_data_bounds(prices, interval)
            df = prices._request_data(interval, start, end)

            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())
            sessions_last_indice = cc.opens[slc] + (interval.as_pdtd * sessions_rows)

            start = cc.opens[slc].iloc[0]
            end = sessions_last_indice.iloc[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            bv = cc.opens[slc].index.weekday == 0  # pylint: disable=compare-to-zero
            assert not (cc.opens[slc][bv] - interval).isin(df.index.left).any()
            assert (cc.opens[slc][~bv] - interval)[1:].isin(df.index.left).all()

            assert sessions_last_indice.isin(df.index.right).all()
            bv = sessions_last_indice.index.weekday == 4
            assert not (sessions_last_indice[bv] + interval).isin(df.index.right).any()
            right_of_last_indice = sessions_last_indice[~bv] + interval
            assert right_of_last_indice[:-1].isin(df.index.right).all()

    def test_prices_inc_247(self, pricess):
        """Verify return from specific fixture."""
        prices = pricess["inc_247"]
        self.verify_limit_intraday_bi(prices)
        cc = prices.cc

        for interval in prices.BaseInterval[:-1]:
            limits = None
            # prices at 2min for BTC only available for same period as 1min data.
            if interval == prices.BaseInterval.T2:
                limits = prices.limits[prices.BaseInterval.T1]

            (start, end), slc = get_data_bounds(prices, interval, limits)
            df = prices._request_data(interval, start, end)

            sessions_durations = cc.closes[slc] - cc.opens[slc]
            sessions_rows = np.ceil(sessions_durations / interval)
            expected_num_rows = int(sessions_rows.sum())
            sessions_last_indice = cc.opens[slc] + (interval.as_pdtd * sessions_rows)

            start = cc.opens[slc].iloc[0]
            end = sessions_last_indice.iloc[-1]
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

            assert cc.opens[slc].isin(df.index.left).all()
            assert (cc.opens[slc] - interval)[1:].isin(df.index.left).all()
            assert (cc.opens[slc] + interval).isin(df.index.left).all()

            assert sessions_last_indice.isin(df.index.right).all()
            assert (sessions_last_indice + interval)[:-1].isin(df.index.right).all()
            assert (sessions_last_indice - interval).isin(df.index.right).all()

    def test_data_unavailable_for_symbol(self, pricess):
        """Verify where data for at least one symbol unavailable."""
        prices = pricess["us_lon"]
        _, slc = get_data_bounds(prices, prices.BaseInterval.T5)

        symbol_us = "MSFT"
        xnys = prices.calendars[symbol_us]
        xlon = prices.calendars["AZN.L"]

        common_index = xnys.opens[slc].index.intersection(xlon.opens[slc].index)
        delta = pd.Timedelta(2, "h")

        # don't use first index
        for i in reversed(range(len(common_index) - 1)):
            session = common_index[i]
            if session in _flakylist:
                continue
            start = xlon.opens[session]
            end = start + delta
            if end < xnys.opens[session]:
                break

        for interval in prices.BaseInterval[:-1]:
            df = prices._request_data(interval, start, end)
            assert df[symbol_us].isna().all(axis=None)
            expected_num_rows = delta // interval
            assertions_intraday(df, interval, prices, start, end, expected_num_rows)

    def test_start_end_session_minutes(self, pricess, one_min):
        """Verify where start and end are session minutes, not open or close."""
        prices = pricess["us_lon"]
        cc = prices.cc
        interval = prices.BaseInterval.T5
        _, slc = get_data_bounds(prices, interval)

        delta = pd.Timedelta(20, "min")
        start = cc.opens[slc].iloc[0] + delta
        end = cc.closes[slc].iloc[-1] - delta

        expected_num_rows, _ = self.get_expected_num_rows_us_lon(interval, cc, slc)
        expected_num_rows -= (delta // interval) * 2

        df = prices._request_data(interval, start, end)
        assertions_intraday(df, interval, prices, start, end, expected_num_rows)

        # verify limit of same return
        start_limit = start - (interval - one_min)
        end_limit = end + (interval - one_min)
        df = prices._request_data(interval, start_limit, end_limit)
        assertions_intraday(df, interval, prices, start, end, expected_num_rows)

        # verify beyond limit
        start_over_limit = start_limit - one_min
        df = prices._request_data(interval, start_over_limit, end_limit)
        assertions_intraday(
            df, interval, prices, start_over_limit, end, expected_num_rows + 1
        )

        end_over_limit = end_limit + one_min
        df = prices._request_data(interval, start_limit, end_over_limit)
        assertions_intraday(
            df, interval, prices, start, end_over_limit, expected_num_rows + 1
        )

        df = prices._request_data(interval, start_over_limit, end_over_limit)
        assertions_intraday(
            df,
            interval,
            prices,
            start_over_limit,
            end_over_limit,
            expected_num_rows + 2,
        )

    def test_start_end_non_session_minutes(
        self, pricess, session_length_xnys, session_length_xlon
    ):
        """Verify where start and end are not session minutes."""
        prices = pricess["us_lon"]
        interval = prices.BaseInterval.T5
        xnys = prices.calendars["MSFT"]
        xlon = prices.calendars["AZN.L"]

        session_start, session_end = get_valid_conforming_sessions(
            prices,
            interval,
            [xnys, xlon],
            [session_length_xnys, session_length_xlon],
            2,
        )
        start = xlon.session_open(session_start)
        end = xnys.session_close(session_end)
        df = prices._request_data(interval, start, end)
        delta = pd.Timedelta(20, "min")
        start_ = start - delta
        end_ = end + delta
        df_not_mins = prices._request_data(interval, start_, end_)
        try:
            # rtol for rare inconsistency in data returned by yahoo API when hit with
            # high frequency of requests, e.g. such as executing the test suite...
            assert_frame_equal(df, df_not_mins, rtol=0.075)
        except AssertionError:
            # ...and even then can fail.
            cols = (("MSFT", "volume"), ("AZN.L", "volume"))
            for df_, col in itertools.product((df, df_not_mins), cols):
                del df_[col]
            assert_frame_equal(df, df_not_mins, rtol=0.075)
            print(
                "test_start_end_non_session_minutes: letting freq_equal assertion"
                " pass with discrepancies in volume column(s)."
            )

    def test_start_none(self, pricess):
        """Verify as expected when start is None."""
        prices = pricess["inc_247"]
        end = pd.Timestamp.now().floor("min")
        start = None
        for interval in prices.BaseInterval[:-1]:
            match = (
                "`start` cannot be None if `interval` is intraday. `interval`"
                f"receieved as 'f{interval}'."
            )
            with pytest.raises(ValueError, match=match):
                prices._request_data(interval, start, end)

    @skip_if_fails_and_today_flakylisted(["XLON"])
    def test_live_indice(self, pricess):
        """Verify return with live indice as expected."""
        prices = pricess["inc_247"]
        start = pd.Timestamp.now(tz=UTC).floor("D") - pd.Timedelta(2, "D")
        for interval in prices.BaseInterval[:-1]:
            now = pd.Timestamp.now(tz=UTC).floor("min")
            end = now + interval
            df = prices._request_data(interval, start, end)
            num_rows = (now - start) / interval
            num_rows = np.ceil(num_rows) if num_rows % 1 else num_rows + 1
            expected_end = start + (num_rows * interval)
            assertions_intraday(df, interval, prices, start, expected_end, num_rows)

        symbol = "BTC-GBP"
        delay_mins = 15
        prices = m.PricesYahoo(symbol, calendars="24/7", delays=delay_mins)
        cal = prices.calendars[symbol]
        cc = calutils.CompositeCalendar([cal])
        delay = pd.Timedelta(delay_mins, "min")
        start = pd.Timestamp.now(tz=UTC).floor("D") - pd.Timedelta(2, "D")
        pp = {
            "minutes": 0,
            "hours": 0,
            "days": 0,
            "weeks": 0,
            "months": 0,
            "years": 0,
            "start": start,
            "end": None,
            "add_a_row": False,
        }
        for interval in prices.BaseInterval[:-1]:
            drg = daterange.GetterIntraday(
                cal, cc, delay, prices.limits[interval][0], False, pp.copy(), interval
            )
            (_, end), _ = drg.daterange
            df = prices._request_data(interval, start, end)
            now = pd.Timestamp.now(tz=UTC).floor("min")
            num_rows = (now - delay - start) / interval
            num_rows = np.ceil(num_rows) if num_rows % 1 else num_rows + 1
            expected_end = start + (num_rows * interval)
            assertions_intraday(df, interval, prices, start, expected_end, num_rows)

    def test_one_min_interval(self, pricess):
        """Verify one minute interval as expected.

        Verifies return on and either side of loops that get and consolidate
        chunks of T1 data.
        """
        prices = pricess["only_247"]
        interval = prices.BaseInterval.T1
        for days in [5, 6, 7, 11, 12, 13]:
            end = pd.Timestamp.now(tz=UTC).ceil("min")
            start = end - pd.Timedelta(days, "D")
            df = prices._request_data(interval, start, end)
            num_expected_rows = days * 24 * 60
            assertions_intraday(df, interval, prices, start, end, num_expected_rows)

    def test_volume_glitch(self, pricess):
        """Verify volume glitch fix being applied where can be."""

        def assertions(
            prices: m.PricesYahoo,
            start: pd.Timestamp,
            end: pd.Timestamp,
            prev_close: pd.Timestamp | None = None,
        ):
            symbol = prices.symbols[0]
            for interval in prices.bis_intraday:
                interval_ = prices._bi_to_source_key(interval)
                hist = prices._ticker.history(
                    interval=interval_, start=start, end=end, adj_timezone=False
                )
                assert isinstance(hist, pd.DataFrame)
                hist_s = hist.loc[symbol].copy()
                if hist_s.iloc[0].isna().any() and hist_s.index[0] < start:
                    # because v rare bug can introduce an initial row with missing
                    # values and indice < start
                    hist_s = hist_s[1:].copy()
                if not hist_s.index[-2] + interval == hist_s.index[-1]:
                    # yahoo can return live indice at end of table
                    hist_s = hist_s[:-1].copy()

                hist0 = hist_s.iloc[0]
                indice = hist0.name

                df = prices._request_data(interval, start, end)[symbol]
                df0_vol = df[indice:indice].volume.iloc[0]

                # verify glitch in hist not present in df
                if prev_close is None:
                    assert df0_vol
                assert (not hist0.volume) | (hist0.volume <= df0_vol)

                # verify volume for df indice
                if prev_close is not None:
                    start_alt = prev_close - interval
                else:
                    start_alt = start - interval
                hist_alt = prices._ticker.history(
                    interval=interval_, start=start_alt, end=end, adj_timezone=False
                )
                assert isinstance(hist_alt, pd.DataFrame)
                hist_alt_s = hist_alt.loc[symbol].copy()

                if not hist_alt_s.index[-2] + interval == hist_alt_s.index[-1]:
                    # yahoo can return live indice at end of table
                    hist_alt_s = hist_alt_s[:-1].copy()

                assert hist_alt_s.loc[indice].name == df[indice:indice].index[0].left
                assert hist_alt_s.loc[indice].volume == df0_vol

                # verify volume for rest of hist is as df
                # hist will be missing any row for which at least one tick not registered
                hist_vol = hist_s[1:].volume
                df_vol = df.pt.indexed_left.loc[hist_vol.index].volume
                try:
                    assert (hist_vol == df_vol).all()
                except AssertionError:
                    # provide for rare inconsistency in yahoo return when receives
                    # high freq of requests, e.g. under execution of test suite.
                    bv = hist_vol != df_vol
                    diff = bv.sum() / len(bv)
                    print(
                        "\ntest_volume_glitch: letting hist_vol == df_vol assertion"
                        f" pass with discrepancies in {diff:.2%} of rows."
                    )
                    # if not working off the same base data then df could have been
                    # evaluated from volume data with different indices to hist, in
                    # which case following bv.all() assertion could (and rarely does)
                    # fail.
                    return
                not_in_hist = df[1:].pt.indexed_left.index.difference(hist_vol.index)
                bv = df.loc[not_in_hist].volume == 0  # pylint: disable=compare-to-zero
                assert bv.all()

        # Verify for us prices
        prices = pricess["us"]
        symbol = prices.symbols[0]
        cal = prices.calendars[symbol]
        now = pd.Timestamp.now(UTC).floor("min")
        session = cal.minute_to_past_session(now, 2)
        session = get_valid_session(session, cal, "previous")
        # extra 30T to cover unaligned end of 1H interval
        end = cal.session_close(session) + pd.Timedelta(30, "min")
        start = cal.session_open(session) + pd.Timedelta(1, "h")
        assertions(prices, start, end)

        # Verify for lon prices
        prices = m.PricesYahoo("AZN.L", calendars="XLON", delays=15)
        cal = prices.calendars["AZN.L"]
        now = pd.Timestamp.now(UTC).floor("min")
        session = cal.minute_to_past_session(now, 2)
        session = get_valid_session(session, cal, "previous")
        # extra 30T to cover unaligned end of 1H interval
        end = cal.session_close(session) + pd.Timedelta(30, "min")
        start = cal.session_open(session)
        prev_close = cal.previous_close(session)
        assertions(prices, start, end, prev_close)


# =========================================================================
# Following are PricesBase methods that are tested here only for
# convenience of using fixtures and helpers defined for tests here.


def test__get_bi_table(pricess):
    """Test `_get_bi_table`.

    Test limited to getting an intraday table and daily table for specific
    dateranges.
    """
    prices = pricess["us"]
    interval = prices.bis.T5
    to = pd.Timestamp.now()
    from_ = to - pd.Timedelta(21, "D")
    (start, _), slc = get_data_bounds(prices, interval, (from_, to))
    end = prices.cc.closes[slc].iloc[-1]
    table = prices._get_bi_table(interval, (start, end))

    bounds, num_rows, sessions_end = expected_table_structure_us(prices, interval, slc)
    assertions_intraday_us(prices, table, interval, bounds, num_rows, slc, sessions_end)

    delta = interval * 3
    start += delta
    end -= delta
    table = prices._get_bi_table(interval, (start, end))
    num_rows -= 6
    assertions_intraday(table, interval, prices, start, end, num_rows)

    sessions = prices.cc.opens[slc].index
    daterange = sessions[0], sessions[-1]
    table = prices._get_bi_table(prices.bi_daily, daterange)
    assertions_daily(table, prices, *daterange)
