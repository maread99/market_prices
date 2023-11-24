"""Common pytest fixtures."""

from __future__ import annotations

from collections import abc
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from hypothesis import settings, HealthCheck
import exchange_calendars as xcals
import pandas as pd
import pytest

from market_prices.intervals import TDInterval
from .utils import Answers, clean_temp_test_dir, TEMP_DIR

# pylint: disable=missing-any-param-doc,redefined-outer-name,
# pylint: disable=unused-argument  # `mock_now` has effect in background

settings.register_profile("ci", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

# pylint: disable=no-member
base_intervals_sample = [
    TDInterval.T1,
    TDInterval.T2,
    TDInterval.T5,
    TDInterval.T15,
    TDInterval.H1,
]
# pylint: enable=no-member

UTC = ZoneInfo("UTC")


@pytest.fixture
def utc() -> abc.Iterator[ZoneInfo]:
    """UTC zoneinfo."""
    yield UTC


@pytest.fixture
def temp_dir() -> abc.Iterator[Path]:
    """Clean temporary test directory.

    Directory cleaned before and after test.
    """
    clean_temp_test_dir()
    yield TEMP_DIR
    clean_temp_test_dir()


@pytest.fixture(scope="session")
def base_intervals() -> abc.Iterator[list[TDInterval]]:
    """Sample of base intervals."""
    yield base_intervals_sample


@pytest.fixture(scope="session", params=base_intervals_sample)
def base_interval(request) -> abc.Iterator[list[TDInterval]]:
    """Parameterized fixture of base interval samples."""
    yield request.param


def _get_ds_intervals(interval: TDInterval) -> list[TDInterval | None]:
    """Sample of ds_intervals that are valid for a given intraday `interval`."""
    all_intervals = TDInterval.intraday_intervals()

    step = interval.as_minutes
    start = step - 1
    stop = all_intervals[-1].as_minutes + 1
    all_ds_intervals = all_intervals[start:stop:step]

    factors = [3, 7, 13, 27, 50, 77]
    if interval is TDInterval.T1:  # pylint: disable=no-member
        factors.extend([120, 360])
    indices = [factor - 1 for factor in factors]

    ds_intervals: list[TDInterval | None] = [
        all_ds_intervals[i] for i in indices if i < len(all_ds_intervals)
    ]
    ds_intervals.insert(0, None)

    last_ds_interval = all_ds_intervals[-1]
    if last_ds_interval not in ds_intervals:
        ds_intervals.append(last_ds_interval)

    return ds_intervals


base_ds_intervals_dict = {bi: _get_ds_intervals(bi) for bi in base_intervals_sample}
base_ds_intervals_list = [
    (bi, dsi, bi if dsi is None else dsi)
    for bi, dsis in base_ds_intervals_dict.items()
    for dsi in dsis
]


@pytest.fixture(scope="session")
def base_ds_intervals() -> abc.Iterator[dict[TDInterval, list[TDInterval | None]]]:
    """Sample of base intervals with valid ds_intervals.

    Yields
    ------
    dict
        key : TDInterval
            sample base interval.
        value : list[TDInterval]
            list of valid ds_interval for sample base interval.
    """
    yield base_ds_intervals_dict


@pytest.fixture(scope="session", params=base_ds_intervals_list)
def base_ds_interval(
    request,
) -> abc.Iterator[tuple[TDInterval, TDInterval | None, TDInterval]]:
    """Parameterized fixture of all sample interval / ds_interval combinations.

    Yields
    ------
    tuple
        [0] base interval
        [1] downsample interval
        [2] base interval if downsample interval is None, otherwise
            downsample interval.
    """
    yield request.param


@pytest.fixture(scope="session")
def zero_td() -> abc.Iterator[pd.Timedelta]:
    """pd.Timedelta with zero value."""
    yield pd.Timedelta(0)


@pytest.fixture(scope="session")
def one_day() -> abc.Iterator[pd.Timedelta]:
    """pd.Timedelta with value as one day."""
    yield pd.Timedelta(1, "D")


@pytest.fixture(scope="session")
def one_min() -> abc.Iterator[pd.Timedelta]:
    """pd.Timedelta with value as one minute."""
    yield pd.Timedelta(1, "T")


@pytest.fixture(scope="session")
def one_sec() -> abc.Iterator[pd.Timedelta]:
    """pd.Timedelta with value as one second."""
    yield pd.Timedelta(1, "S")


_now_utc = pd.Timestamp("2021-11-17 21:59", tz=UTC)


@pytest.fixture(scope="session")
def now_utc() -> abc.Iterator[pd.Timestamp]:
    """Time 'now'.

    Time is the time 'now' for any test that includes the 'mock_now'
    fixture as an argument directly or indirectly. See `mock_now`.__doc__.
    """
    yield _now_utc


@pytest.fixture(scope="session")
def now(now_utc) -> abc.Iterator[pd.Timestamp]:
    """Time 'now' as timezone-naive pd.Timestamp.

    Other that having timezone as None, yields as `now_utc` fixture.
    """
    yield now_utc.tz_convert(None)


@pytest.fixture(scope="session")
def today(now) -> abc.Iterator[pd.Timestamp]:
    """Timestamp representing 'today' according to `now` fixture."""
    yield now.floor("D")


@pytest.fixture(scope="class")
def mock_now(class_mocker, now_utc):
    """Mock pd.Timestamp.now() to fixture `now_utc`.

    The time 'now' will be mocked ot the value of the `now_utc` fixure for
    any test that takes this fixture as an argument directly or indirectly
    - i.e. via a fixture that in turn takes this fixture as an argument.

    Fixtures that take this fixture as an argument include fixtures that
    yield calendars.
    """

    def _mock_now(*_, tz=None, **__) -> pd.Timestamp:
        return now_utc.tz_convert(tz)

    class_mocker.patch("pandas.Timestamp.now", _mock_now)


@pytest.fixture(scope="class")
def side() -> abc.Iterator[str]:
    """Side to be used for any created calendar."""
    yield "left"


_calendar_names = ["24/7", "XHKG", "CMES", "XLON"]


@pytest.fixture(scope="class", params=_calendar_names)
def calendars(request, today, side, mock_now) -> abc.Iterator[xcals.ExchangeCalendar]:
    """Four calendars of distinct behaviour.

    XLON - standard (no breaks, has gaps between sessions and has holidays)
    24/7 - always open
    XHKG - has breaks
    CMES - 24h with gaps at weekends
    """
    yield xcals.get_calendar(request.param, side=side, end=today)


@pytest.fixture(scope="class")
def calendars_with_answers(
    calendars,
) -> abc.Iterator[tuple[xcals.ExchangeCalendar, Answers]]:
    """Calendars with corresponding answers."""
    calendar = calendars
    yield calendar, Answers(calendar)


@pytest.fixture(scope="class")
def calendar_end_extended(today, one_day) -> abc.Iterator[pd.Timestamp]:
    """End date for 'extended' calendars."""
    # Cannot be less than 15 as need to be able to query dates up to 1st of month
    # following now.
    yield today + (one_day * 15)


@pytest.fixture(scope="class", params=_calendar_names)
def calendars_extended(
    request, calendar_end_extended, side, mock_now
) -> abc.Iterator[xcals.ExchangeCalendar]:
    """As `calendars` with last session one week after 'today'."""
    yield xcals.get_calendar(request.param, side=side, end=calendar_end_extended)


@pytest.fixture(scope="class")
def calendars_with_answers_extended(
    calendars_extended,
) -> abc.Iterator[tuple[xcals.ExchangeCalendar, Answers]]:
    """Calendars with corresponding answers."""
    calendar = calendars_extended
    yield calendar, Answers(calendar)


@pytest.fixture(scope="class")
def xlon_calendar(today, side, mock_now) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XLON calendar."""
    yield xcals.get_calendar("XLON", side=side, end=today)


@pytest.fixture(scope="class")
def xlon_calendar_with_answers(
    xlon_calendar,
) -> abc.Iterator[tuple[xcals.ExchangeCalendar, Answers]]:
    """XLON calendar and corresponding answers."""
    yield xlon_calendar, Answers(xlon_calendar)


@pytest.fixture(scope="class")
def xlon_calendar_extended(
    calendar_end_extended, side, mock_now
) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XLON calendar with extended end."""
    yield xcals.get_calendar("XLON", side=side, end=calendar_end_extended)


@pytest.fixture(scope="class")
def xhkg_calendar(today, side, mock_now) -> abc.Iterator[xcals.ExchangeCalendar]:
    """XLON calendar."""
    yield xcals.get_calendar("XHKG", side=side, end=today)
