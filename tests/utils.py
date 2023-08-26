"""Utility functions for test suite."""

from __future__ import annotations

import functools
import pathlib
from collections import abc
from typing import Literal, TYPE_CHECKING
import shelve

import exchange_calendars as xcals
from exchange_calendars.utils import pandas_utils as xcals_pdutils
import numpy as np
import pandas as pd
from pytz import UTC

from market_prices import intervals
from market_prices.utils import pandas_utils as pdutils
from market_prices.prices.base import PricesBase

# pylint: disable=missing-function-docstring, missing-param-doc, too-many-lines

ONE_DAY = pd.Timedelta(1, "D")

_RESOURCES_PATH = pathlib.Path(__file__).parent.joinpath("resources")
STORE_PATH = _RESOURCES_PATH.joinpath("store.h5")
STORE_PBT_PATH = _RESOURCES_PATH.joinpath("store_pbt.h5")
SHELF_PATH = _RESOURCES_PATH.joinpath("pbt")
_INDEXES_SUFFIX = "_indexes"
BI_STR = ["T1", "T2", "T5", "H1", "D1"]

# Frequency information lost on string
multiple_sessions_freq = xcals.get_calendar("XNYS", start="2022").day * 3


def get_store(mode: str = "a") -> pd.HDFStore:
    """Return store holding general data.

    Parameters
    ----------
    mode : default: "a" (append)
        Mode in which store should be opened.
    """
    return pd.HDFStore(STORE_PATH, mode=mode)


def get_store_pbt(mode: str = "a") -> pd.HDFStore:
    """Return store holding PricesBaseTst data.

    Parameters
    ----------
    mode : default: "a" (append)
        Mode in which store should be opened.
    """
    return pd.HDFStore(STORE_PBT_PATH, mode=mode)


def get_shelf() -> shelve.DbfilenameShelf:
    """Return pbt resoureces shelf.

    Shelf has keys as pbt resource keys and values as pd.Timestamp
    corresponding to time when resource was created.
    """
    return shelve.open(SHELF_PATH.as_posix())


def is_key(key: str, store_path: pathlib.Path = STORE_PATH) -> bool:
    """Query if a given key is an existing key of a resources store.

    Parameters
    ----------
    key:
        key to query.

    store_path:
        Path to store.
    """
    with pd.HDFStore(store_path, mode="r") as store:
        rtrn = key in store
    return rtrn


def save_resource(
    resource: pd.DataFrame | pd.Series,
    key: str,
    overwrite: bool = False,
    store_path: pathlib.Path = STORE_PATH,
):
    """Save a resource to a store.

    Parameters
    ----------
    resource
        Pandas object to save.

    key
        key under which to save resource.

    overwrite
        Overwrite any exisiting item stored under `key`. Raises KeyError if
        False and key already exists.

    store_path
        Path to store object which resource is to be stored to.
    """
    if not overwrite and is_key(key):
        raise KeyError(
            f'key "{key}" already exists. To overwrite an existing item pass'
            " `overwrite` as True."
        )

    if isinstance(resource.index, pd.IntervalIndex):
        index = resource.index
        df_indexes = pd.DataFrame(dict(left=index.left, right=index.right))
        df_indexes.to_hdf(store_path, key + _INDEXES_SUFFIX)
        resource = resource.reset_index(drop=True)
    resource.to_hdf(store_path, key)


def get_bi_key(key: str, bi: intervals.BI | str) -> str:
    """Return key for a base interval table of a pbt resource."""
    if isinstance(bi, intervals.BI):
        bi = bi.name
    return key + "_" + bi


def remove_resource_pbt(key: str, store_only: bool = False):
    """Remove a resource from the pbt store.

    Parameters
    ----------
    store_only
        True - remove resource from the store only, not the shelf.
    """
    if not store_only:
        with get_shelf() as shelf:
            if key not in shelf:
                raise KeyError(f'key "{key}" is not in a key of the pbt resources.')

    with get_store_pbt() as store:
        all_keys = list(store.keys())
        for key_ in all_keys:
            if key_[1:].startswith(key):
                del store[key_]

    if not store_only:
        with get_shelf() as shelf:
            del shelf[key]


def save_resource_pbt(
    prices: PricesBase,
    key: str,
    overwrite: bool = False,
):
    """Save all available data for `prices` to the PricesBaseTst store.

    Parameters
    ----------
    prices
        prices instance for which to save all available data. (All available
        data assumed as all stored data after calling prices.request_all_prices.)

    key
        key under which to save resource.

    overwrite
        Overwrite any exisiting item stored under `key`. Raises KeyError if
        False and key already exists.
    """
    with get_shelf() as shelf:
        if not overwrite and key in shelf:
            raise KeyError(
                f'key "{key}" already exists. To overwrite an existing item pass'
                " `overwrite` as True."
            )

        now = pd.Timestamp.now(tz="UTC")
        prices.request_all_prices()
        if pd.Timestamp.now(tz="UTC").floor("T") != now.floor("T"):
            remove_resource_pbt(key, store_only=True)
            raise RuntimeError(
                "Operation aborted as unable to get all data within the same minute."
            )

        for bi in prices.bis:
            df = prices._pdata[bi]._table  # pylint: disable=protected-access
            if df is None:
                # indices of different symbols unaligned
                continue
            if bi.is_daily:
                # cannot store CustomBusinessDay to HDFStore
                df.index.freq = None
            save_resource(df, get_bi_key(key, bi), overwrite, STORE_PBT_PATH)
        shelf[key] = now


def _has_interval_index(key: str, store_path: pathlib.Path) -> bool:
    return is_key(key + _INDEXES_SUFFIX, store_path)


def _regenerate_interval_index(key: str, store_path: pathlib.Path) -> pd.IntervalIndex:
    df = pd.read_hdf(store_path, key + _INDEXES_SUFFIX)
    return pd.IntervalIndex.from_arrays(df.left, df.right, closed="left")


def get_resource(key: str, store_path: pathlib.Path = STORE_PATH) -> pd.DataFrame:
    """Get a resource from the store.

    Parameters
    ----------
    key
        key under which resource saved.

    store_path
        Path to store object which resource is stored to.
    """
    resource = pd.read_hdf(store_path, key)
    if _has_interval_index(key, store_path):
        index = _regenerate_interval_index(key, store_path)
        resource.index = index
    if key.startswith("multiple_sessions_pt"):
        resource.index.left.freq = multiple_sessions_freq
        resource.index.right.freq = multiple_sessions_freq
    return resource


def get_resource_pbt(key: str) -> tuple[dict[str, pd.DataFrame], pd.Timstamp]:
    """Get a resource from the PricesBaseTst store.

    Parameters
    ----------
    key
        key under which resource saved.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.Timstamp]:
        [0] dictionary mapping base interval (as names) with prices tables.
        [1] timestamp when prices tables were created.
    """
    with get_shelf() as shelf:
        if key not in shelf:
            raise ValueError(f"There is no pbt resource under the key {key}")
        now = shelf[key]

    d = {}
    for BI in BI_STR:  # # pylint: disable=invalid-name
        bi_key = get_bi_key(key, BI)
        try:
            d[BI] = get_resource(bi_key, STORE_PBT_PATH)
        except KeyError:
            # indices of symbols unaligned at H1
            assert BI == "H1"
            continue
    return d, now


class Answers:
    """Inputs and expected output for testing a given calendar.

    Inputs and expected outputs are provided by public instance methods and
    properties. These are read directly from the corresponding schedule.

    Parameters
    ----------
    calendar
        Calendar for which require answer info.
    """

    # pylint: disable=too-many-public-methods

    ONE_MIN = pd.Timedelta(1, "T")
    TWO_MIN = pd.Timedelta(2, "T")
    ONE_DAY = pd.Timedelta(1, "D")

    LEFT_SIDES = ["left", "both"]
    RIGHT_SIDES = ["right", "both"]

    def __init__(
        self,
        calendar: xcals.ExchangeCalendar,
    ):
        self._calendar = calendar
        self._answers = calendar.schedule.copy(deep=True)

    # --- Exposed constructor arguments ---

    @property
    def calendar(self) -> xcals.ExchangeCalendar:
        """Corresponding calendar."""
        return self._calendar

    @property
    def name(self) -> str:
        """Name of corresponding calendar."""
        return self.calendar.name

    @property
    def side(self) -> str:
        """Side of calendar for which answers valid."""
        return self.calendar.side

    # --- Properties read from schedule ---

    @property
    def answers(self) -> pd.DataFrame:
        """Answers as correspoding csv."""
        return self._answers

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """Session labels."""
        return self.answers.index

    @property
    def opens(self) -> pd.Series:
        """Market open time for each session."""
        return self.answers.open

    @property
    def closes(self) -> pd.Series:
        """Market close time for each session."""
        return self.answers.close

    @property
    def break_starts(self) -> pd.Series:
        """Break start time for each session."""
        return self.answers.break_start

    @property
    def break_ends(self) -> pd.Series:
        """Break end time for each session."""
        return self.answers.break_end

    # --- get and helper methods ---

    def get_next_session(self, session: pd.Timestamp) -> pd.Timestamp:
        """Get session that immediately follows `session`."""
        if session == self.last_session:
            raise IndexError("Cannot get session later than last answers' session.")
        idx = self.sessions.get_loc(session) + 1
        return self.sessions[idx]

    def get_next_sessions(self, session: pd.Timestamp, count: int) -> pd.Timestamp:
        """Get `count` consecutive sessions starting with `session`."""
        assert count >= 0, "count can only take positive integers."
        start = self.sessions.get_loc(session)
        stop = start + count
        if stop > len(self.sessions):
            raise IndexError("Cannot get sessions later than last answers' session.")
        return self.sessions[start:stop]

    def get_prev_session(self, session: pd.Timestamp) -> pd.Timestamp:
        """Get session that immediately preceeds `session`."""
        if session == self.first_session:
            raise IndexError("Cannot get session earlier than first answers' session.")
        idx = self.sessions.get_loc(session) - 1
        return self.sessions[idx]

    def get_prev_sessions(self, session: pd.Timestamp, count: int) -> pd.Timestamp:
        """Get `count` consecutive sessions ending with `session`."""
        assert count >= 0, "count can only take positive integers."
        stop = self.sessions.get_loc(session) + 1
        start = stop - count
        if start < 0:
            raise IndexError("Cannot get sessions earlier than first answers' session.")
        return self.sessions[start:stop]

    def date_to_session(
        self, date: pd.Timestamp, direction: Literal["next", "previous"]
    ) -> pd.Timestamp:
        """Get session related to `date`.

        If `date` is a session, will return `date`, otherwise will return
        session that immediately preceeds/follows `date` if `direction` is
        "next"/"previous" respectively.
        """
        if date < self.first_session or date > self.last_session:
            msg = (
                "`date` must be within answers' first session and last session,"
                f" although receieved as {date}."
            )
            raise ValueError(msg)
        method = "bfill" if direction == "next" else "ffill"
        idx = self.sessions.get_indexer([date], method)[0]
        return self.sessions[idx]

    def session_has_break(self, session: pd.Timestamp) -> bool:
        """Query if `session` has a break."""
        return session in self.sessions_with_break

    @staticmethod
    def get_sessions_sample(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return sample of given `sessions`.

        Sample includes:
            All sessions within first two years of `sessions`.
            All sessions within last two years of `sessions`.
            All sessions falling:
                within first 3 days of any month.
                from 28th of any month.
                from 14th through 16th of any month.
        """
        if sessions.empty:
            return sessions

        mask = (
            (sessions < sessions[0] + pd.DateOffset(years=2))
            | (sessions > sessions[-1] - pd.DateOffset(years=2))
            | (sessions.day <= 3)
            | (sessions.day >= 28)
            | (14 <= sessions.day) & (sessions.day <= 16)
        )
        return sessions[mask]

    def get_sessions_minutes(
        self, start: pd.Timestamp, end: pd.Timestamp | int = 1
    ) -> pd.DatetimeIndex:
        """Get trading minutes for 1 or more consecutive sessions.

        Parameters
        ----------
        start
            Session from which to get trading minutes.
        end
            Session through which to get trading mintues. Can be passed as:
                pd.Timestamp: return will include trading minutes for `end`
                    session.
                int: where int represents number of consecutive sessions
                    inclusive of `start`, for which require trading
                    minutes. Default is 1, such that by default will return
                    trading minutes for only `start` session.
        """
        idx = self.sessions.get_loc(start)
        stop = idx + end if isinstance(end, int) else self.sessions.get_loc(end) + 1
        indexer = slice(idx, stop)

        dtis = []
        for first, last, last_am, first_pm in zip(
            self.first_minutes[indexer],
            self.last_minutes[indexer],
            self.last_am_minutes[indexer],
            self.first_pm_minutes[indexer],
        ):
            if pd.isna(last_am):
                dtis.append(pd.date_range(first, last, freq="T"))
            else:
                dtis.append(pd.date_range(first, last_am, freq="T"))
                dtis.append(pd.date_range(first_pm, last, freq="T"))

        index = pdutils.indexes_union(dtis)
        assert isinstance(index, pd.DatetimeIndex)
        return index

    def get_session_minutes(
        self,
        session: pd.Timestamp,
        ignore_breaks: bool = False,
    ) -> tuple[pd.DatetimeIndex, ...]:
        """Get trading minutes a single `session`.

        Parameters
        ----------
        ignore_breaks
            If `session` has a break, treat as if were one continuous
            session.

        Returns
        -------
        tuple[pd.DatetimeIndex, ...]
            If `session` has a break, returns 2-tuple where:
                [0] minutes of am session.
                [1] minutes of pm session.
            If `session` does not have a break, returns 1-tuple with
            element holding minutes of session.
        """
        first = self.first_minutes[session]
        last = self.last_minutes[session]
        last_am = self.last_am_minutes[session]
        first_pm = self.first_pm_minutes[session]

        if pd.isna(last_am) or ignore_breaks:
            return (pd.date_range(first, last, freq="T"),)
        else:
            return (
                pd.date_range(first, last_am, freq="T"),
                pd.date_range(first_pm, last, freq="T"),
            )

    def get_session_break_minutes(self, session: pd.Timestamp) -> pd.DatetimeIndex:
        """Get break minutes for single `session`."""
        if not self.session_has_break(session):
            return pd.DatetimeIndex([])
        else:
            minutes = self.get_session_minutes(session)
            # pylint - `get_session_minutes` got 2 items as session had break
            am_mins, pm_mins = minutes  # pylint: disable=unbalanced-tuple-unpacking
        first = am_mins[-1] + self.ONE_MIN
        last = pm_mins[0] - self.ONE_MIN
        return pd.date_range(first, last, freq="T")

    def get_session_edge_minutes(
        self, session: pd.Timestamp, delta: int | pd.Timedelta = 0
    ) -> pd.DatetimeIndex:
        """Get edge trading minutes for a `session`.

        Return will include first and last trading minutes of session and,
        if applicable, subsessions. Passing `delta` will double length
        of return by including trading minutes at `delta` minutes 'inwards'
        from the standard edge minutes. NB `delta` should be less than
        the session/subsession duration - this condition is NOT
        VERIFIED by this method.
        """
        if isinstance(delta, int):
            delta = pd.Timedelta(delta, "T")
        first_minute = self.first_minutes[session]
        last_minute = self.last_minutes[session]
        has_break = self.session_has_break(session)
        if has_break:
            last_am_minute = self.last_am_minutes[session]
            first_pm_minute = self.first_pm_minutes[session]

        minutes = [first_minute, last_minute]
        if delta:
            minutes.append(first_minute + delta)
            minutes.append(last_minute - delta)
        if has_break:
            last_am_minute = self.last_am_minutes[session]
            first_pm_minute = self.first_pm_minutes[session]
            minutes.extend([last_am_minute, first_pm_minute])
            if delta:
                minutes.append(last_am_minute - delta)
                minutes.append(first_pm_minute + delta)

        return pd.DatetimeIndex(minutes)

    # --- Evaluated general calendar properties ---

    @functools.lru_cache(maxsize=4)
    def _has_a_session_with_break(self) -> bool:
        return self.break_starts.notna().any()

    @property
    def has_a_session_with_break(self) -> bool:
        """Does any session of answers have a break."""
        return self._has_a_session_with_break()

    @property
    def has_a_session_without_break(self) -> bool:
        """Does any session of answers not have a break."""
        return self.break_starts.isna().any()

    # --- Evaluated properties for first and last sessions ---

    @property
    def first_session(self) -> pd.Timestamp:
        """First session covered by answers."""
        return self.sessions[0]

    @property
    def last_session(self) -> pd.Timestamp:
        """Last session covered by answers."""
        return self.sessions[-1]

    @property
    def sessions_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last sessions covered by answers."""
        return self.first_session, self.last_session

    @property
    def first_session_open(self) -> pd.Timestamp:
        """Open time of first session covered by answers."""
        return self.opens[0]

    @property
    def last_session_close(self) -> pd.Timestamp:
        """Close time of last session covered by answers."""
        return self.closes[-1]

    @property
    def first_minute(self) -> pd.Timestamp:
        open_ = self.first_session_open
        return open_ if self.side in self.LEFT_SIDES else open_ + self.ONE_MIN

    @property
    def last_minute(self) -> pd.Timestamp:
        close = self.last_session_close
        return close if self.side in self.RIGHT_SIDES else close - self.ONE_MIN

    @property
    def trading_minutes_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last trading minutes covered by answers."""
        return self.first_minute, self.last_minute

    # --- out-of-bounds properties ---

    @property
    def minute_too_early(self) -> pd.Timestamp:
        """Minute earlier than first trading minute."""
        return self.first_minute - self.ONE_MIN

    @property
    def minute_too_late(self) -> pd.Timestamp:
        """Minute later than last trading minute."""
        return self.last_minute + self.ONE_MIN

    @property
    def session_too_early(self) -> pd.Timestamp:
        """Date earlier than first session."""
        return self.first_session - self.ONE_DAY

    @property
    def session_too_late(self) -> pd.Timestamp:
        """Date later than last session."""
        return self.last_session + self.ONE_DAY

    # --- Evaluated properties covering every session. ---

    @functools.lru_cache(maxsize=4)
    def _first_minutes(self) -> pd.Series:
        if self.side in self.LEFT_SIDES:
            minutes = self.opens.copy()
        else:
            minutes = self.opens + self.ONE_MIN
        minutes.name = "first_minutes"
        return minutes

    @property
    def first_minutes(self) -> pd.Series:
        """First trading minute of each session (UTC)."""
        return self._first_minutes()

    @property
    def first_minutes_plus_one(self) -> pd.Series:
        """First trading minute of each session plus one minute."""
        return self.first_minutes + self.ONE_MIN

    @property
    def first_minutes_less_one(self) -> pd.Series:
        """First trading minute of each session less one minute."""
        return self.first_minutes - self.ONE_MIN

    @functools.lru_cache(maxsize=4)
    def _last_minutes(self) -> pd.Series:
        if self.side in self.RIGHT_SIDES:
            minutes = self.closes.copy()
        else:
            minutes = self.closes - self.ONE_MIN
        minutes.name = "last_minutes"
        return minutes

    @property
    def last_minutes(self) -> pd.Series:
        """Last trading minute of each session."""
        return self._last_minutes()

    @property
    def last_minutes_plus_one(self) -> pd.Series:
        """Last trading minute of each session plus one minute."""
        return self.last_minutes + self.ONE_MIN

    @property
    def last_minutes_less_one(self) -> pd.Series:
        """Last trading minute of each session less one minute."""
        return self.last_minutes - self.ONE_MIN

    @functools.lru_cache(maxsize=4)
    def _last_am_minutes(self) -> pd.Series:
        if self.side in self.RIGHT_SIDES:
            minutes = self.break_starts.copy()
        else:
            minutes = self.break_starts - self.ONE_MIN
        minutes.name = "last_am_minutes"
        return minutes

    @property
    def last_am_minutes(self) -> pd.Series:
        """Last pre-break trading minute of each session.

        NaT if session does not have a break.
        """
        return self._last_am_minutes()

    @property
    def last_am_minutes_plus_one(self) -> pd.Series:
        """Last pre-break trading minute of each session plus one minute."""
        return self.last_am_minutes + self.ONE_MIN

    @property
    def last_am_minutes_less_one(self) -> pd.Series:
        """Last pre-break trading minute of each session less one minute."""
        return self.last_am_minutes - self.ONE_MIN

    @functools.lru_cache(maxsize=4)
    def _first_pm_minutes(self) -> pd.Series:
        if self.side in self.LEFT_SIDES:
            minutes = self.break_ends.copy()
        else:
            minutes = self.break_ends + self.ONE_MIN
        minutes.name = "first_pm_minutes"
        return minutes

    @property
    def first_pm_minutes(self) -> pd.Series:
        """First post-break trading minute of each session.

        NaT if session does not have a break.
        """
        return self._first_pm_minutes()

    @property
    def first_pm_minutes_plus_one(self) -> pd.Series:
        """First post-break trading minute of each session plus one minute."""
        return self.first_pm_minutes + self.ONE_MIN

    @property
    def first_pm_minutes_less_one(self) -> pd.Series:
        """First post-break trading minute of each session less one minute."""
        return self.first_pm_minutes - self.ONE_MIN

    # --- Evaluated session sets and ranges that meet a specific condition ---

    @property
    def _mask_breaks(self) -> pd.Series:
        return self.break_starts.notna()

    @functools.lru_cache(maxsize=4)
    def _sessions_with_break(self) -> pd.DatetimeIndex:
        return self.sessions[self._mask_breaks]

    @property
    def sessions_with_break(self) -> pd.DatetimeIndex:
        return self._sessions_with_break()

    @functools.lru_cache(maxsize=4)
    def _sessions_without_break(self) -> pd.DatetimeIndex:
        return self.sessions[~self._mask_breaks]

    @property
    def sessions_without_break(self) -> pd.DatetimeIndex:
        return self._sessions_without_break()

    @property
    def sessions_without_break_run(self) -> pd.DatetimeIndex:
        """Longest run of consecutive sessions without a break."""
        s = self.break_starts.isna()
        if s.empty:
            return pd.DatetimeIndex([])
        return xcals_pdutils.longest_run(s)

    @property
    def sessions_without_break_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Longest session range that does not include a session with a break.

        Returns None if all sessions have a break.
        """
        sessions = self.sessions_without_break_run
        if sessions.empty:
            return None
        return sessions[0], sessions[-1]

    @property
    def _mask_sessions_without_gap_after(self) -> pd.Series:
        if self.side == "neither":
            # will always have gap after if neither open or close are trading
            # minutes (assuming sessions cannot overlap)
            return pd.Series(False, index=self.sessions)

        elif self.side == "both":
            # a trading minute cannot be a minute of more than one session.
            assert not (self.closes == self.opens.shift(-1)).any()
            # there will be no gap if next open is one minute after previous close
            closes_plus_min = self.closes + pd.Timedelta(1, "T")
            return self.opens.shift(-1) == closes_plus_min

        else:
            return self.opens.shift(-1) == self.closes

    @property
    def _mask_sessions_without_gap_before(self) -> pd.Series:
        if self.side == "neither":
            # will always have gap before if neither open or close are trading
            # minutes (assuming sessions cannot overlap)
            return pd.Series(False, index=self.sessions)

        elif self.side == "both":
            # a trading minute cannot be a minute of more than one session.
            assert not (self.closes == self.opens.shift(-1)).any()
            # there will be no gap if previous close is one minute before next open
            opens_minus_one = self.opens - pd.Timedelta(1, "T")
            return self.closes.shift(1) == opens_minus_one

        else:
            return self.closes.shift(1) == self.opens

    @functools.lru_cache(maxsize=4)
    def _sessions_without_gap_after(self) -> pd.DatetimeIndex:
        mask = self._mask_sessions_without_gap_after
        return self.sessions[mask][:-1]

    @property
    def sessions_without_gap_after(self) -> pd.DatetimeIndex:
        """Sessions not followed by a non-trading minute.

        Rather, sessions immediately followed by first trading minute of
        next session.
        """
        return self._sessions_without_gap_after()

    @functools.lru_cache(maxsize=4)
    def _sessions_with_gap_after(self) -> pd.DatetimeIndex:
        mask = self._mask_sessions_without_gap_after
        return self.sessions[~mask][:-1]

    @property
    def sessions_with_gap_after(self) -> pd.DatetimeIndex:
        """Sessions followed by a non-trading minute."""
        return self._sessions_with_gap_after()

    @functools.lru_cache(maxsize=4)
    def _sessions_without_gap_before(self) -> pd.DatetimeIndex:
        mask = self._mask_sessions_without_gap_before
        return self.sessions[mask][1:]

    @property
    def sessions_without_gap_before(self) -> pd.DatetimeIndex:
        """Sessions not preceeded by a non-trading minute.

        Rather, sessions immediately preceeded by last trading minute of
        previous session.
        """
        return self._sessions_without_gap_before()

    @functools.lru_cache(maxsize=4)
    def _sessions_with_gap_before(self) -> pd.DatetimeIndex:
        mask = self._mask_sessions_without_gap_before
        return self.sessions[~mask][1:]

    @property
    def sessions_with_gap_before(self) -> pd.DatetimeIndex:
        """Sessions preceeded by a non-trading minute."""
        return self._sessions_with_gap_before()

    # times are changing...

    @property
    def sessions_unchanging_times_run(self) -> pd.DatetimeIndex:
        """Longest run of sessions that have unchanging times."""
        bv = ~self.sessions.isin(self.sessions_next_time_different)
        s = pd.Series(bv, index=self.sessions)
        return xcals_pdutils.longest_run(s)

    @functools.lru_cache(maxsize=16)
    def _get_sessions_with_times_different_to_next_session(
        self,
        column: str,  # typing.Literal["opens", "closes", "break_starts", "break_ends"]
    ) -> list[pd.DatetimeIndex]:
        """Get sessions with times that differ from times of next session.

        For a given answers column, gets session labels where time differs
        from time of next session.

        Where `column` is a break time ("break_starts" or "break_ends"), return
        will not include sessions when next session has a different `has_break`
        status. For example, if session_0 has a break and session_1 does not have
        a break, or vice versa, then session_0 will not be included to return. For
        sessions followed by a session with a different `has_break` status, see
        `_get_sessions_with_has_break_different_to_next_session`.

        Returns
        -------
        list of pd.Datetimeindex
            [0] sessions with earlier next session
            [1] sessions with later next session
        """
        # column takes string to allow lru_cache (Series not hashable)

        is_break_col = column[0] == "b"
        column_ = getattr(self, column)

        if is_break_col:
            if column_.isna().all():
                return [pd.DatetimeIndex([])] * 2
            column_ = column_.fillna(method="ffill").fillna(method="bfill")

        diff = (column_.shift(-1) - column_)[:-1]
        remainder = diff % pd.Timedelta(24, "H")
        mask = remainder != pd.Timedelta(0)
        sessions = self.sessions[:-1][mask]
        next_session_earlier_mask = remainder[mask] > pd.Timedelta(12, "H")
        next_session_earlier = sessions[next_session_earlier_mask]
        next_session_later = sessions[~next_session_earlier_mask]

        if is_break_col:
            mask = next_session_earlier.isin(self.sessions_without_break)
            next_session_earlier = next_session_earlier.drop(next_session_earlier[mask])
            mask = next_session_later.isin(self.sessions_without_break)
            next_session_later = next_session_later.drop(next_session_later[mask])

        return [next_session_earlier, next_session_later]

    @property
    def _sessions_with_opens_different_to_next_session(
        self,
    ) -> list[pd.DatetimeIndex]:
        return self._get_sessions_with_times_different_to_next_session("opens")

    @property
    def _sessions_with_closes_different_to_next_session(
        self,
    ) -> list[pd.DatetimeIndex]:
        return self._get_sessions_with_times_different_to_next_session("closes")

    @property
    def _sessions_with_break_start_different_to_next_session(
        self,
    ) -> list[pd.DatetimeIndex]:
        return self._get_sessions_with_times_different_to_next_session("break_starts")

    @property
    def _sessions_with_break_end_different_to_next_session(
        self,
    ) -> list[pd.DatetimeIndex]:
        return self._get_sessions_with_times_different_to_next_session("break_ends")

    @property
    def sessions_next_open_earlier(self) -> pd.DatetimeIndex:
        return self._sessions_with_opens_different_to_next_session[0]

    @property
    def sessions_next_open_later(self) -> pd.DatetimeIndex:
        return self._sessions_with_opens_different_to_next_session[1]

    @property
    def sessions_next_open_different(self) -> pd.DatetimeIndex:
        return self.sessions_next_open_earlier.union(self.sessions_next_open_later)

    @property
    def sessions_next_close_earlier(self) -> pd.DatetimeIndex:
        return self._sessions_with_closes_different_to_next_session[0]

    @property
    def sessions_next_close_later(self) -> pd.DatetimeIndex:
        return self._sessions_with_closes_different_to_next_session[1]

    @property
    def sessions_next_close_different(self) -> pd.DatetimeIndex:
        return self.sessions_next_close_earlier.union(self.sessions_next_close_later)

    @property
    def sessions_next_break_start_earlier(self) -> pd.DatetimeIndex:
        return self._sessions_with_break_start_different_to_next_session[0]

    @property
    def sessions_next_break_start_later(self) -> pd.DatetimeIndex:
        return self._sessions_with_break_start_different_to_next_session[1]

    @property
    def sessions_next_break_start_different(self) -> pd.DatetimeIndex:
        earlier = self.sessions_next_break_start_earlier
        later = self.sessions_next_break_start_later
        return earlier.union(later)

    @property
    def sessions_next_break_end_earlier(self) -> pd.DatetimeIndex:
        return self._sessions_with_break_end_different_to_next_session[0]

    @property
    def sessions_next_break_end_later(self) -> pd.DatetimeIndex:
        return self._sessions_with_break_end_different_to_next_session[1]

    @property
    def sessions_next_break_end_different(self) -> pd.DatetimeIndex:
        earlier = self.sessions_next_break_end_earlier
        later = self.sessions_next_break_end_later
        return earlier.union(later)

    @functools.lru_cache(maxsize=4)
    def _get_sessions_with_has_break_different_to_next_session(
        self,
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Get sessions with 'has_break' different to next session.

        Returns
        -------
        tuple[pd.DatetimeIndex, pd.DatetimeIndex]
            [0] Sessions that have a break and are immediately followed by
            a session which does not have a break.
            [1] Sessions that do not have a break and are immediately
            followed by a session which does have a break.
        """
        mask = (self.break_starts.notna() & self.break_starts.shift(-1).isna())[:-1]
        sessions_with_break_next_session_without_break = self.sessions[:-1][mask]

        mask = (self.break_starts.isna() & self.break_starts.shift(-1).notna())[:-1]
        sessions_without_break_next_session_with_break = self.sessions[:-1][mask]

        return (
            sessions_with_break_next_session_without_break,
            sessions_without_break_next_session_with_break,
        )

    @property
    def sessions_with_break_next_session_without_break(self) -> pd.DatetimeIndex:
        return self._get_sessions_with_has_break_different_to_next_session()[0]

    @property
    def sessions_without_break_next_session_with_break(self) -> pd.DatetimeIndex:
        return self._get_sessions_with_has_break_different_to_next_session()[1]

    @functools.lru_cache(maxsize=4)
    def _sessions_next_time_different(self) -> pd.DatetimeIndex:
        session_indexes = [
            self.sessions_next_open_different,
            self.sessions_next_close_different,
            self.sessions_next_break_start_different,
            self.sessions_next_break_end_different,
            self.sessions_with_break_next_session_without_break,
            self.sessions_without_break_next_session_with_break,
        ]
        index = pdutils.indexes_union(session_indexes)
        assert isinstance(index, pd.DatetimeIndex)
        return index

    @property
    def sessions_next_time_different(self) -> pd.DatetimeIndex:
        """Sessions where next session has a different time for any column.

        Includes sessions where next session has a different `has_break`
        status.
        """
        return self._sessions_next_time_different()

    # session blocks...

    def _create_changing_times_session_block(
        self, session: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Create block of sessions with changing times.

        Given a `session` known to have at least one time (open, close,
        break_start or break_end) different from the next session, returns
        a block of consecutive sessions ending with the first session after
        `session` that has the same times as the session that immediately
        preceeds it (i.e. the last two sessions of the block will have the
        same times), or the last calendar session.
        """
        start_idx = self.sessions.get_loc(session)
        end_idx = start_idx + 1
        while self.sessions[end_idx] in self.sessions_next_time_different:
            end_idx += 1
        end_idx += 2  # +1 to include session with same times, +1 to serve as end index
        return self.sessions[start_idx:end_idx]

    def _get_normal_session_block(self) -> pd.DatetimeIndex:
        """Block of 3 sessions with unchanged timings."""
        start_idx = len(self.sessions) // 3
        end_idx = start_idx + 21
        for i in range(start_idx, end_idx):
            times_1 = self.answers.iloc[i].dt.time
            times_2 = self.answers.iloc[i + 1].dt.time
            times_3 = self.answers.iloc[i + 2].dt.time
            one_and_two_equal = (times_1 == times_2) | (times_1.isna() & times_2.isna())
            one_and_three_equal = (times_1 == times_3) | (
                times_1.isna() & times_3.isna()
            )
            if (one_and_two_equal & one_and_three_equal).all():
                break
            assert i < (end_idx - 1), "Unable to evaluate a normal session block!"
        return self.sessions[i : i + 3]

    def _get_session_block(
        self, from_session_of: pd.DatetimeIndex, to_session_of: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Get session block with bounds defined by sessions of given indexes.

        Block will start with middle session of `from_session_of`.

        Block will run to the nearest subsequent session of `to_session_of`
        (or `self.final_session` if this comes first). Block will end with
        the session that immedidately follows this session.
        """
        i = len(from_session_of) // 2
        start_session = from_session_of[i]

        start_idx = self.sessions.get_loc(start_session)
        end_idx = start_idx + 1
        end_session = self.sessions[end_idx]

        while end_session not in to_session_of and end_session != self.last_session:
            end_idx += 1
            end_session = self.sessions[end_idx]

        return self.sessions[start_idx : end_idx + 2]

    @functools.lru_cache(maxsize=4)
    def _session_blocks(self) -> dict[str, pd.DatetimeIndex]:
        # pylint: disable=too-many-locals
        blocks = {}
        blocks["normal"] = self._get_normal_session_block()
        blocks["first_three"] = self.sessions[:3]
        blocks["last_three"] = self.sessions[-3:]

        # blocks here include where:
        #     session 1 has at least one different time from session 0
        #     session 0 has a break and session 1 does not (and vice versa)
        sessions_indexes = (
            ("next_open_earlier", self.sessions_next_open_earlier),
            ("next_open_later", self.sessions_next_open_later),
            ("next_close_earlier", self.sessions_next_close_earlier),
            ("next_close_later", self.sessions_next_close_later),
            ("next_break_start_earlier", self.sessions_next_break_start_earlier),
            ("next_break_start_later", self.sessions_next_break_start_later),
            ("next_break_end_earlier", self.sessions_next_break_end_earlier),
            ("next_break_end_later", self.sessions_next_break_end_later),
            (
                "with_break_to_without_break",
                self.sessions_with_break_next_session_without_break,
            ),
            (
                "without_break_to_with_break",
                self.sessions_without_break_next_session_with_break,
            ),
        )

        for name, index in sessions_indexes:
            if index.empty:
                blocks[name] = pd.DatetimeIndex([])
            else:
                session = index[0]
                blocks[name] = self._create_changing_times_session_block(session)

        # blocks here move from session with gap to session without gap and vice versa
        if (not self.sessions_with_gap_after.empty) and (
            not self.sessions_without_gap_after.empty
        ):
            without_gap_to_with_gap = self._get_session_block(
                self.sessions_without_gap_after, self.sessions_with_gap_after
            )
            with_gap_to_without_gap = self._get_session_block(
                self.sessions_with_gap_after, self.sessions_without_gap_after
            )
        else:
            without_gap_to_with_gap = pd.DatetimeIndex([])
            with_gap_to_without_gap = pd.DatetimeIndex([])

        blocks["without_gap_to_with_gap"] = without_gap_to_with_gap
        blocks["with_gap_to_without_gap"] = with_gap_to_without_gap

        # blocks that adjoin or contain a non_session date
        follows_non_session = pd.DatetimeIndex([])
        preceeds_non_session = pd.DatetimeIndex([])
        contains_non_session = pd.DatetimeIndex([])
        if len(self.non_sessions) > 1:
            diff = self.non_sessions[1:] - self.non_sessions[:-1]
            mask = diff != pd.Timedelta(
                1, "D"
            )  # non_session dates followed by a session
            valid_non_sessions = self.non_sessions[:-1][mask]
            if len(valid_non_sessions) > 1:
                slce = self.sessions.slice_indexer(
                    valid_non_sessions[0], valid_non_sessions[1]
                )
                sessions_between_non_sessions = self.sessions[slce]
                block_length = min(2, len(sessions_between_non_sessions))
                follows_non_session = sessions_between_non_sessions[:block_length]
                preceeds_non_session = sessions_between_non_sessions[-block_length:]
                # take session before and session after non-session
                contains_non_session = self.sessions[slce.stop - 1 : slce.stop + 1]

        blocks["follows_non_session"] = follows_non_session
        blocks["preceeds_non_session"] = preceeds_non_session
        blocks["contains_non_session"] = contains_non_session

        return blocks

    @property
    def session_blocks(self) -> dict[str, pd.DatetimeIndex]:
        """Dictionary of session blocks of a particular behaviour.

        A block comprises either a single session or multiple contiguous
        sessions.

        Keys:
            "normal" - three sessions with unchanging timings.
            "first_three" - answers' first three sessions.
            "last_three" - answers's last three sessions.
            "next_open_earlier" - session 1 open is earlier than session 0
                open.
            "next_open_later" - session 1 open is later than session 0
                open.
            "next_close_earlier" - session 1 close is earlier than session
                0 close.
            "next_close_later" - session 1 close is later than session 0
                close.
            "next_break_start_earlier" - session 1 break_start is earlier
                than session 0 break_start.
            "next_break_start_later" - session 1 break_start is later than
                session 0 break_start.
            "next_break_end_earlier" - session 1 break_end is earlier than
                session 0 break_end.
            "next_break_end_later" - session 1 break_end is later than
                session 0 break_end.
            "with_break_to_without_break" - session 0 has a break, session
                1 does not have a break.
            "without_break_to_with_break" - session 0 does not have a
                break, session 1 does have a break.
            "without_gap_to_with_gap" - session 0 is not followed by a
                gap, session -2 is followed by a gap, session -1 is
                preceeded by a gap.
            "with_gap_to_without_gap" - session 0 is followed by a gap,
                session -2 is not followed by a gap, session -1 is not
                preceeded by a gap.
            "follows_non_session" - one or two sessions where session 0
                is preceeded by a date that is a non-session.
            "follows_non_session" - one or two sessions where session -1
                is followed by a date that is a non-session.
            "contains_non_session" = two sessions with at least one
                non-session date in between.

        If no such session block exists for any key then value will take an
        empty DatetimeIndex (UTC).
        """
        return self._session_blocks()

    def session_block_generator(self) -> abc.Iterator[tuple[str, pd.DatetimeIndex]]:
        """Generator of session blocks of a particular behaviour."""
        for name, block in self.session_blocks.items():
            if not block.empty:
                yield (name, block)

    @functools.lru_cache(maxsize=4)
    def _session_block_minutes(self) -> dict[str, pd.DatetimeIndex]:
        d = {}
        for name, block in self.session_blocks.items():
            if block.empty:
                d[name] = pd.DatetimeIndex([], tz=UTC)
                continue
            d[name] = self.get_sessions_minutes(block[0], len(block))
        return d

    @property
    def session_block_minutes(self) -> dict[str, pd.DatetimeIndex]:
        """Trading minutes for each `session_block`.

        Key:
            Session block name as documented to `session_blocks`.
        Value:
            Trading minutes of corresponding session block.
        """
        return self._session_block_minutes()

    @property
    def sessions_sample(self) -> pd.DatetimeIndex:
        """Sample of normal and unusual sessions.

        Sample comprises set of sessions of all `session_blocks` (see
        `session_blocks` doc). In this way sample includes at least one
        sample of every indentified unique circumstance.
        """
        dtis = list(self.session_blocks.values())
        index = pdutils.indexes_union(dtis)
        assert isinstance(index, pd.DatetimeIndex)
        return index

    # non-sessions...

    @functools.lru_cache(maxsize=4)
    def _non_sessions(self) -> pd.DatetimeIndex:
        all_dates = pd.date_range(
            start=self.first_session, end=self.last_session, freq="D"
        )
        return all_dates.difference(self.sessions)

    @property
    def non_sessions(self) -> pd.DatetimeIndex:
        """Dates (UTC midnight) within answers range that are not sessions."""
        return self._non_sessions()

    @property
    def sessions_range_defined_by_non_sessions(
        self,
    ) -> tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Datetimeindex] | None:
        """Range containing sessions although defined with non-sessions.

        Returns
        -------
        tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Datetimeindex]:
            [0] tuple[pd.Timestamp, pd.Timestamp]:
                [0] range start as non-session date.
                [1] range end as non-session date.
            [1] pd.DatetimeIndex:
                Sessions in range.
        """
        non_sessions = self.non_sessions
        if len(non_sessions) <= 1:
            return None
        limit = len(self.non_sessions) - 2
        i = 0
        start, end = non_sessions[i], non_sessions[i + 1]
        while (end - start) < pd.Timedelta(4, "D"):
            i += 1
            start, end = non_sessions[i], non_sessions[i + 1]
            if i == limit:
                # Unable to evaluate range from consecutive non-sessions
                # that covers >= 3 sessions. Just go with max range...
                start, end = non_sessions[0], non_sessions[-1]
        slice_start, slice_end = self.sessions.searchsorted((start, end))
        return (start, end), self.sessions[slice_start:slice_end]

    @property
    def non_sessions_run(self) -> pd.DatetimeIndex:
        """Longest run of non_sessions."""
        ser = self.sessions.to_series()
        diff = ser.shift(-1) - ser
        max_diff = diff.max()
        if max_diff == pd.Timedelta(1, "D"):
            return pd.DatetimeIndex([])
        session_before_run = diff[diff == max_diff].index[-1]
        run = pd.date_range(
            start=session_before_run + pd.Timedelta(1, "D"),
            periods=(max_diff // pd.Timedelta(1, "D")) - 1,
            freq="D",
        )
        assert run.isin(self.non_sessions).all()
        assert run[0] > self.first_session
        assert run[-1] < self.last_session
        return run

    @property
    def non_sessions_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Longest range covering a period without a session."""
        non_sessions_run = self.non_sessions_run
        if non_sessions_run.empty:
            return None
        else:
            return self.non_sessions_run[0], self.non_sessions_run[-1]

    # --- Evaluated sets of minutes ---

    Minutes = tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]

    @functools.lru_cache(maxsize=4)
    def _evaluate_trading_and_break_minutes(
        self,
    ) -> tuple[
        tuple[tuple[Minutes, pd.Timestamp], ...],
        tuple[tuple[Minutes, pd.Timestamp], ...],
    ]:
        # pylint: disable=too-many-locals
        sessions = self.sessions_sample
        first_mins = self.first_minutes[sessions]
        first_mins_plus_one = first_mins + self.ONE_MIN
        last_mins = self.last_minutes[sessions]
        last_mins_less_one = last_mins - self.ONE_MIN

        trading_mins = []
        break_mins = []

        for session, mins_ in zip(
            sessions,
            zip(first_mins, first_mins_plus_one, last_mins, last_mins_less_one),
        ):
            if TYPE_CHECKING:
                assert isinstance(mins_, tuple)
                assert isinstance(session, pd.Timestamp)
            trading_mins.append((mins_, session))

        if self.has_a_session_with_break:
            last_am_mins = self.last_am_minutes[sessions]
            last_am_mins = last_am_mins[last_am_mins.notna()]
            first_pm_mins = self.first_pm_minutes[last_am_mins.index]

            last_am_mins_less_one = last_am_mins - self.ONE_MIN
            last_am_mins_plus_one = last_am_mins + self.ONE_MIN
            last_am_mins_plus_two = last_am_mins + self.TWO_MIN

            first_pm_mins_plus_one = first_pm_mins + self.ONE_MIN
            first_pm_mins_less_one = first_pm_mins - self.ONE_MIN
            first_pm_mins_less_two = first_pm_mins - self.TWO_MIN

            for session, mins_ in zip(
                last_am_mins.index,
                zip(
                    last_am_mins,
                    last_am_mins_less_one,
                    first_pm_mins,
                    first_pm_mins_plus_one,
                ),
            ):
                if TYPE_CHECKING:
                    assert isinstance(mins_, tuple)
                    assert isinstance(session, pd.Timestamp)
                trading_mins.append((mins_, session))

            for session, mins_ in zip(
                last_am_mins.index,
                zip(
                    last_am_mins_plus_one,
                    last_am_mins_plus_two,
                    first_pm_mins_less_one,
                    first_pm_mins_less_two,
                ),
            ):
                if TYPE_CHECKING:
                    assert isinstance(mins_, tuple)
                    assert isinstance(session, pd.Timestamp)
                break_mins.append((mins_, session))

        return (tuple(trading_mins), tuple(break_mins))

    @property
    def trading_minutes(self) -> tuple[tuple[Minutes, pd.Timestamp], ...]:
        """Edge trading minutes of `sessions_sample`.

        Returns
        -------
        tuple of tuple[tuple[trading_minutes], session]

            tuple[trading_minutes] includes:
                first two trading minutes of a session.
                last two trading minutes of a session.
                If breaks:
                    last two trading minutes of session's am subsession.
                    first two trading minutes of session's pm subsession.

            session
                Session of trading_minutes
        """
        return self._evaluate_trading_and_break_minutes()[0]

    def trading_minutes_only(self) -> abc.Iterator[pd.Timestamp]:
        """Generator of trading minutes of `self.trading_minutes`."""
        for mins, _ in self.trading_minutes:
            for minute in mins:
                yield minute

    @property
    def trading_minute(self) -> pd.Timestamp:
        """A single trading minute."""
        return self.trading_minutes[0][0][0]

    @property
    def break_minutes(self) -> tuple[tuple[Minutes, pd.Timestamp], ...]:
        """Sample of break minutes of `sessions_sample`.

        Returns
        -------
        tuple of tuple[tuple[break_minutes], session]

            tuple[break_minutes]:
                first two minutes of a break.
                last two minutes of a break.

            session
                Session of break_minutes
        """
        return self._evaluate_trading_and_break_minutes()[1]

    def break_minutes_only(self) -> abc.Iterator[pd.Timestamp]:
        """Generator of break minutes of `self.break_minutes`."""
        for mins, _ in self.break_minutes:
            for minute in mins:
                yield minute

    @functools.lru_cache(maxsize=4)
    def _non_trading_minutes(
        self,
    ) -> tuple[
        tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp, pd.Timestamp], ...
    ]:
        non_trading_mins = []

        sessions = self.sessions_sample
        sessions = prev_sessions = sessions[sessions.isin(self.sessions_with_gap_after)]

        next_sessions = self.sessions[self.sessions.get_indexer(sessions) + 1]

        last_mins_plus_one = self.last_minutes[sessions] + self.ONE_MIN
        first_mins_less_one = self.first_minutes[next_sessions] - self.ONE_MIN

        for prev_session, next_session, mins_ in zip(
            prev_sessions, next_sessions, zip(last_mins_plus_one, first_mins_less_one)
        ):
            non_trading_mins.append((mins_, prev_session, next_session))

        return tuple(non_trading_mins)

    @property
    def non_trading_minutes(
        self,
    ) -> tuple[
        tuple[tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp, pd.Timestamp], ...
    ]:
        """non_trading_minutes that edge `sessions_sample`.

        NB. Does not include break minutes.

        Returns
        -------
        tuple of tuple[tuple[non-trading minute], previous session, next session]

            tuple[non-trading minute]
                Two non-trading minutes.
                    [0] first non-trading minute to follow a session.
                    [1] last non-trading minute prior to the next session.

            previous session
                Session that preceeds non-trading minutes.

            next session
                Session that follows non-trading minutes.

        See Also
        --------
        break_minutes
        """
        return self._non_trading_minutes()

    def non_trading_minutes_only(self) -> abc.Iterator[pd.Timestamp]:
        """Generator of non-trading minutes of `self.non_trading_minutes`."""
        for mins, _, _ in self.non_trading_minutes:
            for minute in mins:
                yield minute

    # --- Evaluated minutes of a specific circumstance ---

    def _trading_minute_to_break_minute(
        self, sessions, break_sessions
    ) -> list[list[pd.Timestamp]]:
        times = (self.last_am_minutes[break_sessions] + pd.Timedelta(1, "T")).dt.time

        mask = (self.first_minutes[sessions].dt.time.values < times.values) & (
            times.values < self.last_minutes[sessions].dt.time.values
        )

        minutes = []
        for session, break_session in zip(sessions[mask], break_sessions[mask]):
            break_minutes = self.get_session_break_minutes(break_session)
            trading_minutes = self.get_session_minutes(session)[0]
            bv = np.in1d(trading_minutes.time, break_minutes.time)
            minutes.append([trading_minutes[bv][-1], session, break_session])
        return minutes

    @property
    def trading_minute_to_break_minute_next(self) -> list[list[pd.Timestamp]]:
        """Trading minutes where same minute of next session is a break minute.

        Returns
        -------
        list of list of:
            [0] trading minute
            [1] session of which [0] ia a trading minute
            [2] next session, i.e. session of which a minute with same time as
                [0] is a break minute.
        """
        sessions = self.sessions_without_break_next_session_with_break
        idxr = self.sessions.get_indexer(sessions)
        break_sessions = self.sessions[idxr + 1]
        lst = self._trading_minute_to_break_minute(sessions, break_sessions)

        sessions = self.sessions_next_break_end_later
        idxr = self.sessions.get_indexer(sessions) + 1
        target_sessions = self.sessions[idxr]
        minutes = self.first_pm_minutes[sessions]
        offset_minutes = minutes - sessions + target_sessions
        # only include offset minute if verified as break minute of target
        # (it wont be if the break has shifted by more than the break duration)
        mask = offset_minutes.values > self.last_am_minutes[target_sessions].values
        zipped = zip(minutes[mask], sessions[mask], target_sessions[mask])
        lst.extend(list(zipped))  # type: ignore[arg-type]  # zipped is iterable

        sessions = self.sessions_next_break_start_earlier
        idxr = self.sessions.get_indexer(sessions) + 1
        target_sessions = self.sessions[idxr]
        minutes = self.last_am_minutes[sessions]
        offset_minutes = minutes - sessions + target_sessions
        # only include offset minute if verified as break minute of target
        mask = offset_minutes.values < self.first_pm_minutes[target_sessions].values
        zipped = zip(minutes[mask], sessions[mask], target_sessions[mask])
        lst.extend(list(zipped))  # type: ignore[arg-type]  # zipped is iterable

        return lst

    @property
    def trading_minute_to_break_minute_prev(self) -> list[list[pd.Timestamp]]:
        """Trading minutes where same minute of previous session is a break minute.

        Returns
        -------
        list of list of:
            [0] trading minute
            [1] session of which [0] ia a trading minute
            [2] previous session, i.e. session of which a minute with same time as
                [0] is a break minute.
        """
        break_sessions = self.sessions_with_break_next_session_without_break
        idxr = self.sessions.get_indexer(break_sessions)
        sessions = self.sessions[idxr + 1]
        lst = self._trading_minute_to_break_minute(sessions, break_sessions)

        target_sessions = self.sessions_next_break_end_earlier
        idxr = self.sessions.get_indexer(target_sessions) + 1
        sessions = self.sessions[idxr]  # previous break ends later
        minutes = self.first_pm_minutes[sessions]
        offset_minutes = minutes - sessions + target_sessions
        # only include offset minute if verified as break minute of target
        # (it wont be if the break has shifted by more than the break duration)
        mask = offset_minutes.values > self.last_am_minutes[target_sessions].values
        zipped = zip(minutes[mask], sessions[mask], target_sessions[mask])
        lst.extend(list(zipped))  # type: ignore[arg-type]  # zipped is iterable

        target_sessions = self.sessions_next_break_start_later
        idxr = self.sessions.get_indexer(target_sessions) + 1
        sessions = self.sessions[idxr]  # previous break starts earlier
        minutes = self.last_am_minutes[sessions]
        offset_minutes = minutes - sessions + target_sessions
        # only include offset minute if verified as break minute of target
        mask = offset_minutes.values < self.first_pm_minutes[target_sessions].values
        zipped = zip(minutes[mask], sessions[mask], target_sessions[mask])
        lst.extend(list(zipped))  # type: ignore[arg-type]  # zipped is iterable

        return lst

    # --- method-specific inputs/outputs ---

    def prev_next_open_close_minutes(
        self,
    ) -> abc.Iterator[
        tuple[
            pd.Timestamp,
            tuple[
                pd.Timestamp | None,
                pd.Timestamp | None,
                pd.Timestamp | None,
                pd.Timestamp | None,
            ],
        ]
    ]:
        """Generator of test parameters for prev/next_open/close methods.

        Inputs include following minutes of each session:
            open
            one minute prior to open (not included for first session)
            one minute after open
            close
            one minute before close
            one minute after close (not included for last session)

        NB Assumed that minutes prior to first open and after last close
        will be handled via parse_timestamp.

        Yields
        ------
        2-tuple:
            [0] Input a minute sd pd.Timestamp
            [1] 4 tuple of expected output of corresponding method:
                [0] previous_open as pd.Timestamp | None
                [1] previous_close as pd.Timestamp | None
                [2] next_open as pd.Timestamp | None
                [3] next_close as pd.Timestamp | None

                NB None indicates that corresponding method is expected to
                raise a ValueError for this input.
        """
        # pylint: disable=too-many-locals, too-many-statements
        close_is_next_open_bv = self.closes == self.opens.shift(-1)
        open_was_prev_close_bv = self.opens == self.closes.shift(+1)
        close_is_next_open = close_is_next_open_bv[0]

        # minutes for session 0
        minute = self.opens[0]
        yield (minute, (None, None, self.opens[1], self.closes[0]))

        minute = minute + self.ONE_MIN
        yield (minute, (self.opens[0], None, self.opens[1], self.closes[0]))

        minute = self.closes[0]
        next_open = self.opens[2] if close_is_next_open else self.opens[1]
        yield (minute, (self.opens[0], None, next_open, self.closes[1]))

        minute += self.ONE_MIN
        prev_open = self.opens[1] if close_is_next_open else self.opens[0]
        yield (minute, (prev_open, self.closes[0], next_open, self.closes[1]))

        minute = self.closes[0] - self.ONE_MIN
        yield (minute, (self.opens[0], None, self.opens[1], self.closes[0]))

        # minutes for sessions over [1:-1] except for -1 close and 'close + one_min'
        opens = self.opens[1:-1]
        closes = self.closes[1:-1]
        prev_opens = self.opens[:-2]
        prev_closes = self.closes[:-2]
        next_opens = self.opens[2:]
        next_closes = self.closes[2:]
        opens_after_next = self.opens[3:]
        # add dummy row to equal lengths (won't be used)
        _ = pd.Series(pd.Timestamp("2200-01-01", tz=UTC))
        opens_after_next = opens_after_next.append(_)

        stop = closes[-1]

        for (
            open_,
            close,
            prev_open,
            prev_close,
            next_open,
            next_close,
            open_after_next,
            close_is_next_open,
            open_was_prev_close,
        ) in zip(
            opens,
            closes,
            prev_opens,
            prev_closes,
            next_opens,
            next_closes,
            opens_after_next,
            close_is_next_open_bv[1:-2],
            open_was_prev_close_bv[1:-2],
        ):
            if not open_was_prev_close:
                # only include open minutes if not otherwise duplicating
                # evaluations already made for prior close.
                yield (open_, (prev_open, prev_close, next_open, close))
                yield (open_ - self.ONE_MIN, (prev_open, prev_close, open_, close))
                yield (open_ + self.ONE_MIN, (open_, prev_close, next_open, close))

            yield (close - self.ONE_MIN, (open_, prev_close, next_open, close))

            if close != stop:
                next_open_ = open_after_next if close_is_next_open else next_open
                yield (close, (open_, prev_close, next_open_, next_close))

                open_ = next_open if close_is_next_open else open_
                yield (close + self.ONE_MIN, (open_, close, next_open_, next_close))

        # close and 'close + one_min' for session -2
        minute = self.closes[-2]
        next_open = None if close_is_next_open_bv[-2] else self.opens[-1]
        yield (minute, (self.opens[-2], self.closes[-3], next_open, self.closes[-1]))

        minute += self.ONE_MIN
        prev_open = self.opens[-1] if close_is_next_open_bv[-2] else self.opens[-2]
        yield (minute, (prev_open, self.closes[-2], next_open, self.closes[-1]))

        # minutes for session -1
        if not open_was_prev_close_bv[-1]:
            open_ = self.opens[-1]
            prev_open = self.opens[-2]
            prev_close = self.closes[-2]
            next_open = None
            close = self.closes[-1]
            yield (open_, (prev_open, prev_close, next_open, close))
            yield (open_ - self.ONE_MIN, (prev_open, prev_close, open_, close))
            yield (open_ + self.ONE_MIN, (open_, prev_close, next_open, close))

        minute = self.closes[-1]
        next_open = self.opens[2] if close_is_next_open_bv[-1] else self.opens[1]
        yield (minute, (self.opens[-1], self.closes[-2], None, None))

        minute -= self.ONE_MIN
        yield (minute, (self.opens[-1], self.closes[-2], None, self.closes[-1]))

    # dunder

    def __repr__(self) -> str:
        return f"<Answers: calendar {self.name}, side {self.side}>"
