"""Price table interval definitions and helper functions.

Price table intervals are defined as members of one of the following
enumerations:
    TDInterval: intervals defined by a pd.Timedelta to describe a time
    delta in in terms of minutes, hours or days.

    DOInterval: intervals defined by a pd.DateOffset to describe a time
    delta in terms of months.
"""

from __future__ import annotations

from datetime import timedelta
import enum
import typing

import exchange_calendars as xcals
import pandas as pd

from market_prices import helpers

# pylint doesn't see members of enum or pd.Timedelta attrs of _TDIntervalBase
# pylint: disable=no-member

TIMEDELTA_ARGS = {
    "T1": (0, 0, 0, 0, 1),
    "T2": (0, 0, 0, 0, 2),
    "T5": (0, 0, 0, 0, 5),
    "T10": (0, 0, 0, 0, 10),
    "T15": (0, 0, 0, 0, 15),
    "T30": (0, 0, 0, 0, 30),
    "H1": (0, 0, 0, 0, 0, 1),
    "D1": (1,),
}


class _TDIntervalBase(timedelta, enum.Enum):
    @classmethod
    def as_list(cls) -> list:
        """Return all enum members in a list."""
        try:
            return cls._as_list  # type: ignore[attr-defined]
        except AttributeError:
            cls._as_list = list(cls)  # type: ignore[attr-defined]
            return cls._as_list  # type: ignore[attr-defined]

    @property
    def as_pdtd(self) -> pd.Timedelta:
        """Return as `pd.Timedelta`."""
        return pd.Timedelta(self)

    @property
    def freq_unit(self) -> typing.Literal["min", "h", "D"]:
        """Return unit of pandas frequency represented by the member.

        Returns either "min", "h" or "D".
        """
        unit = self.as_pdtd.resolution_string
        # for pre pandas 2.2 compatibility...
        if unit == "T":
            unit = "min"
        if unit == "H":
            unit = "h"
        return unit

    @property
    def freq_value(self) -> int:
        """Return value of pandas frequency represented by the member."""
        components = self.as_pdtd.components
        if self.freq_unit == "D":
            return components.days
        elif self.freq_unit == "h":
            return components.hours
        else:
            return components.minutes + (components.hours * 60)

    @property
    def is_intraday(self) -> bool:
        """Query if a member represents an intraday interval.

        An interval is considered to be intraday if it is shorter than one
        day. The unit of intraday intervals will be either "min" or "h".
        """
        return self < helpers.ONE_DAY

    @property
    def is_daily(self) -> bool:
        """Query if a member represents a daily interval.

        An interval is considered to be daily if it is one day or a
        multiple of days. The unit of daily intervals is "D".
        """
        return self >= helpers.ONE_DAY

    @property
    def is_monthly(self) -> bool:
        """Query if a member represents a monthly interval.

        An interval is considered to be monthly if it represents one month
        or a multiple of months. The unit of daily intervals is "MS".
        """
        return False

    @property
    def is_one_day(self) -> bool:
        """Query if member represents an interval of length 1 day."""
        return self == helpers.ONE_DAY

    @property
    def is_one_minute(self) -> bool:
        """Query if member represents an interval of length 1 minute."""
        return self == helpers.ONE_MIN

    @property
    def is_gt_one_day(self) -> bool:
        """Query if member represents an interval longer than 1 day."""
        return self.is_daily and not self.is_one_day

    @property
    def as_pdfreq(self) -> str:
        """Return interval represented as a pandas frequency string."""
        return str(self.freq_value) + self.freq_unit

    def as_offset(
        self, calendar: xcals.ExchangeCalendar | None = None, one_less: bool = True
    ) -> pd.offsets.BaseOffset:
        """Return interval represented as an offset.

        Parameters
        ----------
        calendar
            Calendar against which to evaluate custom business day for
            intervals with `self.freq_unit` as "D". Not required for
            intervals with `self.freq_unit` as "min" or "h".

        one_less
            If `self.freq_unit` is "D" then:
            True: return offset representing one less than
            `self.freq_value` trading days. This provides for offsetting
            from the start to the end of a date range, where a a single
            session is represented with start and end as the same value.

            False: return offset representing `self.freq_value` trading
            days.
        """
        if self.freq_unit == "D":
            if calendar is None:
                raise ValueError(
                    "`calendar` must be passed for intervals representing a number"
                    " of trading days."
                )
            return calendar.day * (self.freq_value - one_less)
        return pd.tseries.frequencies.to_offset(self.as_pdfreq)

    @property
    def as_minutes(self) -> int:
        """Return interval as a number of minutes.

        Only available if interval is intraday.

        Raises
        ------
        NotImplementedError
            If interval is 1 day or or higher.
        """
        if self.is_intraday:
            return int(self.total_seconds() / 60)
        else:
            msg = "`as_minutes` only available for intraday intervals."
            raise ValueError(msg)


class TDInterval(_TDIntervalBase):
    """Intervals described by a pd.Timedelta object.

    Intervals can be described in terms of either minutes, hours or days.
    """

    # pylint: disable=exec-used, invalid-name

    _ignore_ = "i, i_, j,"

    for i in range(1, (60 * 22) + 1):
        if not i % 60:
            i_ = i // 60
            exec(f"H{i_} = (0, 0, 0, 0, 0, {i_})")
        else:
            exec(f"T{i} = (0, 0, 0, 0, {i})")

    for j in range(1, 251):
        exec(f"D{j} = ({j},)")

    @classmethod
    def daily_intervals(cls) -> list[TDInterval]:
        """List of those intervals that represent 'one day' or higher."""
        idx = cls.as_list().index(cls.D1)
        return cls.as_list()[idx:]

    @classmethod
    def intraday_intervals(cls) -> list[TDInterval]:
        """List of those intervals that represent no higher than 22 hours."""
        idx = cls.as_list().index(cls.D1)
        return cls.as_list()[:idx]


class DOInterval(enum.Enum):
    """Intervals described by a pd.DateOffset object.

    Intervals can be described in terms of months.
    """

    _ignore_ = "i"  # pylint: disable=invalid-name

    for i in range(1, 37):
        exec(f"M{i} = pd.DateOffset(months={i})")  # pylint: disable=exec-used

    # pylint: disable=missing-return-type-doc
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.freq_value >= other.freq_value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.freq_value > other.freq_value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.freq_value <= other.freq_value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.freq_value < other.freq_value
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, pd.Timestamp):
            return other + self.value
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, pd.Timestamp):
            return other - self.value
        return NotImplemented

    # pylint: enable=missing-return-type-doc

    @property
    def freq_unit(self) -> str:
        """Return unit of pandas frequency represented by the member.

        Returns "MS".
        """
        return "MS"

    @property
    def freq_value(self) -> int:
        """Return value of pandas frequency represented by the member."""
        return self.value.months

    @property
    def is_intraday(self) -> bool:
        """Query if a member represents an intraday interval.

        An interval is considered to be intraday if it is shorter than one
        day. The unit of intraday intervals will be either "min" or "h".
        """
        return False

    @property
    def is_daily(self) -> bool:
        """Query if a member represents a daily interval.

        An interval is considered to be daily if it is one day or a
        multiple of days. The unit of daily intervals is "D".
        """
        return False

    @property
    def is_monthly(self) -> bool:
        """Query if a member represents a monthly interval.

        An interval is considered to be monthly if it represents one month
        or a multiple of months. The unit of daily intervals is "MS".
        """
        return True

    @property
    def is_one_day(self) -> bool:
        """Query if member represents an interval of length 1 day."""
        return False

    @property
    def is_one_minute(self) -> bool:
        """Query if member represents an interval of length 1 minute."""
        return False

    @property
    def is_gt_one_day(self) -> bool:
        """Query if member represents an interval longer than 1 day."""
        return True

    @property
    def as_pdfreq(self) -> str:
        """Return interval represented as a pandas frequency string."""
        return str(self.freq_value) + self.freq_unit

    def as_offset(self, *_, **__) -> pd.DateOffset:  # pylint: disable=missing-param-doc
        """Return interval represented as an offset."""
        return self.value

    @property
    def as_offset_ms(self) -> pd.DateOffset:
        """Return interval represented as a "MS" (month start) offset."""
        offset = pd.tseries.frequencies.to_offset(self.as_pdfreq)
        assert offset is not None
        return offset


class _BaseIntervalMeta(enum.EnumMeta):
    def __getitem__(cls, index) -> _BaseIntervalMeta:  # type: ignore[override]
        return list(cls)[index]

    def __contains__(cls, item) -> bool:
        return item in list(cls)


class _BaseInterval(_TDIntervalBase, metaclass=_BaseIntervalMeta):
    @classmethod
    def daily_bi(cls) -> BI | None:
        """Return daily base interval. None if all base intervals intraday."""
        try:
            return cls.D1
        except AttributeError:
            assert pd.Timedelta(1, "D") not in cls
            return None

    @classmethod
    def intraday_bis(cls) -> list[BI]:
        """Return all intraday base intervals."""
        return [bi for bi in cls if bi.is_intraday]

    @property
    def next(self) -> BI | None:
        """Return next (longer) member.

        Returns None if member represents the longest interval.
        """
        lst = self.as_list()
        i = lst.index(self)
        try:
            return lst[i + 1]
        except IndexError:
            return None

    @property
    def previous(self) -> BI | None:
        """Return previous (shorter) member.

        Returns None if member represents the shortest interval.
        """
        lst = self.as_list()
        i = lst.index(self)
        if not i:
            return None
        return lst[i - 1]


def create_base_intervals_enum(intervals: list[TDInterval]) -> _BaseInterval:
    """Create a _BaseInterval enum for given TDInterval.

    Parameters
    ----------
    intervals
        List of TDInterval to be represented in _BaseInterval enum.
    """
    d = {}
    intervals.sort()
    for intrvl in intervals:
        unit, value = intrvl.freq_unit, intrvl.freq_value
        td_args: tuple
        if unit == "min":
            td_args = (0, 0, 0, 0, value)
        elif unit == "h":
            td_args = (0, 0, 0, 0, 0, value)
        else:
            if intrvl is not TDInterval.D1:
                raise ValueError(
                    "Base Intervals cannot be greater than 1 day although `intervals`"
                    f" included '{intrvl}'."
                )
            td_args = (1,)
        d[intrvl.name] = td_args
    return _BaseInterval("BaseInterval", d)


BI = _BaseInterval
PTInterval = TDInterval | DOInterval | BI

ONE_MIN: TDInterval = TDInterval.T1
ONE_DAY: TDInterval = TDInterval.D1

# Constants able to mimic base intervals.
_BI_CONSTANTS = BI(
    "BI_CONSTANTS",
    dict(T1=TIMEDELTA_ARGS["T1"], D1=TIMEDELTA_ARGS["D1"]),
)

BI_ONE_MIN = _BI_CONSTANTS.T1
BI_ONE_DAY = _BI_CONSTANTS.D1


def to_ptinterval(interval: str | timedelta | pd.Timedelta) -> PTInterval:
    """Verify and parse client `interval` input to a PTInterval.

    Parameters
    ----------
    interval
        As defined for the `interval` parameter of
        `market_prices.prices.base.PricesBase.get`.
    """
    # pylint: disable=too-complex, too-many-branches
    if not isinstance(interval, (str, timedelta, pd.Timedelta)):
        raise TypeError(
            "`interval` should be of type 'str', 'timedelta' or"
            f"'pd.Timedelta', although received type '{type(interval)}'."
        )

    if isinstance(interval, str):
        value, unit = helpers.extract_freq_parts(interval)
        value, unit = int(value), unit.upper()
        valid_units = ["MIN", "T", "H", "D", "M"]
        if unit not in valid_units:
            raise ValueError(
                f"`interval` unit must by one of {valid_units} (or lower-"
                f"case) although evaluated to '{unit}'."
            )
        if unit == "MIN":
            unit = "T"

    else:
        if interval <= timedelta(0):
            raise ValueError("`interval` cannot be negative or zero.")

        if isinstance(interval, timedelta):
            interval = pd.Timedelta(interval)

        error_msg = (
            "An `interval` defined with a timedelta or pd.Timedelta can"
            " only be defined in terms of EITHER minute and/or hours OR"
            f" days, although received as '{interval}'. Note: to define an"
            " interval in terms of months pass as a string, for"
            ' example "1m" for one month.'
        )
        valid_resolutions = ["min", "h", "D"] + ["T", "H"]  # + form pandas pre 2.2
        if interval.resolution_string not in valid_resolutions:
            raise ValueError(error_msg)

        defined_components = [c for c in interval.components if c > 0]
        multiple_components = len(defined_components) > 1
        if multiple_components:
            minutes = interval.components.minutes
            hours = interval.components.hours
            if not minutes + hours == sum(interval.components):
                raise ValueError(error_msg)

    def raise_value_oob_error(component: str, limit: int):
        raise ValueError(
            f"An `interval` defined in terms of {component} cannot have"
            f" a value greater than {limit}, although received `interval`"
            f' as "{interval}".'
        )

    limits = {"T": 1320, "H": 22, "D": 250, "M": 36}
    components = {
        "T": "minutes",
        "H": "hours",
        "D": "days",
        "M": "months",
    }

    if isinstance(interval, str):

        if value > limits[unit]:
            raise_value_oob_error(components[unit], limits[unit])
        if unit == "T" and not value % 60:
            unit, value = "H", value // 60
        Cls = DOInterval if unit == "M" else TDInterval
        return getattr(Cls, unit + str(value))

    else:
        unit = interval.resolution_string
        if unit in ["min", "T"]:  # "T" for compatibility pandas < 2.2
            value = int(interval.total_seconds() // 60)
            unit = "T"
        elif unit in ["h", "H"]:  # "H" for compatibility pandas < 2.2
            value = int(interval.total_seconds() // 3600)
            unit = "H"
        else:
            value = interval.days

        if value > limits[unit]:
            raise_value_oob_error(components[unit], limits[unit])

        return getattr(TDInterval, unit + str(value))


def parse_interval(name: str, obj: str | pd.Timedelta | timedelta, _) -> PTInterval:
    """Parse input to an `interval` parameter."""
    return to_ptinterval(obj)
