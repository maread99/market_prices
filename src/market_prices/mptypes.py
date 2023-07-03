"""Types used for annotations.

Includes:
    Type aliases.
    Custom pydantic types used to parse parameters of public methods.

Internal types are defined to their own section.

Note: types concerning intervals are maintained on the
`market_prices.intervals` module.
"""

from __future__ import annotations

import collections
import enum
from typing import Any, Dict, List, Tuple, TypedDict, Union

import pandas as pd
import pytz
from exchange_calendars import ExchangeCalendar

import pydantic

if int(next(c for c in pydantic.__version__ if c.isdigit())) > 1:
    from pydantic import v1 as pydantic

# pylint: disable=too-few-public-methods  # nature of pydantic types.

# ----------------------------- Type aliases ------------------------------

Symbols = Union[List[str], str]
"""For public parameters that define instrument symbol(s)."""

Calendar = Union[pydantic.StrictStr, ExchangeCalendar]  # pylint: disable=no-member
"""Acceptable types to define a single calendar."""

Calendars = Union[Calendar, List[Calendar], Dict[str, Calendar]]
"""For public parameters that can define calendars by-symbol."""


# ----------------- Custom types with pydantic validators -----------------


def type_error_msg(
    type_: type[object],
    valid_types: type[object] | collections.abc.Sequence[type[object]],
    value: Any,
) -> str:
    """Return error message for a custom type receiving an invalid type.

    Parameters
    ----------
    type_
        Custom type.

    valid_types
        Valid type or types.

    value
        Value, of invalid type, received by parameter annotated with
        `type_`.
    """
    msg = f"{type_.__name__}"
    if isinstance(valid_types, collections.abc.Sequence):
        msg += f" can take any type from {list(valid_types)}"
    else:
        msg += f" takes type {valid_types}"

    msg += f" although receieved <{value}> of type {type(value)}."
    return msg


class LeadSymbol:
    """Type to validate `lead_symbol` parameter.

    Only for annotating `lead_symbol` parameter of public methods of
    PricesBase (or subclass of), excluding __init__.

    A parameter annotated with this class can only take types that can be
    coerced to a `str` which is in PricesBase.symbols.

    The Formal Parameter will be assigned a `str`.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v: str | None, values) -> str:
        valid_types = str
        if not isinstance(v, valid_types):
            raise TypeError(type_error_msg(cls, valid_types, v))

        values["self"]._verify_lead_symbol(v)  # pylint: disable=protected-access
        return v


class Timezone:
    """Type to parse to a timezone.

    A parameter annotated with this class can take:
        - an instance returned by pytz.timezone (i.e. instance of subclass
            of pytz.BaseTzInfo) or a
        - `str` that can be passed to pytz.timezone (for example 'utc' or
            'US/Eastern`)

    The formal parameter will be assigned an instance of a subclass of
    `pytz.BaseTzInfo`.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v) -> pytz.BaseTzInfo:
        if isinstance(v, pytz.BaseTzInfo):
            return v

        valid_types = str, pytz.BaseTzInfo
        if not isinstance(v, str):
            raise TypeError(type_error_msg(cls, valid_types, v))

        return pytz.timezone(v)


class PricesTimezone:
    """Type to parse to a PricesBase parameter to a timezone.

    Only for annotating `tz`, `tzin` or `tzout` parameters of public
    methods of PricesBase (or subclass of).

    A parameter annotated with this class can take
    "pytz.BaseTzInfo | str", where:
        - pytz.BaseTzInfo: any instance returned by pytz.timezone
        - str:
            - valid input to `pytz.timezone`, for example 'utc' or
                'US/Eastern`, to parse to pytz.timezone(<value>)
            - any symbol of `PricesBase.symbols` to parse to the timezone
                associated with that symbol.

    The formal parameter will be assigned an instance of a subclass of
    `pytz.BaseTzInfo`.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v, values: dict) -> pytz.BaseTzInfo:
        if isinstance(v, pytz.BaseTzInfo):
            return v

        valid_types = str, pytz.BaseTzInfo
        if not isinstance(v, str):
            raise TypeError(type_error_msg(cls, valid_types, v))

        if v in values["self"].symbols:
            return values["self"].timezones[v]
        else:
            return pytz.timezone(v)


class Timestamp:
    """Type to parse to a pd.Timestamp.

    A parameter annotated with this class can take any object that is
    acceptable as a single-argument input to pd.Timestamp:
        Union[pd.Timestamp, str, datetime.datetime, int, float]

    The formal parameter will be assigned a pd.Timestamp.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v) -> pd.Timestamp:
        # if v not valid single-argument input to pd.Timestamp then will raise error.
        return pd.Timestamp(v)


class DateTimestamp(Timestamp):
    """Type to parse to a pd.Timestamp and validate as a date.

    Considered a valid date (rather than a time), if:
        - no time component or time component defined as 00:00.
        - tz-naive.

    A parameter annotated with this class can take any object that is
    acceptable as a single-argument input to pd.Timestamp:
        Union[pd.Timestamp, str, datetime.datetime, int, float]

    The formal parameter will be assigned a pd.Timestamp.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate
        yield cls._validate_date

    @classmethod
    def _validate_date(cls, v: pd.Timestamp, field) -> pd.Timestamp:
        if v.tz is not None:
            msg = f"`{field.name}` must be tz-naive, although receieved as {v}."
            raise ValueError(msg)

        if v != v.normalize():  # type: ignore[unreachable]  # mypy doesn't like v.tz
            msg = (
                f"`{field.name}` can not have a time component, although receieved"
                f" as {v}. For an intraday price use .price_at()."
            )
            raise ValueError(msg)
        return v


class TimeTimestamp(Timestamp):
    """Type to parse to a pd.Timestamp and validate as representing a time.

    Considered a valid time (rather than a date) if:
        - time component defined as anything other than 00:00.
        - time component defined as 00:00 and tz-aware.

    A parameter annotated with this class can take any object that is
    acceptable as a single-argument input to pd.Timestamp:
        Union[pd.Timestamp, str, datetime.datetime, int, float]

    The formal parameter will be assigned a pd.Timestamp.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate
        yield cls._validate_time

    @classmethod
    def _validate_time(cls, v: pd.Timestamp, field) -> pd.Timestamp:
        if v == v.normalize() and v.tz is None:
            msg = (  # type: ignore[unreachable]  # mypy doesn't like v.tz
                f"`{field.name}` must have a time component or be tz-aware, although"
                f" receieved as {v}. To define {field.name} as midnight pass as a"
                f" tz-aware pd.Timestamp. For prices as at a session's close use"
                f" .close_at()."
            )
            raise ValueError(msg)
        return v


class PandasFrequency(str):
    """Validated pandas frequency.

    A field annotated with this class:
        can take a string that is a valid pandas frequency, determined as
        being acceptable input to pd.tseries.frequencies.to_offset().

        will be assigned a PandasFrequency.

    Attributes
    ----------
    In addition to inherited str methods:

        as_offset
    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v) -> "PandasFrequency":
        if not isinstance(v, str):
            raise TypeError(type_error_msg(cls, str, v))

        try:
            _ = pd.tseries.frequencies.to_offset(v)
        except ValueError:
            msg = (
                f"PandasFrequency must be a pandas frequency although"
                f" received '{v}'."
            )
            raise ValueError(msg) from None

        return cls(v)

    @property
    def as_offset(
        self,
    ) -> pd.offsets.BaseOffset:  # type: ignore[name-defined]  # is defined
        """Frequency as a pandas offset."""
        return pd.tseries.frequencies.to_offset(self)


class IntervalDatetimeIndex(pd.IntervalIndex):
    """Validated IntervalIndex with left and right as pd.DatetimeIndex."""

    # pylint: disable=abstract-method

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v) -> pd.IntervalIndex:
        if not isinstance(v, pd.IntervalIndex):
            raise ValueError(
                "Parameter must be passed as an instance of pd.IntervalIndex."
            )
        elif not isinstance(v.left, pd.DatetimeIndex):
            raise ValueError(
                "Parameter must have each side as type pd.DatetimeIndex"
                f" although received left side as '{v.left}'."
            )
        return v


# ---------------------------- Internal types -----------------------------
# Interval types are NOT to be used to annotate public parameters.

# ------------------------------ Type aliases -----------------------------

# Aliases with no public use
DateRangeAmb = Tuple[Union[pd.Timestamp, None], Union[pd.Timestamp, None]]
"""For internal types that define a range of dates which can be ambiguous.

tuple[0]: Range start date. If None, earliest available date.
tuple[1]: Range end date. If None, latest available date.
"""

DateRangeReq = Tuple[Union[pd.Timestamp, None], pd.Timestamp]
"""For internal types that define a range of dates over which to request prices.

tuple[0]: Range start date. If None, earliest date for which prices are available.
tuple[1]: Range end date.
"""

DateRange = Tuple[pd.Timestamp, pd.Timestamp]
"""For internal parameters that define an unambiguous range of dates.

tuple[0]: Range start date.
tuple[1]: Range end date.
"""

PytzUTC = type(pytz.UTC)

# -------------------------------- enums ----------------------------------


class Alignment(enum.Enum):
    """Internal representation of `end_alignment` parameter values."""

    BI = "bi"
    FINAL = "final"


class Anchor(enum.Enum):
    """Internal representation of `anchor` parameter values."""

    OPEN = "open"
    WORKBACK = "workback"


class OpenEnd(enum.Enum):
    """Internal representation of `anchor` parameter values."""

    MAINTAIN = "maintain"
    SHORTEN = "shorten"


class Priority(enum.Enum):
    """Internal representation of `priority` parameter values."""

    PERIOD = "period"
    END = "end"


# ----------------------------- Type guards -------------------------------


class PP(TypedDict):
    """Interval type guard for dictionary holding period parameters."""

    minutes: int
    hours: int
    days: int
    weeks: int
    months: int
    years: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    add_a_row: bool
