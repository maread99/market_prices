"""Types used for annotations.

Includes:
    Type aliases.

    Internal types:
        Type aliases
        enums
        TypeGuards

NOTE: types concerning intervals are maintained on the
`market_prices.intervals` module.
"""

import datetime
import enum
from typing import TypedDict

import pandas as pd
from exchange_calendars import ExchangeCalendar


# ----------------------------- Type aliases ------------------------------

Symbols = list[str] | str
"""For public parameters that define instrument symbol(s)."""

Calendar = str | ExchangeCalendar  # pylint: disable=no-member
"""Acceptable types to define a single calendar."""

Calendars = Calendar | list[Calendar] | dict[str, Calendar]
"""For public parameters that can define calendars by-symbol."""

# ----------------------------- Custom types ------------------------------


class PandasFrequency(str):
    """Validated pandas frequency."""

    def __new__(cls, value: str):
        try:
            _ = pd.tseries.frequencies.to_offset(value)
        except ValueError:
            msg = (
                f"PandasFrequency must be a pandas frequency although"
                f" received '{value}'."
            )
            raise ValueError(msg) from None
        return super().__new__(cls, value)

    @property
    def as_offset(
        self,
    ) -> pd.offsets.BaseOffset:
        """Frequency as a pandas offset."""
        return pd.tseries.frequencies.to_offset(self)


# ---------------------------- Internal types -----------------------------
# Interval types are NOT to be used to annotate public parameters.

# ------------------------------ Type aliases -----------------------------

# Aliases with no public use
DateRangeAmb = tuple[pd.Timestamp | None, pd.Timestamp | None]
"""For internal types that define a range of dates which can be ambiguous.

tuple[0]: Range start date. If None, earliest available date.
tuple[1]: Range end date. If None, latest available date.
"""

DateRangeReq = tuple[pd.Timestamp | None, pd.Timestamp]
"""For internal types that define a range of dates over which to request prices.

tuple[0]: Range start date. If None, earliest date for which prices are available.
tuple[1]: Range end date.
"""

DateRange = tuple[pd.Timestamp, pd.Timestamp]
"""For internal parameters that define an unambiguous range of dates.

tuple[0]: Range start date.
tuple[1]: Range end date.
"""

DateTimestamp = pd.Timestamp | str | datetime.datetime | int | float
"""Type to annotate an input that takes a value representing a date.

Used in abstract base classes to identify inputs that should be coerced to
a `pd.Timestamp` and validated as a date with
`market_prices.parsing.to_datetimestamp`.
"""

TimeTimestamp = pd.Timestamp | str | datetime.datetime | int | float
"""Type to annotate an input that takes a value representing a time.

Used in abstract base classes to identify inputs that should be coerced to
a `pd.Timestamp` and validated as a time with
`market_prices.parsing.to_timetimestamp`.
"""

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
