# Typing
`market_prices` is comprehensively typed using both standard library and third-party types. (NB currently typing is NOT declared by way including a py.typed file to the setup package_data.)

Third party types will usually be defined with a full dotted path from the package name. The conventional 'pd' and 'np' abbreviations are employed for pandas and numpy respectively.

## mptypes

`market_prices` defines [type aliases](#Type-aliases) and custom types to annotate some parameters of public methods. Such types are defined in the `mptypes.py` module.

When a parameter takes an mptype the underlying valid types are expressed in the 'Parameters' section of the method's documentation.

### Type aliases
`market_prices` occassionally uses type aliases to represent multiple underlying types that are acceptable input. The underlying types can be inspected by calling the type alias:

```python
>>> from market_prices.mptypes import Symbols, Calendar
>>> Symbols
typing.Union[list[str], str]
>>> Calendar
typing.Union[str, exchange_calendars.exchange_calendar.ExchangeCalendar]
```
