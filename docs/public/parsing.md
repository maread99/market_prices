# Parsing

`market_prices` uses the [pydantic library](https://pydantic-docs.helpmanual.io/) to parse parameters received by public functions and methods. Pydantic ensures that the type passed to a formal parameter conforms with the parameter's type annotation. Passing a object with an invalid type will raise a `pydantic.ValidationError` with a message advising of the invalid inputs and what was expected.

## Coercing

When a parameter receives an invalid type pydantic will try to coerce it to a valid type. For example, a parameter annoated with `int` could be passed a `str` "3" which would be coerced to the `int` 3. It could also be passed a `float` 2.99 which would be coerced the `int` 2!

Parameters that do not allow coercing are typed with a pydantic 'Strict' type, for example `pydantic.StrictInt`.

## Custom pydantic types
`market_prices` defines custom pydantic types for certain parameters. The parsing of custom types may perform additional validations and define default values.

For example, the type `mptypes.PricesTimezone` is defined for parameters that allow a timezone to be specified by way of a symbol or `pytz` timezone object. The parsing process checks that the input is of a valid type and value and then passes through a pytz timezone object to the formal parameter.

The type's documentation includes the requirements for input to be considered valid.
```python
>>> from market_prices.mptypes import DateTimestamp
>>> help(DateTimestamp)
```
    Help on class DateTimestamp in module market_prices.mptypes:

    class DateTimestamp(Timestamp)
     |  Type to parse to a pd.Timestamp and validate as a date.
     |  
     |  Considered a valid date (rather than a time), if:
     |      - no time component or time component defined as 00:00.
     |      - tz-naive.
     |  
     |  A parameter annotated with this class can take any object that is
     |  acceptable as a single-argument input to pd.Timestamp:
     |      Union[pd.Timestamp, str, datetime.datetime, int, float]
     |  
     |  The formal parameter will be assigned a pd.Timestamp.