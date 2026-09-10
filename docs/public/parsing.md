# Parsing

`market_prices` uses the [`valimp`](https://github.com/maread99/valimp) library to parse parameters received by public functions and methods.

## Type validation
 The `valimp.parse` decorator ensures that the objects passed to a function's parameters conform with the corresponding type annotation. Where a parameter takes a container this validation extends to validating the type of the container items. For example, an input for the following parameter would be validated as being a `dict` and each item of that dictionary would be validated as having the key as a `str` and the value as either an `int` or a `float`:

```python
param: dict[str, Union[int, float]]
```

An instance of `valimp.InputsError` is raised if at least one object passed to a function does not confrom with the corresponding type annotation.

## Coercing

An instance of `valimp.Coerce` in a parameter's annotation simply indicates that the object will
be subsequently coerced to a specific type. For example, the following 'start' parameter can take an object of type `pd.Timestamp`, `str`, `datetime.datetime`, `int`, or `float`. In all cases the object will be coerced to a `pd.Timestamp` (NB a None value is never coerced).

```python
start: Annotated[
    Union[pd.Timestamp, str, datetime.datetime, int, float, None],
    Coerce(pd.Timestamp),
] = None,
```

(NB The type annotation is wrapped in `typing.Annotated` and the `valimp.Coerce` instance is passed to the annotated metadata.)

## Ad-hoc validation

An instance of `valimp.Parser` in the type annotation indicates that the input will be subsequently parsed before reaching the decorated function. This parsing may undertake further validation or dynamically assign a default value. For example, the following 'session' parameter will be coerced to a `pd.Timestamp` which in turn will be verified as representing a date (as opposed to a time) by the `parsing.verify_datetimestamp` function.

```python
session: Annotated[
    Union[pd.Timestamp, str, datetime.datetime, int, float, None],
    Coerce(pd.Timestamp),
    Parser(parsing.verify_datetimestamp, parse_none=False),
] = None,
```

In this case if the input does not represent a date then `parsing.verify_datetimestamp` will raise an appropriate error (the parsing functions' documentation offer advices as to what's required for an input to be considered valid).

(NB The `parse_none` argument indicates to the `parse` decorator that a `None` value should not be parsed.)

(NB The type annotation is wrapped in `typing.Annotated` and the `valimp.Parser` instance is passed to the annotated metadata.)