# Method documentation

`market_prices` follows the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for documenting public functions and methods.

## Parameter documentation
Parameters are listed under the 'Parameters' section of a method's docstring.
* A parameter's type is usually only included to the documentation for parameters annotated with a mptype (see [Typing](./typing.md)). For all other parameters user's should refer to a parameter's type annotation in the method signature.
    * For ease of reference, parameter type is included to the documentation of particularly long docstrings, for example PricesBase.get().
* Default values are only documented if the default differs from the default value defined in the method signature, for example if the default value is `None` and the actual default is defined dynamically. In all other cases the user should refer to the method signature for default values.