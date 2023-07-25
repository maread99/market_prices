# Typing and documentation

## Typing

See [typing.md](../public/typing.md).

**The parameters and returns of all methods, properties and functions should be typed.**

It is not necessary to type tests although test helpers should be typed.

### mypy

Code should be type checked with mpyp configured for the project's `mpyp.ini` file. 

Accepted errors should be silenced in-line with `# type: ignore[error_ref]`. An additional comment should be included alongside explaining why the error is accepted.

Errors arising for reasons listed in the comments towards the top fo the `mypy.ini` file do NOT need to be silenced.

### mptypes.py

Types specific to `market_prices` are defined on the `mptypes.py` module. These include type aliases, custom types and internal enums.

The type annotation of any public parameter that takes a type defined on the mptypes module should begin `mptypes.`. This is to explictly declare the type as being specific to `market_prices`.

## Documentation

See [method_doc.md](../public/method_doc.md).

Docstrings should conform to the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

### Public

Maximum line length 75.

#### **Public classes**
Class documentation should **not** include a listing of attributes, properties, methods etc, rather users should be expected to rely on `help(cls)`.

#### **Public functions, methods, properties**
All public properites, methods and functions should be documented.

The documentation of public methods and functions should include a 'Parameters' section listing all parameters.

It is not usually necessary for the parameter type to be defined alongside the parameter name, rather users can expected to inspect the annotations defined in the method signature. Exceptions:
* parameters annotated with a mptype should be documented with the underlying valid types.
* for ease of reference, the type of all parameters should be documentated if the docstring is particularlly long, for example `PricesBase.get.__doc__`.

### Private

Maximum line length 88.

All private properties, methods and functions should be documented, the only exception being if the name 'says it all'.