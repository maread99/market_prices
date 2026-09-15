# LLM Assistant Guide for `market-prices` package
This file provides context for LLM assistants (Claude Code and similar tools) working in this repository.

In all context files, a '@' prefixing a path indicates that the path is defined relative to the project root in which this `AGENTS.md` file is located.

## Skills

Identify all available skills in the @.agents\skills directory

## LLM context

Add the 'agents' label to any PR that amends:
- this @AGENT.md
- any SKILL.md file

## Project Overview

**market-prices** is a Python package to create meaningful OHLCV datasets for financial instruments. It provides for enchanced querying and post-processing of financial price data.

See @pyproject.toml for project metadata and dependencies.

### Repository Layout

```
market_prices/
├── .agents/                            # instructions for LLM coding agents
│   └── skills/                         # skills for LLM coding agents
│       ├── create-pr/
│       │   └── SKILL.md
│       ├── dependencies-management/
│       │   └── SKILL.md
│       ├── update-agents-md/
│       │   └── SKILL.md
│       └── writing-tests/
│           └── SKILL.md
├── .github/
│   ├── workflows/
│   │   ├── build-test.yml
│   │   ├── draft-release-notes.yml
│   │   └── release.yml
│   ├── dependabot.yml
│   └── release-drafter.yml
├── docs/
│   ├── developers/
│   │   ├── other_internals.md          # notes on non-obvious internal design decisions
│   │   ├── releases.md
│   │   ├── serving_data.md
│   │   ├── testing.md
│   │   └── typing_doc.md               # typing conventions and documentation style
│   ├── media/
│   │   ├── readme_pt.png               # screenshot of .pt accessor used in README
│   │   └── splash.png
│   ├── public/
│   │   ├── method_doc.md
│   │   ├── parsing.md
│   │   └── typing.md
│   ├── tutorials/
│   │   ├── resources/                  # CSV data files for tutorial notebooks
│   │   ├── anchor.ipynb
│   │   ├── data_availability.ipynb
│   │   ├── intervals.ipynb
│   │   ├── other_get_options.ipynb
│   │   ├── other_prices_methods.ipynb
│   │   ├── periods.ipynb
│   │   ├── prices.ipynb
│   │   ├── pt_accessor.ipynb
│   │   ├── quickstart.ipynb
│   │   └── specific_query_methods.ipynb
│   └── tutorials_docs.md
├── src/
│   └── market_prices/                  # `PricesYahoo`, `PricesCsv`
│       ├── prices/                     # price-serving classes
│       │   ├── config/                 # configuration data for price sources
│       │   │   └── config_yahoo.py     # includes mappings for Yahoo data
│       │   ├── base.py                 # Core logic in `PricesBase`
│       │   ├── csv.py                  # `PricesCsv`: prices from local CSV files
│       │   └── yahoo.py                # `PricesYahoo`: prices via yahooquery
│       ├── support/                    # support for tutorials and tests
│       │   └── tutorial_helpers.py     # identify data to use in tutorials and tests
│       ├── utils/                      # utility modules
│       │   ├── calendar_utils.py       # includes `CompositeCalendar`
│       │   ├── general_utils.py        # general utils
│       │   └── pandas_utils.py         # pandas-specific utilities and context managers
│       ├── data.py                     # data administrator (by base interval)
│       ├── daterange.py                # derive date ranges for price requests
│       ├── errors.py                   # custom exception classes
│       ├── helpers.py                  # helpers (project-specific)
│       ├── intervals.py                # `TDInterval`, `DOInterval`, `BI` and helpers
│       ├── mptypes.py                  # custom types and aliases
│       ├── parsing.py                  # validates and coerces public input parameters
│       └── pt.py                       # .pt pandas accessor for custom DataFrame ops
├── tests/
│   ├── resources/                      # HDF5 stores and CSV fixtures
│   ├── conftest.py
│   ├── hypstrtgy.py                    # Hypothesis strategies
│   ├── test_base.py
│   ├── test_base_prices.py
│   ├── test_calendar_utils.py
│   ├── test_csv.py
│   ├── test_data.py
│   ├── test_daterange.py
│   ├── test_helpers.py
│   ├── test_intervals.py
│   ├── test_limits.py                  # tests for behaviour at data availability limits
│   ├── test_mpst.py                    # tests for complex Hypothesis strategies
│   ├── test_mptypes.py
│   ├── test_pandas_utils.py
│   ├── test_parsing.py
│   ├── test_pt.py
│   ├── test_tutorial_helpers.py
│   ├── test_yahoo.py
│   └── utils.py
├── .pre-commit-config.yaml
├── .python-version
├── AGENTS.md
├── CLAUDE.md
├── LICENSE.txt
├── MANIFEST.in
├── README.md
├── mypy.ini
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── ruff.toml
└── uv.lock
```

## Technology Stack

| Category | Tools |
|---|---|
| Python | 3.10–3.14 (`.python-version` pins 3.14) |
| Package manager | `uv` |
| Build backend | `setuptools` + `setuptools_scm` |
| Testing | `pytest` |
| Linting/formatting | `ruff` |
| Type checking | `mypy` |
| Git hooks | `pre-commit` |
| Data Manipulation | `pandas`, `numpy` |
| Calendars of Market Hours | `exchange-calendars` |

The current project version is managed by `setuptools_scm` and written to `src/market_prrices/_version.py`.
IMPORTANT: `src/market_prices/_version.py` is auto-generated and you should not edit it.

## Development Workflows

### Setup

```bash
# Install dependencies using uv
uv sync

# Install pre-commit hooks
pre-commit install
```

### Testing

- test with `pytest`
- see @pytest.ini for configuration; options are applied automatically via `addopts`.
- shared fixtures are in @tests/conftest.py
- tests are in @tests/
- doctests are included to some methods/functions
- create any calendar that will be evaluated against a mocked 'now' with explicit bounds defined from that mocked 'now', i.e. pass both `start` and `end` to `exchange_calendars.get_calendar`. This is necessary as `exchange_calendars` otherwise evaluates default bounds against the real clock when it is imported (i.e. before `mock_now` can take effect).

Commands to run tests:
```bash
# All tests (including doctests under src/market_prices/)
pytest

# Tests in specific file
pytest tests/test_module.py

# Specific test
pytest tests/test_module.py::test_name

# With verbose output
pytest -v
```

### Pre-commit Hooks

See @.pre-commit-config.yaml for pre-commit implementation.

Pre-commit runs automatically on `git commit`.

To run manually:
```bash
pre-commit run --all-files
```

### Continuous Integration

GitHub Actions is used for CI. Defined workflows include:
- @.github/workflows/build-test.yml - runs full test suite on matrix of platforms and python versions
- @.github/workflows/release.yml - releases a new version to PyPI

## Code Conventions

### Architecture

The project employs both hierarchal and compositional structure depending on context.

### Formatting

- format to `ruff` (Black compatible).  
- see @ruff.toml for configuration.

```bash
# Format code
ruff format .
```

### Linting

- lint with `ruff`.
- See lint sections of @ruff.toml for configuration (includes excluded files).
- type check with `mypy`.

```bash
# Check lint issues
ruff check .

# Type checking
uv run mypy src/market_prices/
```

### Imports

- No wildcard imports (i.e. no `from x import *`).

### Type Annotations

- Type annotations are required on all public functions and methods.
- See @mypy.ini for configuration
    - `ignore_missing_imports = True` is set globally (many dependencies lack stubs).
- `valimp` library is used for runtime parameter validation:
    - use `@parse` decorator with typed signatures.
    - use`@parse_cls` for dataclasses.

### Docstrings

Public modules, classes, and functions MUST all have docstrings.

Docstrings should follow **NumPy convention**. Familiarise yourself with this as described at https://numpydoc.readthedocs.io/en/latest/format.html. That said, the following should always be adhered to and allowed to override any NumPy convention:
- 75 character line limit for public documentation
- 88 character line limit for private documentation
- formatted to ruff
- parameter types should not be included to the docstring unless this provides useful information that users could not otherwise ascertain from the typed function signature.
- default values should only be noted in function/module docstrings if not defined in the signature - for example if the parameter's default value is None and when received as None the default takes a concrete dynamically evaluated default value. When a default value is included to the parameter documentation it should be defined after a comma at the end of the parameter description, for example:
    - description of parameter 'whatever', defaults to 0.
- **subclasses** documentation should:
    - list only methods and attributes added by the subclass. A note should be included referring users to documentation of base classes for the methods and attributes defined there.
    - include a NOTES section documenting how to implement the subclass (only if not trivial).
- documentation of **subclass methods that extend methods of a base class** should only include any parameters added by the extension. With respect to undocumented parameters a note should be included to refer the user to the corresponding 'super' method(s)' documentation on the corresponding base class or classes.
- **documentation of exceptions and warnings** should be limited to only **unusual** exceptions and warnings that are raised directly by the function/method itself or by any private function/method that is called directly or indirectly by the function/method.
- summary line should be in the imperative mood only when sensical to do so.
- magic methods do not require documentation if their functionality is fully implied by the method name.
- unit tests do not require docstrings.

Example documentation:
```python
def my_func(param1: int, param2: str = "default", param3: None | str = None) -> bool:
    """Short summary line.

    Extended description if needed.

    Parameters
    ----------
    param1
        Description of param1.
    param2
        Description of param2.
    param3
        Description of param3, defaults to value of `param2`.

    Returns
    -------
    bool
        Description of return value.
    """
```

#### Documentation content

- when **revising documentation to reflect changes in implementation**, do NOT fall into the trap of documenting what's 'not the case' in the context of the previous implementation. Rather, just document what *is* the case, in the context of the revised implementation. Documentation that describes what the implementation does NOT do, in terms of a prior or erroneous implementation, is irrelevant and obfuscating to a reader who never knew of that prior implementation. Only ever make comparisons to erroneous or prior implementations if there is GOOD REASON to fear regression!

### Comments

- pay particular attention to comments starting with...: 
    - 'NOTE'
    - 'TODO'
    - 'AIDEV-NOTE' - these comments are specifically addressed to you.
    - 'AIDEV-TODO' - these comments are specifically requesting you do something.
    - 'AIDEV-QUESTION' - these comments are asking a question for specifically you to answer.

---

## Important Notes for AI Agents

1. **NEVER DO RULES**:
    - NEVER schedule a check-in unless specifically asked to.
    - Never edit the file `src/market_prices/_version.py` - this is auto-generated by the build process.

2. **Do not assume that the package is coherent or free of bugs**, or that tests or documentation should be treated as gospel. If you come across a contradiction or a conflict or what you believe to be a bug, then say so.

3. **Use `valimp` for validation of parameters of public API** — see 'Type Annotations' section of this @AGENTS.md file.

4. **NumPy docstring style** — all new public functions/classes must use NumPy-convention docstrings and rules as defined under Docstrings section of this @AGENTS.md file.

5. **Branch naming** — feature branches should follow the `<agent-name>/<description>` pattern with the following substitutions:
    - `<agent-name>` -> your name
    - `<description>` -> very short description of changes
