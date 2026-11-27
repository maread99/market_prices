---
name: writing-tests
description: Conventions for writing, revising and reviewing the `market-prices` test suite (pytest) - test design, datasets and fixtures, layout, naming and documentation. Use whenever adding or changing anything under `tests/`, including when a change to `src/market_prices/` needs new or revised tests.
---
# Writing tests

Tests exist to verify the intended and documented behaviour of the code, to find bugs in existing code and to protect against new code introducing bugs. They are written for a human reviewer, who should be able to see what each test verifies, under what circumstance, and why its expected values are what they are, without having to reverse-engineer the data. Tests should meet these aims efficiently in terms of the amount of code and run time. The order of priority is:
1. correctness - the test verifies intended and documented behaviour and fails if that behaviour breaks.
2. intelligibility.
3. efficiency.

## Test design
- tests use the `pytest` framework and make use of its features (for example fixtures, parametrisation and `tmp_path`) wherever they serve the test.
- test intended and documented behaviour (docstrings, and the documentation under @docs/) to find bugs in the implementation. Never assume that the implementation is correct or that a test should pass - where a test exposes behaviour that contradicts the documentation, report it rather than revising the test to fit.
- tests, and the structure of the test suite, are designed for intelligibility (to a human reviewer) and efficiency (the amount of code, no test repeating another's verification, and run time). Where the two conflict, intelligibility takes precedence. For example:
  - tests use shared contexts (for example, fixtures and datasets) where appropriate, although a separate context is created rather than an existing one being overly manipulated to fit a different pattern. Using a subset of an existing context is acceptable, although where a test needs only a small part of a context (for example, a single row) it defines that part itself, such that the test can be read without looking elsewhere. Changing multiple values across a context so that it tells a different story is not acceptable.
  - a single test combines the testing of multiple aspects of the same verification, but not disparate behaviours. Cases of the same verification are parametrised with `pytest.mark.parametrize`, with `ids` that name each case where the values alone do not.
  - each function is tested directly, over the smallest data that describes the circumstance under test, rather than via a higher-level evaluation that re-tests other functions on the way.
- always test on and either side of each limit where output changes, and seek out edge cases (for example empty input, a single row, the first and last row, a tie).
- do not include tests that add nothing that isn't already verified elsewhere. An exception to this is where the other verifications are sufficiently obscure to make reliance on them unintelligible for this purpose.
- when code or tests are redesigned, don't just check that related tests continue to pass, also consider if they require redesigning or if the changes have made a test redundant as a result of its verifications all now being made elsewhere by more appropriate tests.
- a private function or method can be tested directly where it holds behaviour of its own.
- where a test relies on a property of its data (for example a session being an early close, or a timestamp falling within a break), assert that property, commented `# verify condition`, so that a change to the data fails the test, thereby necessitating a review.
- pass explicitly every argument that a test's expected values depend on, rather than relying on its default, unless the test verifies the default itself. A reader can then see where each expected value comes from, and a change of default does not silently change what the test verifies.
- where a test verifies only that a function passes values through to another that is tested elsewhere (a **delegation test**), pass non-default values, verify that each changes the outcome (such that a value that is not passed through fails the test), name where the other function is tested and say in the docstring that it is a delegation test.
- use mocking (`monkeypatch`, the `mocker` fixture of `pytest-mock`, `unittest.mock`) to stand in for external services and, sparingly, to create a circumstance that data cannot readily produce. Otherwise prefer real data.
- create any calendar that will be evaluated against a mocked 'now' with explicit bounds defined from that mocked 'now', i.e. pass both `start` and `end` to `exchange_calendars.get_calendar`. This is necessary as `exchange_calendars` otherwise evaluates default bounds against the real clock when it is imported (i.e. before `mock_now` can take effect).

## Refactoring and removing tests
- coverage must not drop. Line and branch coverage are a minimum check. To show that revised tests still verify everything the previous tests did, mutation testing (deliberately breaking the code under test and checking that a test fails) is the stronger check.

## Datasets and fixtures
- a dataset that is defined in code is assigned to a module-level constant so that it can be interrogated outside of the tests. Tests do not access the constant directly but via a fixture that returns a COPY, so that no test can change the data another test sees. A dataset of fewer than 10 rows that can be read at a glance can instead be defined inline in the test or fixture that uses it.
- a dataset's docstring describes the circumstance it represents (what happens in which rows, and why), such that a reader can follow the tests that use it without re-deriving the data.
- define test data to include values at the limits of where the output of tested functions/methods changes.
- a dataset or fixture is defined at the narrowest level that shares it:
  - shared by the whole suite - `tests/conftest.py`.
  - shared by the test modules of one package - the `conftest.py` of that package's test directory.
  - used by one test module - that module.
  - used by the tests of one class - a method of that class.
- a test module never imports from another test module. What test modules share goes in a `conftest.py` or in a helper module such as `tests/utils.py`.
- a fixture that returns a class is named in CapWords, as a class is. Every other fixture is named in snake_case.
- a fixture that returns a factory is named `make_<thing>` where the factory creates instances (for example `make_prices`), or `define_<thing>` where it defines classes (for example `define_subclass`).

## Location and naming
- tests are in `tests/`.
- the module under test is imported to the test module as `m`, and the functions, classes etc. under test are accessed via the alias, for example `m.function_under_test`. This makes clear what is under test and what is a collaborator; collaborators from other modules are imported by name.
- doctests can be used instead of unit tests if the doctest fully covers the function/method's functionality.
- within a class of tests, the following names are used where they fit:
  - `test_on_and_either_side_of_<limit>` for the output on and either side of a limit.
  - `test_none` for the circumstances in which a function returns None, an empty input among them.
  - `test_empty` for the return of a function that returns a structure (for example a frame or a Series) for empty input.

## Tests of `PricesBase` and `PricesYahoo`
- @docs/developers/testing.md defines conventions for the tests of `PricesBase` and `PricesYahoo`. It covers the module that holds each of these tests, the `PricesBaseTst` fixtures that serve stored price data, and how a test avoids the sessions for which the Yahoo API is unreliable. Read it before adding or changing any of these tests.

## Documentation
- a test docstring, where included, starts with an imperative, for example 'Verify' or 'Test'.
- a test docstring states WHAT the test verifies and, where not evident from the code, the circumstance it is verified under and WHY it is verified that way (the scenario, scope or design of the test) - i.e. anything about the test as a whole.
- a paragraph of a test docstring that describes one of the following opens with the words given here, such that its opening announces what the paragraph describes:
  - what the test verifies - "This test verifies ...". The paragraph may also state the circumstance under which the test verifies it.
  - what the code under test does - "The <function | method | property | class> under test ...", for example "The method under test ...".
  - the parameters of a parametrised test - "The parameters ..." or, for a single parameter, "The parameter `<name>` ...". A paragraph that describes two or more parameters may instead describe each parameter in a sentence of its own, each sentence opening "The parameter `<name>` ...". A paragraph of a class docstring that describes a parameter shared by the class's tests opens in the same way.

  No paragraph describes more than one of these, such that each of them has a paragraph of its own. Any other content of the docstring (for example the scenario, a property of the data, or why the test is designed as it is) goes in paragraphs of its own, which may open freely. A test docstring is not required to include any of these paragraphs: the rule governs only how such a paragraph opens and what it holds. The summary line, which states what the test verifies in the imperative ("Verify ..."), is not such a paragraph.
- explanation tied to particular lines goes in a code comment at those lines - for example the derivation of an expected value, why a case or assertion gives its result, or what a particular value represents. Where a comment annotates a single line (for example one case of a parametrisation) it goes inline.
- class, fixture and dataset docstrings describe the context shared by the tests that use them.
