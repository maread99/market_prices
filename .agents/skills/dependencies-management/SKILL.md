---
  name: dependencies-management
  description: instructions for updating and managing project dependencies (including Github actions).
---
# Dependencies Management

## Update Dependencies

Instructions to update project dependencies (including Github actions used by CI workflows).

### 1. Prepare the branch

Create a new branch (following the naming convention in @AGENTS.md).

### 2. Update the lock file and environment

First, update `uv` itself to ensure it can correctly parse all configuration (an outdated
`uv` may silently ignore relative durations such as `"2 days"` in `exclude-newer`,
resolving without any timestamp cutoff):

```bash
uv self update
```

If `uv self update` fails due to a network or authentication error (e.g. GitHub API rate
limit), fall back to updating via `pip` and copying the new binary over the old one:

```bash
pip install -U uv
cp /usr/local/bin/uv ~/.local/bin/uv
```

Then update the lock file and environment:

```bash
uv lock --upgrade  # update the lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # sync @requirements.txt with @uv.lock
uv sync --inexact  # update environment to match @uv.lock
```

### 3. Update GitHub Actions versions

For each `uses: <owner>/<repo>@<version>` entry across all files in `.github/workflows/`, retrieve the latest release tag by fetching the releases page with WebFetch:

```
url:    https://github.com/<owner>/<repo>/releases/latest
prompt: What is the latest release tag for this GitHub action?
```

In each case update the version pin to any more recent release. **Preserve the existing pinning style**, so if the current pin is a specific version such as `@v3.0.1`, update to the full latest version string, whilst if it's a major-version tag such as `@v4`, update to any new major-version tag (if the project no longer publishes a major-version tag then switch to the full semver).

### 4. Test

Before running the tests check whether the test environment has live access to the
required network endpoints by running:

```bash
python -c "
import urllib.request, sys

for label, url in [
    ('yahoo', 'https://query1.finance.yahoo.com/v8/finance/chart/MSFT?interval=1d&range=1d'),
    ('raw.github.com', 'https://raw.githubusercontent.com/'),
]:
    try:
        urllib.request.urlopen(url, timeout=10)
        print(f'{label}: reachable')
    except Exception as e:
        print(f'{label}: unreachable ({e})')
"
```

Then, run the test suite with options as determined by the reachability results:
- If **both reachable**: run the full test suite:
  ```bash
  uv run pytest -v
  ```
- If **yahoo unreachable** (regardless of raw.github.com): exclude `tests/test_yahoo.py`
  and, if raw.github.com is also unreachable, also exclude `tests/test_calendar_utils.py`
  (see *Network tests* section below):
  ```bash
  # yahoo unreachable, raw.github.com reachable
  uv run pytest --ignore=tests/test_yahoo.py -v

  # both unreachable
  uv run pytest --ignore=tests/test_yahoo.py --ignore=tests/test_calendar_utils.py -v
  ```
- If **only raw.github.com unreachable**: exclude `tests/test_calendar_utils.py`:
  ```bash
  uv run pytest --ignore=tests/test_calendar_utils.py -v
  ```

**Interpreting the local test results:**
- All tests pass and no raised warning is fixable (see *Fixable warnings* section below) → go to step 6 to raise PR.
- All failing tests are in `tests/test_yahoo.py` and/or `tests/test_calendar_utils.py` and no raised warning is fixable → test failures probably due to a transient network issue (see *Network tests* section below), go to step 6 to raise PR.
- Failure of any other test or a fixable warning raised → proceed to step 5 to fix.

### 5. Fix
Any failing tests and fixable warnings will likely have their origin in changes to the dependencies. To provide support for the latest dependencies MAKE REVISIONS to the code base to fix:
- code causing non-network tests to fail (see *Network tests* section below for notes on network tests).
- all fixable warnings (see *Fixable warnings* section below)

As a general RULE, **change the package code to get the tests passing, not the test code!** You may make changes to the test code only with good reason and only when this does not impair the test's efficacy.

To facilitate identifying the cause of test failures consider researching the changelogs of updated dependencies for versions released since the previously locked version.

Iterate on this process until all non-network tests are passing and all fixable warnings have been fixed.

IMPORTANT: in this step you should not run the full test suite, rather validate fixes by re-running only the previously failing tests. Example to run a specific test:

```bash
pytest tests/test_module.py::test_name
```

### 6. Raise PR

Once local tests pass, commit all changes to the branch and raise a PR.

- **PR title**: Title the PR as `Update Dependencies <MM> <DD> (auto)` where:
  - `<MM>` should be replaced with the first three letters of the current month, the first of which should be capitalized.
  - `<DD>` should be replaced with the current day of the month as represented by two digits.
  Example title: `Update Dependencies Apr 07 (auto)`
- **label**: Add the 'dependencies' label to the PR via `mcp__github__update_pull_request`.
- otherwise comply with the /create-pr skill.

### 7. Inspect CI results

The CI `build-test.yml` workflow will automatically trigger on the PR being raised (against the `master` branch). Under this workflow the full test suite will be run on combinations of supported OS and Python versions. Once the workflow completes, use `mcp__github__pull_request_read` with the `get_check_runs` method to read check statuses for all matrix jobs.

**Interpreting CI results:**
- All checks green → done, no further action is required.
- Failures only in `tests/test_yahoo.py` and/or `tests/test_calendar_utils.py` → possibly
  due to a transient network issue (see *Network tests* section below), **but first check
  the actual error type**:
  - Connection errors, HTTP 403/timeout errors → network issue. Add a comment to the PR
    identifying the failure as a probable network issue and suggest that the owner re-run
    the failing jobs. No further action is required.
  - Assertion errors, import errors, or `AttributeError` → may indicate a
    dependency-caused regression. Investigate before dismissing as a network issue and
    proceed to step 8 if warranted.
- Failures in any other test file → proceed to step 8.

### 8. Fix tests for specific OS/Python configuration

Use the information read from `get_check_runs` to identify any OS/python version configurations for which the test suite has failed.

If the tests failed on specific matrix combinations (e.g. Windows / Python 3.10) then simulate a local matching environment.
- If necessary use a Docker container with the target OS. (To specify different Python versions you will be able to run commands with `uv run` and pass the --python option.)
- create a new branch against the repository's master branch
- overwrite `uv.lock` with the version previously created by step 2
- synchronise the environment by running `uv sync`.

Run the test suite to identify failing tests. For example:

```bash
uv run --isolated --python 3.11 python pytest --ignore=tests/test_yahoo.py -v
```

Then find fixes for the failing tests by following step 5.

Finally commit the necessary changes to your *original* branch (to which previous commits were made). This will trigger the CI on the PR re-run. Return to step 7 (inspect CI results).

### 8. Fallback: raise an issue

ONLY if any test failures cannot be resolved, raise an issue that references the PR and details:
- the failing tests
- any fixes already attempted
- any suggested next steps.

---

**Network tests** — `tests/test_yahoo.py` and `tests/test_calendar_utils.py`

Tests in `tests/test_yahoo.py` require live network access to the Yahoo Finance API via
the `yahooquery` library.

Tests in `tests/test_calendar_utils.py` fetch CSV fixture files from
`raw.githubusercontent.com`, which can return HTTP 403 Forbidden when network access is
restricted.

- **Local test runs**: use the reachability check described in step 4 to decide whether to
  include or exclude tests in `test_yahoo.py` and/or `test_calendar_utils.py`.
- **CI failures in `tests/test_yahoo.py` or `tests/test_calendar_utils.py`**: the cause of
  such failures is usually a dropped, rate-limited, or blocked network connection. However,
  always verify the error type (see step 7) before concluding the failure is purely a
  network issue.

---

**Fixable warnings**
The following warnings are considered UNFIXABLE and you should not attempt to fix them.
- warnings that have their origin in a dependency's code.
- `PricesMissingWarning`.

All other warnings are considered fixable.

## Adding dependencies
```bash
# Add a dependency
uv add <package>
```
