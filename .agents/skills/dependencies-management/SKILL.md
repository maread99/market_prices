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

Update `uv`:
```bash
uv self update --token $GITHUB_TOKEN
```

**IMPORTANT: updating `uv` MUST NOT be skipped.** If the above command fails (e.g. due to a GitHub API rate limit), try the following fallbacks in order until `uv --version` confirms the latest version is active on PATH:

**Fallback 1 — official install script** (preferred; replaces PATH binary directly):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # refresh current shell session
uv --version                 # verify PATH binary is now updated
```

**Fallback 2 — pip** (use only if `curl` is unavailable):

```bash
pip install --user --upgrade uv
uv --version   # check if PATH binary was updated
```

Once `uv` is up to date, run:

```bash
uv lock --upgrade  # update the lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # sync @requirements.txt with @uv.lock
uv sync --inexact  # update environment to match @uv.lock
```

### 3. Update GitHub Actions versions

For each `uses: <owner>/<repo>@<version>` entry across all files in `.github/workflows/`, list the repository's tags with the newest version first:

```bash
git ls-remote --tags --refs --sort=-v:refname https://github.com/<owner>/<repo>
```

In each case update the version pin to the tag of any more recent stable release (ignore the tag of any pre-release, for example `v8.0.0-rc.1`). **Preserve the existing pinning style**, so if the current pin is a specific version such as `@v3.0.1`, update to the full latest version string, whilst if it's a major-version tag such as `@v4`, update to any new major-version tag (if the project no longer publishes a major-version tag then switch to the full semver). Note that the listing places a major-version tag (for example `v4`) after every full version tag of that major version.

### 4. Update pre-commit hook versions

Update the `rev:` pin of each repository in `.pre-commit-config.yaml` to the repository's latest tag:

```bash
uv run pre-commit autoupdate
```

Then run the hooks against all files:

```bash
uv run pre-commit run --all-files
```

Keep any changes that the hooks make. If a hook reports a violation that the hook cannot fix, fix the violation and run the hooks again.

### 5. Test

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
- **otherwise** use the pytest --ignore option to exclude the test module(s) correponding with the unreachable service(s):
  - if yahoo unreachable then '--ignore=tests/test_yahoo.py'
  - if raw.github.com unreachable then '--ignore=tests/test_calendar_utils.py'
  For example, if both yahoo and raw.github.com are unreachable then the command will be:
  ```bash
  uv run pytest --ignore=tests/test_yahoo.py --ignore=tests/test_calendar_utils.py -v
  ```

**Interpreting the local test results:**
- All tests pass and no raised warning is fixable (see *Fixable warnings* section below) → go to step 7 to raise PR.
- All failing tests are in `tests/test_yahoo.py` and/or `tests/test_calendar_utils.py` and no raised warning is fixable → test failures probably due to a transient network issue (see *Network tests* section below), go to step 7 to raise PR. The CI checks will verify the network tests (see step 9).
- Failure of any other test or a fixable warning raised → proceed to step 6 to fix.

### 6. Fix
Any failing tests and fixable warnings will likely have their origin in changes to the dependencies. To provide support for the latest dependencies MAKE REVISIONS to the code base to fix:
- code causing tests to fail.
- all fixable warnings (see *Fixable warnings* section below).

As a general RULE, **change the package code to get the tests passing, not the test code!** You may make changes to the test code only with good reason and only when this does not impair the test's efficacy.

One such reason is a test failure that arises because a new version of `exchange_calendars` corrects a calendar that stored test resources were created with. Fix such a failure by reverting the correction in the class of `tests/utils.py` that reproduces the calendar as at an earlier version (for example `CMESAsAt4132`), and list the change in that class's docstring. If `tests/utils.py` has no such class for the calendar, add one that follows the pattern of the existing classes. Do not edit the stored resources to agree with the corrected calendar.

To facilitate identifying the cause of test failures consider researching the changelogs of updated dependencies for versions released since the previously locked version.

Iterate on this process until:
- all tests are passing with the exception of any requiring unreachable services.
- all fixable warnings have been fixed.

IMPORTANT: in this step you should not run the full test suite, rather validate fixes by re-running only the previously failing tests. Example to run a specific test:

```bash
pytest tests/test_module.py::test_name
```

### 7. Raise PR

Once local tests pass, commit all changes to the branch and raise a PR.

- **PR title**: Title the PR as `Update Dependencies <MM> <DD> (auto)` where:
  - `<MM>` should be replaced with the first three letters of the current month, the first of which should be capitalized.
  - `<DD>` should be replaced with the current day of the month as represented by two digits.
  Example title: `Update Dependencies Apr 07 (auto)`
- **label**: Add the 'dependencies' label to the PR.
- otherwise comply with the package's 'create-pr' skill.

### 8. Subscribe to PR activity

Raising the PR will trigger GitHub CI workflows. Subscribe to the raised PR's activity.

Proceed to step 9 when you receive events indicating that all triggered workflows have completed. A push to the PR cancels any workflow run in progress for an earlier push. An event for a cancelled workflow run of an earlier push does not indicate completion. Continue to wait for the workflow runs of the latest push.

### 9. Inspect CI results

The CI workflow will have run the full test suite against a matrix of OS and Python versions. Use `mcp__github__pull_request_read` with the `get_check_runs` method to read check statuses for all matrix jobs.

If any check is red, read the logs of the failed jobs to identify each failing test and the cause of its failure. To read the logs, call `mcp__github__get_job_logs` with the `run_id` of the failed workflow run, `failed_only: true` and `return_content: true`. If a returned log does not reach back to the test failures, call again with a larger `tail_lines`.

The CI checks are the arbiter of the network tests (see *Network tests* section below), including any network tests that the local test run excluded or that failed locally. A network test that passes in CI needs no fix.

**Interpreting CI results:**
- All checks green → proceed to step 12.
- Every test failure is a network test failure (see *Network tests* section below) → add a comment to the PR that lists the failing tests, identifies the failures as possible network issues and asks the owner to re-run the failed jobs. Suspend the session pending further prompting.
- Any other test failure → proceed to step 10.

### 10. Fix tests for specific OS/Python configuration

Use the information read from `get_check_runs` to identify any OS/python version configurations for which the test suite has failed.

If the tests failed on specific matrix combinations (e.g. Windows / Python 3.10) then simulate a local matching environment.
- If necessary use a Docker container with the target OS. (To specify different Python versions you will be able to run commands with `uv run` and pass the --python option.)
- create a new branch against the repository's master branch.
- update the `uv` package with `uv self update`.
- overwrite `uv.lock` with the version previously created by step 2.
- synchronise the environment by running `uv sync`.

Run the test suite to identify failing tests. For example:

```bash
uv run --isolated --python 3.11 pytest --ignore=tests/test_yahoo.py -v
```

If the local test environment cannot reach a service that a network test requires, the network test cannot be verified locally (see *Network tests* section below). In that case, find the cause of the network test's failure from the CI logs read in step 9.

Then find fixes for the failing tests by following step 6.

Finally commit the necessary changes to your *original* branch (to which previous commits were made). This will trigger the CI on the PR re-run. Return to step 9 (inspect CI results).

### 11. Fallback: raise an issue

ONLY if any test failures cannot be resolved, raise an issue that references the PR and details:
- the failing tests
- any fixes already attempted
- any suggested next steps.

### 12. Tidy up

Perform the following 'tidy up' actions:
- Call `mcp__github__unsubscribe_pr_activity` to unsubscribe from the PR activity.
- Review and if necessary update the PR body to reflect the final circumstances.

---

**Network tests** — `tests/test_yahoo.py` and `tests/test_calendar_utils.py`

Tests in `tests/test_yahoo.py` require live network access to the Yahoo Finance API via the `yahooquery` library.

Tests in `tests/test_calendar_utils.py` fetch CSV fixture files from `raw.githubusercontent.com`, which can return HTTP 403 Forbidden when network access is restricted.

A failure of a test in either module is a *network test failure* if the failure's output shows a network cause, for example an HTTP error (such as 403 Forbidden or 429 Too Many Requests), a connection error, a timeout or prices missing from the data that the Yahoo Finance API returned. A failure in either module that has another cause (for example an error raised by the package code) is not a network test failure.

- **Local test runs**: use the reachability check described in step 5 to decide whether to include or exclude tests in `test_yahoo.py` and/or `test_calendar_utils.py`.
- **CI test runs**: the CI checks are the arbiter of the network tests (see step 9). The cause of a network test failure in CI may be a dropped, rate-limited, or blocked network connection, although this cannot be assumed.

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
