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

```bash
uv lock --upgrade  # update the lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # sync @requirements.txt with @uv.lock
uv sync --inexact  # update environment to match @uv.lock
```

### 3. Update GitHub Actions versions

For each `uses: <owner>/<repo>@<version>` entry across all files in `.github/workflows/`, use `mcp__github__get_latest_release` to retrieve the latest release for that action's repository. Update the version pin to the latest release, preserving the pinning style already used (e.g. if the current pin is a major-version tag such as `@v4`, update to the new major-version tag; if it is a specific version such as `@v3.0.1`, update to the full latest version string).

### 4. Raise the PR

Commit the changed files and raise a PR.

- **PR title**: Title the PR as `Update Dependencies <MM> <DD> (auto)` where:
  - `<MM>` should be replaced with the first three letters of the current month, the first of which should be capitalized.
  - `<DD>` should be replaced with the  current day of the month as represented by two digits.
  Example title: `Update Dependencies Apr 07 (auto)`
- **label**: Add the 'dependencies' label to the PR via `mcp__github__update_pull_request`.

Do NOT run any tests prior to raising the PR - the CI `build-test.yml` workflow will automatically trigger on the PR being raised (against the `master` branch) and this will run the full test suite on all supported OS and Python version combinations.

### 5. Inspect CI results

Once the workflow completes, use `mcp__github__pull_request_read` with the `get_check_runs` method to read check statuses for all matrix jobs.

**Interpreting results:**
- All checks green → done, no further action is required.
- Failures only in `tests/test_yahoo.py` → probably a transient network issue (see *Network tests* section below). Add a comment to the PR identifying the failure as a probable network issue and suggest that the owner re-run the failing jobs. No further action is required.
- Failures in any other test file → proceed to step 6.

### 6. Fix failures locally

The cause of failing tests will most likely be in changes to the dependencies. MAKE REVISIONS to the code base to get all non-network tests passing so that the project supports the latest versions of its dependencies. (See *Network tests* section below for notes on network tests.)

`get_check_runs` returns pass/fail status and any annotations. Use this information to aid in the identification of the failing tests and causes. If this information in insufficient then get a full log of the failing tests by running the full test suite locally (excluding network tests):

```bash
pytest --ignore=tests/test_yahoo.py -v
```

Consider researching the changelogs of any updated dependencies (to cover all changes since the previously locked version) to try and ascertain the cause of the failing tests.

Iterate on this process until all non-network tests are passing locally.

### 7. Commit, push, and verify CI

Once local tests pass, commit the changes and push to the branch. Monitor CI again via `get_check_runs` (step 5). If all non-network checks are green then the PR is complete and no further action is required.

**OS- or Python-version-specific CI failures**: If a failure only appears for a specific matrix combination (e.g. Windows / Python 3.10), reproduce it locally in a matching environment — for example by using `uv python install 3.10` or a Docker container with the target OS — then fix, commit, push, and repeat from step 5.

### 8. Fallback: raise an issue

ONLY if failures cannot be resolved, raise an issue that references the PR and details:
- the failing tests
- any fixes already attempted
- any suggested next steps.

---

**Network tests** — `tests/test_yahoo.py`

All tests in `tests/test_yahoo.py` require live network access to the Yahoo Finance API via the `yahooquery` library . No other test files have this requirement.

- **Local test runs**: always pass `--ignore=tests/test_yahoo.py` unless you specifically intend to test live Yahoo connectivity. It's expected that these tests will fail whenever a local internet connection is not available.
- **CI failures in `tests/test_yahoo.py`**: the cause of such failures is usually a dropped or rate-limited connection during the test run.

## Adding dependencies
```bash
# Add a dependency
uv add <package>
```