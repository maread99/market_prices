---
  name: dependencies-management
  description: instructions for updating and managing project dependencies.
---
# Dependencies Management

## Update Dependencies

Instructions to update project dependencies:

### 1. Prepare the branch

Create a new branch (following the naming convention in @AGENTS.md).

### 2. Update the lock file and environment

```bash
uv lock --upgrade  # update the lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # sync @requirements.txt with @uv.lock
uv sync --inexact  # update environment to match @uv.lock
```

### 3. Raise the PR

Raise the PR before running tests — the CI `build-test.yml` workflow triggers
automatically on PRs to `master` and covers all supported OS and Python version
combinations, providing more authoritative results than a single local run.

- **PR title**: `Update Dependencies <Mon> (auto)` where `<Mon>` is the first
  three capitalised letters of the current month (e.g.
  `Update Dependencies Apr (auto)`).
- **'dependencies' label**: Add it to the PR via
  `mcp__github__update_pull_request`.

### 4. Inspect CI results

Once the workflow completes, use `mcp__github__pull_request_read` with the
`get_check_runs` method to read check statuses for all matrix jobs.

**Interpreting results:**
- All checks green → done.
- Failures only in `tests/test_yahoo.py` → most probably a transient network
  issue (see *Network tests* note below). Request a re-run of the failing jobs;
  these do not require code changes.
- Failures in any other test file → proceed to step 5.

Note: `get_check_runs` returns pass/fail status and any annotations but not the
full log output. For detailed failure messages, run the tests locally (step 5).

### 5. Reproduce and fix failures locally

Run the full test suite locally, excluding the network tests:

```bash
pytest --ignore=tests/test_yahoo.py -v
```

Research the changelogs of any updated dependencies to understand what changed,
then revise the codebase accordingly. Iterate until all local (non-network)
tests pass.

### 6. Commit, push, and verify CI

Once local tests pass, commit the changes and push to the branch. Monitor CI
again via `get_check_runs` (step 4). If all non-network checks are green the
update is complete.

**OS- or Python-version-specific CI failures**: If a failure only appears for
a specific matrix combination (e.g. Windows / Python 3.10), reproduce it in a
matching environment — for example by using `uv python install 3.10` or a
Docker container with the target OS — then fix, commit, push, and repeat
from step 4.

### 7. Fallback: raise an issue

ONLY if failures cannot be resolved, raise an issue referencing the PR that
explains the failing tests, steps already attempted, and suggested next steps.

---

**Network tests** — `tests/test_yahoo.py`

All tests in `tests/test_yahoo.py` require live network access to the Yahoo
Finance API via `yahooquery`. No other test files have this requirement.

- **Local runs**: always pass `--ignore=tests/test_yahoo.py` unless you
  specifically intend to test live Yahoo connectivity. Failures here locally
  are expected without reliable internet access and do not indicate a code
  problem.
- **CI failures in `tests/test_yahoo.py`**: these are usually caused by a
  dropped or rate-limited connection during the test run, not a code change.
  Re-running the affected matrix jobs is sufficient — no code change is needed.

## Adding dependencies
```bash
# Add a dependency
uv add <package>
```