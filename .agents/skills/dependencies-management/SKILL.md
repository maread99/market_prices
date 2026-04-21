---
  name: dependencies-management
  description: instructions for updating and managing project dependencies.
---
# Dependencies Management

## Update Dependencies

Instructions to update project dependencies:
1) Create a new branch
2) Run the following commands:
```bash
uv lock --upgrade  # update the dependencies lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # export @uv.lock to @requirements.txt
uv sync --inexact  # update environment to dependencies as reflected in updated @uv.lock
```
3) Run the project's tests with `pytest -v`. In the case of failing tests (which will probably be due to changes in a dependency) make the necessary revisions to the codebase in order to support the latest dependencies. As deemed necessary, facilitate these revisions by researching changes that have been made in the dependencies. ITERATE on this process until all tests are passing or until you reach a dead end.
4) Commit changes
5) Create and raise a PR
6) ONLY in the event that the tests of step 3 did not pass, raise an issue which references the PR and explains the failing tests, your attempts to fix them and any 'next step' suggestions.

## Adding dependencies
```bash
# Add a dependency
uv add <package>
```