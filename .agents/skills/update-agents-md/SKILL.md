---
  name: update-agents-md
  description: Instructions for updating @AGENT.md file, including: updating the 'Repository Layout' section following the adding or removal of files or folders; updating the 'Technology Stack' section on changes to the pinned python version as defined in @.python-version.
---
# Update agents.md file

## Repository Layout
Update the 'Repository Layout' section to reflect the current layout. Remove all files and folders that are no longer present and add all new files and folders with the following IMPORTANT EXCEPTIONS:
- exclude all files and folders ignored by @.gitignore
- do not include the following files or folders:
    - @.gitignore
    - @.git
    - all files named `__init__.py`
- do not include files and/or folders under the following directories (but do include the folders listed here):
    - @tests/resources
    - @docs/tutorials/resources
List all folders before any files for any given hierarchal level.
Include a minimalist comment offering a top-line explanation for the following files and directories:
- all files and directories under @src
- all other files and directories ONLY if their purpose is not reasonably evident from their name.
The comment should:
- be placed alongside the file/directory name
- not cause the length of the full line (including folder/directory name and all whitespace) to exceed 100 characters.

## Pinned Python version
Update, if necessary, the 'python' field of the 'Technology Stack' table so that it indicates the pinned python version as defined in @.python-version.
