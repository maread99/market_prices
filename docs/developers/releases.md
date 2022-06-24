# Releases

## Versioning
`market_prices` follows [Semantic Versioning](https://semver.org/). Releases should tagged as "v\<MAJOR>.\<MINOR>.\<PATCH>", for example:
* "v0.8.3"
* *v1.12.3"
* *v2.4.12"

setuptools-scm is used to version releases during the build process.

## Draft release notes
Draft release notes for the next release should have been prepared by the [`draft-release-notes.yml` workflow](https://github.com/maread99/market_prices/blob/master/.github/workflows/draft-release-notes.yml). This uses the [Release Drafter action](https://github.com/marketplace/actions/release-drafter).

The latest draft should be at the top of the [releases page](https://github.com/maread99/market_prices/releases).

## Release workflow

### Cut a release
On publishing a release via GitHub the [`release.yml` workflow](https://github.com/maread99/market_prices/blob/master/.github/workflows/release.yml) will upload the distrubtion files to PyPI.

At the GitHub [releases page](https://github.com/maread99/market_prices/releases):
* The draft release notes should already be at the top of the page. Click the pen icon to edit the draft.
* Tag the release. The draft will have suggested a tag for the release. If this tag doesn't reflect the intended version then either select the last commit's tag (if it was added) or create a new tag that reflects the version string (any new tag will be attached to last commit).
* Make sure target is selected as 'refs/head/master'.
* Revise the draft release notes as requried.
* If the release includes new features, select the checkbox for 'Create a discussion for this release' (otherwise leave unchecked).
* Click the 'Publish Release' button.