# Releases

## Versioning
`market_prices` follows [Semantic Versioning](https://semver.org/). Releases should versioned as "v\<MAJOR>.\<MINOR>.\<PATCH>", for example:
* "v0.8.3"
* *v1.12.3"
* *v2.4.12"

## Draft release notes
Draft release notes for the next release should have been prepared by the [`draft-release-notes.yml` workflow](https://github.com/maread99/market_prices/blob/master/.github/workflows/draft-release-notes.yml). This uses the [Release Drafter action](https://github.com/marketplace/actions/release-drafter).

The latest draft should be at the top of the [releases page](https://github.com/maread99/market_prices/releases).

## Release workflow

### Last pre-release commit
* The last pre-release commit (or a prior commit) should include updating `market_prices.__init__.__version__` with the new version string.
* The last pre-release commit can optionally be tagged with the new version string.

### Cut a release
On publishing a release via GitHub the [`release.yml` workflow](https://github.com/maread99/market_prices/blob/master/.github/workflows/release.yml) will upload the distrubtion files to PyPI.

At the GitHub [releases page](https://github.com/maread99/market_prices/releases):
* The draft release notes should already be at the top of the page. Click the 'Draft a new release' button.
* Tag the release. Either choose the last commit's tag (if it was added) or create a new tag with the version string (this new tag will be attached to last commit).
* Make sure target is selected as 'Master'.
* Revise the draft release notes as requried.
* If the release includes new features, select the checkbox for 'Create a discussion for this release' (otherwise leave unchecked).
* Click the 'Publish Release' button.