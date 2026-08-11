# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.5] - 2026-08-10

This **minor release** includes 23 commits.


### Bug Fixes
- Pin exact uv version in setup-uv steps (template v0.29.6)  ([#44](https://github.com/stateful-y/sklearn-optuna/pull/44)) by @gtauzin
- Pin ossf/scorecard-action to the existing v2.4.4 tag by @gtauzin
- Fix a shell injection in the release publish job (template v0.40.1)  ([#65](https://github.com/stateful-y/sklearn-optuna/pull/65)) by @gtauzin
- Stop the nightly coverage upload from silently uploading the wrong report  ([#69](https://github.com/stateful-y/sklearn-optuna/pull/69)) by @gtauzin

### Documentation
- Simplify landing page and update acknowledgements  ([#26](https://github.com/stateful-y/sklearn-optuna/pull/26)) by @gtauzin

### Refactoring
- Move build output to .artifacts/ and CODEOWNERS to .github/  ([#68](https://github.com/stateful-y/sklearn-optuna/pull/68)) by @gtauzin

### Miscellaneous Tasks
- Fix See Also links and root export 404s in the API docs (template v0.26.1)  ([#31](https://github.com/stateful-y/sklearn-optuna/pull/31)) by @gtauzin
- Run pre-commit hooks with prek and filter changelog entries (template v0.27.0)  ([#34](https://github.com/stateful-y/sklearn-optuna/pull/34)) by @gtauzin
- Exempt the docs build scripts from ruff's lint rules (template v0.27.3)  ([#36](https://github.com/stateful-y/sklearn-optuna/pull/36)) by @gtauzin
- Render API page structure from mkdocstrings templates (template v0.28.1)  ([#37](https://github.com/stateful-y/sklearn-optuna/pull/37)) by @gtauzin
- Discover the API surface with Griffe (template v0.28.3)  ([#40](https://github.com/stateful-y/sklearn-optuna/pull/40)) by @gtauzin
- Replace stale git hooks by installing with prek install -f (template v0.28.4)  ([#41](https://github.com/stateful-y/sklearn-optuna/pull/41)) by @gtauzin
- Make the generated docs build engine-independent (template v0.29.3)  ([#42](https://github.com/stateful-y/sklearn-optuna/pull/42)) by @gtauzin
- Migrate docs engine to Zensical (template v0.30.1)  ([#45](https://github.com/stateful-y/sklearn-optuna/pull/45)) by @gtauzin
- Replace Dependabot with Renovate for dependency updates (template v0.31.1)  ([#47](https://github.com/stateful-y/sklearn-optuna/pull/47)) by @gtauzin
- Add pre-push gates and a single CI roll-up check (template v0.32.1)  ([#49](https://github.com/stateful-y/sklearn-optuna/pull/49)) by @gtauzin
- Restrict workflow permissions and add secret scanning (template v0.35.0)  ([#50](https://github.com/stateful-y/sklearn-optuna/pull/50)) by @gtauzin
- Switch Codecov to OIDC and pin the Scorecard action (template v0.36.0)  ([#51](https://github.com/stateful-y/sklearn-optuna/pull/51)) by @gtauzin
- Document signing release tags with gitsign (template v0.37.0)  ([#52](https://github.com/stateful-y/sklearn-optuna/pull/52)) by @gtauzin
- Add a CLAUDE.md project-instructions file for AI assistants (template v0.38.0)  ([#53](https://github.com/stateful-y/sklearn-optuna/pull/53)) by @gtauzin
- Fix three release-pipeline defects (template v0.39.0)  ([#55](https://github.com/stateful-y/sklearn-optuna/pull/55)) by @gtauzin
- Let Renovate see the SBOM tool's version pin (template v0.39.1)  ([#56](https://github.com/stateful-y/sklearn-optuna/pull/56)) by @gtauzin
- Add a nightly job that exercises the release path (template v0.40.0)  ([#57](https://github.com/stateful-y/sklearn-optuna/pull/57)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.4] - 2026-04-19

This **minor release** includes 2 commits.


### Bug Fixes
- Checkout tag ref in build job and fix pre-release version regex by @gtauzin

### Documentation
- Update from copier template v0.18.0 and restructure docs per Diataxis  ([#21](https://github.com/stateful-y/sklearn-optuna/pull/21)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.3] - 2026-03-01

This **minor release** includes 2 commits.


### Features
- Replace WASM export with PEP 723 notebooks and marimo.app playground links  ([#14](https://github.com/stateful-y/sklearn-optuna/pull/14)) by @gtauzin

### Miscellaneous Tasks
- Update copier template to v0.15.0  ([#15](https://github.com/stateful-y/sklearn-optuna/pull/15)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.2] - 2026-02-23

This **minor release** includes 5 commits.


### Documentation
- Align README structure with sklearn-wrap  ([#5](https://github.com/stateful-y/sklearn-optuna/pull/5)) by @gtauzin
- Update notebook examples to match contributing guidelines  ([#6](https://github.com/stateful-y/sklearn-optuna/pull/6)) by @gtauzin
- Reformulate docs text  ([#8](https://github.com/stateful-y/sklearn-optuna/pull/8)) by @gtauzin

### Miscellaneous Tasks
- Update froom copier template 0.13.2  ([#7](https://github.com/stateful-y/sklearn-optuna/pull/7)) by @gtauzin
- Update from copier template 0.13.4  ([#11](https://github.com/stateful-y/sklearn-optuna/pull/11)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.1] - 2026-02-10

This **minor release** includes 1 commit.

- Initial commit

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [Unreleased]

### Added
- Initial project setup
