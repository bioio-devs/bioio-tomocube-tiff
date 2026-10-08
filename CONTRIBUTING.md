# Contributing to bioio-tomocube-tiff

## Prerequisites

**Git LFS** must be installed before you clone. Test resources (cropped `.TIFF`
exports) are stored in LFS. Cloning without LFS causes those files to appear as
text pointer stubs and all fixture-dependent tests to fail.

Install Git LFS: https://git-lfs.github.com/

**just** is used as the task runner: https://just.systems/

## Setting Up a Development Environment

```bash
git clone git@github.com:bioio-devs/bioio-tomocube-tiff.git
cd bioio-tomocube-tiff
just setup-dev
```

`just setup-dev` runs `pip install -e .[lint,test]` and installs the
pre-commit hooks.

## Available Commands

| Command | Description |
|---|---|
| `just build` | Run lint then tests (mirrors CI) |
| `just test` | Run pytest with coverage |
| `just lint` | Run pre-commit on all files |
| `just benchmark` | Run read-performance benchmarks; writes `output.csv` |
| `just clean` | Remove build artefacts |
| `just install` | Install runtime + lint + test dependencies |

## Making Changes

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feat/short-description`.
3. Make your changes.
4. Validate locally: `just build`.
5. Commit and push to your fork.
6. Open a pull request against `main`.

## Test Fixtures

The TIFFs under `bioio_tomocube_tiff/tests/resources/tiff/` are real HTX
ProcessingServer exports cropped to 32 × 32 pixels with every tag and page
preserved, so they reproduce the export's quirks (the FL3D ImageJ header that
reports `slices=12` while the file holds 70 pages). They are tracked by Git LFS,
so run `git lfs pull` after cloning. To regenerate from a full-size export, crop
each page with `tifffile` and copy the `ImageDescription`, `Model`, `Software`,
`DateTime`, `Artist`, `HostComputer` and resolution tags of the first page.

Contributors cannot add fixtures through a pull request; the file must be
committed to LFS by a maintainer.

## Release Process (Maintainers Only)

```bash
just tag-for-release vX.Y.Z
just release
```

A tag beginning with `v` triggers the automated PyPI publish via CI. Version
numbers are managed by `setuptools-scm` and derived from the git tag.
