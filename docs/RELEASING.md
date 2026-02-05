# Releasing & Homebrew Tap

This doc is for maintainers who publish CoLibri releases and keep the Homebrew tap up to date.

## Overview

Recommended distribution path:

1. Tag a CoLibri release in the main repo (this repo).
2. Publish a source artifact (sdist tarball) for that tag.
3. Update the Homebrew tap formula to point at that artifact + SHA-256.
4. Regenerate Homebrew Python resources and push the tap update.

CoLibri’s CLI entry point is defined in `pyproject.toml`:

- `[project.scripts] colibri = "colibri.cli:cli"`

## Create a Release Artifact

1. Bump version in `pyproject.toml`.
2. Create a git tag `vX.Y.Z` and push it.
3. Build artifacts (ensure dev deps are installed first):

```bash
make dev
make build
```

4. Publish the sdist tarball (`dist/colibri-X.Y.Z.tar.gz`) as a release asset and note its SHA-256:

```bash
shasum -a 256 dist/colibri-X.Y.Z.tar.gz
```

## Update the Homebrew Tap

The tap repository is typically a separate repo named `homebrew-colibri` and contains:

- `Formula/colibri.rb`

Steps:

1. Update `url` and `sha256` in `Formula/colibri.rb` to the new release artifact.
2. Regenerate dependency resources:

```bash
brew update-python-resources colibri
```

3. Run `brew audit --strict colibri` (optional but recommended).
4. Push the tap repo changes.

## Files

- Template formula: `packaging/homebrew/colibri.rb`
- Tap formula path (in the tap repo): `Formula/colibri.rb`
