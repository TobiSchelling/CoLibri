# Homebrew Packaging

This folder contains a **template** Homebrew formula for distributing CoLibri via a tap.

Typical layout for the tap repository:

- `Formula/colibri.rb`

Workflow:

1. Copy `packaging/homebrew/colibri.rb` into the tap repo at `Formula/colibri.rb`.
2. Update `url` + `sha256` to a released sdist tarball.
3. Run `brew update-python-resources colibri` in the tap repo to populate resource blocks.

See `docs/RELEASING.md` for the full release + tap update flow.

