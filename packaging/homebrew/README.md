# Homebrew Distribution

## User Installation

```bash
brew tap TobiSchelling/tap
brew install colibri
```

## Maintainer: Updating the Formula

After creating a new release:

1. Tag and push:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

2. Wait for the release workflow to complete and note the SHA256 from the workflow output.

3. Update your tap repository (`homebrew-tap`):
   ```bash
   cd ~/path/to/homebrew-tap
   # Copy the formula
   cp /path/to/CoLibri/packaging/homebrew/colibri.rb Formula/colibri.rb
   # Update the SHA256 in the formula
   # Commit and push
   ```

Alternatively, use `brew bump-formula-pr` if the tap is set up for it.

## Formula Location

The formula in this repo (`packaging/homebrew/colibri.rb`) is the source of truth. Copy it to your tap's `Formula/` directory after updating the version and SHA256.

## Tap Repository Structure

Your tap repository (`TobiSchelling/homebrew-tap`) should have:

```
homebrew-tap/
└── Formula/
    └── colibri.rb
```
