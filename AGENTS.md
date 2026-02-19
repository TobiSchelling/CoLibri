# Repository Guidelines

## Project Structure & Module Organization
`colibri` is a Rust CLI/MCP application. Core source is in `src/`:
- `src/main.rs`: CLI entrypoint.
- `src/cli/`: command handlers (`index`, `search`, `serve`, `doctor`, `import`, TUI config editor).
- `src/sources/`: content source abstraction and markdown source implementation.
- Top-level modules (`config.rs`, `embedding.rs`, `indexer.rs`, `query.rs`, `manifest.rs`, `mcp.rs`, `error.rs`) hold shared domain logic.

Automation and packaging:
- `.github/workflows/`: CI and release pipelines.
- `packaging/homebrew/`: Homebrew formula and tap docs.
- `Makefile`: standard dev commands.

## Build, Test, and Development Commands
- `make check`: fast type-check via `cargo check`.
- `make build`: optimized binary in `target/release/colibri`.
- `make test`: run unit tests.
- `make lint`: `cargo clippy -- -D warnings`.
- `make format`: `cargo fmt`.

Useful direct commands:
- `cargo test test_navigation`: run one test by name.
- `cargo fmt --check && cargo clippy -- -D warnings && cargo test`: local CI parity.

## Coding Style & Naming Conventions
- Rust 2021, stable toolchain (`rust-toolchain.toml`).
- Formatting is enforced by `rustfmt`; lints are enforced by Clippy with warnings denied.
- Naming: `snake_case` for modules/functions/files, `PascalCase` for structs/enums/traits, `SCREAMING_SNAKE_CASE` for constants.
- Keep command-specific code in `src/cli/*`; move reusable logic into shared modules.

## Testing Guidelines
- Prefer colocated unit tests with `#[cfg(test)]` near implementation (current pattern).
- Use descriptive `test_*` names that describe behavior.
- Cover CLI-facing behavior and indexing/search logic changes.
- For retrieval/indexing changes, smoke test manually:
  - `cargo run -- index`
  - `cargo run -- search "example query"`

## Commit & Pull Request Guidelines
- Follow Conventional Commits as used in history: `feat:`, `fix:`, `refactor:`, `docs:`, `style:`, `chore:` (scopes optional, e.g. `feat(import): ...`; use `!` for breaking changes).
- Keep commits focused and atomic.
- PRs should include:
  - What changed and why.
  - User-visible CLI impact (example commands/output when relevant).
  - Any config/schema/release follow-up (e.g., `Cargo.toml`, `Cargo.lock`, `packaging/homebrew/colibri.rb`).
  - Linked issue(s) when available.
