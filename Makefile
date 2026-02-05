.PHONY: install dev lint format test clean build
.PHONY: rust-check rust-build rust-test rust-lint rust-clean setup-rust

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --all-extras

# Run linter
lint:
	uv run ruff check src tests
	uv run mypy src

# Format code
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Run tests
test:
	uv run pytest tests -v

# Build sdist + wheel
build:
	uv run python -m build

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# Check Ollama status
check-ollama:
	@curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "Ollama not running. Start with: ollama serve"

# Pull required Ollama models
pull-models:
	ollama pull nomic-embed-text

# Show help
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Install with dev dependencies"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code"
	@echo "  test         - Run tests"
	@echo "  build        - Build sdist + wheel"
	@echo "  clean        - Clean build artifacts"
	@echo "  check-ollama - Check if Ollama is running"
	@echo "  pull-models  - Pull required Ollama models"
	@echo ""
	@echo "Rust targets:"
	@echo "  setup-rust   - Install Rust toolchain via rustup"
	@echo "  rust-check   - Type-check Rust code (fast)"
	@echo "  rust-build   - Build release binary"
	@echo "  rust-test    - Run Rust tests"
	@echo "  rust-lint    - Run clippy linter"
	@echo "  rust-clean   - Clean Rust build artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# Rust targets (colibri-rs/)
# ─────────────────────────────────────────────────────────────────────────────

RUST_DIR := colibri-rs
# Find cargo: prefer PATH, fallback to ~/.cargo/bin
CARGO := $(shell command -v cargo 2>/dev/null || echo "$(HOME)/.cargo/bin/cargo")

# Install Rust toolchain if not present
setup-rust:
	@command -v rustup >/dev/null 2>&1 || command -v $(HOME)/.cargo/bin/rustup >/dev/null 2>&1 || { \
		echo "Installing Rust via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	}
	@$(CARGO) --version

# Fast type-check without full compilation
rust-check:
	cd $(RUST_DIR) && $(CARGO) check

# Build optimized release binary
rust-build:
	cd $(RUST_DIR) && $(CARGO) build --release
	@echo "Binary: $(RUST_DIR)/target/release/colibri"

# Run Rust tests
rust-test:
	cd $(RUST_DIR) && $(CARGO) test

# Run clippy linter
rust-lint:
	cd $(RUST_DIR) && $(CARGO) clippy -- -D warnings

# Format Rust code
rust-format:
	cd $(RUST_DIR) && $(CARGO) fmt

# Clean Rust build artifacts
rust-clean:
	cd $(RUST_DIR) && $(CARGO) clean
