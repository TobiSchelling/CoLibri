.PHONY: check build test lint format clean setup help

# Find cargo: prefer PATH, fallback to ~/.cargo/bin
CARGO := $(shell command -v cargo 2>/dev/null || echo "$(HOME)/.cargo/bin/cargo")

# Default target
all: build

# Fast type-check without full compilation
check:
	$(CARGO) check

# Build optimized release binary
build:
	$(CARGO) build --release
	@echo "Binary: target/release/colibri"

# Run tests
test:
	$(CARGO) test

# Run clippy linter
lint:
	$(CARGO) clippy -- -D warnings

# Format code
format:
	$(CARGO) fmt

# Clean build artifacts
clean:
	$(CARGO) clean

# Install Rust toolchain if not present
setup:
	@command -v rustup >/dev/null 2>&1 || command -v $(HOME)/.cargo/bin/rustup >/dev/null 2>&1 || { \
		echo "Installing Rust via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	}
	@$(CARGO) --version

# Check Ollama status
check-ollama:
	@curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "Ollama not running. Start with: ollama serve"

# Pull required Ollama models
pull-models:
	ollama pull nomic-embed-text

# Show help
help:
	@echo "Available targets:"
	@echo "  setup        - Install Rust toolchain via rustup"
	@echo "  check        - Type-check code (fast)"
	@echo "  build        - Build release binary"
	@echo "  test         - Run tests"
	@echo "  lint         - Run clippy linter"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  check-ollama - Check if Ollama is running"
	@echo "  pull-models  - Pull required Ollama models"
