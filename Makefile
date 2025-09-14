# Makefile for Side Project CLI
# Make sure to use tabs, not spaces for indentation

.PHONY: help clean dev build test install uninstall

# Default target
help:
	@echo "Available commands:"
	@echo "  make clean      - Clean build artifacts and cache files"
	@echo "  make dev        - Install in development mode"
	@echo "  make build      - Build binary executable"
	@echo "  make test       - Run tests"
	@echo "  make install    - Install package"
	@echo "  make uninstall  - Uninstall package"

# Clean build artifacts and cache
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Clean complete!"

# Development installation
dev:
	@echo "🔧 Installing in development mode..."
	pip install -e .
	@echo "✅ Development installation complete!"
	@echo "💡 You can now use: side-project --help"

# Build binary executable
build: clean
	@echo "🔨 Building binary executable..."
	@if [ -f build.sh ]; then \
		chmod +x build.sh && ./build.sh; \
	else \
		echo "❌ build.sh not found"; \
		exit 1; \
	fi
	@echo "✅ Binary build complete!"
	@echo "💡 Run: ./dist/side-project --help"

# Test the CLI
test:
	@echo "🧪 Testing CLI commands..."
	@echo "Testing help command:"
	side-project --help
	@echo "\nTesting version command:"
	side-project version
	@echo "\nTesting info command:"
	side-project info
	@echo "\nTesting hello command:"
	side-project hello "Makefile Test"
	@echo "✅ All tests passed!"

# Install package
install:
	@echo "📦 Installing package..."
	pip install .
	@echo "✅ Installation complete!"

# Uninstall package
uninstall:
	@echo "🗑️  Uninstalling package..."
	pip uninstall side-project -y
	@echo "✅ Uninstall complete!"

# Quick development cycle
quick: clean dev test
	@echo "🚀 Quick development cycle complete!"

# Full build and test
release: clean dev test build
	@echo "🎉 Release build complete!"
	@echo "💡 Test binary: ./dist/side-project --help"