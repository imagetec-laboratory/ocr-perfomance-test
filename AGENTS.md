# Python CLI Project Development Guide

## Dev Environment Setup

### Prerequisites
- Python >= 3.12.9
- uv package manager (recommended) or pip
- Git for version control

### Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd side-project

# Install in development mode
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

### Development Workflow
- Use `side-project` command to test your CLI during development
- Edit `side_project/config.py` to customize project information
- Add new commands in `side_project/commands.py`
- Register commands in `side_project/main.py`
- Use utilities from `side_project/utils.py` for common functions

### Key Files to Understand
- `pyproject.toml` - Project metadata and dependencies
- `side_project/main.py` - Main CLI app and command registration
- `side_project/config.py` - Project configuration and constants  
- `side_project/commands.py` - CLI command implementations
- `side_project/utils.py` - Shared utility functions

## Testing Instructions

### Manual Testing
```bash
# Test the CLI commands
side-project hello "Developer"
side-project info
side-project version

# Test with different options
side-project hello "World" --greeting "Hi" --count 3
```

### Build Testing
```bash
# Test the build process
./build.sh  # On Unix/macOS
# or
build.bat   # On Windows

# Test the executable
./dist/side-project hello "Test"
```

## Development Guidelines

## Development Commands
- `uv run python -m side_project.main --help` - Run the CLI in development mode
- `uv pip install -e .` - Install the package in editable mode
- `uv pip freeze > requirements-dev.txt` - Update development dependencies

### Adding New Commands
1. Define your command function in `side_project/commands.py`
2. Use Typer decorators for arguments and options
3. Register the command in `side_project/main.py`
4. Test the command thoroughly

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints throughout the codebase
- Add docstrings to all functions and classes
- Use Rich console for beautiful CLI output
- Handle imports for both development and PyInstaller builds

### Configuration Changes
- Update `PROJECT_INFO` in `config.py` for project metadata
- Modify `DEFAULT_VALUES` for command defaults

## Build and Distribution

### Creating Executables
```bash
# Build standalone executable
./build.sh  # Creates dist/side-project

# Test the executable
./dist/side-project --help
```

### Package Structure
- Source code: `side_project/`
- Build output: `build/` and `dist/`
- Configuration: `pyproject.toml`
- Build scripts: `build.sh`, `build.bat`, `create_package.sh`

## PR Instructions

### Before Committing
- Test all CLI commands manually
- Verify the build process works: `./build.sh`
- Check that the executable runs correctly
- Update version in `pyproject.toml` if needed
- Update documentation if adding new features

### PR Title Format
- For new features: `feat: add <feature-description>`
- For bug fixes: `fix: resolve <issue-description>`
- For documentation: `docs: update <what-was-updated>`
- For refactoring: `refactor: improve <what-was-improved>`