# Poetry Migration Guide

This project has been migrated from `requirements.txt` to [Poetry](https://python-poetry.org/) for dependency management.

## What Changed

- ✅ `pyproject.toml` - New Poetry configuration file with all dependencies
- ✅ `poetry.lock` - Lock file for reproducible builds (should be committed)
- ✅ Setup scripts updated to use Poetry
- ⚠️ `requirements.txt` - Kept for backward compatibility (can be regenerated)

## Quick Start

### Install Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### Install Dependencies

```bash
# Install all dependencies (production + dev)
poetry install

# Install only production dependencies
poetry install --no-dev

# Install with optional groups
poetry install --with visualization,optional
```

### Using Poetry

```bash
# Activate the virtual environment
poetry shell

# Run commands within Poetry environment
poetry run python scripts/train_task_classifier.py

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update all dependencies
poetry update

# Update a specific package
poetry update package-name
```

## Dependency Groups

Dependencies are organized into groups in `pyproject.toml`:

- **Main dependencies** - Required for production (installed by default)
- **dev** - Development tools (jupyter, pytest, black, mypy, etc.)
- **optional** - Optional packages (presidio-analyzer, presidio-anonymizer)
- **visualization** - Visualization tools (matplotlib, seaborn)

Install specific groups:
```bash
poetry install --with dev,visualization
```

## Backward Compatibility

If you need a `requirements.txt` file (e.g., for Docker builds), you can generate it:

```bash
# Generate requirements.txt from Poetry
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Generate with dev dependencies
poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev

# Generate with all optional groups
poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev,optional,visualization
```

## Migration from requirements.txt

If you were using `requirements.txt` before:

1. **Remove old virtual environment** (if using venv):
   ```bash
   rm -rf venv/
   ```

2. **Install Poetry** (see above)

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate Poetry shell**:
   ```bash
   poetry shell
   ```

5. **Verify installation**:
   ```bash
   poetry run python -c "import torch; print('PyTorch:', torch.__version__)"
   ```

## Benefits of Poetry

1. **Better dependency resolution** - Automatically resolves version conflicts
2. **Lock files** - `poetry.lock` ensures reproducible builds across environments
3. **Dependency groups** - Organize dependencies by purpose (dev, optional, etc.)
4. **Single tool** - Manage dependencies, virtual environments, and packaging
5. **Faster installs** - Better caching and parallel installation

## Troubleshooting

### Poetry not found

Make sure Poetry is in your PATH:
```bash
# Add Poetry to PATH (Linux/macOS)
export PATH="$HOME/.local/bin:$PATH"

# Or add to ~/.bashrc or ~/.zshrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Lock file conflicts

If `poetry.lock` has conflicts:
```bash
# Update lock file
poetry lock --no-update

# Or update and resolve conflicts
poetry lock
```

### Clear Poetry cache

```bash
poetry cache clear pypi --all
```

## Next Steps

- Run `poetry install` to set up your environment
- Use `poetry shell` to activate the virtual environment
- Continue using the project as before - all scripts work the same way

