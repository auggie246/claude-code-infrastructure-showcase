# uv Project Management

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Creating Projects](#creating-projects)
- [Dependency Management](#dependency-management)
- [Running Code](#running-code)
- [pyproject.toml Configuration](#pyprojecttoml-configuration)
- [Workspace Management](#workspace-management)
- [Common Workflows](#common-workflows)

---

## Overview

**uv** is an extremely fast Python package and project manager written in Rust. It replaces pip, pip-tools, virtualenv, and more with a single, unified tool.

### Why uv?

- **10-100x faster** than pip
- **Unified tooling**: Package management, virtual environments, Python versions
- **Reproducible builds**: Lock files by default
- **Modern standards**: Full PEP compliance
- **Drop-in replacement**: Compatible with existing workflows

---

## Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip (if needed)
pip install uv

# Verify installation
uv --version
```

---

## Creating Projects

### New Project

```bash
# Create new project
uv init my-service
cd my-service

# Create with specific Python version
uv init my-service --python 3.13

# Create library (vs application)
uv init my-library --lib
```

### Project Structure Created

```
my-service/
├── pyproject.toml    # Project configuration
├── README.md         # Documentation
├── .python-version   # Python version pin
└── src/
    └── my_service/
        └── __init__.py
```

### Initialize Existing Directory

```bash
cd existing-project
uv init
```

---

## Dependency Management

### Adding Dependencies

```bash
# Add production dependency
uv add fastapi

# Add multiple dependencies
uv add fastapi uvicorn sqlalchemy pydantic-settings

# Add with version constraint
uv add "fastapi>=0.110.0"
uv add "sqlalchemy>=2.0,<3.0"

# Add from git
uv add git+https://github.com/org/repo.git

# Add optional extras
uv add "httpx[http2]"
```

### Adding Development Dependencies

```bash
# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov

# Add dev tools
uv add --dev ruff mypy pre-commit

# Add typing stubs
uv add --dev types-redis types-requests
```

### Removing Dependencies

```bash
# Remove dependency
uv remove httpx

# Remove dev dependency
uv remove --dev pytest-xdist
```

### Syncing Environment

```bash
# Sync all dependencies (creates venv if needed)
uv sync

# Sync including dev dependencies (default)
uv sync --all-extras

# Sync without dev dependencies
uv sync --no-dev

# Force reinstall
uv sync --reinstall
```

### Lock File

```bash
# Generate/update lock file
uv lock

# Update specific package
uv lock --upgrade-package fastapi

# Update all packages
uv lock --upgrade
```

---

## Running Code

### Running Scripts

```bash
# Run Python file
uv run python src/my_service/main.py

# Run module
uv run python -m my_service

# Run with uvicorn
uv run uvicorn src.my_service.main:app --reload

# Run any command in the project environment
uv run pytest
uv run ruff check .
uv run mypy src/
```

### Running Project Scripts

Define scripts in `pyproject.toml`:

```toml
[project.scripts]
my-service = "my_service.main:main"

[tool.uv.scripts]
dev = "uvicorn src.my_service.main:app --reload"
test = "pytest tests/"
lint = "ruff check src/"
format = "ruff format src/"
typecheck = "mypy src/"
```

Then run:

```bash
# Run defined script
uv run dev
uv run test
uv run lint
```

---

## pyproject.toml Configuration

### Complete Example

```toml
[project]
name = "my-service"
version = "0.1.0"
description = "My FastAPI microservice"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["fastapi", "api", "microservice"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Framework :: FastAPI",
    "Programming Language :: Python :: 3.13",
]

dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.30.0",
    "alembic>=1.14.0",
    "sentry-sdk[fastapi]>=2.0.0",
    "httpx>=0.28.0",
    "redis>=5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
    "httpx>=0.28.0",  # For TestClient
]

[project.scripts]
my-service = "my_service.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[tool.uv.scripts]
dev = "uvicorn src.my_service.main:app --reload --host 0.0.0.0 --port 8000"
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src/my_service --cov-report=html"
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "mypy src/"
migrate = "alembic upgrade head"
makemigrations = "alembic revision --autogenerate -m"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "-v --tb=short"

[tool.ruff]
target-version = "py313"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.lint.isort]
known-first-party = ["my_service"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_ignores = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

---

## Workspace Management

### Monorepo Setup

For multiple services in one repo:

```toml
# Root pyproject.toml
[tool.uv.workspace]
members = ["services/*", "packages/*"]
```

```
monorepo/
├── pyproject.toml          # Workspace root
├── uv.lock                 # Shared lock file
├── services/
│   ├── api-gateway/
│   │   └── pyproject.toml
│   ├── user-service/
│   │   └── pyproject.toml
│   └── notification-service/
│       └── pyproject.toml
└── packages/
    └── shared-utils/
        └── pyproject.toml
```

### Workspace Commands

```bash
# Sync all workspace members
uv sync

# Run command in specific member
uv run --package user-service pytest

# Add dependency to specific member
uv add --package user-service redis
```

---

## Common Workflows

### Initial Project Setup

```bash
# 1. Create project
uv init my-service --python 3.13
cd my-service

# 2. Add core dependencies
uv add fastapi "uvicorn[standard]" sqlalchemy pydantic-settings
uv add sentry-sdk httpx

# 3. Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy

# 4. Sync environment
uv sync

# 5. Verify setup
uv run python -c "import fastapi; print(fastapi.__version__)"
```

### Daily Development

```bash
# Start dev server
uv run dev

# Run tests
uv run test

# Lint and format
uv run lint
uv run format

# Type check
uv run typecheck
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run linting
        run: uv run ruff check src/

      - name: Run type checking
        run: uv run mypy src/

      - name: Run tests
        run: uv run pytest --cov
```

### Updating Dependencies

```bash
# Check for outdated packages
uv pip list --outdated

# Update specific package
uv lock --upgrade-package fastapi
uv sync

# Update all packages
uv lock --upgrade
uv sync

# Commit lock file
git add uv.lock
git commit -m "chore: update dependencies"
```

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev)
RUN uv sync --no-dev --frozen

# Copy application
COPY src/ src/

# Run application
CMD ["uv", "run", "uvicorn", "src.my_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Create project | `uv init my-project` |
| Add dependency | `uv add package` |
| Add dev dependency | `uv add --dev package` |
| Remove dependency | `uv remove package` |
| Sync environment | `uv sync` |
| Update lock | `uv lock --upgrade` |
| Run script | `uv run script` |
| Run Python | `uv run python file.py` |
| Install Python | `uv python install 3.13` |

---

## Best Practices

1. **Always commit `uv.lock`**: Ensures reproducible builds
2. **Use `uv sync --frozen` in CI**: Fails if lock is outdated
3. **Pin Python version**: Use `.python-version` file
4. **Define scripts**: Use `[tool.uv.scripts]` for common commands
5. **Separate dev dependencies**: Use `--dev` flag
6. **Regular updates**: Schedule dependency updates
