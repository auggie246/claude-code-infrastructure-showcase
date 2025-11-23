---
name: python-backend-guidelines
description: Comprehensive Python 3.13 backend development guide using uv for project management. Use when creating FastAPI routes, services, repositories, middleware, or working with SQLAlchemy, Pydantic validation, Sentry error tracking, dependency injection, or async patterns. Covers layered architecture (routers → controllers → services → repositories), BaseController pattern, error handling, performance monitoring, testing with pytest, and modern Python best practices.
---

# Python Backend Development Guidelines

## Purpose

Establish consistency and best practices across Python backend microservices using modern Python 3.13 patterns with uv for project and environment management.

## When to Use This Skill

Automatically activates when working on:
- Creating or modifying FastAPI routes, endpoints, APIs
- Building controllers, services, repositories in Python
- Implementing middleware (auth, validation, error handling)
- Database operations with SQLAlchemy
- Error tracking with Sentry
- Input validation with Pydantic
- Configuration management with pydantic-settings
- Python backend testing with pytest
- Project setup with uv

---

## Quick Start

### New Backend Feature Checklist

- [ ] **Router**: Clean definition, delegate to controller
- [ ] **Controller**: Extend BaseController
- [ ] **Service**: Business logic with DI
- [ ] **Repository**: Database access (if complex)
- [ ] **Validation**: Pydantic schema
- [ ] **Sentry**: Error tracking
- [ ] **Tests**: Unit + integration tests
- [ ] **Config**: Use Settings class

### New Microservice Checklist

- [ ] `uv init` with proper pyproject.toml
- [ ] Directory structure (see [architecture-overview.md](resources/architecture-overview.md))
- [ ] Sentry initialization in lifespan
- [ ] Settings class with pydantic-settings
- [ ] BaseController class
- [ ] Middleware stack
- [ ] Exception handlers
- [ ] Testing framework with pytest

---

## Architecture Overview

### Layered Architecture

```
HTTP Request
    ↓
Routers (routing only)
    ↓
Controllers (request handling)
    ↓
Services (business logic)
    ↓
Repositories (data access)
    ↓
Database (SQLAlchemy)
```

**Key Principle:** Each layer has ONE responsibility.

See [architecture-overview.md](resources/architecture-overview.md) for complete details.

---

## Directory Structure

```
service/
├── pyproject.toml           # uv project config
├── uv.lock                   # Locked dependencies
├── src/
│   └── service_name/
│       ├── __init__.py
│       ├── main.py          # FastAPI app + lifespan
│       ├── config/          # Settings classes
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── api/             # Routers
│       │   ├── __init__.py
│       │   ├── deps.py      # Dependency injection
│       │   └── v1/
│       │       ├── __init__.py
│       │       └── routes/
│       ├── controllers/     # Request handlers
│       ├── services/        # Business logic
│       ├── repositories/    # Data access
│       ├── models/          # SQLAlchemy models
│       ├── schemas/         # Pydantic schemas
│       ├── middleware/      # Custom middleware
│       └── core/            # Core utilities
│           ├── exceptions.py
│           └── base_controller.py
└── tests/
    ├── conftest.py
    ├── unit/
    └── integration/
```

**Naming Conventions:**
- Modules: `snake_case` - `user_service.py`
- Classes: `PascalCase` - `UserController`
- Functions: `snake_case` - `get_user_by_id`
- Constants: `UPPER_SNAKE` - `DEFAULT_TIMEOUT`

---

## Core Principles (7 Key Rules)

### 1. Routers Only Route, Controllers Control

```python
# ❌ NEVER: Business logic in routers
@router.post("/submit")
async def submit(request: Request):
    # 200 lines of logic
    ...

# ✅ ALWAYS: Delegate to controller
@router.post("/submit")
async def submit(
    data: SubmitRequest,
    controller: UserController = Depends(get_user_controller)
) -> SubmitResponse:
    return await controller.submit(data)
```

### 2. All Controllers Extend BaseController

```python
class UserController(BaseController):
    def __init__(self, user_service: UserService):
        super().__init__()
        self.user_service = user_service

    async def get_user(self, user_id: str) -> UserResponse:
        try:
            user = await self.user_service.find_by_id(user_id)
            return self.success(user)
        except Exception as e:
            return self.handle_error(e, "get_user")
```

### 3. All Errors to Sentry

```python
try:
    await operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

### 4. Use Settings Class, NEVER os.environ

```python
# ❌ NEVER
timeout = os.environ.get("TIMEOUT_MS")

# ✅ ALWAYS
from .config import settings
timeout = settings.timeout_ms
```

### 5. Validate All Input with Pydantic

```python
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
```

### 6. Use Repository Pattern for Data Access

```python
# Service → Repository → Database
users = await user_repository.find_active()
```

### 7. Comprehensive Testing Required

```python
@pytest.mark.asyncio
async def test_create_user(user_service: UserService):
    user = await user_service.create(UserCreate(email="test@example.com"))
    assert user.id is not None
```

---

## uv Project Management

### Essential Commands

```bash
# Create new project
uv init my-service
cd my-service

# Add dependencies
uv add fastapi uvicorn sqlalchemy pydantic-settings
uv add sentry-sdk httpx

# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy

# Sync environment
uv sync

# Run application
uv run uvicorn src.service_name.main:app --reload

# Run tests
uv run pytest

# Lock dependencies
uv lock
```

See [uv-project-management.md](resources/uv-project-management.md) for complete details.

---

## Common Imports

```python
# FastAPI
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Validation
from pydantic import BaseModel, Field, EmailStr, field_validator

# Database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Sentry
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

# Config
from pydantic_settings import BaseSettings, SettingsConfigDict

# Typing
from typing import Annotated, AsyncGenerator
from collections.abc import Callable
```

---

## Quick Reference

### HTTP Status Codes

| Code | Use Case | FastAPI Constant |
|------|----------|------------------|
| 200 | Success | `status.HTTP_200_OK` |
| 201 | Created | `status.HTTP_201_CREATED` |
| 400 | Bad Request | `status.HTTP_400_BAD_REQUEST` |
| 401 | Unauthorized | `status.HTTP_401_UNAUTHORIZED` |
| 403 | Forbidden | `status.HTTP_403_FORBIDDEN` |
| 404 | Not Found | `status.HTTP_404_NOT_FOUND` |
| 500 | Server Error | `status.HTTP_500_INTERNAL_SERVER_ERROR` |

### Type Hints (Python 3.13)

```python
# Use built-in generics (no typing import needed)
def process(items: list[str]) -> dict[str, int]: ...

# Use | for unions
def get_user(id: int) -> User | None: ...

# Annotated for DI
UserDep = Annotated[User, Depends(get_current_user)]
```

---

## Anti-Patterns to Avoid

❌ Business logic in routers
❌ Direct `os.environ` usage
❌ Missing error handling
❌ No input validation
❌ Direct SQLAlchemy everywhere (no repository)
❌ `print()` instead of logging/Sentry
❌ Sync operations in async code
❌ Missing type hints

---

## Navigation Guide

| Need to... | Read this |
|------------|-----------|
| Understand architecture | [architecture-overview.md](resources/architecture-overview.md) |
| Set up uv project | [uv-project-management.md](resources/uv-project-management.md) |
| Create routes/controllers | [routing-and-controllers.md](resources/routing-and-controllers.md) |
| Organize business logic | [services-and-repositories.md](resources/services-and-repositories.md) |
| Validate input | [validation-patterns.md](resources/validation-patterns.md) |
| Add error tracking | [sentry-and-monitoring.md](resources/sentry-and-monitoring.md) |
| Create middleware | [middleware-guide.md](resources/middleware-guide.md) |
| Database access | [database-patterns.md](resources/database-patterns.md) |
| Manage config | [configuration.md](resources/configuration.md) |
| Handle async/errors | [async-and-errors.md](resources/async-and-errors.md) |
| Write tests | [testing-guide.md](resources/testing-guide.md) |
| See examples | [complete-examples.md](resources/complete-examples.md) |

---

## Resource Files

### [architecture-overview.md](resources/architecture-overview.md)
Layered architecture, request lifecycle, separation of concerns

### [uv-project-management.md](resources/uv-project-management.md)
uv setup, pyproject.toml, dependency management, scripts

### [routing-and-controllers.md](resources/routing-and-controllers.md)
Router definitions, BaseController, error handling, examples

### [services-and-repositories.md](resources/services-and-repositories.md)
Service patterns, DI with FastAPI, repository pattern, caching

### [validation-patterns.md](resources/validation-patterns.md)
Pydantic models, validators, serialization

### [sentry-and-monitoring.md](resources/sentry-and-monitoring.md)
Sentry init, error capture, performance monitoring

### [middleware-guide.md](resources/middleware-guide.md)
Auth, logging, error boundaries, context vars

### [database-patterns.md](resources/database-patterns.md)
SQLAlchemy async, repositories, transactions, optimization

### [configuration.md](resources/configuration.md)
pydantic-settings, environment configs, secrets

### [async-and-errors.md](resources/async-and-errors.md)
Async patterns, custom exceptions, error handlers

### [testing-guide.md](resources/testing-guide.md)
pytest-asyncio, fixtures, mocking, coverage

### [complete-examples.md](resources/complete-examples.md)
Full examples, migration guide from Flask/Django

---

## Related Skills

- **error-tracking** - Sentry integration patterns
- **skill-developer** - Meta-skill for creating and managing skills

---

**Skill Status**: COMPLETE
**Line Count**: < 500
**Progressive Disclosure**: 12 resource files
