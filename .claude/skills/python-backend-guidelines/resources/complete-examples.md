# Complete Examples

## Table of Contents
- [Full Project Setup](#full-project-setup)
- [Complete User Module](#complete-user-module)
- [Authentication Flow](#authentication-flow)
- [CRUD API Example](#crud-api-example)

---

## Full Project Setup

### 1. Initialize Project

```bash
# Create project
uv init my-fastapi-service --python 3.13
cd my-fastapi-service

# Add dependencies
uv add fastapi "uvicorn[standard]" sqlalchemy asyncpg pydantic-settings
uv add sentry-sdk httpx alembic

# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy httpx

# Create structure
mkdir -p src/my_service/{api/v1/routes,config,controllers,services,repositories,models,schemas,middleware,core}
mkdir -p tests/{unit,integration}
touch src/my_service/__init__.py
touch src/my_service/{main,app}.py
```

### 2. Main Application

```python
# src/my_service/main.py
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.v1.router import api_router
from .core.database import engine
from .middleware.logging import LoggingMiddleware
from .middleware.request_id import RequestIDMiddleware


def init_sentry() -> None:
    """Initialize Sentry SDK."""
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    init_sentry()
    yield
    # Shutdown
    await engine.dispose()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Middleware (order matters)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Routes
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
```

### 3. Configuration

```python
# src/my_service/config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "my-service"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/mydb"
    )

    # Security
    secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Sentry
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1


settings = Settings()
```

---

## Complete User Module

### Model

```python
# src/my_service/models/user.py
from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
```

### Schemas

```python
# src/my_service/schemas/user.py
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    name: str | None = Field(None, min_length=1, max_length=100)
    is_active: bool | None = None


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserList(BaseModel):
    items: list[UserResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
```

### Repository

```python
# src/my_service/repositories/user_repository.py
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.user import User
from ..schemas.user import UserCreate, UserUpdate


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_by_id(self, user_id: int) -> User | None:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def find_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        result = await self.session.execute(
            select(User)
            .offset(skip)
            .limit(limit)
            .order_by(User.created_at.desc())
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(User)
        )
        return result.scalar_one()

    async def create(
        self,
        email: str,
        name: str,
        hashed_password: str,
    ) -> User:
        user = User(
            email=email,
            name=name,
            hashed_password=hashed_password,
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def update(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> User | None:
        user = await self.find_by_id(user_id)
        if not user:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def delete(self, user_id: int) -> bool:
        user = await self.find_by_id(user_id)
        if not user:
            return False

        await self.session.delete(user)
        await self.session.flush()
        return True
```

### Service

```python
# src/my_service/services/user_service.py
import logging
from passlib.context import CryptContext
import sentry_sdk

from ..repositories.user_repository import UserRepository
from ..schemas.user import UserCreate, UserUpdate
from ..models.user import User
from ..core.exceptions import UserNotFoundError, UserAlreadyExistsError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo

    async def create_user(self, data: UserCreate) -> User:
        """Create a new user."""
        logger.info("Creating user", extra={"email": data.email})

        # Check if email exists
        existing = await self.user_repo.find_by_email(data.email)
        if existing:
            raise UserAlreadyExistsError(
                f"Email {data.email} is already registered"
            )

        # Hash password
        hashed_password = pwd_context.hash(data.password)

        # Create user
        return await self.user_repo.create(
            email=data.email.lower(),
            name=data.name,
            hashed_password=hashed_password,
        )

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        return await self.user_repo.find_by_id(user_id)

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[User], int]:
        """List users with pagination."""
        users = await self.user_repo.find_all(skip=skip, limit=limit)
        total = await self.user_repo.count()
        return users, total

    async def update_user(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> User | None:
        """Update a user."""
        # Check email uniqueness if changing
        if data.email:
            existing = await self.user_repo.find_by_email(data.email)
            if existing and existing.id != user_id:
                raise UserAlreadyExistsError(
                    f"Email {data.email} is already registered"
                )

        return await self.user_repo.update(user_id, data)

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        return await self.user_repo.delete(user_id)
```

### Controller

```python
# src/my_service/controllers/user_controller.py
from fastapi import HTTPException, status
import sentry_sdk

from ..services.user_service import UserService
from ..schemas.user import UserCreate, UserUpdate, UserResponse, UserList
from ..core.exceptions import UserAlreadyExistsError


class UserController:
    def __init__(self, user_service: UserService) -> None:
        self.user_service = user_service

    async def create(self, data: UserCreate) -> UserResponse:
        """Create a new user."""
        try:
            user = await self.user_service.create_user(data)
            return UserResponse.model_validate(user)
        except UserAlreadyExistsError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )

    async def get(self, user_id: int) -> UserResponse:
        """Get user by ID."""
        user = await self.user_service.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        return UserResponse.model_validate(user)

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> UserList:
        """List users."""
        users, total = await self.user_service.list_users(
            skip=skip,
            limit=limit,
        )
        return UserList(
            items=[UserResponse.model_validate(u) for u in users],
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(users) < total,
        )

    async def update(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> UserResponse:
        """Update a user."""
        try:
            user = await self.user_service.update_user(user_id, data)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found",
                )
            return UserResponse.model_validate(user)
        except UserAlreadyExistsError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )

    async def delete(self, user_id: int) -> None:
        """Delete a user."""
        deleted = await self.user_service.delete_user(user_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
```

### Routes

```python
# src/my_service/api/v1/routes/users.py
from fastapi import APIRouter, Depends, Query, status
from typing import Annotated

from ....schemas.user import UserCreate, UserUpdate, UserResponse, UserList
from ....controllers.user_controller import UserController
from ...deps import get_user_controller

router = APIRouter(prefix="/users", tags=["users"])

UserControllerDep = Annotated[UserController, Depends(get_user_controller)]


@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
)
async def create_user(
    data: UserCreate,
    controller: UserControllerDep,
) -> UserResponse:
    """Create a new user."""
    return await controller.create(data)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
)
async def get_user(
    user_id: int,
    controller: UserControllerDep,
) -> UserResponse:
    """Get user by ID."""
    return await controller.get(user_id)


@router.get(
    "/",
    response_model=UserList,
    summary="List users",
)
async def list_users(
    controller: UserControllerDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> UserList:
    """List users with pagination."""
    return await controller.list(skip=skip, limit=limit)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
)
async def update_user(
    user_id: int,
    data: UserUpdate,
    controller: UserControllerDep,
) -> UserResponse:
    """Update a user."""
    return await controller.update(user_id, data)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
)
async def delete_user(
    user_id: int,
    controller: UserControllerDep,
) -> None:
    """Delete a user."""
    await controller.delete(user_id)
```

### Dependencies

```python
# src/my_service/api/deps.py
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import async_session_maker
from ..repositories.user_repository import UserRepository
from ..services.user_service import UserService
from ..controllers.user_controller import UserController


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]


def get_user_service(user_repo: UserRepoDep) -> UserService:
    return UserService(user_repo)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def get_user_controller(user_service: UserServiceDep) -> UserController:
    return UserController(user_service)
```

---

## Running the Application

### Development

```bash
# Run with hot reload
uv run uvicorn src.my_service.main:app --reload --host 0.0.0.0 --port 8000

# Or using pyproject.toml script
uv run dev
```

### Production

```bash
# Run with multiple workers
uv run uvicorn src.my_service.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/my_service --cov-report=html
```

---

## Summary

This example demonstrates:
- Complete layered architecture
- Proper dependency injection
- Type-safe schemas with Pydantic
- Async database operations
- Error handling patterns
- Testing structure
- Configuration management
