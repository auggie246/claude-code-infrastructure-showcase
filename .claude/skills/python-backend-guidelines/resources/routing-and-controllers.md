# Routing and Controllers

## Table of Contents
- [Router Basics](#router-basics)
- [Route Organization](#route-organization)
- [BaseController Pattern](#basecontroller-pattern)
- [Request Handling](#request-handling)
- [Response Formatting](#response-formatting)
- [Error Handling](#error-handling)
- [Dependency Injection](#dependency-injection)

---

## Router Basics

### Creating a Router

```python
# src/api/v1/routes/users.py
from fastapi import APIRouter, Depends, status
from typing import Annotated

from ....schemas.user import UserCreate, UserResponse, UserList
from ....controllers.user_controller import UserController
from ...deps import get_user_controller

router = APIRouter(prefix="/users", tags=["users"])

UserControllerDep = Annotated[UserController, Depends(get_user_controller)]


@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user with email and name.",
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
    summary="Get user by ID",
)
async def get_user(
    user_id: int,
    controller: UserControllerDep,
) -> UserResponse:
    """Retrieve a user by their ID."""
    return await controller.get(user_id)


@router.get(
    "/",
    response_model=UserList,
    summary="List users",
)
async def list_users(
    controller: UserControllerDep,
    skip: int = 0,
    limit: int = 100,
) -> UserList:
    """List all users with pagination."""
    return await controller.list(skip=skip, limit=limit)
```

### Route Decorators

```python
# HTTP Methods
@router.get("/")      # Read
@router.post("/")     # Create
@router.put("/{id}")  # Full update
@router.patch("/{id}")  # Partial update
@router.delete("/{id}")  # Delete

# With all options
@router.post(
    "/",
    response_model=UserResponse,           # Response schema
    status_code=status.HTTP_201_CREATED,   # Success status
    summary="Create user",                  # OpenAPI summary
    description="Detailed description...",  # OpenAPI description
    tags=["users"],                         # OpenAPI tags
    deprecated=False,                       # Mark as deprecated
    response_description="The created user",
    responses={
        400: {"description": "Validation error"},
        409: {"description": "User already exists"},
    },
)
async def create_user(...): ...
```

---

## Route Organization

### API Version Structure

```
src/
└── api/
    ├── __init__.py
    ├── deps.py              # Shared dependencies
    └── v1/
        ├── __init__.py
        ├── router.py        # Main v1 router
        └── routes/
            ├── __init__.py
            ├── users.py
            ├── posts.py
            └── health.py
```

### Main Router Assembly

```python
# src/api/v1/router.py
from fastapi import APIRouter

from .routes import users, posts, health

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(users.router)
api_router.include_router(posts.router)
```

### App Integration

```python
# src/main.py
from fastapi import FastAPI
from .api.v1.router import api_router

app = FastAPI(title="My Service", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")
```

### Health Check Route

```python
# src/api/v1/routes/health.py
from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")
```

---

## BaseController Pattern

### BaseController Implementation

```python
# src/controllers/base.py
from abc import ABC
from typing import Any, TypeVar, Generic
import logging

from fastapi import HTTPException, status
import sentry_sdk

from ..schemas.base import PaginatedResponse

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BaseController(ABC, Generic[T]):
    """Base controller with common functionality."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def success(self, data: T) -> T:
        """Return successful response."""
        return data

    def paginated(
        self,
        items: list[T],
        total: int,
        skip: int,
        limit: int,
    ) -> PaginatedResponse[T]:
        """Return paginated response."""
        return PaginatedResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )

    def handle_error(
        self,
        error: Exception,
        operation: str,
        *,
        user_message: str | None = None,
    ) -> None:
        """Handle and log errors appropriately."""
        self.logger.error(
            "Error in %s: %s",
            operation,
            str(error),
            exc_info=True,
        )

        # Capture to Sentry
        sentry_sdk.capture_exception(error)

        # Re-raise HTTPException as-is
        if isinstance(error, HTTPException):
            raise error

        # Convert known errors
        if isinstance(error, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=user_message or str(error),
            )

        # Unknown errors become 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=user_message or "An unexpected error occurred",
        )

    def not_found(self, resource: str, identifier: Any) -> HTTPException:
        """Create a 404 error."""
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id '{identifier}' not found",
        )

    def bad_request(self, message: str) -> HTTPException:
        """Create a 400 error."""
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )

    def forbidden(self, message: str = "Access denied") -> HTTPException:
        """Create a 403 error."""
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
        )
```

### Concrete Controller

```python
# src/controllers/user_controller.py
from fastapi import HTTPException, status

from .base import BaseController
from ..schemas.user import UserCreate, UserUpdate, UserResponse, UserList
from ..services.user_service import UserService
from ..core.exceptions import UserNotFoundError, UserAlreadyExistsError


class UserController(BaseController[UserResponse]):
    """Controller for user operations."""

    def __init__(self, user_service: UserService) -> None:
        super().__init__()
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
            self.handle_error(e, "create_user")

    async def get(self, user_id: int) -> UserResponse:
        """Get user by ID."""
        try:
            user = await self.user_service.get_by_id(user_id)
            if not user:
                raise self.not_found("User", user_id)
            return UserResponse.model_validate(user)
        except HTTPException:
            raise
        except Exception as e:
            self.handle_error(e, "get_user")

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> UserList:
        """List users with pagination."""
        try:
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
        except Exception as e:
            self.handle_error(e, "list_users")

    async def update(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> UserResponse:
        """Update a user."""
        try:
            user = await self.user_service.update_user(user_id, data)
            if not user:
                raise self.not_found("User", user_id)
            return UserResponse.model_validate(user)
        except HTTPException:
            raise
        except Exception as e:
            self.handle_error(e, "update_user")

    async def delete(self, user_id: int) -> None:
        """Delete a user."""
        try:
            deleted = await self.user_service.delete_user(user_id)
            if not deleted:
                raise self.not_found("User", user_id)
        except HTTPException:
            raise
        except Exception as e:
            self.handle_error(e, "delete_user")
```

---

## Request Handling

### Path Parameters

```python
@router.get("/{user_id}")
async def get_user(user_id: int) -> UserResponse:
    """user_id is automatically validated as int."""
    ...

# With validation
from fastapi import Path

@router.get("/{user_id}")
async def get_user(
    user_id: int = Path(..., gt=0, description="User ID"),
) -> UserResponse:
    ...
```

### Query Parameters

```python
@router.get("/")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    search: str | None = None,
    active: bool = True,
) -> UserList:
    ...

# With validation
from fastapi import Query

@router.get("/")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: str | None = Query(None, min_length=1, max_length=100),
) -> UserList:
    ...
```

### Request Body

```python
@router.post("/")
async def create_user(
    data: UserCreate,  # Pydantic model from body
) -> UserResponse:
    ...

# Multiple body params
from fastapi import Body

@router.post("/with-metadata")
async def create_with_meta(
    user: UserCreate,
    metadata: dict = Body(...),
) -> UserResponse:
    ...
```

### Headers

```python
from fastapi import Header

@router.get("/")
async def get_items(
    x_request_id: str = Header(...),
    authorization: str | None = Header(None),
) -> list[Item]:
    ...
```

---

## Response Formatting

### Response Models

```python
# src/schemas/base.py
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: list[T]
    total: int
    skip: int
    limit: int
    has_more: bool


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    code: str | None = None


# src/schemas/user.py
from pydantic import BaseModel, EmailStr, ConfigDict


class UserResponse(BaseModel):
    """User response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    name: str
    is_active: bool


class UserList(PaginatedResponse[UserResponse]):
    """Paginated list of users."""
    pass
```

### Custom Responses

```python
from fastapi.responses import JSONResponse, StreamingResponse

@router.get("/custom")
async def custom_response() -> JSONResponse:
    return JSONResponse(
        content={"message": "Custom"},
        status_code=200,
        headers={"X-Custom-Header": "value"},
    )

@router.get("/stream")
async def stream_response() -> StreamingResponse:
    async def generate():
        for i in range(10):
            yield f"data: {i}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
```

---

## Error Handling

### Global Exception Handlers

```python
# src/main.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import sentry_sdk

from .core.exceptions import AppError

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """Handle custom application errors."""
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected errors."""
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )
```

---

## Dependency Injection

### Complete Dependencies Setup

```python
# src/api/deps.py
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import async_session_maker
from ..repositories.user_repository import UserRepository
from ..services.user_service import UserService
from ..controllers.user_controller import UserController


# Database session
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


SessionDep = Annotated[AsyncSession, Depends(get_session)]


# Repositories
def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]


# Services
def get_user_service(user_repo: UserRepoDep) -> UserService:
    return UserService(user_repo)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


# Controllers
def get_user_controller(user_service: UserServiceDep) -> UserController:
    return UserController(user_service)


UserControllerDep = Annotated[UserController, Depends(get_user_controller)]
```

### Using Dependencies in Routes

```python
from ..deps import UserControllerDep

@router.get("/{user_id}")
async def get_user(
    user_id: int,
    controller: UserControllerDep,
) -> UserResponse:
    return await controller.get(user_id)
```

---

## Best Practices

1. **Keep routers thin**: Only routing logic, delegate to controllers
2. **Use type hints everywhere**: Enable static analysis and docs
3. **Document with docstrings**: Auto-generates OpenAPI docs
4. **Use Annotated types**: Cleaner dependency injection
5. **Validate at edges**: Use Path, Query, Body validators
6. **Handle errors in controllers**: Not in routers
7. **Return Pydantic models**: Type-safe responses
