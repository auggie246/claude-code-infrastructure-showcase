# Async Patterns and Error Handling

## Table of Contents
- [Async Fundamentals](#async-fundamentals)
- [Async Best Practices](#async-best-practices)
- [Custom Exceptions](#custom-exceptions)
- [Exception Handlers](#exception-handlers)
- [Error Response Patterns](#error-response-patterns)
- [Retry Patterns](#retry-patterns)

---

## Async Fundamentals

### Basic Async/Await

```python
import asyncio


async def fetch_user(user_id: int) -> User:
    """Async function."""
    return await user_repository.find_by_id(user_id)


async def fetch_users(user_ids: list[int]) -> list[User]:
    """Fetch multiple users concurrently."""
    tasks = [fetch_user(uid) for uid in user_ids]
    return await asyncio.gather(*tasks)
```

### Concurrent Execution

```python
import asyncio


async def process_order(order_id: int) -> OrderResult:
    """Process order with concurrent operations."""
    # Run independent operations concurrently
    user, inventory, payment_methods = await asyncio.gather(
        fetch_user(order.user_id),
        check_inventory(order.items),
        fetch_payment_methods(order.user_id),
    )

    # Sequential operations that depend on results
    validated_order = await validate_order(order, inventory)
    result = await process_payment(validated_order, payment_methods[0])

    return result
```

### TaskGroup (Python 3.11+)

```python
import asyncio


async def fetch_all_data() -> dict:
    """Use TaskGroup for structured concurrency."""
    results = {}

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_users())
        task2 = tg.create_task(fetch_posts())
        task3 = tg.create_task(fetch_comments())

    # All tasks complete here, or exception is raised
    results["users"] = task1.result()
    results["posts"] = task2.result()
    results["comments"] = task3.result()

    return results
```

### Timeouts

```python
import asyncio


async def fetch_with_timeout(url: str) -> Response:
    """Fetch with timeout."""
    try:
        async with asyncio.timeout(30):  # 30 second timeout
            return await http_client.get(url)
    except TimeoutError:
        raise ServiceTimeoutError(f"Request to {url} timed out")


# Or using wait_for
async def fetch_with_wait_for(url: str) -> Response:
    try:
        return await asyncio.wait_for(
            http_client.get(url),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise ServiceTimeoutError(f"Request to {url} timed out")
```

---

## Async Best Practices

### Never Block the Event Loop

```python
# ❌ Bad: Blocking call in async function
async def process_file(path: str) -> str:
    with open(path) as f:  # Blocking!
        return f.read()


# ✅ Good: Use async file I/O
import aiofiles

async def process_file(path: str) -> str:
    async with aiofiles.open(path) as f:
        return await f.read()


# ✅ Good: Run blocking code in thread pool
import asyncio

async def process_file(path: str) -> str:
    def read_file():
        with open(path) as f:
            return f.read()

    return await asyncio.to_thread(read_file)
```

### Use Async Context Managers

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """Async context manager for database connection."""
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)


# Usage
async def query_users() -> list[User]:
    async with get_db_connection() as conn:
        return await conn.fetch("SELECT * FROM users")
```

### Async Generators

```python
from typing import AsyncGenerator


async def stream_users(batch_size: int = 100) -> AsyncGenerator[User, None]:
    """Stream users in batches."""
    offset = 0
    while True:
        users = await user_repo.find_all(skip=offset, limit=batch_size)
        if not users:
            break

        for user in users:
            yield user

        offset += batch_size


# Usage
async def process_all_users() -> None:
    async for user in stream_users():
        await process_user(user)
```

---

## Custom Exceptions

### Exception Hierarchy

```python
# src/core/exceptions.py
from fastapi import status


class AppError(Exception):
    """Base application error."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        super().__init__(self.message)


class NotFoundError(AppError):
    """Resource not found."""
    status_code = status.HTTP_404_NOT_FOUND
    code = "NOT_FOUND"
    message = "Resource not found"


class ValidationError(AppError):
    """Validation error."""
    status_code = status.HTTP_400_BAD_REQUEST
    code = "VALIDATION_ERROR"
    message = "Validation failed"


class UnauthorizedError(AppError):
    """Authentication required."""
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "UNAUTHORIZED"
    message = "Authentication required"


class ForbiddenError(AppError):
    """Access denied."""
    status_code = status.HTTP_403_FORBIDDEN
    code = "FORBIDDEN"
    message = "Access denied"


class ConflictError(AppError):
    """Resource conflict."""
    status_code = status.HTTP_409_CONFLICT
    code = "CONFLICT"
    message = "Resource conflict"


# Domain-specific errors
class UserNotFoundError(NotFoundError):
    code = "USER_NOT_FOUND"
    message = "User not found"


class UserAlreadyExistsError(ConflictError):
    code = "USER_EXISTS"
    message = "User already exists"


class InsufficientFundsError(ValidationError):
    code = "INSUFFICIENT_FUNDS"
    message = "Insufficient funds for transaction"


class ExternalServiceError(AppError):
    """External service failure."""
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "EXTERNAL_SERVICE_ERROR"
    message = "External service unavailable"
```

### Using Custom Exceptions

```python
class UserService:
    async def get_by_id(self, user_id: int) -> User:
        user = await self.user_repo.find_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user

    async def create(self, data: UserCreate) -> User:
        existing = await self.user_repo.find_by_email(data.email)
        if existing:
            raise UserAlreadyExistsError(
                f"Email {data.email} is already registered"
            )
        return await self.user_repo.create(data)
```

---

## Exception Handlers

### Global Exception Handler

```python
# src/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import sentry_sdk

from .core.exceptions import AppError

app = FastAPI()


@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """Handle custom application errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def global_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected errors."""
    # Log to Sentry
    sentry_sdk.capture_exception(exc)

    # Don't expose internal errors
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
            }
        },
    )
```

### Middleware Error Handling

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
import sentry_sdk


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log error
            sentry_sdk.capture_exception(e)

            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                    }
                },
            )
```

---

## Error Response Patterns

### Standardized Error Response

```python
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Error detail schema."""
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: ErrorDetail


class ValidationErrorResponse(BaseModel):
    """Validation error with details."""
    error: ErrorDetail
    details: list[dict]
```

### HTTP Exception Wrapper

```python
from fastapi import HTTPException, status


def http_error(
    status_code: int,
    code: str,
    message: str,
) -> HTTPException:
    """Create standardized HTTP error."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


# Usage
raise http_error(404, "USER_NOT_FOUND", "User not found")
```

---

## Retry Patterns

### Simple Retry Decorator

```python
import asyncio
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


# Usage
@retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
async def fetch_external_data(url: str) -> dict:
    return await http_client.get(url)
```

### Circuit Breaker

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for external services."""

    failure_threshold: int = 5
    reset_timeout: timedelta = timedelta(seconds=60)
    half_open_max_calls: int = 3

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time: datetime | None = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if datetime.utcnow() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        # HALF_OPEN
        return self.half_open_calls < self.half_open_max_calls

    def record_success(self) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failures = 0

    def record_failure(self) -> None:
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()

        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN


# Usage
circuit = CircuitBreaker()


async def call_external_service() -> dict:
    if not circuit.can_execute():
        raise ExternalServiceError("Circuit breaker is open")

    try:
        result = await http_client.get(url)
        circuit.record_success()
        return result
    except Exception as e:
        circuit.record_failure()
        raise
```

---

## Best Practices

1. **Always await**: Never forget `await` for coroutines
2. **Use TaskGroup**: For structured concurrency (Python 3.11+)
3. **Handle timeouts**: Always set timeouts for external calls
4. **Don't block**: Use `asyncio.to_thread` for blocking operations
5. **Custom exceptions**: Create domain-specific exception hierarchy
6. **Global handlers**: Catch all errors at app level
7. **Log to Sentry**: Capture all unexpected errors
8. **Retry wisely**: Use exponential backoff, circuit breakers
