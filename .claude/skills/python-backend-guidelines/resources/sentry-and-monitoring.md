# Sentry and Monitoring

## Table of Contents
- [Sentry Setup](#sentry-setup)
- [Error Capture](#error-capture)
- [Performance Monitoring](#performance-monitoring)
- [Context and Tags](#context-and-tags)
- [Custom Instrumentation](#custom-instrumentation)
- [Best Practices](#best-practices)

---

## Sentry Setup

### Installation

```bash
uv add "sentry-sdk[fastapi]"
```

### Basic Initialization

```python
# src/main.py
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from fastapi import FastAPI

from .config import settings


def init_sentry() -> None:
    """Initialize Sentry SDK."""
    if not settings.sentry_dsn:
        return

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=settings.app_version,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        profiles_sample_rate=settings.sentry_profiles_sample_rate,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],
        # Don't send PII by default
        send_default_pii=False,
        # Filter sensitive data
        before_send=filter_sensitive_data,
    )


def filter_sensitive_data(event: dict, hint: dict) -> dict | None:
    """Filter sensitive data before sending to Sentry."""
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        sensitive = ["authorization", "cookie", "x-api-key"]
        for key in sensitive:
            headers.pop(key, None)

    return event


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    init_sentry()
    yield
    # Shutdown
    await sentry_sdk.flush(timeout=2.0)


app = FastAPI(lifespan=lifespan)
```

### Configuration Settings

```python
# src/config/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Sentry
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1  # 10% of transactions
    sentry_profiles_sample_rate: float = 0.1
    environment: str = "development"
    app_version: str = "1.0.0"
```

---

## Error Capture

### Automatic Capture

FastAPI integration automatically captures:
- Unhandled exceptions
- HTTP 500 errors
- Request context

### Manual Capture

```python
import sentry_sdk


# Capture exception
try:
    risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise


# Capture message
sentry_sdk.capture_message("Something noteworthy happened")


# Capture with level
sentry_sdk.capture_message(
    "User performed unusual action",
    level="warning",
)
```

### Exception Handler Integration

```python
# src/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sentry_sdk

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle all uncaught exceptions."""
    # Capture to Sentry with request context
    with sentry_sdk.push_scope() as scope:
        scope.set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
        })
        sentry_sdk.capture_exception(exc)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
```

### In Controllers

```python
class UserController(BaseController):
    async def create(self, data: UserCreate) -> UserResponse:
        try:
            user = await self.user_service.create_user(data)
            return UserResponse.model_validate(user)
        except UserAlreadyExistsError as e:
            # Known error - capture but don't alert
            sentry_sdk.capture_exception(e)
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            # Unknown error - capture and re-raise
            sentry_sdk.capture_exception(e)
            raise
```

---

## Performance Monitoring

### Automatic Instrumentation

The FastAPI integration automatically creates transactions for:
- HTTP requests
- Database queries (with SQLAlchemy integration)

### Custom Transactions

```python
import sentry_sdk


async def process_batch(items: list[Item]) -> None:
    """Process items with transaction tracking."""
    with sentry_sdk.start_transaction(
        op="batch.process",
        name="Process Item Batch",
    ) as transaction:
        for item in items:
            with transaction.start_child(
                op="item.process",
                description=f"Process item {item.id}",
            ):
                await process_item(item)
```

### Custom Spans

```python
async def complex_operation() -> Result:
    """Operation with multiple spans."""
    with sentry_sdk.start_span(op="db.query", description="Fetch users"):
        users = await fetch_users()

    with sentry_sdk.start_span(op="transform", description="Transform data"):
        transformed = transform_users(users)

    with sentry_sdk.start_span(op="external.api", description="Call external API"):
        result = await call_external_api(transformed)

    return result
```

### Database Query Monitoring

```python
# Automatic with SQLAlchemy integration
# Queries are automatically instrumented

# For additional context
from sentry_sdk import set_tag

async def get_users_by_department(dept_id: int) -> list[User]:
    set_tag("department_id", dept_id)
    return await user_repo.find_by_department(dept_id)
```

---

## Context and Tags

### Setting User Context

```python
import sentry_sdk
from fastapi import Request, Depends


async def set_sentry_user(request: Request) -> None:
    """Set Sentry user context from request."""
    user = getattr(request.state, "user", None)
    if user:
        sentry_sdk.set_user({
            "id": str(user.id),
            "email": user.email,
            "username": user.name,
        })


# Use as dependency
@router.get("/protected")
async def protected_route(
    _: None = Depends(set_sentry_user),
) -> dict:
    ...
```

### Setting Tags

```python
import sentry_sdk

# Global tags (set once)
sentry_sdk.set_tag("service", "user-service")

# Request-specific tags
sentry_sdk.set_tag("tenant_id", tenant.id)
sentry_sdk.set_tag("feature_flag", "new_checkout")
```

### Setting Context

```python
import sentry_sdk


# Custom context
sentry_sdk.set_context("order", {
    "order_id": order.id,
    "total": str(order.total),
    "items_count": len(order.items),
})

# In exception handling
with sentry_sdk.push_scope() as scope:
    scope.set_context("user_action", {
        "action": "checkout",
        "cart_id": cart.id,
    })
    scope.set_tag("payment_method", "credit_card")
    sentry_sdk.capture_exception(error)
```

### Breadcrumbs

```python
import sentry_sdk


# Automatic breadcrumbs from logging
import logging
logger = logging.getLogger(__name__)
logger.info("User started checkout")  # Creates breadcrumb

# Manual breadcrumbs
sentry_sdk.add_breadcrumb(
    category="user.action",
    message="Added item to cart",
    level="info",
    data={"item_id": item.id, "quantity": quantity},
)
```

---

## Custom Instrumentation

### Decorators

```python
import functools
from typing import Callable, TypeVar, ParamSpec
import sentry_sdk

P = ParamSpec("P")
R = TypeVar("R")


def traced(op: str, description: str | None = None) -> Callable:
    """Decorator to trace function execution."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            desc = description or func.__name__
            with sentry_sdk.start_span(op=op, description=desc):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            desc = description or func.__name__
            with sentry_sdk.start_span(op=op, description=desc):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Usage
class PaymentService:
    @traced("payment.process", "Process payment")
    async def process_payment(self, order: Order) -> Payment:
        ...
```

### Context Managers

```python
from contextlib import contextmanager
import sentry_sdk


@contextmanager
def sentry_operation(
    op: str,
    description: str,
    **extra_data,
):
    """Context manager for Sentry spans with extra data."""
    with sentry_sdk.start_span(op=op, description=description) as span:
        for key, value in extra_data.items():
            span.set_data(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status("error")
            raise


# Usage
with sentry_operation("db.query", "Fetch users", limit=100):
    users = await fetch_users(limit=100)
```

### Middleware Instrumentation

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk
import time


class SentryMiddleware(BaseHTTPMiddleware):
    """Middleware for additional Sentry context."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # Set request context
        sentry_sdk.set_context("request_info", {
            "path": request.url.path,
            "method": request.method,
            "query_params": dict(request.query_params),
        })

        # Track timing
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Add response info
            duration = time.perf_counter() - start_time
            sentry_sdk.set_context("response_info", {
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            })

            return response

        except Exception as e:
            duration = time.perf_counter() - start_time
            sentry_sdk.set_context("error_info", {
                "duration_ms": round(duration * 1000, 2),
            })
            raise
```

---

## Best Practices

### 1. Always Capture Errors

```python
# ❌ Bad: Error lost
try:
    await operation()
except Exception:
    pass

# ✅ Good: Error captured
try:
    await operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise  # Or handle appropriately
```

### 2. Add Context Before Capture

```python
# ❌ Bad: No context
sentry_sdk.capture_exception(error)

# ✅ Good: Rich context
with sentry_sdk.push_scope() as scope:
    scope.set_tag("operation", "checkout")
    scope.set_context("order", order.to_dict())
    scope.set_user({"id": user.id})
    sentry_sdk.capture_exception(error)
```

### 3. Use Appropriate Sample Rates

```python
# Production
sentry_traces_sample_rate = 0.1  # 10%

# Development
sentry_traces_sample_rate = 1.0  # 100%

# High-traffic endpoints - use sampling
sentry_sdk.init(
    traces_sampler=custom_sampler,
)

def custom_sampler(sampling_context: dict) -> float:
    path = sampling_context.get("asgi_scope", {}).get("path", "")
    if path == "/health":
        return 0.0  # Don't trace health checks
    if path.startswith("/api/v1/high-traffic"):
        return 0.01  # 1% for high traffic
    return 0.1  # 10% default
```

### 4. Filter Sensitive Data

```python
def before_send(event: dict, hint: dict) -> dict | None:
    # Remove sensitive fields
    if "extra" in event:
        for key in ["password", "token", "secret"]:
            event["extra"].pop(key, None)

    # Don't send certain errors
    if "exc_info" in hint:
        exc_type = hint["exc_info"][0]
        if exc_type == CancelledError:
            return None

    return event
```

### 5. Group Errors Properly

```python
# Set fingerprint for custom grouping
sentry_sdk.set_tag("error_type", "payment_failed")
sentry_sdk.capture_exception(
    error,
    fingerprint=["payment-error", str(payment_provider)],
)
```

### 6. Don't Over-Capture

```python
# ❌ Bad: Capturing expected errors
try:
    user = await get_user(user_id)
except UserNotFoundError:
    sentry_sdk.capture_exception(e)  # Don't do this
    raise HTTPException(404)

# ✅ Good: Only capture unexpected errors
try:
    user = await get_user(user_id)
except UserNotFoundError:
    raise HTTPException(404)  # Expected, don't capture
except Exception as e:
    sentry_sdk.capture_exception(e)  # Unexpected, capture
    raise
```

---

## Quick Reference

| Task | Code |
|------|------|
| Capture exception | `sentry_sdk.capture_exception(e)` |
| Capture message | `sentry_sdk.capture_message("msg")` |
| Set user | `sentry_sdk.set_user({"id": "123"})` |
| Set tag | `sentry_sdk.set_tag("key", "value")` |
| Set context | `sentry_sdk.set_context("name", {...})` |
| Add breadcrumb | `sentry_sdk.add_breadcrumb(...)` |
| Start span | `sentry_sdk.start_span(op="x")` |
| Push scope | `with sentry_sdk.push_scope() as scope:` |
