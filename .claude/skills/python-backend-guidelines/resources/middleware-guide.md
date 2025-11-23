# Middleware Guide

## Table of Contents
- [Middleware Basics](#middleware-basics)
- [Common Middleware](#common-middleware)
- [Authentication Middleware](#authentication-middleware)
- [Logging Middleware](#logging-middleware)
- [Context Variables](#context-variables)
- [Middleware Order](#middleware-order)

---

## Middleware Basics

### Creating Middleware

```python
# Using BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class CustomMiddleware(BaseHTTPMiddleware):
    """Custom middleware example."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # Before request processing
        print(f"Request: {request.method} {request.url}")

        # Process request
        response = await call_next(request)

        # After request processing
        print(f"Response: {response.status_code}")

        return response


# Register middleware
app.add_middleware(CustomMiddleware)
```

### Pure ASGI Middleware

```python
from starlette.types import ASGIApp, Receive, Scope, Send


class PureASGIMiddleware:
    """Pure ASGI middleware (more performant)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Before processing
        path = scope["path"]

        # Process request
        await self.app(scope, receive, send)

        # After processing (limited access to response)
```

---

## Common Middleware

### CORS Middleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
```

### Request ID Middleware

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # Get or generate request ID
        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4()),
        )

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response
```

### Timing Middleware

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
import logging

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Log request timing."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        duration_ms = round(duration * 1000, 2)

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        return response
```

---

## Authentication Middleware

### JWT Authentication

```python
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from ..config import settings


class JWTBearer(HTTPBearer):
    """JWT Bearer authentication."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> str | None:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme",
                )

            token = credentials.credentials
            payload = self.verify_token(token)

            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                )

            return payload

        return None

    def verify_token(self, token: str) -> dict | None:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            return payload
        except JWTError:
            return None


# Dependency
jwt_bearer = JWTBearer()


async def get_current_user(
    payload: dict = Depends(jwt_bearer),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Get current authenticated user."""
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = await user_service.get_by_id(int(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user
```

### API Key Authentication

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


async def verify_api_key(
    api_key: str = Security(API_KEY_HEADER),
) -> str:
    """Verify API key."""
    # In production, check against database or cache
    valid_keys = {"key1", "key2", "key3"}

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


# Usage
@router.get("/protected")
async def protected_route(
    api_key: str = Depends(verify_api_key),
) -> dict:
    return {"message": "Access granted"}
```

---

## Logging Middleware

### Structured Logging

```python
import logging
import json
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

logger = logging.getLogger("api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging middleware."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # Extract request info
        request_id = getattr(request.state, "request_id", "-")
        start_time = time.perf_counter()

        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            },
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            raise

        # Calculate duration
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response
```

### JSON Formatter

```python
import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Configure logging
def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
```

---

## Context Variables

### Using contextvars

```python
# src/core/context.py
from contextvars import ContextVar
from typing import Optional

# Context variables
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
user_id_ctx: ContextVar[Optional[int]] = ContextVar("user_id", default=None)
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def get_request_id() -> str:
    """Get current request ID."""
    return request_id_ctx.get()


def get_user_id() -> Optional[int]:
    """Get current user ID."""
    return user_id_ctx.get()
```

### Context Middleware

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

from .core.context import request_id_ctx, user_id_ctx


class ContextMiddleware(BaseHTTPMiddleware):
    """Set context variables for request."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # Set request ID
        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4()),
        )
        token = request_id_ctx.set(request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
```

### Using Context in Services

```python
from ..core.context import get_request_id, get_user_id
import sentry_sdk


class UserService:
    async def create_user(self, data: UserCreate) -> User:
        # Context is available anywhere in the call stack
        request_id = get_request_id()
        user_id = get_user_id()

        # Use in logging
        self.logger.info(
            "Creating user",
            extra={
                "request_id": request_id,
                "created_by": user_id,
            },
        )

        # Use in Sentry
        sentry_sdk.set_tag("request_id", request_id)

        return await self.user_repo.create(data)
```

---

## Middleware Order

### Recommended Order

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

# Order matters! Last added = first executed

# 1. Compression (outermost)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request ID (needs to be early)
app.add_middleware(RequestIDMiddleware)

# 4. Context variables
app.add_middleware(ContextMiddleware)

# 5. Logging (after context is set)
app.add_middleware(LoggingMiddleware)

# 6. Timing
app.add_middleware(TimingMiddleware)

# 7. Error handling (innermost)
app.add_middleware(ErrorHandlingMiddleware)
```

### Execution Flow

```
Request In:
  1. GZipMiddleware (decompress)
  2. CORSMiddleware (add headers)
  3. RequestIDMiddleware (set ID)
  4. ContextMiddleware (set context)
  5. LoggingMiddleware (log request)
  6. TimingMiddleware (start timer)
  7. ErrorHandlingMiddleware
  8. → Route Handler

Response Out:
  8. ← Route Handler
  7. ErrorHandlingMiddleware (catch errors)
  6. TimingMiddleware (stop timer)
  5. LoggingMiddleware (log response)
  4. ContextMiddleware (cleanup)
  3. RequestIDMiddleware (add header)
  2. CORSMiddleware
  1. GZipMiddleware (compress)
```

---

## Best Practices

1. **Keep middleware thin**: Do one thing well
2. **Handle errors**: Don't let middleware crash the request
3. **Use context vars**: For request-scoped data
4. **Order matters**: Add middleware in correct order
5. **Pure ASGI for performance**: When BaseHTTPMiddleware is too slow
6. **Log at boundaries**: Request start/end
7. **Add request IDs**: For tracing across services
8. **Clean up**: Reset context vars in finally blocks
