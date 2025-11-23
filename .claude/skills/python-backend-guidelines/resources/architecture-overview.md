# Architecture Overview

## Table of Contents
- [Layered Architecture](#layered-architecture)
- [Request Lifecycle](#request-lifecycle)
- [Layer Responsibilities](#layer-responsibilities)
- [Dependency Flow](#dependency-flow)
- [Project Structure](#project-structure)

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP Request                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Middleware Stack                         │
│  (CORS, Auth, Logging, Error Handling)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Routers                               │
│  - Route definitions only                                    │
│  - Request/Response typing                                   │
│  - Dependency injection                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Controllers                             │
│  - Request handling                                          │
│  - Input extraction                                          │
│  - Response formatting                                       │
│  - Error handling                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Services                              │
│  - Business logic                                            │
│  - Orchestration                                             │
│  - Transaction management                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Repositories                            │
│  - Data access abstraction                                   │
│  - Query building                                            │
│  - Result mapping                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Database (SQLAlchemy)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Request Lifecycle

### 1. Request Arrives

```python
# Uvicorn receives HTTP request
# FastAPI routes to appropriate handler
```

### 2. Middleware Processing

```python
# 1. CORS middleware
# 2. Request ID middleware (adds trace ID)
# 3. Auth middleware (validates token)
# 4. Logging middleware (logs request)
```

### 3. Route Handler

```python
@router.post("/users", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    controller: UserController = Depends(get_user_controller)
) -> UserResponse:
    return await controller.create(data)
```

### 4. Controller Processing

```python
async def create(self, data: UserCreate) -> UserResponse:
    try:
        user = await self.user_service.create_user(data)
        return UserResponse.model_validate(user)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 5. Service Logic

```python
async def create_user(self, data: UserCreate) -> User:
    # Business rules
    if await self.user_repo.exists_by_email(data.email):
        raise UserAlreadyExistsError(data.email)

    # Create user
    return await self.user_repo.create(data)
```

### 6. Repository Access

```python
async def create(self, data: UserCreate) -> User:
    user = User(**data.model_dump())
    self.session.add(user)
    await self.session.commit()
    await self.session.refresh(user)
    return user
```

---

## Layer Responsibilities

### Routers (API Layer)

**DO:**
- Define routes and HTTP methods
- Specify request/response schemas
- Inject dependencies
- Document with OpenAPI

**DON'T:**
- Contain business logic
- Access database directly
- Handle complex error cases

```python
# ✅ Good router
router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    controller: UserController = Depends(get_user_controller)
) -> UserResponse:
    return await controller.get(user_id)
```

### Controllers (Request Handling)

**DO:**
- Extract and validate request data
- Call appropriate services
- Format responses
- Handle errors appropriately
- Log operations

**DON'T:**
- Contain business rules
- Access database directly
- Know about other controllers

```python
# ✅ Good controller
class UserController(BaseController):
    def __init__(self, user_service: UserService):
        super().__init__()
        self.user_service = user_service

    async def get(self, user_id: int) -> UserResponse:
        user = await self.user_service.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserResponse.model_validate(user)
```

### Services (Business Logic)

**DO:**
- Implement business rules
- Orchestrate multiple repositories
- Manage transactions
- Emit events/notifications

**DON'T:**
- Handle HTTP concerns
- Format responses
- Access request objects

```python
# ✅ Good service
class UserService:
    def __init__(
        self,
        user_repo: UserRepository,
        email_service: EmailService
    ):
        self.user_repo = user_repo
        self.email_service = email_service

    async def create_user(self, data: UserCreate) -> User:
        # Business rule: check uniqueness
        if await self.user_repo.exists_by_email(data.email):
            raise UserAlreadyExistsError(data.email)

        # Create user
        user = await self.user_repo.create(data)

        # Side effect: send welcome email
        await self.email_service.send_welcome(user.email)

        return user
```

### Repositories (Data Access)

**DO:**
- Abstract database operations
- Build queries
- Map results to domain objects
- Handle database-specific concerns

**DON'T:**
- Contain business logic
- Know about HTTP
- Call other repositories

```python
# ✅ Good repository
class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_id(self, user_id: int) -> User | None:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(self, data: UserCreate) -> User:
        user = User(**data.model_dump())
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
```

---

## Dependency Flow

```
┌────────────────┐
│    Settings    │  ← Loaded once at startup
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Database    │  ← Connection pool
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Repository   │  ← Per-request session
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Service     │  ← Stateless, injected repos
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Controller   │  ← Stateless, injected services
└────────────────┘
```

### Dependency Injection with FastAPI

```python
# src/api/deps.py
from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_session)]

def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)

UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]

def get_user_service(user_repo: UserRepoDep) -> UserService:
    return UserService(user_repo)

UserServiceDep = Annotated[UserService, Depends(get_user_service)]

def get_user_controller(user_service: UserServiceDep) -> UserController:
    return UserController(user_service)

UserControllerDep = Annotated[UserController, Depends(get_user_controller)]
```

---

## Project Structure

### Complete Structure

```
my-service/
├── pyproject.toml              # uv project config
├── uv.lock                     # Locked dependencies
├── README.md
├── .env                        # Local environment
├── .env.example                # Environment template
├── alembic/                    # Database migrations
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
├── src/
│   └── my_service/
│       ├── __init__.py
│       ├── main.py             # FastAPI app entry point
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py     # pydantic-settings
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── deps.py         # Dependency injection
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── router.py   # Main v1 router
│       │       └── routes/
│       │           ├── __init__.py
│       │           ├── users.py
│       │           └── health.py
│       │
│       ├── controllers/
│       │   ├── __init__.py
│       │   ├── base.py         # BaseController
│       │   └── user_controller.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── user_service.py
│       │
│       ├── repositories/
│       │   ├── __init__.py
│       │   ├── base.py         # BaseRepository
│       │   └── user_repository.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # SQLAlchemy Base
│       │   └── user.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── base.py         # Base schemas
│       │   └── user.py         # User Pydantic models
│       │
│       ├── middleware/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── auth.py
│       │
│       └── core/
│           ├── __init__.py
│           ├── exceptions.py   # Custom exceptions
│           ├── database.py     # Database setup
│           └── security.py     # Auth utilities
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_services/
    │   └── test_repositories/
    └── integration/
        ├── __init__.py
        └── test_api/
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, lifespan, middleware setup |
| `config/settings.py` | Environment configuration |
| `api/deps.py` | Dependency injection factories |
| `core/database.py` | SQLAlchemy engine and session |
| `core/exceptions.py` | Custom exception classes |
| `controllers/base.py` | BaseController class |
| `repositories/base.py` | BaseRepository class |
| `models/base.py` | SQLAlchemy declarative base |

---

## Best Practices Summary

1. **Single Responsibility**: Each layer does ONE thing
2. **Dependency Injection**: Always inject dependencies
3. **No Circular Imports**: Layers only depend downward
4. **Type Everything**: Full type hints throughout
5. **Async by Default**: Use async/await consistently
6. **Test at Each Layer**: Unit tests per layer, integration for flows
