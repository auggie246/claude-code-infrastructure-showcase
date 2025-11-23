# Services and Repositories

## Table of Contents
- [Service Layer](#service-layer)
- [Repository Pattern](#repository-pattern)
- [Dependency Injection](#dependency-injection)
- [Transaction Management](#transaction-management)
- [Caching Patterns](#caching-patterns)
- [Service Composition](#service-composition)

---

## Service Layer

### Service Responsibilities

Services contain **business logic**:
- Business rules and validation
- Orchestration of multiple repositories
- Transaction boundaries
- External service integration
- Event emission

### Base Service

```python
# src/services/base.py
from abc import ABC
import logging
from typing import TypeVar, Generic

import sentry_sdk

T = TypeVar("T")


class BaseService(ABC, Generic[T]):
    """Base service with common functionality."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_operation(self, operation: str, **kwargs) -> None:
        """Log a service operation."""
        self.logger.info(
            "Service operation: %s",
            operation,
            extra=kwargs,
        )

    def capture_error(self, error: Exception, operation: str) -> None:
        """Capture error to Sentry with context."""
        sentry_sdk.set_context("operation", {"name": operation})
        sentry_sdk.capture_exception(error)
```

### Concrete Service

```python
# src/services/user_service.py
from .base import BaseService
from ..repositories.user_repository import UserRepository
from ..schemas.user import UserCreate, UserUpdate
from ..models.user import User
from ..core.exceptions import UserNotFoundError, UserAlreadyExistsError


class UserService(BaseService[User]):
    """Service for user business logic."""

    def __init__(self, user_repo: UserRepository) -> None:
        super().__init__()
        self.user_repo = user_repo

    async def create_user(self, data: UserCreate) -> User:
        """Create a new user with business rules."""
        self.log_operation("create_user", email=data.email)

        # Business rule: Check email uniqueness
        existing = await self.user_repo.find_by_email(data.email)
        if existing:
            raise UserAlreadyExistsError(f"Email {data.email} already registered")

        # Business rule: Normalize email
        normalized_data = UserCreate(
            email=data.email.lower().strip(),
            name=data.name.strip(),
        )

        return await self.user_repo.create(normalized_data)

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        self.log_operation("get_user", user_id=user_id)
        return await self.user_repo.find_by_id(user_id)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        return await self.user_repo.find_by_email(email.lower())

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        *,
        active_only: bool = True,
    ) -> tuple[list[User], int]:
        """List users with pagination."""
        self.log_operation("list_users", skip=skip, limit=limit)

        if active_only:
            users = await self.user_repo.find_active(skip=skip, limit=limit)
            total = await self.user_repo.count_active()
        else:
            users = await self.user_repo.find_all(skip=skip, limit=limit)
            total = await self.user_repo.count()

        return users, total

    async def update_user(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> User | None:
        """Update user with business rules."""
        self.log_operation("update_user", user_id=user_id)

        user = await self.user_repo.find_by_id(user_id)
        if not user:
            return None

        # Business rule: If email changes, check uniqueness
        if data.email and data.email != user.email:
            existing = await self.user_repo.find_by_email(data.email)
            if existing:
                raise UserAlreadyExistsError(
                    f"Email {data.email} already registered"
                )

        return await self.user_repo.update(user_id, data)

    async def delete_user(self, user_id: int) -> bool:
        """Soft delete a user."""
        self.log_operation("delete_user", user_id=user_id)
        return await self.user_repo.soft_delete(user_id)

    async def activate_user(self, user_id: int) -> User | None:
        """Activate a user account."""
        return await self.user_repo.update(
            user_id,
            UserUpdate(is_active=True),
        )

    async def deactivate_user(self, user_id: int) -> User | None:
        """Deactivate a user account."""
        return await self.user_repo.update(
            user_id,
            UserUpdate(is_active=False),
        )
```

---

## Repository Pattern

### Repository Responsibilities

Repositories handle **data access**:
- Database queries
- Result mapping
- Query building
- No business logic

### Base Repository

```python
# src/repositories/base.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(ABC, Generic[T]):
    """Base repository with common CRUD operations."""

    model: type[T]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_by_id(self, id: int) -> T | None:
        """Find entity by ID."""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[T]:
        """Find all entities with pagination."""
        result = await self.session.execute(
            select(self.model)
            .offset(skip)
            .limit(limit)
            .order_by(self.model.id)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Count all entities."""
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def create(self, data: Any) -> T:
        """Create a new entity."""
        entity = self.model(**data.model_dump())
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def update(self, id: int, data: Any) -> T | None:
        """Update an entity."""
        entity = await self.find_by_id(id)
        if not entity:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(entity, field, value)

        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, id: int) -> bool:
        """Hard delete an entity."""
        entity = await self.find_by_id(id)
        if not entity:
            return False

        await self.session.delete(entity)
        await self.session.commit()
        return True

    async def exists(self, id: int) -> bool:
        """Check if entity exists."""
        result = await self.session.execute(
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == id)
        )
        return result.scalar_one() > 0
```

### Concrete Repository

```python
# src/repositories/user_repository.py
from sqlalchemy import select, func

from .base import BaseRepository
from ..models.user import User
from ..schemas.user import UserCreate, UserUpdate


class UserRepository(BaseRepository[User]):
    """Repository for User data access."""

    model = User

    async def find_by_email(self, email: str) -> User | None:
        """Find user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def exists_by_email(self, email: str) -> bool:
        """Check if email exists."""
        result = await self.session.execute(
            select(func.count())
            .select_from(User)
            .where(User.email == email)
        )
        return result.scalar_one() > 0

    async def find_active(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Find all active users."""
        result = await self.session.execute(
            select(User)
            .where(User.is_active == True)
            .offset(skip)
            .limit(limit)
            .order_by(User.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_active(self) -> int:
        """Count active users."""
        result = await self.session.execute(
            select(func.count())
            .select_from(User)
            .where(User.is_active == True)
        )
        return result.scalar_one()

    async def soft_delete(self, user_id: int) -> bool:
        """Soft delete by setting is_active to False."""
        user = await self.find_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        await self.session.commit()
        return True

    async def search(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Search users by name or email."""
        search_pattern = f"%{query}%"
        result = await self.session.execute(
            select(User)
            .where(
                (User.name.ilike(search_pattern))
                | (User.email.ilike(search_pattern))
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
```

---

## Dependency Injection

### FastAPI Depends Pattern

```python
# src/api/deps.py
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import async_session_maker
from ..repositories.user_repository import UserRepository
from ..repositories.post_repository import PostRepository
from ..services.user_service import UserService
from ..services.post_service import PostService


# Session dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


# Repository dependencies
def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


def get_post_repository(session: SessionDep) -> PostRepository:
    return PostRepository(session)


UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]
PostRepoDep = Annotated[PostRepository, Depends(get_post_repository)]


# Service dependencies
def get_user_service(user_repo: UserRepoDep) -> UserService:
    return UserService(user_repo)


def get_post_service(
    post_repo: PostRepoDep,
    user_repo: UserRepoDep,
) -> PostService:
    return PostService(post_repo, user_repo)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]
PostServiceDep = Annotated[PostService, Depends(get_post_service)]
```

---

## Transaction Management

### Automatic Transactions (Recommended)

```python
# Session commits automatically with context manager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
        # Commit happens here if no exception
```

### Manual Transaction Control

```python
class OrderService(BaseService[Order]):
    def __init__(
        self,
        session: AsyncSession,
        order_repo: OrderRepository,
        inventory_repo: InventoryRepository,
    ) -> None:
        super().__init__()
        self.session = session
        self.order_repo = order_repo
        self.inventory_repo = inventory_repo

    async def create_order(self, data: OrderCreate) -> Order:
        """Create order with inventory update in transaction."""
        try:
            # Start transaction
            async with self.session.begin():
                # Check inventory
                item = await self.inventory_repo.find_by_id(data.item_id)
                if item.quantity < data.quantity:
                    raise InsufficientInventoryError()

                # Create order
                order = await self.order_repo.create(data)

                # Update inventory
                await self.inventory_repo.decrease_quantity(
                    data.item_id,
                    data.quantity,
                )

                return order
            # Commit happens here
        except Exception as e:
            # Rollback happens automatically
            self.capture_error(e, "create_order")
            raise
```

### Nested Transactions (Savepoints)

```python
async def complex_operation(self) -> None:
    async with self.session.begin():
        # Main transaction
        await self.do_step_1()

        try:
            async with self.session.begin_nested():
                # Savepoint
                await self.do_optional_step()
        except OptionalStepError:
            # Savepoint rolled back, main continues
            pass

        await self.do_step_2()
    # Main transaction commits
```

---

## Caching Patterns

### Simple In-Memory Cache

```python
from functools import lru_cache
from datetime import datetime, timedelta


class CachedUserService(UserService):
    """User service with caching."""

    def __init__(self, user_repo: UserRepository) -> None:
        super().__init__(user_repo)
        self._cache: dict[int, tuple[User, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user with caching."""
        # Check cache
        if user_id in self._cache:
            user, cached_at = self._cache[user_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return user

        # Cache miss - fetch from DB
        user = await self.user_repo.find_by_id(user_id)
        if user:
            self._cache[user_id] = (user, datetime.utcnow())

        return user

    def invalidate_cache(self, user_id: int) -> None:
        """Invalidate cache for user."""
        self._cache.pop(user_id, None)
```

### Redis Cache

```python
import json
from redis.asyncio import Redis


class RedisCachedUserService(UserService):
    """User service with Redis caching."""

    def __init__(
        self,
        user_repo: UserRepository,
        redis: Redis,
    ) -> None:
        super().__init__(user_repo)
        self.redis = redis
        self.cache_ttl = 300  # 5 minutes

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user with Redis caching."""
        cache_key = f"user:{user_id}"

        # Try cache
        cached = await self.redis.get(cache_key)
        if cached:
            return User(**json.loads(cached))

        # Cache miss
        user = await self.user_repo.find_by_id(user_id)
        if user:
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(user.to_dict()),
            )

        return user

    async def invalidate_cache(self, user_id: int) -> None:
        """Invalidate Redis cache."""
        await self.redis.delete(f"user:{user_id}")
```

---

## Service Composition

### Service Orchestration

```python
class OrderFulfillmentService(BaseService):
    """Service that orchestrates multiple services."""

    def __init__(
        self,
        order_service: OrderService,
        inventory_service: InventoryService,
        payment_service: PaymentService,
        notification_service: NotificationService,
    ) -> None:
        super().__init__()
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.notification_service = notification_service

    async def fulfill_order(self, order_id: int) -> OrderResult:
        """Orchestrate order fulfillment."""
        self.log_operation("fulfill_order", order_id=order_id)

        # Step 1: Get order
        order = await self.order_service.get_by_id(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        # Step 2: Reserve inventory
        await self.inventory_service.reserve(
            order.item_id,
            order.quantity,
        )

        try:
            # Step 3: Process payment
            payment = await self.payment_service.charge(
                order.user_id,
                order.total,
            )

            # Step 4: Confirm inventory deduction
            await self.inventory_service.confirm_reservation(
                order.item_id,
                order.quantity,
            )

            # Step 5: Update order status
            await self.order_service.mark_fulfilled(order_id)

            # Step 6: Send notification
            await self.notification_service.send_order_confirmation(
                order.user_id,
                order_id,
            )

            return OrderResult(success=True, order=order)

        except PaymentError as e:
            # Rollback inventory reservation
            await self.inventory_service.release_reservation(
                order.item_id,
                order.quantity,
            )
            raise
```

---

## Best Practices

1. **Services own business logic**: Never in repositories
2. **Repositories are dumb**: Only data access, no decisions
3. **Use dependency injection**: Never instantiate in constructors
4. **Transaction boundaries in services**: Not repositories
5. **Cache at service layer**: Repositories don't know about caching
6. **Compose services**: Don't call repos from other repos
7. **Log in services**: Repositories are silent
