# Database Patterns

## Table of Contents
- [SQLAlchemy Setup](#sqlalchemy-setup)
- [Model Definitions](#model-definitions)
- [Async Session Management](#async-session-management)
- [Repository Pattern](#repository-pattern)
- [Transactions](#transactions)
- [Query Patterns](#query-patterns)
- [Migrations with Alembic](#migrations-with-alembic)

---

## SQLAlchemy Setup

### Installation

```bash
uv add sqlalchemy asyncpg alembic
```

### Database Configuration

```python
# src/core/database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from ..config import settings


# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # Log SQL in debug mode
    pool_pre_ping=True,   # Check connections before use
    pool_size=5,
    max_overflow=10,
    # For testing, use NullPool
    # poolclass=NullPool,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

---

## Model Definitions

### Base Model

```python
# src/models/base.py
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""

    # Common columns
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
```

### Model with Relationships

```python
# src/models/user.py
from sqlalchemy import String, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    posts: Mapped[list["Post"]] = relationship(
        "Post",
        back_populates="author",
        lazy="selectin",
    )
    profile: Mapped["Profile"] = relationship(
        "Profile",
        back_populates="user",
        uselist=False,
    )


class Post(Base):
    __tablename__ = "posts"

    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(String)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # Relationships
    author: Mapped["User"] = relationship("User", back_populates="posts")


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    bio: Mapped[str | None] = mapped_column(String(500), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="profile")
```

### Model with Enum

```python
from enum import Enum as PyEnum

from sqlalchemy import Enum


class OrderStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Order(Base):
    __tablename__ = "orders"

    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus),
        default=OrderStatus.PENDING,
    )
```

---

## Async Session Management

### Session Dependency

```python
# src/api/deps.py
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import async_session_maker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_maker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

### Using in Routes

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    session: SessionDep,
) -> UserResponse:
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404)
    return UserResponse.model_validate(user)
```

---

## Repository Pattern

### Base Repository

```python
# src/repositories/base.py
from typing import TypeVar, Generic, Any

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Base repository with common operations."""

    model: type[T]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_by_id(self, id: int) -> T | None:
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[T]:
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def create(self, **data: Any) -> T:
        instance = self.model(**data)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, id: int, **data: Any) -> T | None:
        instance = await self.find_by_id(id)
        if not instance:
            return None

        for key, value in data.items():
            if value is not None:
                setattr(instance, key, value)

        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, id: int) -> bool:
        instance = await self.find_by_id(id)
        if not instance:
            return False

        await self.session.delete(instance)
        await self.session.flush()
        return True
```

### Concrete Repository

```python
# src/repositories/user_repository.py
from sqlalchemy import select, func

from .base import BaseRepository
from ..models.user import User


class UserRepository(BaseRepository[User]):
    model = User

    async def find_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def find_active(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        result = await self.session.execute(
            select(User)
            .where(User.is_active == True)
            .offset(skip)
            .limit(limit)
            .order_by(User.created_at.desc())
        )
        return list(result.scalars().all())

    async def search(self, query: str) -> list[User]:
        pattern = f"%{query}%"
        result = await self.session.execute(
            select(User).where(
                (User.name.ilike(pattern)) | (User.email.ilike(pattern))
            )
        )
        return list(result.scalars().all())
```

---

## Transactions

### Automatic Transaction (Recommended)

```python
# Session commits on successful completion
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()  # Auto-commit on success
        except Exception:
            await session.rollback()  # Rollback on error
            raise
```

### Explicit Transaction

```python
async def transfer_funds(
    session: AsyncSession,
    from_account: int,
    to_account: int,
    amount: Decimal,
) -> None:
    """Transfer with explicit transaction."""
    async with session.begin():
        # Deduct from source
        await session.execute(
            update(Account)
            .where(Account.id == from_account)
            .values(balance=Account.balance - amount)
        )

        # Add to destination
        await session.execute(
            update(Account)
            .where(Account.id == to_account)
            .values(balance=Account.balance + amount)
        )
    # Commits automatically when exiting context
```

### Savepoints (Nested Transactions)

```python
async def create_order_with_items(
    session: AsyncSession,
    order_data: OrderCreate,
) -> Order:
    """Create order with optional items."""
    async with session.begin():
        # Main transaction
        order = Order(**order_data.model_dump(exclude={"items"}))
        session.add(order)
        await session.flush()

        for item_data in order_data.items:
            try:
                async with session.begin_nested():
                    # Savepoint for each item
                    item = OrderItem(order_id=order.id, **item_data.model_dump())
                    session.add(item)
            except IntegrityError:
                # Savepoint rolled back, continue with other items
                continue

        return order
```

---

## Query Patterns

### Basic Queries

```python
from sqlalchemy import select, and_, or_, desc, asc


# Select all
result = await session.execute(select(User))
users = result.scalars().all()

# Select one
result = await session.execute(
    select(User).where(User.id == 1)
)
user = result.scalar_one_or_none()

# Filter with conditions
result = await session.execute(
    select(User).where(
        and_(
            User.is_active == True,
            User.created_at > some_date,
        )
    )
)

# OR conditions
result = await session.execute(
    select(User).where(
        or_(User.name == "Alice", User.name == "Bob")
    )
)

# Ordering
result = await session.execute(
    select(User).order_by(desc(User.created_at))
)

# Pagination
result = await session.execute(
    select(User).offset(10).limit(20)
)
```

### Joins and Relationships

```python
from sqlalchemy.orm import selectinload, joinedload


# Eager load relationships
result = await session.execute(
    select(User).options(selectinload(User.posts))
)

# Join query
result = await session.execute(
    select(User, Post)
    .join(Post, User.id == Post.author_id)
    .where(Post.title.ilike("%python%"))
)

# Subquery
subq = (
    select(Post.author_id)
    .where(Post.created_at > some_date)
    .distinct()
    .scalar_subquery()
)
result = await session.execute(
    select(User).where(User.id.in_(subq))
)
```

### Aggregations

```python
from sqlalchemy import func


# Count
result = await session.execute(
    select(func.count()).select_from(User)
)
count = result.scalar_one()

# Group by
result = await session.execute(
    select(User.is_active, func.count(User.id))
    .group_by(User.is_active)
)

# Having
result = await session.execute(
    select(Post.author_id, func.count(Post.id).label("post_count"))
    .group_by(Post.author_id)
    .having(func.count(Post.id) > 5)
)
```

---

## Migrations with Alembic

### Setup

```bash
# Initialize Alembic
uv run alembic init alembic
```

### Configure alembic/env.py

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from src.models.base import Base
from src.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Migration Commands

```bash
# Generate migration
uv run alembic revision --autogenerate -m "Add users table"

# Apply migrations
uv run alembic upgrade head

# Rollback one step
uv run alembic downgrade -1

# Show current revision
uv run alembic current

# Show history
uv run alembic history
```

---

## Best Practices

1. **Always use async**: `AsyncSession`, `create_async_engine`
2. **Expire on commit**: Set `expire_on_commit=False`
3. **Flush before refresh**: Call `flush()` before `refresh()`
4. **Use selectinload**: For eager loading collections
5. **Index foreign keys**: Always index FK columns
6. **Soft deletes**: Prefer `is_deleted` flag over hard deletes
7. **Migrations**: Always use Alembic for schema changes
