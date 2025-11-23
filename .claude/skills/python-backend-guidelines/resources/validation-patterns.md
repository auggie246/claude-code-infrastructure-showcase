# Validation Patterns

## Table of Contents
- [Pydantic Basics](#pydantic-basics)
- [Field Validation](#field-validation)
- [Custom Validators](#custom-validators)
- [Schema Patterns](#schema-patterns)
- [Request/Response Models](#requestresponse-models)
- [Serialization](#serialization)

---

## Pydantic Basics

### Simple Model

```python
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for creating a user."""
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    age: int | None = Field(None, ge=0, le=150)
```

### Model Configuration

```python
from pydantic import BaseModel, ConfigDict


class UserResponse(BaseModel):
    """Schema for user response."""

    model_config = ConfigDict(
        from_attributes=True,      # Allow ORM model conversion
        str_strip_whitespace=True, # Strip whitespace from strings
        str_min_length=1,          # Minimum string length
        validate_assignment=True,  # Validate on attribute assignment
        extra="forbid",            # Forbid extra fields
    )

    id: int
    email: str
    name: str
    is_active: bool
```

---

## Field Validation

### Built-in Constraints

```python
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal


class ProductCreate(BaseModel):
    # String constraints
    name: str = Field(..., min_length=1, max_length=200)
    sku: str = Field(..., pattern=r"^[A-Z]{3}-\d{4}$")
    description: str = Field(default="", max_length=5000)

    # Numeric constraints
    price: Decimal = Field(..., gt=0, decimal_places=2)
    quantity: int = Field(..., ge=0, le=10000)
    discount: float = Field(default=0.0, ge=0.0, le=1.0)

    # Date constraints
    available_from: datetime = Field(default_factory=datetime.utcnow)

    # List constraints
    tags: list[str] = Field(default_factory=list, max_length=10)
```

### Common Field Types

```python
from pydantic import (
    BaseModel,
    EmailStr,
    HttpUrl,
    IPvAnyAddress,
    SecretStr,
    constr,
    conint,
    confloat,
)
from uuid import UUID
from datetime import date, datetime, time
from enum import Enum


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class CompleteExample(BaseModel):
    # Email validation
    email: EmailStr

    # URL validation
    website: HttpUrl | None = None

    # IP address
    ip_address: IPvAnyAddress | None = None

    # Secret (hidden in logs/repr)
    password: SecretStr

    # UUID
    external_id: UUID

    # Constrained types
    username: constr(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    age: conint(ge=0, le=150)
    score: confloat(ge=0.0, le=100.0)

    # Enum
    status: Status = Status.PENDING

    # Dates and times
    birth_date: date
    created_at: datetime
    preferred_time: time | None = None
```

---

## Custom Validators

### Field Validators

```python
from pydantic import BaseModel, field_validator, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    password_confirm: str

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace")
        return v.strip().title()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        return v
```

### Model Validators

```python
from pydantic import BaseModel, model_validator


class DateRange(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_dates(self) -> "DateRange":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class UserCreate(BaseModel):
    password: str
    password_confirm: str

    @model_validator(mode="after")
    def passwords_match(self) -> "UserCreate":
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self
```

### Before Validators (Pre-processing)

```python
from pydantic import BaseModel, field_validator, BeforeValidator
from typing import Annotated


def normalize_email(v: str) -> str:
    return v.lower().strip()


# Using Annotated
NormalizedEmail = Annotated[str, BeforeValidator(normalize_email)]


class UserCreate(BaseModel):
    email: NormalizedEmail

    # Or using decorator
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower().strip()
        return v
```

---

## Schema Patterns

### Base Schema Pattern

```python
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime


class UserBase(BaseSchema):
    """Base user fields."""
    email: str
    name: str


class UserCreate(UserBase):
    """Schema for creating user."""
    password: str


class UserUpdate(BaseSchema):
    """Schema for updating user."""
    email: str | None = None
    name: str | None = None


class UserResponse(UserBase, TimestampMixin):
    """Schema for user response."""
    id: int
    is_active: bool
```

### Partial Update Schema

```python
from pydantic import BaseModel
from typing import Any


class UserUpdate(BaseModel):
    """All fields optional for partial update."""
    email: str | None = None
    name: str | None = None
    is_active: bool | None = None

    def get_update_dict(self) -> dict[str, Any]:
        """Get only set fields for update."""
        return self.model_dump(exclude_unset=True)
```

### Nested Schemas

```python
class AddressCreate(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class UserWithAddressCreate(BaseModel):
    email: EmailStr
    name: str
    address: AddressCreate


class AddressResponse(AddressCreate):
    id: int


class UserWithAddressResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    address: AddressResponse | None = None
```

---

## Request/Response Models

### Paginated Response

```python
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: list[T]
    total: int
    skip: int = Field(ge=0)
    limit: int = Field(ge=1, le=1000)
    has_more: bool

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        skip: int,
        limit: int,
    ) -> "PaginatedResponse[T]":
        return cls(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total,
        )


class UserList(PaginatedResponse[UserResponse]):
    """Paginated list of users."""
    pass
```

### Error Response

```python
class ErrorDetail(BaseModel):
    """Single error detail."""
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str | list[ErrorDetail]
    code: str | None = None


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    detail: list[ErrorDetail]
```

### API Response Wrapper

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response."""
    success: bool
    data: T | None = None
    error: str | None = None
    meta: dict | None = None


# Usage
class UserAPIResponse(APIResponse[UserResponse]):
    pass


class UserListAPIResponse(APIResponse[list[UserResponse]]):
    meta: dict = {"pagination": True}
```

---

## Serialization

### From ORM Models

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pydantic import BaseModel, ConfigDict


# SQLAlchemy Model
class User(DeclarativeBase):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str]
    name: Mapped[str]


# Pydantic Schema
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str


# Usage
user_orm = User(id=1, email="test@example.com", name="Test")
user_response = UserResponse.model_validate(user_orm)
```

### Custom Serialization

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime


class UserResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


class MoneyResponse(BaseModel):
    amount: Decimal
    currency: str

    @field_serializer("amount")
    def serialize_amount(self, value: Decimal) -> str:
        return f"{value:.2f}"
```

### Exclude Fields

```python
from pydantic import BaseModel, Field


class UserInternal(BaseModel):
    id: int
    email: str
    password_hash: str = Field(exclude=True)
    internal_notes: str = Field(exclude=True)


# Or at serialization time
user.model_dump(exclude={"password_hash", "internal_notes"})
user.model_dump(include={"id", "email"})
```

---

## FastAPI Integration

### Request Validation

```python
from fastapi import APIRouter, Query, Path, Body

router = APIRouter()


@router.get("/users")
async def list_users(
    skip: int = Query(0, ge=0, description="Number to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    search: str | None = Query(None, min_length=1, max_length=100),
) -> UserList:
    ...


@router.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., gt=0, description="User ID"),
) -> UserResponse:
    ...


@router.post("/users")
async def create_user(
    data: UserCreate = Body(..., description="User data"),
) -> UserResponse:
    ...
```

### Response Model

```python
@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_user(user_id: int) -> UserResponse:
    ...
```

---

## Best Practices

1. **Separate schemas**: Create, Update, Response for each entity
2. **Use ConfigDict**: Enable `from_attributes` for ORM support
3. **Validate early**: Use Pydantic at API boundaries
4. **Custom validators**: For complex business rules
5. **Type safety**: Use proper type hints everywhere
6. **Documentation**: Add Field descriptions for OpenAPI
7. **Reuse base schemas**: Reduce duplication with inheritance
