# Testing Guide

## Table of Contents
- [Testing Setup](#testing-setup)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [API Testing](#api-testing)
- [Fixtures](#fixtures)
- [Mocking](#mocking)
- [Test Organization](#test-organization)

---

## Testing Setup

### Installation

```bash
uv add --dev pytest pytest-asyncio pytest-cov httpx
```

### pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "-v --tb=short --strict-markers"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── conftest.py          # Unit test fixtures
│   ├── test_services/
│   │   ├── __init__.py
│   │   └── test_user_service.py
│   └── test_repositories/
│       ├── __init__.py
│       └── test_user_repository.py
└── integration/
    ├── __init__.py
    ├── conftest.py          # Integration fixtures
    └── test_api/
        ├── __init__.py
        └── test_users.py
```

---

## Unit Tests

### Testing Services

```python
# tests/unit/test_services/test_user_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.user_service import UserService
from src.schemas.user import UserCreate
from src.models.user import User
from src.core.exceptions import UserAlreadyExistsError


@pytest.fixture
def mock_user_repo() -> AsyncMock:
    """Mock user repository."""
    return AsyncMock()


@pytest.fixture
def user_service(mock_user_repo: AsyncMock) -> UserService:
    """User service with mocked dependencies."""
    return UserService(user_repo=mock_user_repo)


class TestUserService:
    """Tests for UserService."""

    async def test_create_user_success(
        self,
        user_service: UserService,
        mock_user_repo: AsyncMock,
    ) -> None:
        """Test successful user creation."""
        # Arrange
        user_data = UserCreate(email="test@example.com", name="Test User")
        expected_user = User(id=1, email="test@example.com", name="Test User")

        mock_user_repo.find_by_email.return_value = None
        mock_user_repo.create.return_value = expected_user

        # Act
        result = await user_service.create_user(user_data)

        # Assert
        assert result.id == 1
        assert result.email == "test@example.com"
        mock_user_repo.find_by_email.assert_awaited_once_with("test@example.com")
        mock_user_repo.create.assert_awaited_once()

    async def test_create_user_already_exists(
        self,
        user_service: UserService,
        mock_user_repo: AsyncMock,
    ) -> None:
        """Test user creation with existing email."""
        # Arrange
        user_data = UserCreate(email="existing@example.com", name="Test")
        mock_user_repo.find_by_email.return_value = User(id=1, email="existing@example.com")

        # Act & Assert
        with pytest.raises(UserAlreadyExistsError):
            await user_service.create_user(user_data)

    async def test_get_by_id_found(
        self,
        user_service: UserService,
        mock_user_repo: AsyncMock,
    ) -> None:
        """Test getting user by ID when found."""
        expected_user = User(id=1, email="test@example.com", name="Test")
        mock_user_repo.find_by_id.return_value = expected_user

        result = await user_service.get_by_id(1)

        assert result == expected_user
        mock_user_repo.find_by_id.assert_awaited_once_with(1)

    async def test_get_by_id_not_found(
        self,
        user_service: UserService,
        mock_user_repo: AsyncMock,
    ) -> None:
        """Test getting user by ID when not found."""
        mock_user_repo.find_by_id.return_value = None

        result = await user_service.get_by_id(999)

        assert result is None
```

### Testing Repositories

```python
# tests/unit/test_repositories/test_user_repository.py
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.user_repository import UserRepository
from src.models.user import User


@pytest.fixture
async def user_repository(
    db_session: AsyncSession,
) -> UserRepository:
    """User repository with test session."""
    return UserRepository(db_session)


class TestUserRepository:
    """Tests for UserRepository."""

    async def test_create(
        self,
        user_repository: UserRepository,
    ) -> None:
        """Test creating a user."""
        user = await user_repository.create(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )

        assert user.id is not None
        assert user.email == "test@example.com"

    async def test_find_by_email(
        self,
        user_repository: UserRepository,
    ) -> None:
        """Test finding user by email."""
        # Create user first
        await user_repository.create(
            email="find@example.com",
            name="Find Me",
            hashed_password="hashed",
        )

        # Find by email
        found = await user_repository.find_by_email("find@example.com")

        assert found is not None
        assert found.email == "find@example.com"

    async def test_find_by_email_not_found(
        self,
        user_repository: UserRepository,
    ) -> None:
        """Test finding non-existent user."""
        found = await user_repository.find_by_email("nonexistent@example.com")
        assert found is None
```

---

## Integration Tests

### API Integration Tests

```python
# tests/integration/test_api/test_users.py
import pytest
from httpx import AsyncClient

from src.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestUsersAPI:
    """Integration tests for users API."""

    async def test_create_user(self, client: AsyncClient) -> None:
        """Test POST /api/v1/users."""
        response = await client.post(
            "/api/v1/users",
            json={
                "email": "newuser@example.com",
                "name": "New User",
                "password": "SecurePass123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["name"] == "New User"
        assert "id" in data
        assert "password" not in data

    async def test_create_user_invalid_email(
        self,
        client: AsyncClient,
    ) -> None:
        """Test validation error for invalid email."""
        response = await client.post(
            "/api/v1/users",
            json={
                "email": "not-an-email",
                "name": "Test",
                "password": "pass",
            },
        )

        assert response.status_code == 422

    async def test_get_user(self, client: AsyncClient) -> None:
        """Test GET /api/v1/users/{user_id}."""
        # Create user first
        create_response = await client.post(
            "/api/v1/users",
            json={
                "email": "getuser@example.com",
                "name": "Get User",
                "password": "SecurePass123",
            },
        )
        user_id = create_response.json()["id"]

        # Get user
        response = await client.get(f"/api/v1/users/{user_id}")

        assert response.status_code == 200
        assert response.json()["id"] == user_id

    async def test_get_user_not_found(
        self,
        client: AsyncClient,
    ) -> None:
        """Test 404 for non-existent user."""
        response = await client.get("/api/v1/users/99999")
        assert response.status_code == 404

    async def test_list_users(self, client: AsyncClient) -> None:
        """Test GET /api/v1/users with pagination."""
        response = await client.get(
            "/api/v1/users",
            params={"skip": 0, "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)
```

---

## API Testing

### TestClient Setup

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.core.database import get_session
from src.models.base import Base


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session scope."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncSession:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncClient:
    """Create test client with overridden dependencies."""

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()
```

### Testing with Authentication

```python
@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    """Get authentication headers."""
    # Create test user
    await client.post(
        "/api/v1/users",
        json={
            "email": "auth@example.com",
            "password": "SecurePass123",
        },
    )

    # Login
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "auth@example.com",
            "password": "SecurePass123",
        },
    )
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


async def test_protected_endpoint(
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    """Test accessing protected endpoint."""
    response = await client.get(
        "/api/v1/users/me",
        headers=auth_headers,
    )
    assert response.status_code == 200
```

---

## Fixtures

### Common Fixtures

```python
# tests/conftest.py
import pytest
from datetime import datetime


@pytest.fixture
def sample_user_data() -> dict:
    """Sample user data for tests."""
    return {
        "email": "sample@example.com",
        "name": "Sample User",
        "password": "SecurePass123",
    }


@pytest.fixture
def sample_user() -> User:
    """Sample user model."""
    return User(
        id=1,
        email="sample@example.com",
        name="Sample User",
        is_active=True,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
async def created_user(
    client: AsyncClient,
    sample_user_data: dict,
) -> dict:
    """Create a user and return response data."""
    response = await client.post(
        "/api/v1/users",
        json=sample_user_data,
    )
    return response.json()
```

### Factory Fixtures

```python
import factory
from factory.alchemy import SQLAlchemyModelFactory

from src.models.user import User


class UserFactory(SQLAlchemyModelFactory):
    """Factory for creating test users."""

    class Meta:
        model = User
        sqlalchemy_session_persistence = "commit"

    email = factory.Sequence(lambda n: f"user{n}@example.com")
    name = factory.Faker("name")
    hashed_password = "hashed_password"
    is_active = True


@pytest.fixture
def user_factory(db_session: AsyncSession):
    """User factory with session."""
    UserFactory._meta.sqlalchemy_session = db_session
    return UserFactory
```

---

## Mocking

### Mocking External Services

```python
from unittest.mock import AsyncMock, patch


async def test_send_notification(
    user_service: UserService,
) -> None:
    """Test with mocked notification service."""
    with patch(
        "src.services.notification_service.send_email",
        new_callable=AsyncMock,
    ) as mock_send:
        mock_send.return_value = True

        await user_service.create_and_notify(user_data)

        mock_send.assert_awaited_once()


async def test_external_api_call() -> None:
    """Test with mocked HTTP client."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": "test"},
        )

        result = await fetch_external_data()

        assert result == {"data": "test"}
```

### Mocking Database

```python
@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session
```

---

## Test Organization

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_services/test_user_service.py

# Run specific test
uv run pytest tests/unit/test_services/test_user_service.py::TestUserService::test_create_user_success

# Run by marker
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not slow"

# Run with verbose output
uv run pytest -v

# Run and stop on first failure
uv run pytest -x
```

### Test Markers

```python
import pytest


@pytest.mark.unit
async def test_unit_example() -> None:
    """Unit test."""
    pass


@pytest.mark.integration
async def test_integration_example() -> None:
    """Integration test."""
    pass


@pytest.mark.slow
async def test_slow_example() -> None:
    """Slow test."""
    pass


@pytest.mark.skip(reason="Not implemented yet")
async def test_skip_example() -> None:
    pass


@pytest.mark.parametrize(
    "email,expected",
    [
        ("valid@example.com", True),
        ("invalid", False),
        ("", False),
    ],
)
async def test_validate_email(email: str, expected: bool) -> None:
    """Parametrized test."""
    assert validate_email(email) == expected
```

---

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Mock at boundaries**: Mock external services, not internals
4. **Use fixtures**: Share setup across tests
5. **Test edge cases**: Empty inputs, nulls, large data
6. **Integration tests**: Test full request flow
7. **CI/CD**: Run tests on every commit
8. **Coverage**: Aim for 80%+ coverage
