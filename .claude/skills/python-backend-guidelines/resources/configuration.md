# Configuration Management

## Table of Contents
- [pydantic-settings Overview](#pydantic-settings-overview)
- [Settings Class](#settings-class)
- [Environment Files](#environment-files)
- [Nested Configuration](#nested-configuration)
- [Secrets Management](#secrets-management)
- [Environment-Specific Config](#environment-specific-config)

---

## pydantic-settings Overview

### Installation

```bash
uv add pydantic-settings
```

### Why pydantic-settings?

- **Type-safe**: Full type validation
- **Environment variables**: Auto-load from environment
- **Validation**: Built-in Pydantic validation
- **IDE support**: Autocomplete and type checking
- **Dotenv support**: Load from `.env` files

---

## Settings Class

### Basic Settings

```python
# src/config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "my-service"
    debug: bool = False
    environment: str = "development"
    app_version: str = "1.0.0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost/db",
        description="Database connection URL",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Security
    secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Sentry
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1

    # External APIs
    external_api_url: str = ""
    external_api_key: str | None = None
    external_api_timeout: int = 30


# Singleton instance
settings = Settings()
```

### Usage

```python
from .config import settings

# Access settings
print(settings.database_url)
print(settings.debug)

# In FastAPI
from fastapi import FastAPI

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version=settings.app_version,
)
```

---

## Environment Files

### .env File

```env
# Application
APP_NAME=my-service
DEBUG=false
ENVIRONMENT=production

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:password@db.example.com:5432/mydb

# Redis
REDIS_URL=redis://redis.example.com:6379/0

# Security
SECRET_KEY=your-super-secret-key-at-least-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Sentry
SENTRY_DSN=https://xxx@sentry.io/xxx
SENTRY_TRACES_SAMPLE_RATE=0.1

# External APIs
EXTERNAL_API_URL=https://api.example.com
EXTERNAL_API_KEY=your-api-key
```

### .env.example (Template)

```env
# Application
APP_NAME=my-service
DEBUG=true
ENVIRONMENT=development

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/mydb

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=development-secret-key-change-in-production

# Sentry (optional in development)
SENTRY_DSN=

# External APIs
EXTERNAL_API_URL=
EXTERNAL_API_KEY=
```

### Multiple Environment Files

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),  # Load both, .env.local overrides
        env_file_encoding="utf-8",
    )
```

---

## Nested Configuration

### Nested Settings Classes

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Database configuration."""
    url: str = "postgresql+asyncpg://localhost/db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class RedisSettings(BaseModel):
    """Redis configuration."""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10
    decode_responses: bool = True


class SentrySettings(BaseModel):
    """Sentry configuration."""
    dsn: str | None = None
    traces_sample_rate: float = 0.1
    profiles_sample_rate: float = 0.1
    environment: str = "development"


class Settings(BaseSettings):
    """Main settings with nested configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",  # Use __ for nested keys
    )

    app_name: str = "my-service"
    debug: bool = False

    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    sentry: SentrySettings = SentrySettings()


settings = Settings()
```

### Environment Variables for Nested

```env
# Nested with __ delimiter
DATABASE__URL=postgresql+asyncpg://localhost/prod_db
DATABASE__POOL_SIZE=20
DATABASE__ECHO=false

REDIS__URL=redis://redis.example.com:6379/0
REDIS__MAX_CONNECTIONS=50

SENTRY__DSN=https://xxx@sentry.io/xxx
SENTRY__TRACES_SAMPLE_RATE=0.05
SENTRY__ENVIRONMENT=production
```

### Usage

```python
# Access nested settings
print(settings.database.url)
print(settings.redis.max_connections)
print(settings.sentry.dsn)
```

---

## Secrets Management

### Using SecretStr

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_password: SecretStr
    api_key: SecretStr
    jwt_secret: SecretStr


settings = Settings()

# SecretStr hides value in repr/str
print(settings.api_key)  # SecretStr('**********')

# Get actual value when needed
actual_key = settings.api_key.get_secret_value()
```

### Loading from Files

```python
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",  # Docker secrets directory
    )

    database_password: SecretStr
    api_key: SecretStr
```

With files:
```
/run/secrets/database_password  # Contains the password
/run/secrets/api_key            # Contains the API key
```

### AWS Secrets Manager / Vault

```python
import boto3
from functools import lru_cache


@lru_cache
def get_secret(secret_name: str) -> str:
    """Fetch secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]


class Settings(BaseSettings):
    @property
    def database_password(self) -> str:
        return get_secret("myapp/database/password")
```

---

## Environment-Specific Config

### Using Environment Detection

```python
from enum import Enum
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    environment: Environment = Environment.DEVELOPMENT

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def cors_origins(self) -> list[str]:
        if self.is_development:
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
        return ["https://myapp.com"]

    @property
    def log_level(self) -> str:
        if self.is_development:
            return "DEBUG"
        return "INFO"
```

### Factory Pattern

```python
from functools import lru_cache


class BaseSettings(BaseSettings):
    """Base settings for all environments."""
    app_name: str = "my-service"


class DevelopmentSettings(BaseSettings):
    """Development-specific settings."""
    debug: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(BaseSettings):
    """Production-specific settings."""
    debug: bool = False
    log_level: str = "INFO"


@lru_cache
def get_settings() -> BaseSettings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development")

    settings_map = {
        "development": DevelopmentSettings,
        "production": ProductionSettings,
    }

    settings_class = settings_map.get(env, DevelopmentSettings)
    return settings_class()


settings = get_settings()
```

---

## FastAPI Integration

### Settings Dependency

```python
from functools import lru_cache
from fastapi import Depends
from typing import Annotated


@lru_cache
def get_settings() -> Settings:
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/info")
async def app_info(settings: SettingsDep) -> dict:
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "version": settings.app_version,
    }
```

### Configuration in Lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Use settings
    if settings.sentry.dsn:
        init_sentry(settings.sentry)

    yield

    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)
```

---

## Best Practices

1. **Never hardcode**: Always use settings for config values
2. **Use .env.example**: Document required variables
3. **Validate early**: Settings are validated at startup
4. **Use SecretStr**: For sensitive values
5. **Cache settings**: Use `@lru_cache` for singleton
6. **Type everything**: Full type hints for IDE support
7. **Default sensibly**: Safe defaults for development
8. **Never commit .env**: Add to .gitignore

### .gitignore

```gitignore
# Environment files
.env
.env.local
.env.*.local

# Keep example
!.env.example
```

---

## Quick Reference

| Task | Code |
|------|------|
| Load from env | `model_config = SettingsConfigDict(env_file=".env")` |
| Nested config | `env_nested_delimiter="__"` |
| Secret value | `SecretStr`, `.get_secret_value()` |
| Required field | `field: str = Field(...)` |
| With default | `field: str = "default"` |
| Validation | `Field(..., min_length=32)` |
| Environment check | `settings.environment == "production"` |
