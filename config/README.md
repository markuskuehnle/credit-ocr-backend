# Configuration Files

This directory contains all configuration files for the credit OCR demo backend application.

## Configuration Files

### `application.conf`
Main application configuration including:
- **generative_llm**: Ollama LLM service settings (URL, model name)
- **app**: Application-level settings (debug, log level, timezone)

### `database.conf`
Database configuration for PostgreSQL:
- Connection settings (host, port, database name, user, password)
- Connection pool settings

### `redis.conf`
Redis configuration for Celery task queue:
- Redis connection settings
- Celery-specific configuration

### `azure.conf`
Azure services configuration:
- **storage**: Azure Blob Storage settings (using Azurite for local development)
- **form_recognizer**: Azure Form Recognizer settings (for production OCR)

### `document_types.conf`
Document type definitions and field mappings (unchanged as requested):
- Expected fields for each document type
- Field descriptions and validation rules
- Field mappings for OCR extraction

## Architecture

The application uses the following services:

1. **Ollama** (port 11435) - LLM service for field extraction
2. **Azurite** (port 10000) - Azure Storage emulator for blob storage
3. **PostgreSQL** (port 5432) - Database for metadata
4. **Redis** (port 6379) - Message broker for Celery tasks

All services are defined in `docker-compose.yml` for easy local development.

## Usage

The application loads configuration using the `AppConfig` class:

```python
from src.config import AppConfig

# Load configuration from config directory
app_config = AppConfig("config")

# Access configuration values
llm_url = app_config.generative_llm.url
db_host = app_config.database.host
redis_url = app_config.redis.broker_url
azure_storage = app_config.azure.storage.connection_string
```

## Environment Overrides

For production deployments, you can override configuration values using environment variables. The configuration system will check for environment variables before falling back to the config files.

## Testing

For testing, the application uses the same configuration files but may override specific values through the test environment setup.

## Storage

The application uses Azure Blob Storage with Azurite for local development. Azurite provides a local emulator for Azure Storage services, making it easy to develop and test without requiring a cloud Azure account.

## LLM Service

The application uses Ollama for LLM inference. Ollama runs locally and can host various open-source models like llama3.1:8b for field extraction from documents. 