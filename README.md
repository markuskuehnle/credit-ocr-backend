# Development Setup

## DMS Mock Setup

Quick-start for the DMS mock:

```bash
docker compose up -d          # starts Postgres + Azurite
docker compose exec postgres psql -U dms -d dms_meta -f /schema/schema.sql
```

### Connection Details

**PostgreSQL:**
- Host: localhost:5432
- Database: dms_meta
- Username: dms
- Password: dms

**Azurite (Blob Storage):**
- Connection string: `DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;`
- Endpoint: http://localhost:10000

## Testing

### Optimized Test Runner

The project includes an optimized test runner that ensures proper cleanup and efficient test execution:

```bash
# Run all tests with proper cleanup
./run_tests.py

# Run specific test file
./run_tests.py tests/test_dms_mock.py

# Run with verbose output
./run_tests.py -v

# Run with coverage reporting
./run_tests.py --coverage

# Run tests matching a pattern
./run_tests.py "tests/test_*_mock.py"
```

### Test Categories

**DMS Mock Tests** (`tests/test_dms_mock.py`):
- Tests PostgreSQL and Azurite integration
- Uses `@pytest.mark.no_global_setup` to avoid Ollama startup
- Session-scoped containers for efficiency

**LLM Tests** (`tests/test_field_extraction.py`):
- Tests that require Ollama/LLM functionality
- Automatically starts Ollama container when needed
- Session-scoped for performance

**Other Tests**:
- Unit tests that don't require external services
- Fast execution without container startup

### Manual Test Execution

If you prefer to run tests manually:

```bash
# Run all tests
pytest tests/

# Run only DMS mock tests (no Ollama needed)
pytest tests/test_dms_mock.py

# Run only LLM tests
pytest tests/test_field_extraction.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=src tests/
```

### Container Management

The test setup automatically:
- Starts only the containers needed for each test category
- Uses session-scoped fixtures to avoid multiple container startups
- Ensures proper cleanup after tests complete
- Handles container cleanup on test interruption (Ctrl+C)

### Troubleshooting

If containers don't clean up properly:

```bash
# Stop all running containers
docker stop $(docker ps -q)

# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -f
```