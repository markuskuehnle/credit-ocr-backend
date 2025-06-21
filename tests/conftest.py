import pytest
import logging
import requests
from pathlib import Path
import json
import atexit
import time
import os
import re

from _pytest.config import Config as PytestConfig
from src.config import AppConfig
from tests.environment.environment import (
    setup_environment,
    teardown_environment,
)

logger = logging.getLogger(__name__)
app_config = AppConfig("tests/resources/test_application.conf")

MODEL_CHECK_CACHE = {"generative": None, "embedding": None}

# Global container tracking for cleanup
_active_containers = []


def cleanup_all_containers():
    """Clean up all active containers on exit."""
    for container in _active_containers:
        try:
            if container and hasattr(container, 'stop'):
                # Check if container is still running before trying to stop it
                if hasattr(container, '_container') and container._container:
                    try:
                        container._container.reload()
                        if container._container.status == 'running':
                            container.stop()
                            logger.info(f"Stopped container: {container}")
                        else:
                            logger.info(f"Container already stopped: {container}")
                    except Exception:
                        # Container might not exist anymore, which is fine
                        logger.debug(f"Container no longer exists: {container}")
                else:
                    logger.debug(f"Container object invalid: {container}")
        except Exception as e:
            # Only log as warning if it's not a "not found" error
            if "404" not in str(e) and "Not Found" not in str(e):
                logger.warning(f"Failed to stop container {container}: {e}")
            else:
                logger.debug(f"Container already removed: {container}")
    
    # Clear the list after cleanup
    _active_containers.clear()


# Register cleanup function to run on exit
atexit.register(cleanup_all_containers)


def get_azurite_port_from_env() -> int:
    """Extract Azurite port from AZURE_STORAGE_CONNECTION_STRING environment variable."""
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    m = re.search(r"BlobEndpoint=http://localhost:(\d+)/", conn_str)
    return int(m.group(1)) if m else 10000


def wait_for_azurite_ready(max_retries: int = 30, delay: float = 1.0) -> bool:
    """Wait for Azurite to be ready by checking the blob service endpoint."""
    logger.info("Waiting for Azurite to be ready...")
    
    # Get the port from the environment variable set by DMS mock
    port = get_azurite_port_from_env()
    logger.info(f"Checking Azurite at port: {port}")
    
    # Try different endpoints
    endpoints = [
        f"http://localhost:{port}/devstoreaccount1",
        f"http://localhost:{port}/devstoreaccount1?restype=account",
        f"http://localhost:{port}/"
    ]
    
    for attempt in range(max_retries):
        for endpoint in endpoints:
            try:
                logger.debug(f"Attempt {attempt + 1}: Checking {endpoint}")
                response = requests.get(endpoint, timeout=5)
                logger.info(f"Azurite responded with status {response.status_code} from {endpoint}")
                # Accept any 2xx or 4xx status (4xx means service is up but endpoint not found)
                if 200 <= response.status_code < 500:
                    logger.info(f"Azurite is ready after {attempt + 1} attempts")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Request failed for {endpoint}: {e}")
                continue
        
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    logger.error(f"Azurite failed to become ready after {max_retries} attempts")
    return False


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    """Global test setup - only runs for tests that need Ollama."""
    # Skip global setup for tests marked with no_global_setup
    if request.node.get_closest_marker("no_global_setup"):
        yield
        return
    
    logger.info("Starting global test environment (Ollama)")
    ollama_container = setup_environment()
    _active_containers.append(ollama_container)
    
    yield
    
    logger.info("Cleaning up global test environment")
    teardown_environment()
    if ollama_container in _active_containers:
        _active_containers.remove(ollama_container)


@pytest.fixture(scope="session", autouse=True)
def dms_mock_environment():
    """Provide DMS mock environment for all tests (includes Azurite for blob storage)."""
    from src.dms_mock.environment import DmsMockEnvironment
    
    logger.info("Starting DMS mock environment (Postgres + Azurite)")
    dms_environment = DmsMockEnvironment()
    dms_environment.start()
    _active_containers.extend([dms_environment.postgres_container, dms_environment.azurite_container])
    
    # Wait for Azurite to be ready before creating containers
    if not wait_for_azurite_ready():
        raise RuntimeError("Azurite failed to start properly")
    
    # Ensure credit-docs container is created in Azurite
    logger.info("Initializing credit-docs containers in Azurite")
    from src.creditsystem.storage import ensure_all_credit_docs_containers
    ensure_all_credit_docs_containers()
    
    yield dms_environment
    
    logger.info("Cleaning up DMS mock environment")
    dms_environment.stop()
    # Remove from active containers list
    if dms_environment.postgres_container in _active_containers:
        _active_containers.remove(dms_environment.postgres_container)
    if dms_environment.azurite_container in _active_containers:
        _active_containers.remove(dms_environment.azurite_container)


# Session finish hook to print available models
def pytest_sessionfinish(session: PytestConfig, exitstatus: int):
    """Print model status after tests and ensure cleanup."""
    if hasattr(session, 'app_config'):
        logger.info("\n\n=== Model Status Summary ===")
        logger.info(f"Generative LLM URL: {session.app_config.generative_llm.url}")
        logger.info(f"Generative LLM Model: {session.app_config.generative_llm.model_name}")

        def log_model_status(base_url: str, model_type: str, model_name: str):
            try:
                res = requests.get(f"{base_url}/api/tags", timeout=3)
                if res.status_code == 200:
                    tags = res.json().get("models", [])
                    available = [m.get("name") for m in tags]
                    logger.info(
                        f"[{model_type.capitalize()}] Available models: {', '.join(available) or 'None'}"
                    )

                    is_loaded = model_name in available
                    MODEL_CHECK_CACHE[model_type] = is_loaded
                    logger.info(
                        f"[{model_type.capitalize()}] Required model '{model_name}': {'LOADED' if is_loaded else 'NOT LOADED'}"
                    )
                else:
                    logger.warning(
                        f"[{model_type.capitalize()}] Could not fetch models (status {res.status_code})"
                    )
            except Exception as e:
                logger.warning(
                    f"[{model_type.capitalize()}] Failed to query tags endpoint: {e}"
                )
                MODEL_CHECK_CACHE[model_type] = False

        log_model_status(
            session.app_config.generative_llm.url,
            "generative",
            session.app_config.generative_llm.model_name,
        )
    
    # Final cleanup
    cleanup_all_containers()


@pytest.fixture(scope="session")
def app_config():
    """Provide application configuration for tests."""
    config_path = Path("tests/resources/test_application.conf")
    return AppConfig(str(config_path))


@pytest.fixture(scope="session")
def document_config():
    """Load document configuration for testing."""
    config_path = Path("config/document_types.conf")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)