import pytest
import logging
import requests
from pathlib import Path
import json
import atexit

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


@pytest.fixture(scope="session")
def dms_mock_environment():
    """Provide DMS mock environment for tests that need it."""
    from src.dms_mock.environment import DmsMockEnvironment
    
    logger.info("Starting DMS mock environment")
    dms_environment = DmsMockEnvironment()
    dms_environment.start()
    _active_containers.extend([dms_environment.postgres_container, dms_environment.azurite_container])
    
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