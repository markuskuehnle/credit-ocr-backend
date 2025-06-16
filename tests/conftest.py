import pytest
import logging
import requests
from pathlib import Path
import json

from _pytest.config import Config as PytestConfig
from src.config import AppConfig
from tests.environment.environment import (
    setup_environment,
    teardown_environment,
)

logger = logging.getLogger(__name__)
app_config = AppConfig("tests/resources/test_application.conf")

MODEL_CHECK_CACHE = {"generative": None, "embedding": None}


@pytest.fixture(scope="session", autouse=True)
def setup():
    setup_environment()
    yield
    teardown_environment()


# Session finish hook to print available models
def pytest_sessionfinish(session: PytestConfig, exitstatus: int):
    """Print model status after tests."""
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