import pytest
import logging
import requests
from pathlib import Path
import json
import atexit
import time
import os
import re
import uuid
from testcontainers.redis import RedisContainer
from testcontainers.core.container import DockerContainer
from src.dms_mock.environment import DmsMockEnvironment
from tests.environment.ollama import start_ollama
import subprocess

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
_global_dms_environment = None  # Store the global DMS environment


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
                        
                        # Remove the container after stopping
                        try:
                            container._container.remove()
                            logger.info(f"Removed container: {container}")
                        except Exception as e:
                            logger.debug(f"Could not remove container {container}: {e}")
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


def _upload_test_documents():
    """Upload test documents to blob storage for testing."""
    from pathlib import Path
    from src.creditsystem.storage import get_storage, Stage
    import json
    import uuid
    
    storage = get_storage()
    tmp_dir = Path("tests/tmp")
    
    # Generate test document IDs dynamically
    test_documents = [
        (str(uuid.uuid4()), "sample_creditrequest.pdf"),  # test-doc-complete-1
        (str(uuid.uuid4()), "sample_creditrequest.pdf"),  # test-doc-ocr-1
        (str(uuid.uuid4()), "sample_creditrequest.pdf"),  # test-doc-post-1
        (str(uuid.uuid4()), "sample_creditrequest.pdf"),  # test-doc-llm-1
        (str(uuid.uuid4()), "sample_creditrequest.pdf"),  # test-doc-viz-1
    ]
    
    # Upload PDFs to RAW stage
    for doc_id, pdf_file in test_documents:
        pdf_path = tmp_dir / pdf_file
        if pdf_path.exists():
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            storage.upload_blob(doc_id, Stage.RAW, ".pdf", pdf_data)
            logger.info(f"Uploaded {pdf_file} as {doc_id}.pdf to RAW stage")
    
    # Upload OCR results for documents that need them
    ocr_results_file = tmp_dir / "sample_creditrequest_ocr_result.json"
    if ocr_results_file.exists():
        with open(ocr_results_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Upload to OCR_RAW stage for documents that need OCR
        for doc_id, _ in test_documents:
            ocr_storage_data = {
                "document_uuid": doc_id,
                "timestamp": "2024-01-01T12:00:00Z",
                "ocr_results": ocr_data,
                "metadata": {"source": "test"}
            }
            storage.upload_blob(doc_id, Stage.OCR_RAW, ".json", 
                              json.dumps(ocr_storage_data).encode('utf-8'))
            logger.info(f"Uploaded OCR results for {doc_id} to OCR_RAW stage")
    
    # Upload clean OCR results for documents that need them
    clean_ocr_file = tmp_dir / "sample_creditrequest_normalized.json"
    if clean_ocr_file.exists():
        with open(clean_ocr_file, 'r', encoding='utf-8') as f:
            clean_ocr_data = json.load(f)
        
        # Upload to OCR_CLEAN stage for documents that need clean OCR
        for doc_id, _ in test_documents:
            clean_storage_data = {
                "document_id": doc_id,
                "normalized_lines": clean_ocr_data,
                "original_lines": clean_ocr_data,  # Use same data for simplicity
                "timestamp": "2024-01-01T12:00:00Z"
            }
            storage.upload_blob(doc_id, Stage.OCR_CLEAN, ".json", 
                              json.dumps(clean_storage_data).encode('utf-8'))
            logger.info(f"Uploaded clean OCR results for {doc_id} to OCR_CLEAN stage")
    
    # Upload LLM results for the complete test document
    llm_results_file = tmp_dir / "sample_creditrequest_extracted_fields.json"
    if llm_results_file.exists():
        with open(llm_results_file, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        # Upload to LLM stage for the first test document
        first_doc_id = test_documents[0][0]
        llm_storage_data = {
            "document_id": first_doc_id,
            "extracted_fields": llm_data.get("extracted_fields", {}),
            "missing_fields": llm_data.get("missing_fields", []),
            "validation_results": llm_data.get("validation_results", {}),
            "timestamp": "2024-01-01T12:00:00Z"
        }
        storage.upload_blob(first_doc_id, Stage.LLM, ".json", 
                          json.dumps(llm_storage_data).encode('utf-8'))
        logger.info(f"Uploaded LLM results for {first_doc_id} to LLM stage")


def cleanup_existing_containers():
    """Clean up any existing containers with old naming patterns."""
    try:
        # Remove containers with old naming patterns
        old_containers = ["dms-postgres", "azurite-blob-storages"]
        for container_name in old_containers:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    check=False
                )
                logger.info(f"Cleaned up old container: {container_name}")
            except Exception as e:
                logger.debug(f"Could not clean up container {container_name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to cleanup existing containers: {e}")


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
def test_environment():
    """Start DMS mock (Postgres + Azurite), Redis, and Ollama before any test runs. Set env vars. Cleanup after session."""
    global _global_dms_environment
    
    # Clean up any existing containers first
    cleanup_existing_containers()
    
    logger.info("[conftest] Starting DMS mock environment (Postgres + Azurite)")
    dms_env = DmsMockEnvironment()
    dms_env.start()
    _global_dms_environment = dms_env  # Store globally
    
    # Set DB env vars
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = str(dms_env.postgres_port)
    os.environ["POSTGRES_DB"] = "dms_meta"
    os.environ["POSTGRES_USER"] = "dms"
    os.environ["POSTGRES_PASSWORD"] = "dms"
    
    # Set Azurite env var
    azurite_port = dms_env.azurite_port
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = (
        f"DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:{azurite_port}/devstoreaccount1;"
    )
    logger.info(f"[conftest] Set AZURE_STORAGE_CONNECTION_STRING with port {azurite_port}")

    # Start Redis
    logger.info("[conftest] Starting Redis test container")
    redis_container = RedisContainer()
    redis_container.start()
    redis_port = redis_container.get_exposed_port(6379)
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = str(redis_port)

    yield  # Run tests

    # Cleanup
    logger.info("[conftest] Stopping Redis and DMS mock environment")
    
    try:
        redis_container.stop()
        # Remove the container using the underlying Docker container object
        if hasattr(redis_container, '_container') and redis_container._container:
            redis_container._container.remove()
        logger.info("Stopped and removed Redis container")
    except Exception as e:
        logger.warning(f"Failed to stop/remove Redis container: {e}")
    
    dms_env.stop()
    _global_dms_environment = None  # Clear global reference


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
    return AppConfig("config")


@pytest.fixture(scope="session")
def document_config():
    """Provide document configuration for tests."""
    return app_config.document_types


@pytest.fixture(scope="session")
def dms_mock_environment():
    """Provide DMS mock environment for tests that need it."""
    global _global_dms_environment
    
    if _global_dms_environment is None:
        raise RuntimeError("DMS mock environment not available. Make sure test_environment fixture runs first.")
    
    # Return the global environment that's already started
    yield _global_dms_environment

test_document_id = str(uuid.uuid4())