import pytest
from celery.contrib.testing.worker import start_worker
from testcontainers.redis import RedisContainer
from src.tasks.celery_app import celery_app
from src.tasks.pipeline_tasks import run_full_pipeline
from src.creditsystem.storage import get_storage, Stage
import time

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer() as redis:
        yield redis

@pytest.fixture(scope="session")
def celery_config(redis_container):
    redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"
    return {
        "broker_url": redis_url,
        "result_backend": redis_url,
    }

@pytest.fixture(scope="session")
def celery_app_for_test(celery_config):
    celery_app.conf.broker_url = celery_config["broker_url"]
    celery_app.conf.result_backend = celery_config["result_backend"]
    celery_app.conf.task_always_eager = True
    return celery_app

@pytest.fixture(scope="session")
def celery_worker_for_test(celery_app_for_test):
    with start_worker(celery_app_for_test, perform_ping_check=False) as worker:
        yield worker

def test_end_to_end_pipeline(dms_mock_environment, celery_app_for_test, celery_worker_for_test):
    """Test the full extraction pipeline runs via Celery."""
    test_document_id = "test-doc-pipeline-1"

    # Run the full pipeline task
    run_full_pipeline.delay(test_document_id)

    # Give the pipeline time to run
    # With task_always_eager=True, this should be synchronous, but we add a small delay for safety.
    time.sleep(5)

    # Verify all stages have data in storage
    storage_client = get_storage()

    # Raw PDF
    raw_pdf = storage_client.download_blob(test_document_id, Stage.RAW, ".pdf")
    assert raw_pdf is not None, "Raw PDF not found in storage"

    # Raw OCR
    raw_ocr = storage_client.download_blob(test_document_id, Stage.OCR_RAW, ".json")
    assert raw_ocr is not None, "Raw OCR result not found in storage"

    # Clean OCR
    clean_ocr = storage_client.download_blob(test_document_id, Stage.OCR_CLEAN, ".json")
    assert clean_ocr is not None, "Clean OCR result not found in storage"

    # LLM results
    llm_results = storage_client.download_blob(test_document_id, Stage.LLM, ".json")
    assert llm_results is not None, "LLM result not found in storage"

    # Visualization
    visualization = storage_client.download_blob(test_document_id, Stage.ANNOTATED, ".png")
    assert visualization is not None, "Annotated visualization not found in storage" 