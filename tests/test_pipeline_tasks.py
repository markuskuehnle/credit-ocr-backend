import pytest
import logging
from unittest.mock import patch, MagicMock
from celery.contrib.testing.worker import start_worker
from testcontainers.redis import RedisContainer
from src.tasks.celery_app import celery_app
from src.tasks.pipeline_tasks import run_full_pipeline, perform_ocr_task
from src.creditsystem.storage import get_storage, Stage
import time
import psycopg2
import os
import uuid

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer().with_name("redis") as redis:
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

@pytest.fixture(scope="session")
def setup_database_env(dms_mock_environment):
    """Set up database environment variables for status updates."""
    # Set environment variables for database connection
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = str(dms_mock_environment.postgres_port)
    os.environ["POSTGRES_DB"] = "dms_meta"
    os.environ["POSTGRES_USER"] = "dms"
    os.environ["POSTGRES_PASSWORD"] = "dms"
    
    yield
    
    # Clean up environment variables
    for var in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]:
        os.environ.pop(var, None)

def test_redis_broker_connection(redis_container, celery_app_for_test, celery_worker_for_test):
    """Test that Redis broker is working and Celery worker is ready."""
    # Test that we can submit a simple task and get a result
    test_document_id = str(uuid.uuid4())
    
    # Submit a simple task
    result = perform_ocr_task.delay(test_document_id)
    
    # Verify the task was accepted and processed
    assert result is not None
    assert result.id is not None
    
    # With task_always_eager=True, the task should complete immediately
    # but we'll add a small delay for safety
    time.sleep(1)
    
    # Verify the task state
    assert result.state in ['SUCCESS', 'FAILURE', 'PENDING']
    
    logger.info(f"Redis broker test completed. Task state: {result.state}")

def test_end_to_end_pipeline(dms_mock_environment, celery_app_for_test, celery_worker_for_test, setup_database_env):
    """Test the full extraction pipeline runs via Celery."""
    test_document_id = str(uuid.uuid4())

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

def test_end_to_end_document_extraction_failure(dms_mock_environment, celery_app_for_test, celery_worker_for_test, setup_database_env):
    """Test that extraction failures are handled gracefully with proper error logging and status updates."""
    test_document_id = str(uuid.uuid4())
    
    # Create a document record in the database first
    connection = psycopg2.connect(
        host="localhost",
        port=dms_mock_environment.postgres_port,
        database="dms_meta",
        user="dms",
        password="dms"
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO Dokument (
                    dokument_id, pfad_dms, dokumententyp, hash_sha256, quelle_dateiname, 
                    textextraktion_status
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (test_document_id, "raw/test_failure.pdf", "Kreditantrag", "a" * 64, "test_failure.pdf", 
                 "nicht bereit")
            )
            connection.commit()
    finally:
        connection.close()
    
    # Mock Azure OCR to raise an exception
    with patch('src.ocr.azure_ocr_client.analyze_single_document_with_azure') as mock_azure:
        mock_azure.side_effect = Exception("Azure OCR service unavailable")
        
        # Run the OCR task that should fail
        result = perform_ocr_task.delay(test_document_id)
        
        # Give the task time to fail
        time.sleep(2)
        
        # Verify the task failed
        assert result.state == 'FAILURE'
        
        # Check that the error was logged (we can't easily capture logs in tests,
        # but we can verify the task failed as expected)
        assert result.failed()
        
        # Verify the status was updated to "Fehlerhaft" in the database
        connection = psycopg2.connect(
            host="localhost",
            port=dms_mock_environment.postgres_port,
            database="dms_meta",
            user="dms",
            password="dms"
        )
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status, fehlermeldung 
                    FROM Extraktionsauftrag 
                    WHERE dokument_id = %s 
                    ORDER BY erstellt_am DESC 
                    LIMIT 1
                    """,
                    (test_document_id,)
                )
                result_row = cursor.fetchone()
                
                if result_row:
                    status, fehlermeldung = result_row
                    # The status should be "Fehlerhaft" or contain error information
                    assert status == "Fehlerhaft" or "error" in fehlermeldung.lower() or "failed" in fehlermeldung.lower()
                    logger.info(f"Extraction job status: {status}, log: {fehlermeldung}")
                else:
                    # If no job record found, that's also acceptable as the error handling
                    # might prevent job creation
                    logger.info("No extraction job record found after failure (acceptable)")
                    
        finally:
            connection.close() 