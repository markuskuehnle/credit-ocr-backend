import pytest
import uuid
from pathlib import Path
from typing import Dict, Any

import psycopg2
from azure.storage.blob import BlobServiceClient
from src.dms_mock.environment import DmsMockEnvironment

# Mark all tests in this module to not use the global setup
pytestmark = pytest.mark.no_global_setup


@pytest.fixture(scope="session")
def postgres_connection(dms_mock_environment):
    """Provide PostgreSQL connection for tests."""
    return dms_mock_environment.get_postgres_connection()


@pytest.fixture(scope="session")
def blob_service_client(dms_mock_environment):
    """Provide Azure Blob Service client for tests."""
    return dms_mock_environment.get_blob_service_client()


def test_can_create_document_record_in_postgres(postgres_connection):
    """Test that we can create a document record in PostgreSQL."""
    document_id = str(uuid.uuid4())
    blob_path = "raw/Kreditantrag/test.pdf"
    mime_type = "application/pdf"
    hash_sha256 = "a" * 64  # Mock SHA256 hash
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, linked_entity, linked_entity_id, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, blob_path, mime_type, hash_sha256, "test.pdf", 
             "Kreditantrag", "KREDITANTRAG", "123", "not ready")
        )
        postgres_connection.commit()
    
    # Verify the record was created
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, blob_path, mime_type, hash_sha256, source_filename, 
                   document_type, linked_entity, linked_entity_id, textextraction_status 
            FROM dokument WHERE id = %s
            """, 
            (document_id,)
        )
        result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == document_id
    assert result[1] == blob_path
    assert result[2] == mime_type
    assert result[3] == hash_sha256
    assert result[4] == "test.pdf"
    assert result[5] == "Kreditantrag"
    assert result[6] == "KREDITANTRAG"
    assert result[7] == "123"
    assert result[8] == "not ready"


def test_can_upload_pdf_file_to_blob_storage(blob_service_client):
    """Test that we can upload a PDF file to blob storage."""
    container_name = "documents"
    blob_name = "credit_request_001.pdf"
    test_content = b"%PDF-1.4\nTest PDF content"
    
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    
    blob_client.upload_blob(test_content, overwrite=True)
    
    # Verify the blob was uploaded
    downloaded_content = blob_client.download_blob().readall()
    assert downloaded_content == test_content


def test_can_create_extraction_task_for_document(postgres_connection):
    """Test that we can create an extraction task for a document."""
    # First create a document
    document_id = str(uuid.uuid4())
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, "raw/test.pdf", "application/pdf", "a" * 64, "test.pdf", 
             "Kreditantrag", "not ready")
        )
        postgres_connection.commit()
    
    # Create extraction job
    job_id = str(uuid.uuid4())
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO extraction_job (id, dokument_id, state) VALUES (%s, %s, %s)",
            (job_id, document_id, "PENDING")
        )
        postgres_connection.commit()
    
    # Verify the job was created
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, dokument_id, state FROM extraction_job WHERE id = %s",
            (job_id,)
        )
        result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == job_id
    assert result[1] == document_id
    assert result[2] == "PENDING"


def test_can_update_textextraction_status_of_document(postgres_connection):
    """Test that we can update the text extraction status of a document."""
    document_id = str(uuid.uuid4())
    
    # Create document with 'not ready' status
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, "raw/test.pdf", "application/pdf", "a" * 64, "test.pdf", 
             "Kreditantrag", "not ready")
        )
        postgres_connection.commit()
    
    # Update text extraction status to 'done'
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE dokument SET textextraction_status = %s WHERE id = %s",
            ("done", document_id)
        )
        postgres_connection.commit()
    
    # Verify the status was updated
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT textextraction_status FROM dokument WHERE id = %s", (document_id,))
        result = cursor.fetchone()
    
    assert result[0] == "done"


def test_can_complete_extraction_job(postgres_connection):
    """Test that we can mark an extraction job as completed."""
    # Create document and job
    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, "raw/test.pdf", "application/pdf", "a" * 64, "test.pdf", 
             "Kreditantrag", "not ready")
        )
        cursor.execute(
            "INSERT INTO extraction_job (id, dokument_id, state) VALUES (%s, %s, %s)",
            (job_id, document_id, "PENDING")
        )
        postgres_connection.commit()
    
    # Complete the job
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE extraction_job SET state = %s, finished_at = NOW() WHERE id = %s",
            ("SUCCESS", job_id)
        )
        postgres_connection.commit()
    
    # Verify the job was completed
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "SELECT state, finished_at FROM extraction_job WHERE id = %s",
            (job_id,)
        )
        result = cursor.fetchone()
    
    assert result[0] == "SUCCESS"
    assert result[1] is not None  # finished_at should be set


def test_cascade_delete_removes_extraction_jobs(postgres_connection):
    """Test that deleting a document cascades to remove its extraction jobs."""
    # Create document and multiple jobs
    document_id = str(uuid.uuid4())
    job_id_1 = str(uuid.uuid4())
    job_id_2 = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, "raw/test.pdf", "application/pdf", "a" * 64, "test.pdf", 
             "Kreditantrag", "not ready")
        )
        cursor.execute(
            "INSERT INTO extraction_job (id, dokument_id, state) VALUES (%s, %s, %s), (%s, %s, %s)",
            (job_id_1, document_id, "PENDING", job_id_2, document_id, "SUCCESS")
        )
        postgres_connection.commit()
    
    # Delete the document
    with postgres_connection.cursor() as cursor:
        cursor.execute("DELETE FROM dokument WHERE id = %s", (document_id,))
        postgres_connection.commit()
    
    # Verify both jobs were deleted
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM extraction_job WHERE dokument_id = %s", (document_id,))
        result = cursor.fetchone()
    
    assert result[0] == 0


def test_can_retrieve_document_with_its_extraction_jobs(postgres_connection):
    """Test that we can retrieve a document with all its extraction jobs."""
    # Create document and jobs
    document_id = str(uuid.uuid4())
    job_id_1 = str(uuid.uuid4())
    job_id_2 = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dokument (
                id, blob_path, mime_type, hash_sha256, source_filename, 
                document_type, textextraction_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, "raw/test.pdf", "application/pdf", "a" * 64, "test.pdf", 
             "Kreditantrag", "not ready")
        )
        cursor.execute(
            "INSERT INTO extraction_job (id, dokument_id, state) VALUES (%s, %s, %s), (%s, %s, %s)",
            (job_id_1, document_id, "PENDING", job_id_2, document_id, "SUCCESS")
        )
        postgres_connection.commit()
    
    # Retrieve document with jobs
    with postgres_connection.cursor() as cursor:
        cursor.execute("""
            SELECT d.id, d.blob_path, d.textextraction_status, 
                   e.id as job_id, e.state, e.created_at
            FROM dokument d
            LEFT JOIN extraction_job e ON d.id = e.dokument_id
            WHERE d.id = %s
            ORDER BY e.created_at
        """, (document_id,))
        results = cursor.fetchall()
    
    assert len(results) == 2  # One document with two jobs
    assert results[0][0] == document_id  # Document ID
    assert results[0][1] == "raw/test.pdf"  # Blob path
    assert results[0][2] == "not ready"  # Text extraction status
    assert results[0][3] in [job_id_1, job_id_2]  # Job ID
    assert results[1][3] in [job_id_1, job_id_2]  # Other job ID


def test_can_upload_credit_request_pdf_to_dms(dms_mock_environment):
    """Test that we can upload a credit request PDF to the DMS with proper document type."""
    from pathlib import Path
    
    # Get the sample PDF file
    sample_pdf_path = Path("tests/tmp/sample_creditrequest.pdf")
    assert sample_pdf_path.exists(), f"Sample PDF not found: {sample_pdf_path}"
    
    # Get DMS service
    dms_service = dms_mock_environment.get_dms_service()
    
    # Upload document with type 'Kreditantrag' and link to a credit application
    document_id = dms_service.upload_document(
        sample_pdf_path, 
        "Kreditantrag",
        source_filename="credit_application_form.pdf",
        linked_entity="KREDITANTRAG",
        linked_entity_id="12345"
    )
    
    # Verify document was created in database
    document = dms_service.get_document(document_id)
    assert document is not None
    assert document["id"] == document_id
    assert document["blob_path"].startswith("raw/Kreditantrag/")
    assert document["blob_path"].endswith(".pdf")
    assert document["mime_type"] == "application/pdf"
    assert document["textextraction_status"] == "not ready"
    assert document["source_filename"] == "credit_application_form.pdf"
    assert document["document_type"] == "Kreditantrag"
    assert document["linked_entity"] == "KREDITANTRAG"
    assert document["linked_entity_id"] == "12345"
    assert len(document["hash_sha256"]) == 64  # SHA256 hash is 64 characters
    
    # Verify document can be downloaded from blob storage
    downloaded_content = dms_service.download_document(document_id)
    assert downloaded_content is not None
    assert len(downloaded_content) > 0
    
    # Verify original file and downloaded content match
    with open(sample_pdf_path, 'rb') as f:
        original_content = f.read()
    assert downloaded_content == original_content
    
    # Verify document appears in list by type
    kreditantrag_documents = dms_service.list_documents_by_type("Kreditantrag")
    assert len(kreditantrag_documents) >= 1
    
    # Find our uploaded document in the list
    uploaded_doc = next((doc for doc in kreditantrag_documents if doc["id"] == document_id), None)
    assert uploaded_doc is not None
    assert uploaded_doc["blob_path"] == document["blob_path"]
    
    # Test text extraction status updates
    assert dms_service.update_textextraction_status(document_id, "ready")
    updated_document = dms_service.get_document(document_id)
    assert updated_document["textextraction_status"] == "ready"
    
    # Test extraction job creation and management
    job_id = dms_service.create_extraction_job(document_id, "PENDING")
    assert job_id is not None
    
    # Update job status
    assert dms_service.update_extraction_job(job_id, "RUNNING", "Processing document...")
    assert dms_service.update_extraction_job(job_id, "SUCCESS", "Extraction completed successfully")
    
    # Get extraction jobs for the document
    jobs = dms_service.get_extraction_jobs(document_id)
    assert len(jobs) >= 1
    
    # Find our job
    our_job = next((job for job in jobs if job["id"] == job_id), None)
    assert our_job is not None
    assert our_job["state"] == "SUCCESS"
    assert our_job["worker_log"] == "Extraction completed successfully"
    assert our_job["finished_at"] is not None


def test_can_upload_credit_request_pdf_to_dms_with_path(dms_mock_environment):
    """Test that we can upload a credit request PDF to the DMS with proper document type using a path."""
    from pathlib import Path
    
    # Get the sample PDF file
    sample_pdf_path = Path("tests/tmp/sample_creditrequest.pdf")
    assert sample_pdf_path.exists(), f"Sample PDF not found: {sample_pdf_path}"
    
    # Get DMS service
    dms_service = dms_mock_environment.get_dms_service()
    
    # Upload document with type 'Kreditantrag' (credit application form)
    document_id = dms_service.upload_document(
        sample_pdf_path, 
        "Kreditantrag",
        source_filename="credit_request_form.pdf",
        linked_entity="KREDITANTRAG",
        linked_entity_id="67890"
    )
    
    # Verify document was created in database
    document = dms_service.get_document(document_id)
    assert document is not None
    assert document["id"] == document_id
    assert document["blob_path"].startswith("raw/Kreditantrag/")
    assert document["blob_path"].endswith(".pdf")
    assert document["mime_type"] == "application/pdf"
    assert document["textextraction_status"] == "not ready"
    assert document["source_filename"] == "credit_request_form.pdf"
    assert document["document_type"] == "Kreditantrag"
    assert document["linked_entity"] == "KREDITANTRAG"
    assert document["linked_entity_id"] == "67890"
    assert len(document["hash_sha256"]) == 64  # SHA256 hash is 64 characters
    
    # Verify document can be downloaded from blob storage
    downloaded_content = dms_service.download_document(document_id)
    assert downloaded_content is not None
    assert len(downloaded_content) > 0
    
    # Verify original file and downloaded content match
    with open(sample_pdf_path, 'rb') as f:
        original_content = f.read()
    assert downloaded_content == original_content
    
    # Verify document appears in list by type
    kreditantrag_documents = dms_service.list_documents_by_type("Kreditantrag")
    assert len(kreditantrag_documents) >= 1
    
    # Find our uploaded document in the list
    uploaded_doc = next((doc for doc in kreditantrag_documents if doc["id"] == document_id), None)
    assert uploaded_doc is not None
    assert uploaded_doc["blob_path"] == document["blob_path"] 