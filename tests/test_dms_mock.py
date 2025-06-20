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
    blob_path = "documents/credit_request_001.pdf"
    mime_type = "application/pdf"
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dokument (id, blob_path, mime_type) VALUES (%s, %s, %s)",
            (document_id, blob_path, mime_type)
        )
        postgres_connection.commit()
    
    # Verify the record was created
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT id, blob_path, mime_type, ocr_status FROM dokument WHERE id = %s", (document_id,))
        result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == document_id
    assert result[1] == blob_path
    assert result[2] == mime_type
    assert result[3] == "PENDING"


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
            "INSERT INTO dokument (id, blob_path, mime_type) VALUES (%s, %s, %s)",
            (document_id, "documents/test.pdf", "application/pdf")
        )
        postgres_connection.commit()
    
    # Create extraction task
    task_id = str(uuid.uuid4())
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO extraktionsauftrag (id, dokument_id, state) VALUES (%s, %s, %s)",
            (task_id, document_id, "PENDING")
        )
        postgres_connection.commit()
    
    # Verify the task was created
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, dokument_id, state FROM extraktionsauftrag WHERE id = %s",
            (task_id,)
        )
        result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == task_id
    assert result[1] == document_id
    assert result[2] == "PENDING"


def test_can_update_ocr_status_of_document(postgres_connection):
    """Test that we can update the OCR status of a document."""
    document_id = str(uuid.uuid4())
    
    # Create document with PENDING status
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dokument (id, blob_path, mime_type, ocr_status) VALUES (%s, %s, %s, %s)",
            (document_id, "documents/test.pdf", "application/pdf", "PENDING")
        )
        postgres_connection.commit()
    
    # Update OCR status to COMPLETED
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE dokument SET ocr_status = %s WHERE id = %s",
            ("COMPLETED", document_id)
        )
        postgres_connection.commit()
    
    # Verify the status was updated
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT ocr_status FROM dokument WHERE id = %s", (document_id,))
        result = cursor.fetchone()
    
    assert result[0] == "COMPLETED"


def test_can_complete_extraction_task(postgres_connection):
    """Test that we can mark an extraction task as completed."""
    # Create document and task
    document_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dokument (id, blob_path, mime_type) VALUES (%s, %s, %s)",
            (document_id, "documents/test.pdf", "application/pdf")
        )
        cursor.execute(
            "INSERT INTO extraktionsauftrag (id, dokument_id, state) VALUES (%s, %s, %s)",
            (task_id, document_id, "PENDING")
        )
        postgres_connection.commit()
    
    # Complete the task
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE extraktionsauftrag SET state = %s, finished_at = NOW() WHERE id = %s",
            ("COMPLETED", task_id)
        )
        postgres_connection.commit()
    
    # Verify the task was completed
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "SELECT state, finished_at FROM extraktionsauftrag WHERE id = %s",
            (task_id,)
        )
        result = cursor.fetchone()
    
    assert result[0] == "COMPLETED"
    assert result[1] is not None  # finished_at should be set


def test_cascade_delete_removes_extraction_tasks(postgres_connection):
    """Test that deleting a document cascades to remove its extraction tasks."""
    # Create document and multiple tasks
    document_id = str(uuid.uuid4())
    task_id_1 = str(uuid.uuid4())
    task_id_2 = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dokument (id, blob_path, mime_type) VALUES (%s, %s, %s)",
            (document_id, "documents/test.pdf", "application/pdf")
        )
        cursor.execute(
            "INSERT INTO extraktionsauftrag (id, dokument_id, state) VALUES (%s, %s, %s), (%s, %s, %s)",
            (task_id_1, document_id, "PENDING", task_id_2, document_id, "COMPLETED")
        )
        postgres_connection.commit()
    
    # Delete the document
    with postgres_connection.cursor() as cursor:
        cursor.execute("DELETE FROM dokument WHERE id = %s", (document_id,))
        postgres_connection.commit()
    
    # Verify both tasks were deleted
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM extraktionsauftrag WHERE dokument_id = %s", (document_id,))
        result = cursor.fetchone()
    
    assert result[0] == 0


def test_can_retrieve_document_with_its_extraction_tasks(postgres_connection):
    """Test that we can retrieve a document with all its extraction tasks."""
    # Create document and tasks
    document_id = str(uuid.uuid4())
    task_id_1 = str(uuid.uuid4())
    task_id_2 = str(uuid.uuid4())
    
    with postgres_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dokument (id, blob_path, mime_type) VALUES (%s, %s, %s)",
            (document_id, "documents/test.pdf", "application/pdf")
        )
        cursor.execute(
            "INSERT INTO extraktionsauftrag (id, dokument_id, state) VALUES (%s, %s, %s), (%s, %s, %s)",
            (task_id_1, document_id, "PENDING", task_id_2, document_id, "COMPLETED")
        )
        postgres_connection.commit()
    
    # Retrieve document with tasks
    with postgres_connection.cursor() as cursor:
        cursor.execute("""
            SELECT d.id, d.blob_path, d.ocr_status, 
                   e.id as task_id, e.state, e.created_at
            FROM dokument d
            LEFT JOIN extraktionsauftrag e ON d.id = e.dokument_id
            WHERE d.id = %s
            ORDER BY e.created_at
        """, (document_id,))
        results = cursor.fetchall()
    
    assert len(results) == 2  # One document with two tasks
    assert results[0][0] == document_id  # Document ID
    assert results[0][1] == "documents/test.pdf"  # Blob path
    assert results[0][2] == "PENDING"  # OCR status
    assert results[0][3] in [task_id_1, task_id_2]  # Task ID
    assert results[1][3] in [task_id_1, task_id_2]  # Other task ID 