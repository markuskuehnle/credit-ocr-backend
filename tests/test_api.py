"""
Tests for the FastAPI endpoints
"""

import pytest
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Set TESTING environment variable to prevent API from starting its own DMS mock
os.environ["TESTING"] = "1"

from src.api.main import app

client = TestClient(app)

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    # Use the existing sample PDF from the test data
    pdf_path = Path(__file__).parent.parent / "data" / "sample_creditrequest.pdf"
    if pdf_path.exists():
        return pdf_path
    else:
        # Create a minimal PDF if the sample doesn't exist
        from reportlab.pdfgen import canvas
        from io import BytesIO
        
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        p.drawString(100, 750, "Sample Credit Request Document")
        p.drawString(100, 700, "This is a test document for API testing.")
        p.save()
        
        return BytesIO(buffer.getvalue())

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "database" in data
    assert "redis" in data
    assert "ocr_service" in data
    assert "llm_service" in data

def test_upload_documents(sample_pdf):
    """Test document upload endpoint."""
    credit_request_id = "test-credit-request-123"
    
    # Test with a PDF file
    if isinstance(sample_pdf, (str, Path)):
        # It's a file path
        with open(sample_pdf, "rb") as f:
            files = [("files", ("test_document.pdf", f, "application/pdf"))]
            data = {"document_type": "Gehaltsnachweis"}
            
            response = client.post(
                f"/credit-request/{credit_request_id}/documents",
                files=files,
                data=data
            )
    else:
        # It's a BytesIO object
        sample_pdf.seek(0)  # Reset to beginning
        files = [("files", ("test_document.pdf", sample_pdf, "application/pdf"))]
        data = {"document_type": "Gehaltsnachweis"}
        
        response = client.post(
            f"/credit-request/{credit_request_id}/documents",
            files=files,
            data=data
        )
    
    assert response.status_code == 200
    
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1
    assert "document_id" in result[0]
    assert result[0]["status"] == "Extraktion ausstehend"

def test_get_documents():
    """Test getting documents for a credit request."""
    credit_request_id = "test-credit-request-123"
    
    response = client.get(f"/credit-request/{credit_request_id}/documents")
    assert response.status_code == 200
    
    documents = response.json()
    assert isinstance(documents, list)

def test_get_document_status():
    """Test getting document status."""
    # First upload a document to get a document ID
    credit_request_id = "test-credit-request-456"
    
    # Upload a document first
    sample_pdf = Path(__file__).parent.parent / "data" / "sample_creditrequest.pdf"
    if sample_pdf.exists():
        with open(sample_pdf, "rb") as f:
            files = [("files", ("test_document.pdf", f, "application/pdf"))]
            data = {"document_type": "Gehaltsnachweis"}
            
            upload_response = client.post(
                f"/credit-request/{credit_request_id}/documents",
                files=files,
                data=data
            )
        
        if upload_response.status_code == 200:
            document_id = upload_response.json()[0]["document_id"]
            
            # Now test getting the status
            response = client.get(f"/document/{document_id}/status")
            assert response.status_code == 200
            
            status_data = response.json()
            assert "document_id" in status_data
            assert "status" in status_data
            assert status_data["document_id"] == document_id

def test_get_nonexistent_document_status():
    """Test getting status for a non-existent document."""
    document_id = "non-existent-document-id"
    
    response = client.get(f"/document/{document_id}/status")
    assert response.status_code == 404

def test_upload_invalid_file_type():
    """Test uploading an invalid file type."""
    credit_request_id = "test-credit-request-789"
    
    # Create a text file (not allowed)
    from io import BytesIO
    text_file = BytesIO(b"This is a text file, not a PDF")
    
    files = [("files", ("test.txt", text_file, "text/plain"))]
    data = {"document_type": "Test"}
    
    response = client.post(
        f"/credit-request/{credit_request_id}/documents",
        files=files,
        data=data
    )
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_no_files():
    """Test uploading without files."""
    credit_request_id = "test-credit-request-999"
    
    response = client.post(f"/credit-request/{credit_request_id}/documents")
    assert response.status_code == 422  # FastAPI returns 422 for missing required fields
    # The error detail structure may vary, so we'll just check the status code 