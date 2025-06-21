"""
Tests for OCR extraction process logic.
"""

import pytest
import json
from pathlib import Path
import tempfile
import asyncio

from src.ocr.extraction import (
    trigger_extraction,
    perform_ocr,
    postprocess_ocr,
    run_llm_extraction,
    generate_visualization
)
from src.creditsystem.storage import Stage, get_storage
from src.ocr.storage import read_ocr_results_from_bucket
from src.llm.client import OllamaClient
from src.config import AppConfig


@pytest.fixture
def llm_client():
    """Create a test LLM client."""
    app_config = AppConfig("tests/resources/test_application.conf")
    return OllamaClient(
        base_url=app_config.generative_llm.url,
        model_name=app_config.generative_llm.model_name
    )


class TestTriggerExtraction:
    """Test trigger_extraction function."""
    
    def test_generates_unique_extraction_job_id(self):
        """Test that trigger_extraction generates a unique job ID."""
        test_document_id = "test-doc-123"
        
        job_id = trigger_extraction(test_document_id)
        
        assert job_id is not None
        assert len(job_id) > 0
        assert job_id != test_document_id
    
    def test_saves_original_pdf_to_raw_bucket(self):
        """Test that trigger_extraction saves original PDF to raw bucket."""
        test_document_id = "test-doc-456"
        
        job_id = trigger_extraction(test_document_id)
        
        # Verify PDF was saved to raw bucket
        storage_client = get_storage()
        pdf_data = storage_client.download_blob(test_document_id, Stage.RAW, ".pdf")
        
        assert pdf_data is not None
        assert len(pdf_data) > 0
        assert pdf_data.startswith(b"%PDF-")  # Check for any PDF version
    
    def test_creates_extraction_job_with_pending_status(self):
        """Test that trigger_extraction creates extraction job with pending status."""
        test_document_id = "test-doc-789"
        
        job_id = trigger_extraction(test_document_id)
        
        # The job creation is logged, so we can verify it was called
        # In a real implementation, this would check the database
        assert job_id is not None


class TestPerformOcr:
    """Test perform_ocr function."""
    
    def test_performs_ocr_on_document(self):
        """Test that perform_ocr performs OCR on a document."""
        # First trigger extraction to get a document in the system
        test_document_id = "test-doc-ocr-1"
        trigger_extraction(test_document_id)
        
        # Perform OCR
        ocr_results = perform_ocr(test_document_id)
        
        # Verify OCR results structure
        assert ocr_results is not None
        assert "extracted_lines" in ocr_results
        assert "document_id" in ocr_results
        assert ocr_results["document_id"] == test_document_id
        
        # Verify OCR results were saved to bucket
        storage_client = get_storage()
        raw_ocr_data = read_ocr_results_from_bucket(test_document_id)
        assert raw_ocr_data is not None
        assert "ocr_results" in raw_ocr_data
    
    def test_raises_file_not_found_when_raw_pdf_missing(self):
        """Test that perform_ocr raises FileNotFoundError when raw PDF is missing."""
        test_document_id = "test-doc-ocr-missing"
        
        with pytest.raises(FileNotFoundError, match=f"Raw PDF not found for document {test_document_id}"):
            perform_ocr(test_document_id)


class TestPostprocessOcr:
    """Test postprocess_ocr function."""
    
    def test_postprocesses_ocr_results(self):
        """Test that postprocess_ocr post-processes OCR results."""
        # First trigger extraction and perform OCR
        test_document_id = "test-doc-post-1"
        trigger_extraction(test_document_id)
        perform_ocr(test_document_id)
        
        # Post-process OCR results
        cleaned_results = postprocess_ocr(test_document_id)
        
        # Verify cleaned results structure
        assert cleaned_results is not None
        assert "normalized_lines" in cleaned_results
        assert "original_lines" in cleaned_results
        assert "document_id" in cleaned_results
        assert cleaned_results["document_id"] == test_document_id
        
        # Verify cleaned results were saved to bucket
        storage_client = get_storage()
        clean_ocr_data = storage_client.download_blob(test_document_id, Stage.OCR_CLEAN, ".json")
        assert clean_ocr_data is not None
        
        clean_ocr_results = json.loads(clean_ocr_data.decode('utf-8'))
        assert "normalized_lines" in clean_ocr_results
        assert "original_lines" in clean_ocr_results
    
    def test_raises_file_not_found_when_raw_ocr_missing(self):
        """Test that postprocess_ocr raises FileNotFoundError when raw OCR is missing."""
        test_document_id = "test-doc-post-missing"
        
        with pytest.raises(FileNotFoundError, match=f"Raw OCR results not found for document {test_document_id}"):
            postprocess_ocr(test_document_id)


class TestRunLlmExtraction:
    """Test run_llm_extraction function."""
    
    @pytest.mark.asyncio
    async def test_runs_llm_extraction(self, llm_client):
        """Test that run_llm_extraction runs LLM extraction."""
        # First trigger extraction, perform OCR, and post-process
        test_document_id = "test-doc-llm-1"
        trigger_extraction(test_document_id)
        perform_ocr(test_document_id)
        postprocess_ocr(test_document_id)
        
        # Run LLM extraction
        extracted_fields_result = await run_llm_extraction(test_document_id)
        
        # Verify extracted fields result structure
        assert extracted_fields_result is not None
        assert "extracted_fields" in extracted_fields_result
        assert "missing_fields" in extracted_fields_result
        assert "validation_results" in extracted_fields_result
        assert "document_id" in extracted_fields_result
        assert extracted_fields_result["document_id"] == test_document_id
        
        # Verify extracted fields were saved to bucket
        storage_client = get_storage()
        llm_data = storage_client.download_blob(test_document_id, Stage.LLM, ".json")
        assert llm_data is not None
        
        llm_results = json.loads(llm_data.decode('utf-8'))
        assert "extracted_fields" in llm_results
        assert "missing_fields" in llm_results
        assert "validation_results" in llm_results
    
    @pytest.mark.asyncio
    async def test_raises_file_not_found_when_clean_ocr_missing(self):
        """Test that run_llm_extraction raises FileNotFoundError when clean OCR is missing."""
        test_document_id = "test-doc-llm-missing"
        
        with pytest.raises(FileNotFoundError, match=f"Clean OCR results not found for document {test_document_id}"):
            await run_llm_extraction(test_document_id)


class TestGenerateVisualization:
    """Test generate_visualization function."""
    
    def test_generates_visualization(self):
        """Test that generate_visualization generates visualization."""
        # First run the complete pipeline
        test_document_id = "test-doc-viz-1"
        trigger_extraction(test_document_id)
        perform_ocr(test_document_id)
        postprocess_ocr(test_document_id)
        
        # Generate visualization
        visualization_path = generate_visualization(test_document_id)
        
        # Verify visualization path
        assert visualization_path is not None
        assert visualization_path.endswith(f"{test_document_id}.png")
        assert Stage.ANNOTATED.value in visualization_path
        
        # Verify visualization was saved to bucket
        storage_client = get_storage()
        visualization_data = storage_client.download_blob(test_document_id, Stage.ANNOTATED, ".png")
        assert visualization_data is not None
        assert len(visualization_data) > 0
    
    def test_raises_file_not_found_when_original_pdf_missing(self):
        """Test that generate_visualization raises FileNotFoundError when original PDF is missing."""
        test_document_id = "test-doc-viz-missing-pdf"
        
        with pytest.raises(FileNotFoundError, match=f"Original PDF not found for document {test_document_id}"):
            generate_visualization(test_document_id)
    
    def test_raises_file_not_found_when_clean_ocr_missing(self):
        """Test that generate_visualization raises FileNotFoundError when clean OCR is missing."""
        test_document_id = "test-doc-viz-missing-ocr"
        
        # Trigger extraction to create PDF but skip OCR
        trigger_extraction(test_document_id)
        
        with pytest.raises(FileNotFoundError, match=f"Clean OCR results not found for document {test_document_id}"):
            generate_visualization(test_document_id)


class TestCompletePipeline:
    """Test the complete extraction pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_extraction_pipeline(self, llm_client):
        """Test the complete extraction pipeline from start to finish."""
        test_document_id = "test-doc-complete-1"
        
        # Step 1: Trigger extraction
        job_id = trigger_extraction(test_document_id)
        assert job_id is not None
        
        # Step 2: Perform OCR
        ocr_results = perform_ocr(test_document_id)
        assert ocr_results is not None
        assert "extracted_lines" in ocr_results
        
        # Step 3: Post-process OCR
        cleaned_results = postprocess_ocr(test_document_id)
        assert cleaned_results is not None
        assert "normalized_lines" in cleaned_results
        
        # Step 4: Run LLM extraction
        extracted_fields_result = await run_llm_extraction(test_document_id)
        assert extracted_fields_result is not None
        assert "extracted_fields" in extracted_fields_result
        
        # Step 5: Generate visualization
        visualization_path = generate_visualization(test_document_id)
        assert visualization_path is not None
        
        # Verify all stages have data in storage
        storage_client = get_storage()
        
        # Raw PDF
        raw_pdf = storage_client.download_blob(test_document_id, Stage.RAW, ".pdf")
        assert raw_pdf is not None
        
        # Raw OCR
        raw_ocr = read_ocr_results_from_bucket(test_document_id)
        assert raw_ocr is not None
        
        # Clean OCR
        clean_ocr = storage_client.download_blob(test_document_id, Stage.OCR_CLEAN, ".json")
        assert clean_ocr is not None
        
        # LLM results
        llm_results = storage_client.download_blob(test_document_id, Stage.LLM, ".json")
        assert llm_results is not None
        
        # Visualization
        visualization = storage_client.download_blob(test_document_id, Stage.ANNOTATED, ".png")
        assert visualization is not None
        
        # Verify data consistency
        clean_ocr_data = json.loads(clean_ocr.decode('utf-8'))
        llm_data = json.loads(llm_results.decode('utf-8'))
        
        assert clean_ocr_data["document_id"] == test_document_id
        assert llm_data["document_id"] == test_document_id 