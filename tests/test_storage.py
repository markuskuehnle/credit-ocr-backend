"""
Tests for the credit system storage helper.
"""

import pytest
from pathlib import PurePosixPath

from src.creditsystem.storage import Stage, get_storage


class TestStage:
    """Test the Stage enum."""
    
    def test_stage_values_are_correct(self):
        """Test that stage values match expected bucket names."""
        assert Stage.RAW.value == "credit-docs-raw"
        assert Stage.OCR_RAW.value == "credit-docs-ocr-raw"
        assert Stage.OCR_CLEAN.value == "credit-docs-ocr-clean"
        assert Stage.LLM.value == "credit-docs-llm"
        assert Stage.ANNOTATED.value == "credit-docs-annotated"


class TestBlobStorage:
    """Test the BlobStorage class."""
    
    def test_blob_path_builds_correct_paths(self):
        """Test that blob_path builds correct paths for different stages."""
        storage = get_storage()
        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        
        # Test PDF extension
        raw_path = storage.blob_path(test_uuid, Stage.RAW, ".pdf")
        assert raw_path == PurePosixPath("123e4567-e89b-12d3-a456-426614174000.pdf")
        
        # Test JSON extension
        ocr_raw_path = storage.blob_path(test_uuid, Stage.OCR_RAW, ".json")
        assert ocr_raw_path == PurePosixPath("123e4567-e89b-12d3-a456-426614174000.json")
        
        # Test without leading dot
        llm_path = storage.blob_path(test_uuid, Stage.LLM, "json")
        assert llm_path == PurePosixPath("123e4567-e89b-12d3-a456-426614174000.json")
    
    def test_blob_path_handles_all_stages(self):
        """Test that blob_path works for all processing stages."""
        storage = get_storage()
        test_uuid = "test-uuid-123"
        
        all_stages = [Stage.RAW, Stage.OCR_RAW, Stage.OCR_CLEAN, Stage.LLM, Stage.ANNOTATED]
        
        for stage in all_stages:
            path = storage.blob_path(test_uuid, stage, ".pdf")
            assert isinstance(path, PurePosixPath)
            assert path.name == f"{test_uuid}.pdf"
    
    def test_singleton_pattern(self):
        """Test that BlobStorage follows singleton pattern."""
        storage1 = get_storage()
        storage2 = get_storage()
        
        assert storage1 is storage2
        assert id(storage1) == id(storage2)
    
    def test_blob_client_returns_client(self):
        """Test that blob_client returns a BlobClient instance."""
        storage = get_storage()
        test_uuid = "test-uuid-456"
        
        blob_client = storage.blob_client(test_uuid, Stage.RAW, ".pdf")
        
        from azure.storage.blob import BlobClient
        assert isinstance(blob_client, BlobClient)
        assert blob_client.blob_name == f"{test_uuid}.pdf"
        assert blob_client.container_name == Stage.RAW.value
    
    def test_upload_and_download_blob(self):
        """Test uploading and downloading a blob."""
        storage = get_storage()
        test_uuid = "test-uuid-789"
        test_data = b"test content for blob storage"
        
        # Upload blob
        storage.upload_blob(test_uuid, Stage.RAW, ".txt", test_data)
        
        # Verify blob exists
        assert storage.blob_exists(test_uuid, Stage.RAW, ".txt")
        
        # Download and verify content
        downloaded_data = storage.download_blob(test_uuid, Stage.RAW, ".txt")
        assert downloaded_data == test_data
        
        # Clean up
        storage.delete_blob(test_uuid, Stage.RAW, ".txt")
        assert not storage.blob_exists(test_uuid, Stage.RAW, ".txt")
    
    def test_all_stages_blob_operations(self):
        """Test blob operations for all processing stages."""
        storage = get_storage()
        test_uuid = "test-uuid-all-stages"
        test_data = b"test data for all stages"
        
        all_stages = [Stage.RAW, Stage.OCR_RAW, Stage.OCR_CLEAN, Stage.LLM, Stage.ANNOTATED]
        
        for stage in all_stages:
            # Upload to each stage
            storage.upload_blob(test_uuid, stage, ".json", test_data)
            
            # Verify exists
            assert storage.blob_exists(test_uuid, stage, ".json")
            
            # Download and verify
            downloaded_data = storage.download_blob(test_uuid, stage, ".json")
            assert downloaded_data == test_data
            
            # Clean up
            storage.delete_blob(test_uuid, stage, ".json")
            assert not storage.blob_exists(test_uuid, stage, ".json")
    
    def test_list_blobs_in_stage(self):
        """Test listing blobs in a specific stage."""
        storage = get_storage()
        test_uuid = "test-uuid-list"
        test_data = b"test data for listing"
        
        # Upload a blob to OCR_RAW stage
        storage.upload_blob(test_uuid, Stage.OCR_RAW, ".json", test_data)
        
        # List blobs in OCR_RAW stage
        blob_names = storage.list_blobs_in_stage(Stage.OCR_RAW)
        
        # Should contain our test blob
        expected_blob_name = f"{test_uuid}.json"
        assert expected_blob_name in blob_names
        
        # Clean up
        storage.delete_blob(test_uuid, Stage.OCR_RAW, ".json")
        
        # List again should be empty or not contain our blob
        blob_names_after_cleanup = storage.list_blobs_in_stage(Stage.OCR_RAW)
        assert expected_blob_name not in blob_names_after_cleanup 