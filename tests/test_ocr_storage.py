import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.ocr.storage import (
    write_ocr_results_to_bucket,
    read_ocr_results_from_bucket,
    delete_ocr_results_from_bucket,
    list_ocr_results_in_bucket
)
from src.creditsystem.storage import Stage


class TestWriteOcrResultsToBucket:
    """Test writing OCR results to bucket."""
    
    def test_writes_ocr_results_with_metadata_to_bucket(self):
        """Test that OCR results are written with metadata to bucket."""
        test_document_uuid = "test-uuid-123"
        test_ocr_results = {"text": "sample text", "confidence": 0.95}
        test_metadata = {"source": "azure", "version": "1.0"}
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_path.return_value = "ocr-raw/test-uuid-123.json"
            
            result_path = write_ocr_results_to_bucket(
                document_uuid=test_document_uuid,
                ocr_results=test_ocr_results,
                metadata=test_metadata
            )
        
        mock_storage.upload_blob.assert_called_once()
        upload_call_args = mock_storage.upload_blob.call_args
        
        assert upload_call_args[1]['uuid'] == test_document_uuid
        assert upload_call_args[1]['stage'] == Stage.OCR_RAW
        assert upload_call_args[1]['ext'] == ".json"
        assert upload_call_args[1]['overwrite'] is True
        
        # Verify the uploaded data structure
        uploaded_data = json.loads(upload_call_args[1]['data'].decode('utf-8'))
        assert uploaded_data['document_uuid'] == test_document_uuid
        assert uploaded_data['ocr_results'] == test_ocr_results
        assert uploaded_data['metadata'] == test_metadata
        assert 'timestamp' in uploaded_data
        
        assert result_path == "ocr-raw/test-uuid-123.json"
    
    def test_writes_ocr_results_without_metadata_to_bucket(self):
        """Test that OCR results are written without metadata to bucket."""
        test_document_uuid = "test-uuid-456"
        test_ocr_results = {"text": "another sample", "confidence": 0.88}
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_path.return_value = "ocr-raw/test-uuid-456.json"
            
            result_path = write_ocr_results_to_bucket(
                document_uuid=test_document_uuid,
                ocr_results=test_ocr_results
            )
        
        mock_storage.upload_blob.assert_called_once()
        upload_call_args = mock_storage.upload_blob.call_args
        
        uploaded_data = json.loads(upload_call_args[1]['data'].decode('utf-8'))
        assert uploaded_data['metadata'] == {}
        assert result_path == "ocr-raw/test-uuid-456.json"
    
    def test_includes_timestamp_in_uploaded_data(self):
        """Test that timestamp is included in the uploaded data."""
        test_document_uuid = "test-uuid-timestamp"
        test_ocr_results = {"text": "timestamp test"}
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_path.return_value = "ocr-raw/test-uuid-timestamp.json"
            
            write_ocr_results_to_bucket(
                document_uuid=test_document_uuid,
                ocr_results=test_ocr_results
            )
        
        upload_call_args = mock_storage.upload_blob.call_args
        uploaded_data = json.loads(upload_call_args[1]['data'].decode('utf-8'))
        
        # Verify timestamp is ISO format and recent
        timestamp_str = uploaded_data['timestamp']
        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        time_difference = abs((datetime.utcnow() - parsed_timestamp.replace(tzinfo=None)).total_seconds())
        
        assert time_difference < 5  # Should be within 5 seconds


class TestReadOcrResultsFromBucket:
    """Test reading OCR results from bucket."""
    
    def test_reads_existing_ocr_results_from_bucket(self):
        """Test that existing OCR results are read from bucket."""
        test_document_uuid = "test-uuid-read"
        expected_ocr_data = {
            "document_uuid": test_document_uuid,
            "timestamp": "2024-01-01T12:00:00",
            "ocr_results": {"text": "read test", "confidence": 0.92},
            "metadata": {"source": "test"}
        }
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_exists.return_value = True
            mock_storage.download_blob.return_value = json.dumps(expected_ocr_data).encode('utf-8')
            
            result_data = read_ocr_results_from_bucket(test_document_uuid)
        
        mock_storage.blob_exists.assert_called_once_with(test_document_uuid, Stage.OCR_RAW, ".json")
        mock_storage.download_blob.assert_called_once_with(test_document_uuid, Stage.OCR_RAW, ".json")
        assert result_data == expected_ocr_data
    
    def test_returns_none_when_ocr_results_not_found(self):
        """Test that None is returned when OCR results don't exist."""
        test_document_uuid = "test-uuid-not-found"
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_exists.return_value = False
            
            result_data = read_ocr_results_from_bucket(test_document_uuid)
        
        mock_storage.blob_exists.assert_called_once_with(test_document_uuid, Stage.OCR_RAW, ".json")
        mock_storage.download_blob.assert_not_called()
        assert result_data is None
    
    def test_returns_none_when_download_fails(self):
        """Test that None is returned when download fails."""
        test_document_uuid = "test-uuid-download-fail"
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_exists.return_value = True
            mock_storage.download_blob.return_value = None
            
            result_data = read_ocr_results_from_bucket(test_document_uuid)
        
        assert result_data is None
    
    def test_returns_none_when_json_parsing_fails(self):
        """Test that None is returned when JSON parsing fails."""
        test_document_uuid = "test-uuid-json-fail"
        invalid_json_data = b"invalid json data"
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.blob_exists.return_value = True
            mock_storage.download_blob.return_value = invalid_json_data
            
            result_data = read_ocr_results_from_bucket(test_document_uuid)
        
        assert result_data is None


class TestDeleteOcrResultsFromBucket:
    """Test deleting OCR results from bucket."""
    
    def test_deletes_existing_ocr_results_from_bucket(self):
        """Test that existing OCR results are deleted from bucket."""
        test_document_uuid = "test-uuid-delete"
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.delete_blob.return_value = True
            
            deletion_success = delete_ocr_results_from_bucket(test_document_uuid)
        
        mock_storage.delete_blob.assert_called_once_with(test_document_uuid, Stage.OCR_RAW, ".json")
        assert deletion_success is True
    
    def test_returns_false_when_ocr_results_not_found_for_deletion(self):
        """Test that False is returned when OCR results don't exist for deletion."""
        test_document_uuid = "test-uuid-delete-not-found"
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.delete_blob.return_value = False
            
            deletion_success = delete_ocr_results_from_bucket(test_document_uuid)
        
        mock_storage.delete_blob.assert_called_once_with(test_document_uuid, Stage.OCR_RAW, ".json")
        assert deletion_success is False


class TestListOcrResultsInBucket:
    """Test listing OCR results in bucket."""
    
    def test_lists_all_ocr_result_files_in_bucket(self):
        """Test that all OCR result files are listed from bucket."""
        # Mock the list_blobs_in_stage method to return blob names
        mock_blob_names = [
            "uuid1.json",
            "uuid2.json", 
            "uuid3.json",
            "uuid4.json"
        ]
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.list_blobs_in_stage.return_value = mock_blob_names
            
            document_uuids = list_ocr_results_in_bucket()
        
        expected_uuids = ["uuid1", "uuid2", "uuid3", "uuid4"]
        assert document_uuids == expected_uuids
    
    def test_returns_empty_list_when_no_ocr_results_exist(self):
        """Test that empty list is returned when no OCR results exist."""
        mock_blob_names = []
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.list_blobs_in_stage.return_value = mock_blob_names
            
            document_uuids = list_ocr_results_in_bucket()
        
        assert document_uuids == []
    
    def test_returns_empty_list_when_exception_occurs(self):
        """Test that empty list is returned when exception occurs during listing."""
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.list_blobs_in_stage.side_effect = Exception("Storage error")
            
            document_uuids = list_ocr_results_in_bucket()
        
        assert document_uuids == []
    
    def test_filters_only_json_files_from_ocr_raw_stage(self):
        """Test that only JSON files from OCR_RAW stage are included."""
        # Mock blob names that include both JSON and non-JSON files
        mock_blob_names = [
            "uuid1.json",
            "uuid2.txt",  # Should be filtered out (not JSON)
            "uuid3.json",
            "uuid4.txt",  # Should be filtered out (not JSON)
            "uuid5.json"
        ]
        
        with patch('src.ocr.storage.get_storage') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            mock_storage.list_blobs_in_stage.return_value = mock_blob_names
            
            document_uuids = list_ocr_results_in_bucket()
        
        # Only JSON files should be included
        expected_uuids = ["uuid1", "uuid3", "uuid5"]
        assert document_uuids == expected_uuids 