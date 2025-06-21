import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.creditsystem.storage import get_storage, Stage

logger = logging.getLogger(__name__)


def write_ocr_results_to_bucket(
    document_uuid: str,
    ocr_results: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Write OCR results to blob storage bucket.
    
    Args:
        document_uuid: Unique identifier for the document
        ocr_results: OCR processing results as dictionary
        metadata: Optional metadata to include with the results
        
    Returns:
        Path to the stored OCR results file
    """
    storage_client = get_storage()
    
    # Prepare the complete data structure
    complete_ocr_data = {
        "document_uuid": document_uuid,
        "timestamp": datetime.utcnow().isoformat(),
        "ocr_results": ocr_results,
        "metadata": metadata or {}
    }
    
    # Convert to JSON bytes
    ocr_data_bytes = json.dumps(complete_ocr_data, indent=2, ensure_ascii=False).encode('utf-8')
    
    # Upload to OCR_RAW stage with JSON extension
    storage_client.upload_blob(
        uuid=document_uuid,
        stage=Stage.OCR_RAW,
        ext=".json",
        data=ocr_data_bytes,
        overwrite=True
    )
    
    # Get the blob path for return value
    blob_path = storage_client.blob_path(document_uuid, Stage.OCR_RAW, ".json")
    
    logger.info(f"OCR results written to bucket: {Stage.OCR_RAW.value}/{blob_path}")
    return str(blob_path)


def read_ocr_results_from_bucket(document_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Read OCR results from blob storage bucket.
    
    Args:
        document_uuid: Unique identifier for the document
        
    Returns:
        OCR results dictionary or None if not found
    """
    storage_client = get_storage()
    
    # Check if blob exists
    if not storage_client.blob_exists(document_uuid, Stage.OCR_RAW, ".json"):
        logger.warning(f"OCR results not found for document: {document_uuid}")
        return None
    
    # Download blob data
    blob_data = storage_client.download_blob(document_uuid, Stage.OCR_RAW, ".json")
    
    if blob_data is None:
        logger.error(f"Failed to download OCR results for document: {document_uuid}")
        return None
    
    # Parse JSON data
    try:
        ocr_data = json.loads(blob_data.decode('utf-8'))
        logger.info(f"OCR results read from bucket for document: {document_uuid}")
        return ocr_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OCR results JSON for document {document_uuid}: {e}")
        return None


def delete_ocr_results_from_bucket(document_uuid: str) -> bool:
    """
    Delete OCR results from blob storage bucket.
    
    Args:
        document_uuid: Unique identifier for the document
        
    Returns:
        True if deleted successfully, False otherwise
    """
    storage_client = get_storage()
    
    success = storage_client.delete_blob(document_uuid, Stage.OCR_RAW, ".json")
    
    if success:
        logger.info(f"OCR results deleted from bucket for document: {document_uuid}")
    else:
        logger.warning(f"OCR results not found for deletion: {document_uuid}")
    
    return success


def list_ocr_results_in_bucket() -> list[str]:
    """
    List all OCR result files in the bucket.
    
    Returns:
        List of document UUIDs that have OCR results
    """
    storage_client = get_storage()
    
    try:
        blob_names = storage_client.list_blobs_in_stage(Stage.OCR_RAW)
        
        # Extract UUIDs from blob names (remove .json extension)
        document_uuids = []
        for blob_name in blob_names:
            if blob_name.endswith('.json'):
                uuid_part = blob_name.replace('.json', '')
                document_uuids.append(uuid_part)
        
        logger.info(f"Found {len(document_uuids)} OCR result files in bucket")
        return document_uuids
    except Exception as e:
        logger.error(f"Failed to list OCR results in bucket: {e}")
        return [] 