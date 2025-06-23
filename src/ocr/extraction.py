import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
import tempfile
import asyncio
import psycopg2
import os

from src.creditsystem.storage import get_storage, Stage
from src.ocr.storage import write_ocr_results_to_bucket, read_ocr_results_from_bucket
from src.ocr.azure_ocr_client import analyze_single_document_with_azure
from src.ocr.postprocess import extract_text_lines_with_bbox_and_confidence, normalize_ocr_lines
from src.llm.field_extractor import extract_fields_with_llm as llm_extract_fields
from src.visualization.pdf_visualizer import visualize_extracted_fields
from src.config import DocumentProcessingConfig, AppConfig

logger = logging.getLogger(__name__)


def _get_database_connection():
    """Get database connection for status updates."""
    # Try to get connection details from environment (DMS mock)
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "dms_meta")
    user = os.getenv("POSTGRES_USER", "dms")
    password = os.getenv("POSTGRES_PASSWORD", "dms")
    
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=5  # Add timeout to prevent hanging
        )
        return connection
    except Exception as e:
        logger.warning(f"Failed to connect to database for status updates: {e}")
        return None


def trigger_extraction(document_id: str) -> str:
    """
    Trigger document extraction process.
    
    Fetches metadata from DMS, saves original PDF to raw bucket,
    and creates extraction job entry with pending status.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        Extraction job ID
    """
    storage_client = get_storage()
    
    # Generate extraction job ID
    extraction_job_id = str(uuid.uuid4())
    
    # Fetch metadata from DMS (placeholder - implement actual DMS integration)
    document_metadata = _fetch_document_metadata_from_dms(document_id)
    
    # Save original PDF to credit-docs-raw bucket
    pdf_data = _fetch_original_pdf_from_dms(document_id)
    storage_client.upload_blob(
        uuid=document_id,
        stage=Stage.RAW,
        ext=".pdf",
        data=pdf_data,
        overwrite=True
    )
    
    # Create extraction job entry with pending status
    extraction_job = {
        "job_id": extraction_job_id,
        "document_id": document_id,
        "status": "Extraktion ausstehend",
        "metadata": document_metadata,
        "created_at": _get_current_timestamp()
    }
    
    _save_extraction_job(extraction_job)
    
    logger.info(f"Extraction triggered for document {document_id} with job ID {extraction_job_id}")
    return extraction_job_id


def perform_ocr(document_id: str) -> Dict[str, Any]:
    """
    Perform OCR on document.
    
    Loads raw PDF from credit-docs-raw bucket, calls Azure OCR,
    saves raw JSON to credit-docs-ocr-raw bucket, and updates status.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        OCR results dictionary
    """
    storage_client = get_storage()
    
    # Load raw PDF from credit-docs-raw bucket
    pdf_data = storage_client.download_blob(document_id, Stage.RAW, ".pdf")
    if pdf_data is None:
        raise FileNotFoundError(f"Raw PDF not found for document {document_id}")
    
    # Save PDF to temporary file for Azure OCR
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name
    
    try:
        # Call Azure OCR using the existing function
        azure_ocr_result = analyze_single_document_with_azure(temp_pdf_path)
        
        # Extract text lines with bounding boxes and confidence
        ocr_lines = extract_text_lines_with_bbox_and_confidence(azure_ocr_result)
        
        # Prepare OCR results structure
        ocr_results = {
            "azure_raw_result": azure_ocr_result.to_dict(),
            "extracted_lines": ocr_lines,
            "document_id": document_id,
            "processing_timestamp": _get_current_timestamp()
        }
        
        # Save raw OCR results to credit-docs-ocr-raw bucket
        write_ocr_results_to_bucket(
            document_uuid=document_id,
            ocr_results=ocr_results,
            metadata={"source": "azure_ocr", "stage": "raw"}
        )
        
        # Update extraction job status
        _update_extraction_job_status(document_id, "OCR abgeschlossen")
        
        logger.info(f"OCR completed for document {document_id}")
        return ocr_results
        
    finally:
        # Clean up temporary file
        Path(temp_pdf_path).unlink(missing_ok=True)


def postprocess_ocr(document_id: str) -> Dict[str, Any]:
    """
    Post-process OCR results.
    
    Loads raw OCR JSON, normalizes reading order, handles rotation,
    cleans garbage, saves to credit-docs-ocr-clean bucket, and updates status.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        Cleaned OCR results dictionary
    """
    storage_client = get_storage()
    
    # Load raw OCR results
    raw_ocr_data = read_ocr_results_from_bucket(document_id)
    if raw_ocr_data is None:
        raise FileNotFoundError(f"Raw OCR results not found for document {document_id}")
    
    # Get the extracted lines from raw OCR results
    raw_ocr_lines = raw_ocr_data["ocr_results"]["extracted_lines"]
    
    # Normalize OCR lines using the existing function
    normalized_ocr_lines = normalize_ocr_lines(raw_ocr_lines)
    
    # Prepare cleaned OCR results
    cleaned_ocr_results = {
        "normalized_lines": normalized_ocr_lines,
        "original_lines": raw_ocr_lines,
        "document_id": document_id,
        "processing_timestamp": _get_current_timestamp(),
        "processing_metadata": {
            "reading_order": "normalized",
            "rotation_corrected": True,
            "garbage_cleaned": True
        }
    }
    
    # Save cleaned OCR results to credit-docs-ocr-clean bucket
    storage_client.upload_blob(
        uuid=document_id,
        stage=Stage.OCR_CLEAN,
        ext=".json",
        data=json.dumps(cleaned_ocr_results, indent=2, ensure_ascii=False).encode('utf-8'),
        overwrite=True
    )
    
    # Update extraction job status
    _update_extraction_job_status(document_id, "Extraktion abgeschlossen")
    
    logger.info(f"OCR post-processing completed for document {document_id}")
    return cleaned_ocr_results


async def run_llm_extraction(document_id: str) -> Dict[str, Any]:
    """
    Run LLM field extraction.
    
    Loads clean OCR JSON, prompts LLM for field extraction,
    saves structured fields to credit-docs-llm bucket, and updates status.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        Extracted fields dictionary
    """
    storage_client = get_storage()
    
    # Load clean OCR results
    clean_ocr_data = storage_client.download_blob(document_id, Stage.OCR_CLEAN, ".json")
    if clean_ocr_data is None:
        raise FileNotFoundError(f"Clean OCR results not found for document {document_id}")
    
    clean_ocr_results = json.loads(clean_ocr_data.decode('utf-8'))
    
    # Get normalized lines for LLM processing
    normalized_lines = clean_ocr_results["normalized_lines"]
    original_lines = clean_ocr_results["original_lines"]
    
    # Load document configuration
    doc_config = DocumentProcessingConfig.from_json("config/document_types.conf")
    credit_request_config = doc_config.document_types["credit_request"]
    
    # Create LLM client using configuration
    from src.llm.client import OllamaClient
    app_config = AppConfig("tests/resources/test_application.conf")
    llm_client = OllamaClient(
        base_url=app_config.generative_llm.url,
        model_name=app_config.generative_llm.model_name
    )
    
    # Extract fields using LLM with the existing function
    extracted_fields_result = await llm_extract_fields(
        ocr_lines=normalized_lines,
        doc_config=credit_request_config,
        llm_client=llm_client,
        original_ocr_lines=original_lines
    )
    
    # Save each extracted field to the database
    extracted_fields = extracted_fields_result.get("extracted_fields", {})
    for field_name, field_data in extracted_fields.items():
        # Handle both simple values and complex field data
        if isinstance(field_data, dict):
            field_value = field_data.get("value", str(field_data))
            field_position = field_data.get("position")
            field_confidence = field_data.get("confidence")
        else:
            field_value = str(field_data)
            field_position = None
            field_confidence = None
        
        # Save the field to the database
        save_extracted_field(
            document_id=document_id,
            field_name=field_name,
            value=field_value,
            position=field_position,
            confidence=field_confidence
        )
    
    # Prepare final result structure
    final_result = {
        "extracted_fields": extracted_fields_result["extracted_fields"],
        "missing_fields": extracted_fields_result["missing_fields"],
        "validation_results": extracted_fields_result["validation_results"],
        "document_id": document_id,
        "processing_timestamp": _get_current_timestamp(),
        "llm_metadata": {
            "model_used": "ollama",
            "extraction_method": "llm_assisted"
        }
    }
    
    # Save extracted fields to credit-docs-llm bucket
    storage_client.upload_blob(
        uuid=document_id,
        stage=Stage.LLM,
        ext=".json",
        data=json.dumps(final_result, indent=2, ensure_ascii=False).encode('utf-8'),
        overwrite=True
    )
    
    # Update extraction job status
    _update_extraction_job_status(document_id, "Fertig")
    
    logger.info(f"LLM extraction completed for document {document_id}")
    return final_result


def generate_visualization(document_id: str) -> str:
    """
    Generate visualization with bounding boxes.
    
    Loads original PDF and extracted fields, generates PDF overlay
    with bounding boxes, saves to credit-docs-annotated bucket, and updates status.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        Path to generated visualization
    """
    storage_client = get_storage()
    
    # Load original PDF
    pdf_data = storage_client.download_blob(document_id, Stage.RAW, ".pdf")
    if pdf_data is None:
        raise FileNotFoundError(f"Original PDF not found for document {document_id}")
    
    # Load clean OCR results for visualization
    clean_ocr_data = storage_client.download_blob(document_id, Stage.OCR_CLEAN, ".json")
    if clean_ocr_data is None:
        raise FileNotFoundError(f"Clean OCR results not found for document {document_id}")
    
    clean_ocr_results = json.loads(clean_ocr_data.decode('utf-8'))
    normalized_lines = clean_ocr_results["normalized_lines"]
    
    # Load document configuration
    doc_config = DocumentProcessingConfig.from_json("config/document_types.conf")
    credit_request_config = doc_config.document_types["credit_request"]
    
    # Save PDF to temporary file for visualization
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_pdf_path = temp_pdf.name
    
    try:
        # Create temporary output path for visualization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        # Generate visualization using the existing function
        visualize_extracted_fields(
            pdf_path=Path(temp_pdf_path),
            normalized_data=normalized_lines,
            output_path=Path(temp_output_path),
            doc_config=credit_request_config
        )
        
        # Find the first generated PNG (e.g., *_page1.png)
        output_dir = Path(temp_output_path).parent
        output_stem = Path(temp_output_path).stem
        page1_png = next(output_dir.glob(f"{output_stem}_page1.png"), None)
        if page1_png is None or not page1_png.exists():
            raise FileNotFoundError(f"Visualization PNG not found: {output_stem}_page1.png")
        
        # Read the generated visualization
        with open(page1_png, "rb") as f:
            annotated_pdf_data = f.read()
        
        # Save annotated PDF to credit-docs-annotated bucket
        storage_client.upload_blob(
            uuid=document_id,
            stage=Stage.ANNOTATED,
            ext=".png",
            data=annotated_pdf_data,
            overwrite=True
        )
        
        # Update extraction job status
        _update_extraction_job_status(document_id, "Geprüft")
        
        logger.info(f"Visualization generated for document {document_id}")
        return f"{Stage.ANNOTATED.value}/{document_id}.png"
        
    finally:
        # Clean up temporary files
        Path(temp_pdf_path).unlink(missing_ok=True)
        Path(temp_output_path).unlink(missing_ok=True)
        # Also clean up generated page PNGs
        for page_png in Path(temp_output_path).parent.glob(f"{Path(temp_output_path).stem}_page*.png"):
            page_png.unlink(missing_ok=True)


# Helper functions (implementations would depend on your specific DMS and LLM setup)

def _fetch_document_metadata_from_dms(document_id: str) -> Dict[str, Any]:
    """Fetch document metadata from DMS."""
    # Placeholder implementation
    return {
        "document_id": document_id,
        "source": "dms",
        "timestamp": _get_current_timestamp()
    }


def _fetch_original_pdf_from_dms(document_id: str) -> bytes:
    """Fetch original PDF from DMS."""
    # For testing, use the sample PDF file
    sample_pdf_path = Path("tests/tmp/sample_creditrequest.pdf")
    if sample_pdf_path.exists():
        return sample_pdf_path.read_bytes()
    else:
        # Fallback to placeholder if sample file doesn't exist
        return b"%PDF-1.4\n%Sample PDF content\n%%EOF"


def _save_extraction_job(extraction_job: Dict[str, Any]) -> None:
    """Save extraction job to database."""
    connection = _get_database_connection()
    if connection is None:
        logger.warning("Database connection not available, skipping job save")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO Extraktionsauftrag (auftrag_id, dokument_id, status, fehlermeldung) 
                VALUES (%s, %s, %s, %s)
                """,
                (
                    extraction_job['job_id'],
                    extraction_job['document_id'],
                    extraction_job['status'],
                    f"Job created with status: {extraction_job['status']}"
                )
            )
            connection.commit()
            logger.info(f"Extraction job saved to database: {extraction_job['job_id']}")
    except Exception as e:
        logger.error(f"Failed to save extraction job to database: {e}")
    finally:
        connection.close()


def _update_extraction_job_status(document_id: str, status: str) -> None:
    """Update extraction job status in database."""
    connection = _get_database_connection()
    if connection is None:
        logger.warning("Database connection not available, skipping status update")
        return
    
    try:
        with connection.cursor() as cursor:
            # Update the most recent extraction job for this document
            cursor.execute(
                """
                UPDATE Extraktionsauftrag 
                SET status = %s, fehlermeldung = %s, abgeschlossen_am = CASE 
                    WHEN %s IN ('Fertig', 'Geprüft', 'Fehlerhaft') THEN NOW() 
                    ELSE abgeschlossen_am 
                END
                WHERE dokument_id = %s 
                AND auftrag_id = (
                    SELECT auftrag_id FROM Extraktionsauftrag 
                    WHERE dokument_id = %s 
                    ORDER BY erstellt_am DESC 
                    LIMIT 1
                )
                """,
                (status, f"Status updated to: {status}", status, document_id, document_id)
            )
            connection.commit()
            logger.info(f"Updated status for document {document_id}: {status}")
    except Exception as e:
        logger.error(f"Failed to update status for document {document_id}: {e}")
    finally:
        connection.close()


def _get_current_timestamp() -> str:
    """Get current timestamp as string."""
    from datetime import datetime
    return datetime.now().isoformat()


def save_extracted_field(document_id: str, field_name: str, value: str, position: Optional[Dict[str, Any]] = None, confidence: Optional[float] = None) -> None:
    """
    Save extracted field to ExtrahierteDaten table.
    
    Args:
        document_id: UUID of the document
        field_name: Name of the extracted field
        value: Extracted value
        position: Position information in the document (optional)
        confidence: Confidence score for the extraction (optional)
    """
    connection = _get_database_connection()
    if connection is None:
        logger.warning("Database connection not available, skipping field save")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ExtrahierteDaten (dokument_id, feldname, wert, position_im_dokument, konfidenzscore)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    field_name,
                    value,
                    json.dumps(position) if position else None,
                    confidence
                )
            )
            connection.commit()
            logger.info(f"Saved extracted field '{field_name}' for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to save extracted field '{field_name}' for document {document_id}: {e}")
    finally:
        connection.close() 