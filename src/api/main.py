"""
FastAPI application for Credit OCR Demo Backend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import logging
import os
import uuid
from pathlib import Path

from src.config import AppConfig
from src.creditsystem.storage import get_storage, Stage
from src.ocr.extraction import trigger_extraction
from src.dms_mock.environment import DmsMockEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit OCR Demo Backend",
    description="Backend API for credit request document processing and OCR extraction",
    version="1.0.0"
)

# Global configuration
app_config = AppConfig()

# Initialize storage
storage = get_storage()

# Initialize DMS mock environment (for development)
dms_environment = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global dms_environment
    
    logger.info("Starting Credit OCR Demo Backend")
    
    # Initialize DMS mock environment for development (but not in test mode)
    if os.getenv("ENVIRONMENT", "development") == "development" and not os.getenv("TESTING"):
        logger.info("Initializing DMS mock environment")
        dms_environment = DmsMockEnvironment()
        dms_environment.start()
        
        # Set environment variables for the application
        os.environ["POSTGRES_HOST"] = "localhost"
        os.environ["POSTGRES_PORT"] = str(dms_environment.postgres_port)
        os.environ["POSTGRES_DB"] = "dms_meta"
        os.environ["POSTGRES_USER"] = "dms"
        os.environ["POSTGRES_PASSWORD"] = "dms"
        
        # Set Azure storage connection string
        azure_connection_string = f"DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:{dms_environment.azurite_port}/devstoreaccount1;"
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = azure_connection_string
        
        logger.info(f"DMS mock environment started - PostgreSQL: {dms_environment.postgres_port}, Azurite: {dms_environment.azurite_port}")
    else:
        logger.info("Skipping DMS mock environment initialization (test mode or production)")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global dms_environment
    
    if dms_environment:
        logger.info("Stopping DMS mock environment")
        dms_environment.stop()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check backend health (DB, Redis, etc.)."""
    health_status = {
        "database": "unknown",
        "redis": "unknown", 
        "ocr_service": "unknown",
        "llm_service": "unknown"
    }
    
    try:
        # Check database connection
        from src.ocr.extraction import _get_database_connection
        conn = _get_database_connection()
        if conn:
            conn.close()
            health_status["database"] = "ok"
        else:
            health_status["database"] = "error"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "error"
    
    try:
        # Check Redis connection
        import redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        health_status["redis"] = "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = "error"
    
    try:
        # Check Azure storage (OCR service)
        storage.blob_service_client.get_service_properties()
        health_status["ocr_service"] = "ok"
    except Exception as e:
        logger.error(f"OCR service health check failed: {e}")
        health_status["ocr_service"] = "error"
    
    try:
        # Check Ollama (LLM service)
        import aiohttp
        import asyncio
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        ollama_port = int(os.getenv('OLLAMA_PORT', 11435))
        
        async def check_ollama():
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{ollama_host}:{ollama_port}/api/tags") as response:
                    return response.status == 200
        
        if asyncio.run(check_ollama()):
            health_status["llm_service"] = "ok"
        else:
            health_status["llm_service"] = "error"
    except Exception as e:
        logger.error(f"LLM service health check failed: {e}")
        health_status["llm_service"] = "error"
    
    return health_status

# Document upload endpoint
@app.post("/credit-request/{credit_request_id}/documents")
async def upload_documents(
    credit_request_id: str,
    files: List[UploadFile] = File(...),
    document_type: Optional[str] = Form(None)
):
    """Upload one or multiple documents for a credit request. Triggers extraction jobs for each document."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
    
    results = []
    
    for file in files:
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Read file content
            file_content = await file.read()
            
            # Upload to RAW stage
            filename = f"{document_id}.pdf" if file.filename.lower().endswith('.pdf') else f"{document_id}{Path(file.filename).suffix}"
            file_ext = Path(file.filename).suffix.lower()
            storage.upload_blob(document_id, Stage.RAW, file_ext, file_content)
            
            # Store document metadata in database
            from src.ocr.extraction import store_document_metadata
            store_document_metadata(
                document_id=document_id,
                credit_request_id=credit_request_id,
                filename=file.filename,
                document_type=document_type or "Unknown",
                status="Extraktion ausstehend"
            )
            
            # Trigger extraction
            job_id = trigger_extraction(document_id)
            
            results.append({
                "document_id": document_id,
                "status": "Extraktion ausstehend"
            })
            
            logger.info(f"Document uploaded: {document_id}, extraction job: {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload document {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")
    
    return results

# Get documents for credit request
@app.get("/credit-request/{credit_request_id}/documents")
async def get_documents(credit_request_id: str):
    """Returns all uploaded documents and their extraction status."""
    
    try:
        from src.ocr.extraction import get_documents_for_credit_request
        documents = get_documents_for_credit_request(credit_request_id)
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to get documents for credit request {credit_request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

# Get document status
@app.get("/document/{document_id}/status")
async def get_document_status(document_id: str):
    """Returns extraction status and potential error info for a document."""
    
    try:
        from src.ocr.extraction import get_document_status
        status = get_document_status(document_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

# Get extracted fields
@app.get("/document/{document_id}/extracted-fields")
async def get_extracted_fields(document_id: str):
    """Returns all extracted fields from the LLM postprocessing step."""
    
    try:
        # Check if document exists
        from src.ocr.extraction import get_document_status
        status = get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get LLM results from storage
        llm_filename = f"{document_id}.json"
        try:
            llm_content = storage.download_blob(document_id, Stage.LLM, ".json")
            import json
            fields = json.loads(llm_content)
            return fields
        except Exception as e:
            logger.error(f"Failed to get LLM results for document {document_id}: {e}")
            raise HTTPException(status_code=404, detail="Extracted fields not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get extracted fields for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get extracted fields: {str(e)}")

# Get annotated PDF overlay
@app.get("/document/{document_id}/overlay")
async def get_document_overlay(document_id: str):
    """Returns the annotated PDF with bounding boxes as a binary blob."""
    
    try:
        # Check if document exists
        from src.ocr.extraction import get_document_status
        status = get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get annotated PDF from storage
        overlay_filename = f"{document_id}_annotated.pdf"
        try:
            overlay_content = storage.download_blob(document_id, Stage.ANNOTATED, "_annotated.pdf")
            
            # Save to temporary file for response
            temp_dir = Path("/tmp/credit-ocr-overlays")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / overlay_filename
            
            with open(temp_file, "wb") as f:
                f.write(overlay_content)
            
            return FileResponse(
                path=temp_file,
                media_type="application/pdf",
                filename=overlay_filename
            )
            
        except Exception as e:
            logger.error(f"Failed to get overlay for document {document_id}: {e}")
            raise HTTPException(status_code=404, detail="Overlay not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get overlay for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get overlay: {str(e)}")

# Reprocess document
@app.post("/document/{document_id}/reprocess")
async def reprocess_document(document_id: str):
    """Re-run the full extraction pipeline for a failed or outdated document."""
    
    try:
        # Check if document exists
        from src.ocr.extraction import get_document_status
        status = get_document_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if already processing
        if status.get("status") in ["Extraktion ausstehend", "OCR läuft", "LLM läuft"]:
            raise HTTPException(status_code=409, detail="Document is already being processed")
        
        # Update status to pending
        from src.ocr.extraction import update_document_status
        update_document_status(document_id, "Extraktion ausstehend")
        
        # Trigger extraction
        job_id = trigger_extraction(document_id)
        
        return {
            "document_id": document_id,
            "status": "Extraktion ausstehend"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reprocess document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 