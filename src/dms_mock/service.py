"""
DMS Service for handling document operations.
Provides functions to upload files and manage document metadata.
"""

import logging
import uuid
import hashlib
from pathlib import Path
from typing import Optional
import mimetypes

import psycopg2
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class DmsService:
    """Service for DMS operations."""
    
    def __init__(self, postgres_connection, blob_service_client):
        self.postgres_connection = postgres_connection
        self.blob_service_client = blob_service_client
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def upload_document(self, file_path: Path, document_type: str, 
                       source_filename: Optional[str] = None,
                       linked_entity: Optional[str] = None,
                       linked_entity_id: Optional[str] = None) -> str:
        """
        Upload a document to the DMS with the specified document type.
        
        Args:
            file_path: Path to the file to upload
            document_type: Type of document (e.g., 'Kunden-Ausweis', 'Grundbuchauszug')
            source_filename: Original filename from scanner/upload
            linked_entity: Entity type (e.g., 'KUNDE', 'KREDITANTRAG', 'IMMOBILIE')
            linked_entity_id: ID in the core system
            
        Returns:
            Document ID (UUID) of the created document record
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Calculate file hash
        file_hash = self._calculate_sha256(file_path)
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Use provided source filename or fall back to file name
        if source_filename is None:
            source_filename = file_path.name
        
        # Create blob path with document type and ID
        file_extension = file_path.suffix
        blob_name = f"raw/{document_type}/{document_id}{file_extension}"
        
        logger.info(f"Uploading document {document_id} of type '{document_type}' to blob storage")
        
        # Upload file to blob storage
        container_client = self.blob_service_client.get_container_client("documents")
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(file_path, 'rb') as file_data:
            blob_client.upload_blob(file_data, overwrite=True)
        
        logger.info(f"File uploaded to blob storage: {blob_name}")
        
        # Create database record
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO dokument (
                    id, blob_path, mime_type, hash_sha256, source_filename, 
                    document_type, linked_entity, linked_entity_id, textextraction_status
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (document_id, blob_name, mime_type, file_hash, source_filename,
                 document_type, linked_entity, linked_entity_id, "not ready")
            )
            self.postgres_connection.commit()
        
        logger.info(f"Document record created in database with ID: {document_id}")
        
        return document_id
    
    def get_document(self, document_id: str) -> Optional[dict]:
        """
        Retrieve document metadata by ID.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Document metadata dictionary or None if not found
        """
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, blob_path, mime_type, uploaded_at, hash_sha256, 
                       source_filename, document_type, linked_entity, linked_entity_id, 
                       textextraction_status
                FROM dokument 
                WHERE id = %s
                """,
                (document_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return {
                    "id": result[0],
                    "blob_path": result[1],
                    "mime_type": result[2],
                    "uploaded_at": result[3],
                    "hash_sha256": result[4],
                    "source_filename": result[5],
                    "document_type": result[6],
                    "linked_entity": result[7],
                    "linked_entity_id": result[8],
                    "textextraction_status": result[9]
                }
            return None
    
    def list_documents_by_type(self, document_type: str) -> list:
        """
        List all documents of a specific type.
        
        Args:
            document_type: Type of documents to list
            
        Returns:
            List of document metadata dictionaries
        """
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, blob_path, mime_type, uploaded_at, hash_sha256, 
                       source_filename, document_type, linked_entity, linked_entity_id, 
                       textextraction_status
                FROM dokument 
                WHERE document_type = %s
                ORDER BY uploaded_at DESC
                """,
                (document_type,)
            )
            results = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "blob_path": row[1],
                    "mime_type": row[2],
                    "uploaded_at": row[3],
                    "hash_sha256": row[4],
                    "source_filename": row[5],
                    "document_type": row[6],
                    "linked_entity": row[7],
                    "linked_entity_id": row[8],
                    "textextraction_status": row[9]
                }
                for row in results
            ]
    
    def download_document(self, document_id: str) -> Optional[bytes]:
        """
        Download document content from blob storage.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Document content as bytes or None if not found
        """
        document = self.get_document(document_id)
        if not document:
            return None
        
        container_client = self.blob_service_client.get_container_client("documents")
        blob_client = container_client.get_blob_client(document["blob_path"])
        
        try:
            blob_data = blob_client.download_blob()
            return blob_data.readall()
        except Exception as e:
            logger.error(f"Failed to download document {document_id}: {e}")
            return None
    
    def update_textextraction_status(self, document_id: str, status: str) -> bool:
        """
        Update the text extraction status of a document.
        
        Args:
            document_id: UUID of the document
            status: New status ('not ready', 'ready', 'in progress', 'done', 'error')
            
        Returns:
            True if update was successful, False otherwise
        """
        valid_statuses = ['not ready', 'ready', 'in progress', 'done', 'error']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                "UPDATE dokument SET textextraction_status = %s WHERE id = %s",
                (status, document_id)
            )
            self.postgres_connection.commit()
            
            return cursor.rowcount > 0
    
    def create_extraction_job(self, document_id: str, state: str = "PENDING") -> str:
        """
        Create an extraction job for a document.
        
        Args:
            document_id: UUID of the document
            state: Initial state of the job
            
        Returns:
            Job ID (UUID) of the created extraction job
        """
        job_id = str(uuid.uuid4())
        
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO extraction_job (id, dokument_id, state) 
                VALUES (%s, %s, %s)
                """,
                (job_id, document_id, state)
            )
            self.postgres_connection.commit()
        
        logger.info(f"Extraction job created with ID: {job_id} for document: {document_id}")
        return job_id
    
    def update_extraction_job(self, job_id: str, state: str, worker_log: str = None) -> bool:
        """
        Update an extraction job.
        
        Args:
            job_id: UUID of the extraction job
            state: New state
            worker_log: Optional log message
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.postgres_connection.cursor() as cursor:
            if worker_log:
                cursor.execute(
                    """
                    UPDATE extraction_job 
                    SET state = %s, worker_log = %s, finished_at = CASE 
                        WHEN %s IN ('SUCCESS', 'FAILURE') THEN NOW() 
                        ELSE finished_at 
                    END
                    WHERE id = %s
                    """,
                    (state, worker_log, state, job_id)
                )
            else:
                cursor.execute(
                    """
                    UPDATE extraction_job 
                    SET state = %s, finished_at = CASE 
                        WHEN %s IN ('SUCCESS', 'FAILURE') THEN NOW() 
                        ELSE finished_at 
                    END
                    WHERE id = %s
                    """,
                    (state, state, job_id)
                )
            self.postgres_connection.commit()
            
            return cursor.rowcount > 0
    
    def get_extraction_jobs(self, document_id: str) -> list:
        """
        Get all extraction jobs for a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            List of extraction job dictionaries
        """
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, dokument_id, created_at, finished_at, state, worker_log
                FROM extraction_job 
                WHERE dokument_id = %s
                ORDER BY created_at DESC
                """,
                (document_id,)
            )
            results = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "dokument_id": row[1],
                    "created_at": row[2],
                    "finished_at": row[3],
                    "state": row[4],
                    "worker_log": row[5]
                }
                for row in results
            ] 