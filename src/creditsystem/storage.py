"""
Credit System Storage Helper

Provides blob storage management for the credit system backend with
five processing stages: raw, ocr-raw, ocr-clean, llm, annotated.
"""

import os
import threading
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional
import logging

from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceExistsError

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Processing stages for credit documents."""
    RAW = "raw"
    OCR_RAW = "ocr-raw"
    OCR_CLEAN = "ocr-clean"
    LLM = "llm"
    ANNOTATED = "annotated"


class BlobStorage:
    """Thread-safe singleton for blob storage operations."""
    
    _instance: Optional['BlobStorage'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        # Get connection string from environment or default to Azurite
        self.connection_string = os.getenv(
            'AZURE_STORAGE_CONNECTION_STRING',
            'DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;'
            'AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;'
            'BlobEndpoint=http://localhost:10000/devstoreaccount1;'
        )
        
        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Container name for credit documents
        self.container_name = "credit-docs"
        
        # Lazy initialization - don't create container until first use
        self._container_initialized = False
        self._container_lock = threading.Lock()
        
        self._initialized = True
        logger.info(f"BlobStorage initialized with container: {self.container_name}")
    
    def _ensure_container_exists(self) -> None:
        """Ensure the credit-docs container exists."""
        if self._container_initialized:
            return
            
        with self._container_lock:
            if self._container_initialized:
                return
                
            try:
                container_client = self.blob_service_client.get_container_client(self.container_name)
                container_client.create_container()
                logger.info(f"Container '{self.container_name}' created successfully")
            except ResourceExistsError:
                logger.debug(f"Container '{self.container_name}' already exists")
            except Exception as e:
                logger.error(f"Failed to create container '{self.container_name}': {e}")
                raise
            
            self._container_initialized = True
    
    def ensure_container_ready(self) -> None:
        """Ensure the credit-docs container is ready for use (for test setup)."""
        self._ensure_container_exists()
    
    def blob_path(self, uuid: str, stage: Stage, ext: str) -> PurePosixPath:
        """
        Build blob path for a document at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension (e.g., '.pdf', '.json')
            
        Returns:
            PurePosixPath representing the blob path
        """
        if not ext.startswith('.'):
            ext = f'.{ext}'
        
        path = PurePosixPath(stage.value) / f"{uuid}{ext}"
        return path
    
    def blob_client(self, uuid: str, stage: Stage, ext: str) -> BlobClient:
        """
        Get blob client for a document at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension (e.g., '.pdf', '.json')
            
        Returns:
            BlobClient for the specified blob
        """
        self._ensure_container_exists()
        blob_path = self.blob_path(uuid, stage, ext)
        container_client = self.blob_service_client.get_container_client(self.container_name)
        return container_client.get_blob_client(str(blob_path))
    
    def upload_blob(self, uuid: str, stage: Stage, ext: str, data: bytes, overwrite: bool = True) -> None:
        """
        Upload data to a blob at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension
            data: Data to upload
            overwrite: Whether to overwrite existing blob
        """
        blob_client = self.blob_client(uuid, stage, ext)
        blob_client.upload_blob(data, overwrite=overwrite)
        logger.info(f"Uploaded blob: {self.blob_path(uuid, stage, ext)}")
    
    def download_blob(self, uuid: str, stage: Stage, ext: str) -> Optional[bytes]:
        """
        Download data from a blob at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension
            
        Returns:
            Blob data as bytes or None if not found
        """
        try:
            blob_client = self.blob_client(uuid, stage, ext)
            blob_data = blob_client.download_blob()
            data = blob_data.readall()
            logger.info(f"Downloaded blob: {self.blob_path(uuid, stage, ext)}")
            return data
        except Exception as e:
            logger.warning(f"Failed to download blob {self.blob_path(uuid, stage, ext)}: {e}")
            return None
    
    def blob_exists(self, uuid: str, stage: Stage, ext: str) -> bool:
        """
        Check if a blob exists at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            blob_client = self.blob_client(uuid, stage, ext)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
    
    def delete_blob(self, uuid: str, stage: Stage, ext: str) -> bool:
        """
        Delete a blob at a specific stage.
        
        Args:
            uuid: Document UUID
            stage: Processing stage
            ext: File extension
            
        Returns:
            True if blob was deleted, False if it didn't exist
        """
        try:
            blob_client = self.blob_client(uuid, stage, ext)
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {self.blob_path(uuid, stage, ext)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete blob {self.blob_path(uuid, stage, ext)}: {e}")
            return False


def get_storage() -> BlobStorage:
    """Get the singleton storage instance."""
    return BlobStorage()


def ensure_credit_docs_container() -> None:
    """Ensure the credit-docs container is ready for use (for test setup)."""
    storage = get_storage()
    storage.ensure_container_ready() 