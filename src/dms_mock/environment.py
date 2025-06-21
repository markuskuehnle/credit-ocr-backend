"""
DMS Mock Environment for development and testing.
Provides a local Postgres and Azurite setup for document management system emulation.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import psycopg2
from azure.storage.blob import BlobServiceClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

logger = logging.getLogger(__name__)

class DmsMockEnvironment:
    """Manages DMS mock environment with PostgreSQL and Azurite."""
    
    def __init__(self):
        self.postgres_container: Optional[DockerContainer] = None
        self.azurite_container: Optional[DockerContainer] = None
        self.postgres_connection = None
        self.blob_service_client = None
        self.postgres_port = None
        self.azurite_port = None
        self._started = False
    
    def start(self) -> None:
        """Start PostgreSQL and Azurite containers."""
        if self._started:
            logger.warning("DMS mock environment already started")
            return
            
        logger.info("Starting DMS mock environment")
        
        try:
            # Start PostgreSQL with random port
            self.postgres_container = (
                DockerContainer("postgres:15-alpine")
                .with_env("POSTGRES_DB", "dms_meta")
                .with_env("POSTGRES_USER", "dms")
                .with_env("POSTGRES_PASSWORD", "dms")
                .with_bind_ports(5432, None)  # Use random port
                .with_name("dms-postgres")
            )
            self.postgres_container.start()
            
            # Get the assigned port
            self.postgres_port = self.postgres_container.get_exposed_port(5432)
            logger.info(f"PostgreSQL started on port {self.postgres_port}")
            
            # Wait for PostgreSQL to be ready
            wait_for_logs(self.postgres_container, ".*database system is ready to accept connections.*", timeout=60)
            
            # Start Azurite with random port
            self.azurite_container = (
                DockerContainer("mcr.microsoft.com/azure-storage/azurite:latest")
                .with_command(["azurite", "--location", "/data", "--blobHost", "0.0.0.0"])
                .with_bind_ports(10000, None)  # Use random port
                .with_name("dms-azurite")
            )
            self.azurite_container.start()
            
            # Get the assigned port
            self.azurite_port = self.azurite_container.get_exposed_port(10000)
            logger.info(f"Azurite started on port {self.azurite_port}")
            
            # Wait for Azurite to be ready
            wait_for_logs(self.azurite_container, ".*Azurite Blob service is starting.*", timeout=60)
            
            # Initialize database schema
            self._setup_database()
            
            # Initialize blob storage
            self._setup_blob_storage()
            
            self._started = True
            logger.info("DMS mock environment started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start DMS mock environment: {e}")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop all containers."""
        logger.info("Stopping DMS mock environment")
        
        # Close database connection
        if self.postgres_connection:
            try:
                self.postgres_connection.close()
                logger.info("Closed PostgreSQL connection")
            except Exception as e:
                logger.warning(f"Failed to close PostgreSQL connection: {e}")
            finally:
                self.postgres_connection = None
        
        # Stop containers
        if self.azurite_container:
            try:
                self.azurite_container.stop()
                logger.info("Stopped Azurite container")
            except Exception as e:
                logger.warning(f"Failed to stop Azurite container: {e}")
            finally:
                self.azurite_container = None
        
        if self.postgres_container:
            try:
                self.postgres_container.stop()
                logger.info("Stopped PostgreSQL container")
            except Exception as e:
                logger.warning(f"Failed to stop PostgreSQL container: {e}")
            finally:
                self.postgres_container = None
        
        self._started = False
        logger.info("DMS mock environment stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def _setup_database(self) -> None:
        """Initialize database schema."""
        # Try to find schema.sql in project root or src/dms_mock/
        candidates = [
            Path(__file__).parent.parent.parent / "schema.sql",
            Path(__file__).parent / "schema.sql"
        ]
        schema_path = next((p for p in candidates if p.exists()), None)
        if not schema_path:
            raise FileNotFoundError("Could not find schema.sql for DMS mock environment.")
        
        # Connect to PostgreSQL
        self.postgres_connection = psycopg2.connect(
            host="localhost",
            port=self.postgres_port,
            database="dms_meta",
            user="dms",
            password="dms"
        )
        
        # Execute schema
        with self.postgres_connection.cursor() as cursor:
            with open(schema_path, 'r') as f:
                cursor.execute(f.read())
            self.postgres_connection.commit()
        
        logger.info("Database schema initialized")
    
    def _setup_blob_storage(self) -> None:
        """Initialize blob storage client."""
        connection_string = (
            "DefaultEndpointsProtocol=http;"
            "AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
            f"BlobEndpoint=http://localhost:{self.azurite_port}/devstoreaccount1;"
        )
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create default container
        container_client = self.blob_service_client.get_container_client("documents")
        try:
            container_client.create_container()
        except Exception:
            # Container might already exist
            pass
        
        logger.info("Blob storage initialized")
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection."""
        if not self._started:
            raise RuntimeError("DMS mock environment not started")
        return self.postgres_connection
    
    def get_blob_service_client(self):
        """Get Azure Blob Service client."""
        if not self._started:
            raise RuntimeError("DMS mock environment not started")
        return self.blob_service_client
    
    def get_dms_service(self):
        """Get DMS service for document operations."""
        if not self._started:
            raise RuntimeError("DMS mock environment not started")
        
        from .service import DmsService
        return DmsService(self.postgres_connection, self.blob_service_client) 