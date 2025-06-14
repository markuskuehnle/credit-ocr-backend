from typing import Any
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from dotenv import load_dotenv
import os
import logging

load_dotenv()

AZURE_OCR_ENDPOINT: str = os.getenv("AZURE_OCR_ENDPOINT", "")
AZURE_OCR_KEY: str = os.getenv("AZURE_OCR_KEY", "")

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not AZURE_OCR_ENDPOINT or not AZURE_OCR_KEY:
    logger.error("Azure OCR credentials are missing. Check your .env file.")
    raise EnvironmentError("Missing Azure OCR credentials.")


document_analysis_client: DocumentAnalysisClient = DocumentAnalysisClient(
    endpoint=AZURE_OCR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_OCR_KEY)
)


def analyze_single_document_with_azure(document_path: str) -> Any:
    """Analyzes a single document using Azure Document Intelligence and returns the raw result."""

    try:
        with open(document_path, "rb") as document_file:
            logger.info("Sending document to Azure OCR: %s", document_path)

            poller = document_analysis_client.begin_analyze_document(
                model_id="prebuilt-layout",
                document=document_file,
            )
            result = poller.result()

            logger.info("Result retrieved from Azure OCR for %s", document_path)
            return result

    except FileNotFoundError:
        logger.error("File not found: %s", document_path)
        raise

    except AzureError as azure_err:
        logger.error("Azure OCR request failed: %s", str(azure_err))
        raise

    except Exception as e:
        logger.exception("Unexpected error during OCR processing")
        raise
