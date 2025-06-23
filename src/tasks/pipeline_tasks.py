import asyncio
import logging
import traceback
from celery import chain
from src.tasks.celery_app import celery_app
from src.ocr.extraction import (
    trigger_extraction,
    perform_ocr,
    postprocess_ocr,
    run_llm_extraction,
    generate_visualization,
    _update_extraction_job_status,
)

logger = logging.getLogger(__name__)


def handle_extraction_error(document_id: str, exception: Exception, task_name: str) -> None:
    """
    Handle extraction errors by logging and updating status.
    
    Args:
        document_id: The document ID that failed
        exception: The exception that occurred
        task_name: Name of the task that failed
    """
    error_message = f"Task {task_name} failed for document {document_id}: {str(exception)}"
    logger.error(error_message, exc_info=True)
    
    # Update extraction job status to "Fehlerhaft"
    try:
        _update_extraction_job_status(document_id, "Fehlerhaft")
        logger.info(f"Updated status to 'Fehlerhaft' for document {document_id}")
    except Exception as status_update_error:
        logger.error(f"Failed to update status for document {document_id}: {status_update_error}")


@celery_app.task(bind=True)
def trigger_extraction_task(self, document_id: str) -> str:
    """Celery task to trigger the extraction process."""
    task_name = "trigger_extraction"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        trigger_extraction(document_id)
        logger.info(f"Successfully completed {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise


@celery_app.task(bind=True)
def perform_ocr_task(self, document_id: str) -> str:
    """Celery task to perform OCR."""
    task_name = "perform_ocr"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        perform_ocr(document_id)
        logger.info(f"Successfully completed {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise


@celery_app.task(bind=True)
def postprocess_ocr_task(self, document_id: str) -> str:
    """Celery task to post-process OCR results."""
    task_name = "postprocess_ocr"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        postprocess_ocr(document_id)
        logger.info(f"Successfully completed {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise


@celery_app.task(bind=True)
def run_llm_extraction_task(self, document_id: str) -> str:
    """Celery task to run LLM extraction."""
    task_name = "run_llm_extraction"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        asyncio.run(run_llm_extraction(document_id))
        logger.info(f"Successfully completed {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise


@celery_app.task(bind=True)
def generate_visualization_task(self, document_id: str) -> str:
    """Celery task to generate visualization."""
    task_name = "generate_visualization"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        generate_visualization(document_id)
        logger.info(f"Successfully completed {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise


@celery_app.task(bind=True)
def run_full_pipeline(self, document_id: str):
    """Celery task to run the full extraction pipeline."""
    task_name = "run_full_pipeline"
    logger.info(f"Starting {task_name} for document {document_id}")
    
    try:
        pipeline = chain(
            trigger_extraction_task.s(document_id=document_id),
            perform_ocr_task.s(),
            postprocess_ocr_task.s(),
            run_llm_extraction_task.s(),
            generate_visualization_task.s(),
        )
        pipeline.apply_async()
        logger.info(f"Successfully initiated {task_name} for document {document_id}")
        return document_id
    except Exception as e:
        handle_extraction_error(document_id, e, task_name)
        # Re-raise the exception to mark the task as failed
        raise 