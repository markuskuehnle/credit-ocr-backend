import asyncio
from celery import chain
from src.tasks.celery_app import celery_app
from src.ocr.extraction import (
    trigger_extraction,
    perform_ocr,
    postprocess_ocr,
    run_llm_extraction,
    generate_visualization,
)

@celery_app.task
def trigger_extraction_task(document_id: str) -> str:
    """Celery task to trigger the extraction process."""
    trigger_extraction(document_id)
    return document_id

@celery_app.task
def perform_ocr_task(document_id: str) -> str:
    """Celery task to perform OCR."""
    perform_ocr(document_id)
    return document_id

@celery_app.task
def postprocess_ocr_task(document_id: str) -> str:
    """Celery task to post-process OCR results."""
    postprocess_ocr(document_id)
    return document_id

@celery_app.task
def run_llm_extraction_task(document_id: str) -> str:
    """Celery task to run LLM extraction."""
    asyncio.run(run_llm_extraction(document_id))
    return document_id

@celery_app.task
def generate_visualization_task(document_id: str) -> str:
    """Celery task to generate visualization."""
    generate_visualization(document_id)
    return document_id

@celery_app.task
def run_full_pipeline(document_id: str):
    """Celery task to run the full extraction pipeline."""
    pipeline = chain(
        trigger_extraction_task.s(document_id=document_id),
        perform_ocr_task.s(),
        postprocess_ocr_task.s(),
        run_llm_extraction_task.s(),
        generate_visualization_task.s(),
    )
    pipeline.apply_async() 