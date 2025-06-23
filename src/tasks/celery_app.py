from celery import Celery

# Create a Celery instance
celery_app = Celery(
    "credit_ocr_demo_backend",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["src.tasks.pipeline_tasks"],
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Berlin",
    enable_utc=True,
)

if __name__ == "__main__":
    celery_app.start() 