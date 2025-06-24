from celery import Celery
from src.config import AppConfig

# Load configuration
app_config = AppConfig()

# Create a Celery instance
celery_app = Celery(
    "credit_ocr_demo_backend",
    broker=app_config.redis.broker_url,
    backend=app_config.redis.result_backend,
    include=["src.tasks.pipeline_tasks"],
)

# Configuration from config files
celery_app.conf.update(
    task_serializer=app_config.redis.task_serializer,
    accept_content=app_config.redis.accept_content,
    result_serializer=app_config.redis.result_serializer,
    timezone=app_config.redis.timezone,
    enable_utc=app_config.redis.enable_utc,
)

if __name__ == "__main__":
    celery_app.start() 