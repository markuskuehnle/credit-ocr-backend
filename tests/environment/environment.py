import os
import time
import requests
import logging
import sys
from pathlib import Path
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from src.config import AppConfig

from tests.environment.ollama import start_ollama

FORMAT = "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)s]: %(message)s"
logging.basicConfig(stream=sys.stdout, format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)

# Load configuration
app_config = AppConfig("config")

curr_dir = str(Path(__file__).parent)


def setup_environment():
    logger.info("Setup test environment")

    # Start the generative model with ollama
    ollama_generative = start_ollama(
        model_name=app_config.generative_llm.model_name,
        port=int(app_config.generative_llm.url.split(":")[-1]),
        cache_dir="ollama_cache_generative",
    )
    
    return ollama_generative

def teardown_environment():
    logging.info("Tearing down the test environment, nothing do to")
