import logging
from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

logger = logging.getLogger(__name__)
curr_dir = str(Path(__file__).parent)


def start_ollama(model_name: str, port: int, cache_dir: str) -> DockerContainer:
    container = (
        DockerContainer("ollama/ollama:0.5.13")
        .with_bind_ports(11434, port)
        .with_volume_mapping(f"{curr_dir}/../data/{cache_dir}", "/root/.ollama", "rw")
        .with_kwargs(mem_limit="8g")  # Increase container memory to 8 GB
        .with_name("ollama")
    )
    container.start()

    # Wait until Ollama outputs a log indicating it's ready.
    wait_for_logs(container, ".*Listening on.*", timeout=60)

    # Trigger a model pull by calling the /api/pull endpoint.
    logger.info(f"Pulling {model_name} model (this may take a while)...")
    response = requests.post(
        f"http://127.0.0.1:{port}/api/pull",
        headers={"Content-Type": "application/json"},
        json={"name": model_name},
    )
    logger.info(f"Done pulling {model_name}")
    logger.info(f"Response: {response}")
    return container
