from abc import ABC, abstractmethod
import aiohttp
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass

class OllamaClient(LLMClient):
    """Client for Ollama LLM service."""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
    async def generate(self, prompt: str) -> str:
        """Generate a response from Ollama."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")
                    
            except Exception as e:
                logger.exception("Error calling Ollama API")
                raise 