from openai import OpenAI, AuthenticationError, APIConnectionError
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(f"llm_config.{__name__}")

LLM_CLIENTS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-fp16")
    },
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    }
}

class LLMConfig:
    def __init__(self, client_name: str = "ollama", model: str = None):
        client_name = client_name.lower().strip()
        
        if client_name not in LLM_CLIENTS:
            raise ValueError(f"Unsupported LLM client: {client_name}. Supported clients are: {list(LLM_CLIENTS.keys())}")

        self.client_name = client_name
        self.config = LLM_CLIENTS[self.client_name]
        self.model = model if model else self.config["model"]

        if not self.config["api_key"] or not self.config["base_url"]:
            raise ValueError(f"Invalid configuration for '{self.client_name}': API key or base URL missing")

        try:
            self.client = OpenAI(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"]
            )
            logger.info(f"Successfully initialized LLM client for '{self.client_name}'")
        except AuthenticationError as e:
            logger.error(f"Authentication failed for '{self.client_name}': {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Failed to connect to '{self.client_name}' API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing LLM client for '{self.client_name}': {e}")
            raise
    