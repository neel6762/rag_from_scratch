from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)

LLM_CLIENTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2:3b-instruct-fp16"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini"
    }
}

class LLMConfig:

    def __init__(self, client_name: str = "ollama"):
        
        client_name = client_name.lower().strip()
        if client_name not in LLM_CLIENTS:
            raise ValueError(f"Unsupported LLM client: {client_name}. Supported clients are: {list(LLM_CLIENTS.keys())}")

        self.client_name = client_name
        config = LLM_CLIENTS[self.client_name]
        
        self.model = config["model"]
        
        try:
            self.client = OpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"]
            )
            self.client.models.list()
            logger.info(f"Successfully initialized LLM client for '{self.client_name}'")

        except Exception as e:
            logger.error(f"Error initializing LLM client for '{self.client_name}': {e}")
            raise
        
if __name__ == "__main__":

    try:
        ollama_config = LLMConfig(client_name="ollama")
        response = ollama_config.client.chat.completions.create(
            model=ollama_config.model,
            messages=[{"role": "user", "content": "Hello from Ollama!"}]
        )
        print(f"Ollama response: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f"Failed to get response from Ollama: {e}")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in .env file. Skipping OpenAI example.")
    else:
        try:
            openai_config = LLMConfig(client_name="openai")
            response = openai_config.client.chat.completions.create(
                model=openai_config.model,
                messages=[{"role": "user", "content": "Hello from OpenAI!"}]
            )
            print(f"OpenAI response: {response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"Failed to get response from OpenAI: {e}")