import logging
from rag.config import LLMConfig
from rag.loader import Loader

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')

logger = logging.getLogger(__name__)

class Indexer:
    
    def __init__(self, data: dict[str, any], llm_config: LLMConfig):
        self.data = data
        self.llm_config = llm_config