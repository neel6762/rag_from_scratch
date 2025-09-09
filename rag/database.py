import chromadb
import logging

logging.basicConfig(level=logging.ERROR, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)


class ChromaDatabase:
    """Stores the vectorized data in a chroma database"""

    def __init__(self):
        """
        Initialize the ChromaDatabase class.

        Args:
            data (dict[str, any]): The data to store in the database.
        """
        self.collection_name = "rag_collection"

        try:
            self._db_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self._db_client.get_or_create_collection(name=self.collection_name)
            logger.info(f"ChromaDatabase initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDatabase: {e}")
            raise e