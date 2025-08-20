import logging
from pydantic import BaseModel
from .llm_config import LLMConfig

logging.basicConfig(level=logging.ERROR, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)

class Vectorizer:
    """Splits and vectorizes the document
    """

    def __init__(
            self,
            data: dict[str, any],
            chunk_size: int = 500,
            chunk_overlap: int = 150,
            client_name: str = "openai",
            embedding_model: str = "text-embedding-3-small",
        ) -> None:
        """Initialize the Vectorizer class.

        Args:
            data (dict[str, any]): The data to vectorize.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
            client_name (str): The name of the client to use.
            embedding_model (str): The embedding model to use.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data = data
        self.chunks = None
        self.embedding_model = embedding_model
        self.client = LLMConfig(client_name=client_name)


    def vectorize_docs(self):
        """Vectorize the documents, by splitting them into chunks and then vectorizing each chunk.
        """
        self.chunks = {}
        self.embeddings = {}

        for file_name, file_content in self.data.items(): 
            self.chunks[file_name] = self._split_document(file_content)
            self.embeddings[file_name] = self._get_embeddings(self.chunks[file_name])    

    def _split_document(self, file_content: str) -> dict[str, list[str]]:
        """
        Split the data into chunks.

        Returns:
            dict[str, list[str]]: A dictionary of the split data.
        """
        
        chunks = []
        file_content = file_content.split(" ")
        
        for i in range(0, len(file_content), self.chunk_size - self.chunk_overlap):
            try:
                chunk = file_content[i:i+self.chunk_size]
            except IndexError:
                chunk = file_content[i:]
            except Exception as e:
                logger.error(f"Error splitting document: {e}")
                continue
            chunks.append(" ".join(chunk))
        
        return chunks

    def _get_embeddings(self, chunks: list[str]):
        """Get the embeddings of the chunks.
        """
        return self.client.client.embeddings.create(
            input=chunks,
            model=self.embedding_model
        )

    def get_file_chunks(self, file_name: str) -> list[str]:
        """Get the chunks of a file.

        Args:
            file_name (str): The name of the file to get the chunks of.

        Returns:
            list[str]: The chunks of the file.
        """
        
        if file_name not in list(self.data.keys()):
            logger.error(f"File {file_name} not found in data")
            raise ValueError(f"File {file_name} not found in data")
        
        return self.chunks[file_name]

    def get_file_embeddings(self, file_name: str) -> list[list[float]]:
        """Get the embeddings of a file.

        Args:
            file_name (str): The name of the file to get the embeddings of.

        Returns:
            list[list[float]]: The embeddings of the file.
        """

        if file_name not in list(self.data.keys()):
            logger.error(f"File {file_name} not found in data")
            raise ValueError(f"File {file_name} not found in data")
        
        return self.embeddings[file_name]