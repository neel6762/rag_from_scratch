import logging
from .llm_config import LLMConfig
import chromadb

from .schemas import DocumentSchema

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')
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
        except Exception as e:
            logger.error(f"Error initializing ChromaDatabase: {e}")
            raise e


class Vectorizer:
    """Splits and vectorizes the document"""

    def __init__(
            self,
            data: dict[str, any],
            chunk_size: int = 500,
            chunk_overlap: int = 150,
            client_name: str = "openai",
            embedding_model: str = "text-embedding-3-small",
        ) -> None:
        """
        Initialize the Vectorizer class.

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
        self.embedding_model = embedding_model
        self.llm_client = LLMConfig(client_name=client_name)
        self.db_client = ChromaDatabase()


    def vectorize_docs(self):
        """
        Vectorize the documents, by splitting them, generating embeddings and storing them in the database.
        """
        for file_name, file_content in self.data.items(): 
            logger.info(f"Vectorizing file: {file_name}")

            chunks = self._split_document(file_content)
            logger.info(f"Number of chunks: {len(chunks)}")
        
            embeddings = self._get_embeddings(chunks)
            logger.info(f"Number of embeddings: {len(embeddings)}")
            
            # create the document schema to store in the database
            document_schema = DocumentSchema(
                file_name = file_name,
                metadata = {"file_name": file_name, "file_type": file_name.split(".")[-1]},
                chunks = chunks,
                embeddings = embeddings
            )

            # store the document schema in the database
            document_dict = document_schema.model_dump()
            try:
                # Create unique IDs for each chunk
                chunk_ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
                
                logger.info(f"Storing {len(chunk_ids)} chunks in database")
                
                self.db_client.collection.add(
                    documents = document_dict["chunks"],
                    embeddings = document_dict["embeddings"],
                    ids = chunk_ids,
                    metadatas = [document_dict["metadata"]] * len(chunks)
                )
                logger.info("Successfully stored document in database")
            except Exception as e:
                logger.error(f"Error storing document in database: {e}")
                continue


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
        response = self.llm_client.client.embeddings.create(
            input=chunks,
            model=self.embedding_model
        )
        return [embedding.embedding for embedding in response.data]

    def get_file_chunks(self, file_name: str) -> list[str]:
        """Get the chunks of a file from the database

        Args:
            file_name (str): The name of the file to get the chunks of.

        Returns:
            list[str]: The chunks of the file.
        """
        # Get all items from the database and filter by file_name metadata
        all_data = self.db_client.collection.get(include=["documents", "metadatas"])
        
        # Find chunks that belong to this file
        file_chunks = []
        for i, metadata in enumerate(all_data.get("metadatas", [])):
            if metadata and metadata.get("file_name") == file_name:
                file_chunks.append(all_data["documents"][i])
        
        if not file_chunks:
            logger.error(f"File {file_name} not found in database")
            raise ValueError(f"File {file_name} not found in database")
        
        return file_chunks

    def get_file_embeddings(self, file_name: str) -> list[list[float]]:
        """Get the embeddings of a file from the database

        Args:
            file_name (str): The name of the file to get the embeddings of.

        Returns:
            list[list[float]]: The embeddings of the file.
        """
        # Get all items from the database and filter by file_name metadata
        all_data = self.db_client.collection.get(include=["embeddings", "metadatas", "documents"])
        
        # Check if embeddings exist in the data
        embeddings_data = all_data.get("embeddings")
        if embeddings_data is None or len(embeddings_data) == 0:
            logger.error("No embeddings found in database")
            raise ValueError("No embeddings found in database")
        
        # Find embeddings that belong to this file
        file_embeddings = []
        for i, metadata in enumerate(all_data.get("metadatas", [])):
            if metadata and metadata.get("file_name") == file_name:
                if i < len(all_data["embeddings"]):
                    file_embeddings.append(all_data["embeddings"][i])
        
        if not file_embeddings:
            logger.error(f"File {file_name} not found in database")
            raise ValueError(f"File {file_name} not found in database")
        
        return file_embeddings