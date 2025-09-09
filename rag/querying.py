import bm25s
import logging
import Stemmer

from .database import ChromaDatabase
from .llm_config import LLMConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve the data from the database"""

    def __init__(
            self, 
            method: str = "similarity",
            top_k: int = 5,
            llm_client: LLMConfig = LLMConfig(client_name="openai"),
            embedding_model: str = "text-embedding-3-small",
            db_client: ChromaDatabase = ChromaDatabase()
    ):
        """
        Initialize the Retriever class.

        Args:  
            method (str): The method to use for retrieval. Options: keyword_based, semantic_search, hybrid
            top_k (int): The number of results to retrieve.
            llm_client (LLMConfig): The LLM client to use.
            embedding_model (str): The embedding model to use.
            db_client (ChromaDatabase): The database client to use.
        """
        self.retrieval_method = method
        self.top_k = top_k
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.db_client = db_client

    def _keyword_based(self, all_data, query, query_embeddings):
        """Retrieve the data from the database using keyword-based retrieval."""
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]
    
        # Preprocess documents (tokenization + stemming)
        stemmer = Stemmer.Stemmer("english")
    
        def _preprocess_text(text):
            # Tokenize and stem
            tokens = text.lower().split()
            stemmed_tokens = stemmer.stemWords(tokens)
            return stemmed_tokens
    
        # Preprocess all documents
        corpus_tokens = [_preprocess_text(doc) for doc in documents]
        
        # Preprocess the query (BM25 needs tokens, not embeddings)
        query_tokens = _preprocess_text(query)

        # Create and index the BM25 retriever
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        results, scores = retriever.retrieve(bm25s.tokenize([query], stopwords="en"), k=self.top_k)

        retrieved_docs = []
        for i, (doc_idx, score) in enumerate(zip(results[0], scores[0])):
            retrieved_docs.append({
                "document": documents[doc_idx],
                "metadata": metadatas[doc_idx],
                "score": float(score),
                "rank": i + 1
            })
    
        return retrieved_docs

    def _semantic_search(self, all_data, query_embeddings):
        """Retrieve the data from the database using semantic search."""
        pass
    
    def _hybrid(self, all_data, query_embeddings):
        """Retrieve the data from the database using hybrid retrieval."""
        pass
    
    def retrieve(self, query: str):
        """Retrieve the data from the database using the retrieval method."""

        # vectorize the query
        query_embeddings = self.llm_client.client.embeddings.create(
            input=query,
            model=self.embedding_model
        )

        all_data = self.db_client.collection.get(include=["documents", "metadatas"])
        print(len(all_data["documents"]))

        if self.retrieval_method == "keyword_based":
            return self._keyword_based(all_data, query, query_embeddings)
        elif self.retrieval_method == "semantic_search":
            return self._semantic_search(all_data, query_embeddings)
        elif self.retrieval_method == "hybrid":
            return self._hybrid(all_data, query_embeddings)
        else:
            raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")
