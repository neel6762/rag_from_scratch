from rag import Indexer, Loader, LLMConfig
import os

def test_indexer():
    ...

def test_indexer_with_loader():
    
    loader = Loader(data_dir="data")
    data = loader.load_files()
    indexer = Indexer(data=data, llm_config=LLMConfig(client_name="ollama"))
    
    assert indexer.data is not None, "Data should not be None"
    assert isinstance(indexer.data, dict), "Data should be a dictionary"


if __name__ == "__main__":
    test_loader()
    test_indexer_with_loader()