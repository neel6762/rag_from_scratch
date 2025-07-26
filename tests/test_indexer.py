from rag import Indexer, Loader, LLMConfig
import os

def test_indexer():
    ...

def test_indexer_with_loader():
    
    loader = Loader(data_dir="data")
    data = loader.load_files()
    llm_client = LLMConfig(client_name="ollama")
    indexer = Indexer(data=data, llm_client=llm_client)

    assert indexer.data is not None, "Data should not be None"
    assert isinstance(indexer.data, dict), "Data should be a dictionary"