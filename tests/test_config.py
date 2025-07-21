from rag import LLMConfig

def test_ollama_llm_config():

    llm_config = LLMConfig(client_name="ollama")
    assert llm_config.client_name == "ollama", "Client name should be ollama"

def test_openai_llm_config():
    llm_config = LLMConfig(client_name="openai")
    assert llm_config.client_name == "openai", "Client name should be openai"
