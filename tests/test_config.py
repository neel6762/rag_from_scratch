from rag import LLMConfig

def test_ollama_llm_config():

    ollama_config = LLMConfig(client_name="ollama")
    assert ollama_config.client_name == "ollama", "Client name should be ollama"

    try:
        response = ollama_config.client.chat.completions.create(
            model=ollama_config.model,
            messages=[{"role": "user", "content": "Hello from Ollama!"}]
        )
        assert response.choices[0].message.content is not None, "Response should not be None"
    except Exception as e:
        assert False, f"Failed to get response from Ollama: {e}"        


def test_openai_llm_config():
    openai_config = LLMConfig(client_name="openai")
    assert openai_config.client_name == "openai", "Client name should be openai"

    try:
        response = openai_config.client.chat.completions.create(
            model=openai_config.model,
            messages=[{"role": "user", "content": "Hello from OpenAI!"}]
        )
        assert response.choices[0].message.content is not None, "Response should not be None"
    except Exception as e:
        assert False, f"Failed to get response from OpenAI: {e}"