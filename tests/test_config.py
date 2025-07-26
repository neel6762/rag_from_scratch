from rag import LLMConfig
import pytest

@pytest.mark.parametrize("client_name", ["ollama", "openai", "random"])
def test_llm_config(client_name):
    
    if client_name == "random":
        with pytest.raises(ValueError):
            LLMConfig(client_name=client_name)
        return

    llm_config = LLMConfig(client_name=client_name)
    assert llm_config.client_name == client_name, "Client name should be " + client_name

    try:
        response = llm_config.client.chat.completions.create(
            model=llm_config.model,
            messages=[{"role": "user", "content": "Hello from " + client_name + "!"}]
        )
        assert response.choices[0].message.content is not None, "Response should not be None"
    except Exception as e:
        assert False, f"Failed to get response from {client_name}: {e}"        
