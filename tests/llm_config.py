from rag import LLMConfig
import pytest

@pytest.mark.parametrize(
    "client_name, model", 
    [("ollama", "phi4-mini-reasoning:3.8b"),
     ("openai", "gpt-4o-mini"),
     ("random", None)]
)   
def test_llm_config(client_name, model):

    if client_name == "random":
        with pytest.raises(ValueError):
            LLMConfig(client_name=client_name)
        return

    llm_config = LLMConfig(client_name=client_name)
    assert llm_config.client_name == client_name, "Client name should be " + client_name

    try:
        response = llm_config.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello from " + client_name + "!"}]
        )
        assert response.choices[0].message.content is not None, "Response should not be None"
    except Exception as e:
        assert False, f"Failed to get response from {client_name}: {e}"        