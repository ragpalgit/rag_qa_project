"""
llm.py
Step 8 in the flow: LLM call via OpenRouter (free-tier models, e.g. DeepSeek, Llama).
"""
from langchain_openai import ChatOpenAI
import config

_llm = None


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a singleton ChatOpenAI client pointed at OpenRouter's free models."""
    global _llm
    if _llm is None:
        print(f"[llm] Using model: {config.LLM_MODEL} via OpenRouter")
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            default_headers={
                # Optional but recommended by OpenRouter for free-tier routing/analytics
                "HTTP-Referer": "https://localhost",
                "X-Title": "RAG QA Project",
            },
        )
    return _llm


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Say hello in one short sentence.")
    print(response.content)