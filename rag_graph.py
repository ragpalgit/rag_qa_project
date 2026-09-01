"""
rag_graph.py
Step 7 in the flow: LangGraph pipeline with two nodes:
  START -> Retrieve Node -> Generate Node -> END
"""
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from retriever import get_retriever
from llm import get_llm


class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


PROMPT = ChatPromptTemplate.from_template(
    """You are a precise, grounded QA assistant for a document-based knowledge base.

Rules:
1. Answer using ONLY information found in the "Context" section below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond exactly:
   "I don't have enough information in the provided context to answer that."
3. Do not speculate, guess, or fill gaps with assumptions.
4. Be concise and directly answer the question in 2-5 sentences unless the question
   requires a list or step-by-step explanation.
5. Do not mention "the context" or "the document" in your answer — just answer naturally,
   as if you know the material.

Context:
{context}

Question: {question}

Answer:"""
)


def retrieve_node(state: RAGState) -> dict:
    """Gets context from the retriever.

    Note: for noisy/conversational questions, you can insert a query-rewriting
    LLM call here (e.g. 'rewrite this as a clean search query') before retrieval
    to improve similarity search quality. Kept simple here since test questions
    are already well-formed.
    """
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"context": [d.page_content for d in docs]}


def generate_node(state: RAGState) -> dict:
    """Generates an answer using the LLM, grounded in retrieved context."""
    llm = get_llm()
    chain = PROMPT | llm
    result = chain.invoke({
        "context": "\n\n".join(state["context"]),
        "question": state["question"],
    })
    return {"answer": result.content}


def build_rag_graph():
    """Compile the LangGraph: START -> retrieve -> generate -> END."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# Compiled app, ready to import elsewhere (e.g. run_rag_tests.py)
rag_app = build_rag_graph()


if __name__ == "__main__":
    result = rag_app.invoke({"question": "What is RAG?"})
    print("Answer:", result["answer"])
    print("\nRetrieved context chunks:", len(result["context"]))
