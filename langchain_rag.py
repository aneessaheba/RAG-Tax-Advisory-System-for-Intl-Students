"""
LangChain RAG Chain — wraps the HybridRetriever as a LangChain-compatible
retriever and builds a full LCEL (LangChain Expression Language) RAG chain.

Usage:
    from langchain_rag import build_rag_chain
    chain = build_rag_chain(retriever, api_key)
    answer = chain.invoke({"question": "...", "student_info": {...}})
"""
import os
from typing import List
from dotenv import load_dotenv

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

load_dotenv()


class HybridRetrieverWrapper(BaseRetriever):
    """
    Wraps our custom HybridRetriever (BM25 + vector + RRF) as a
    LangChain BaseRetriever so it can plug into any LangChain chain.
    """
    hybrid_retriever: object = Field(description="HybridRetriever instance")
    top_k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        chunks, confidence = self.hybrid_retriever.retrieve(query, top_k=self.top_k)
        docs = []
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    **chunk["metadata"],
                    "chunk_id": chunk["chunk_id"],
                    "confidence": round(confidence, 4),
                }
            ))
        return docs


def format_docs(docs: List[Document]) -> str:
    """Format retrieved LangChain Documents into a labeled context string."""
    parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[{title} - p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful tax advisor for international students in the U.S.

Student profile:
- Visa: {visa_type}
- Home country: {home_country}
- Tax year: {tax_year}
- Income types: {income_types}
- State: {state}

Use ONLY the provided reference documents to answer. If the documents don't cover something,
say so clearly. Always remind the student this is general guidance, not professional tax advice.

--- REFERENCE DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Student's question: {question}

Provide a clear, helpful answer:""")


def build_rag_chain(hybrid_retriever, api_key: str):
    """
    Build a full LangChain LCEL RAG chain.

    Chain flow:
      input (question + student_info)
        -> HybridRetrieverWrapper (BM25 + vector + RRF)
        -> format_docs
        -> RAG_PROMPT (with student profile injected)
        -> ChatGoogleGenerativeAI (Gemini 2.0 Flash)
        -> StrOutputParser
    """
    lc_retriever = HybridRetrieverWrapper(hybrid_retriever=hybrid_retriever, top_k=5)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
    )

    def build_chain_input(inputs: dict) -> dict:
        question = inputs["question"]
        info = inputs["student_info"]
        docs = lc_retriever.get_relevant_documents(question)
        return {
            "context": format_docs(docs),
            "question": question,
            "visa_type": info.get("visa_type", "F-1"),
            "home_country": info.get("home_country", "Unknown"),
            "tax_year": info.get("tax_year", "2024"),
            "income_types": ", ".join(info.get("income_types", ["None"])),
            "state": info.get("state", "CA"),
        }

    chain = (
        RunnableLambda(build_chain_input)
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    """Quick test of the LangChain RAG chain."""
    import chromadb
    from retriever import HybridRetriever

    api_key = os.environ.get("GEMINI_API_KEY")
    BASE_DIR = os.path.dirname(__file__)
    CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')

    db_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db_client.get_collection(name="tax_docs")
    retriever = HybridRetriever(collection)

    chain = build_rag_chain(retriever, api_key)

    answer = chain.invoke({
        "question": "Do F-1 students need to file Form 8843?",
        "student_info": {
            "visa_type": "F-1",
            "home_country": "India",
            "tax_year": "2024",
            "income_types": ["On-campus job"],
            "state": "CA",
        }
    })
    print(answer)
