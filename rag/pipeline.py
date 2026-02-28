"""RAG pipeline: load → split → embed → FAISS store → retrieve → generate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.document_loader import load_document

logger = logging.getLogger(__name__)

# Map provider display name → exact string key used in _get_llm
_PROVIDER_OPENAI = "OpenAI"
_PROVIDER_ANTHROPIC = "Anthropic"
_PROVIDER_OLLAMA = "Ollama (Local)"


class RAGPipeline:
    """Encapsulates the full Retrieval-Augmented Generation pipeline."""

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 4

    def __init__(self) -> None:
        self.vector_store = None
        self.mcp_enabled: bool = False
        self._embeddings = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, file_paths: list[str]) -> None:
        """Load, chunk, embed and index documents into a FAISS vector store."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        raw_docs: list[dict] = []
        for path in file_paths:
            try:
                raw_docs.extend(load_document(path))
            except Exception as exc:
                print(f"Warning: could not load {path!r}: {exc}")

        if not raw_docs:
            raise ValueError("No documents could be loaded from the provided paths.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        lc_docs = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in raw_docs
        ]
        chunks = splitter.split_documents(lc_docs)

        embeddings = self._get_embeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        prompt: str,
        model: str,
        provider: str,
        api_key: str,
    ) -> dict[str, Any]:
        """Retrieve relevant context chunks and generate an answer with an LLM."""
        if self.vector_store is None:
            raise RuntimeError("Documents have not been indexed yet.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.TOP_K}
        )
        relevant_docs = retriever.invoke(prompt)

        context_chunks = [
            {
                "content": doc.page_content,
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "page": doc.metadata.get("page", "?"),
            }
            for doc in relevant_docs
        ]
        context_text = "\n\n".join(
            f"[Source: {c['source']}, p.{c['page']}]\n{c['content']}"
            for c in context_chunks
        )

        system_prompt = (
            "You are a helpful assistant. Use the provided document context to "
            "answer the question accurately. If the context does not contain "
            "enough information, say so clearly."
        )
        full_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer:"
        )

        llm = self._get_llm(model=model, provider=provider, api_key=api_key)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt),
        ]
        response = llm.invoke(messages)

        answer = response.content if hasattr(response, "content") else str(response)

        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            usage = {
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "total_tokens": meta.get("total_tokens", 0),
            }

        return {
            "answer": answer,
            "context": context_chunks,
            "model": model,
            "provider": provider,
            "usage": usage,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self):
        """Return an embeddings instance (HuggingFace local, or OpenAI fallback)."""
        if self._embeddings is not None:
            return self._embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as exc:
            logger.warning(
                "HuggingFace embeddings unavailable (%s); "
                "falling back to OpenAIEmbeddings — ensure OPENAI_API_KEY is set.",
                exc,
            )
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings()
        return self._embeddings

    def _get_llm(self, model: str, provider: str, api_key: str):
        """Return the appropriate LangChain chat model."""
        if provider == _PROVIDER_OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, api_key=api_key, temperature=0.7)
        elif provider == _PROVIDER_ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, api_key=api_key, temperature=0.7)
        elif provider == _PROVIDER_OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
