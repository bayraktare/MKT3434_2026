"""
RAG Engine for MKT3434 Term Project
Supports TXT, PDF, DOCX, JSON, CSV, and Markdown document formats.
Uses LangChain for document loading, chunking, embeddings, and retrieval.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".json", ".csv", ".md"}


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Loads documents in various formats, creates vector embeddings, and
    performs semantic search to provide context-aware LLM responses.
    """

    def __init__(self) -> None:
        self._vectorstore = None
        self._chain = None
        self._model_name: str = "gpt-4o-mini"
        self._api_key: Optional[str] = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_documents(self, file_paths: List[str], api_key: Optional[str] = None) -> None:
        """Load, chunk, and index documents from the given file paths."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        self._api_key = api_key
        documents = []

        for path in file_paths:
            ext = Path(path).suffix.lower()
            try:
                docs = self._load_single_document(path, ext)
                documents.extend(docs)
                logger.info("Loaded '%s' → %d document section(s)", path, len(docs))
            except Exception as exc:
                logger.error("Failed to load '%s': %s", path, exc)
                raise

        if not documents:
            raise ValueError("No content could be loaded from the provided files.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        logger.info("Created %d chunks from %d document(s)", len(chunks), len(documents))

        embeddings = self._get_embeddings(api_key)
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=".chroma_db",
        )

        self._ready = True
        self._build_chain()
        logger.info("RAG engine is ready")

    def _load_single_document(self, path: str, ext: str):
        """Return LangChain documents for a single file."""
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(path)
        elif ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(path, encoding="utf-8")
        elif ext == ".json":
            from langchain_community.document_loaders import JSONLoader
            loader = JSONLoader(path, jq_schema=".", text_content=False)
        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    # ------------------------------------------------------------------
    # Embeddings & LLM helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self, api_key: Optional[str] = None):
        """Return the appropriate embeddings model."""
        if api_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=api_key)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def set_model(self, model_name: str, api_key: Optional[str] = None) -> None:
        """Update the LLM model used for generating answers."""
        self._model_name = model_name
        self._api_key = api_key
        if self._ready:
            self._build_chain()

    def _build_chain(self) -> None:
        """Build the LangChain RetrievalQA chain."""
        from langchain.chains import RetrievalQA

        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        self._chain = RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info("RetrievalQA chain built with model '%s'", self._model_name)

    def _get_llm(self):
        """Return the configured LLM instance."""
        model = self._model_name
        if model.startswith("ollama/"):
            from langchain_community.llms import Ollama
            return Ollama(model=model.split("/", 1)[1])
        if model.startswith("claude"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, prompt: str) -> str:
        """Execute a RAG query and return the LLM answer."""
        if not self._ready or self._chain is None:
            raise RuntimeError("Engine not ready. Please load and index documents first.")

        result = self._chain.invoke({"query": prompt})
        answer: str = result.get("result", "No answer generated.")

        sources = result.get("source_documents", [])
        if sources:
            src_names = [doc.metadata.get("source", "unknown") for doc in sources[:3]]
            logger.info("Sources used for answer: %s", src_names)

        return answer

    def is_ready(self) -> bool:
        """Return True if documents have been loaded and indexed."""
        return self._ready
