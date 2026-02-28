"""
rag_pipeline.py
───────────────
RAG (Retrieval-Augmented Generation) Pipeline

Fully implemented: document loading, chunking, embedding, vector store,
and multi-provider LLM querying with streaming support.
"""

from __future__ import annotations
import os
import json
import csv
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A complete RAG pipeline.

    Public interface (called by main.py):
        build_index(file_paths, chunk_size, top_k) -> dict
        query(prompt, model, stream_callback)      -> (str, dict)
        is_ready()                                 -> bool
        last_retrieved_chunks()                    -> list[str]
    """

    def __init__(self):
        self._ready = False
        self._top_k = 5
        self._last_chunks: list[str] = []
        self._vectorstore = None
        self._embed_model_name = "all-MiniLM-L6-v2"
        self._embeddings = None

    # ─────────────────────────────────────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────────────────────────────────────

    def build_index(
        self,
        file_paths: list[str],
        chunk_size: int = 512,
        top_k: int = 5,
    ) -> dict:
        """
        Load documents, split into chunks, embed, and store in a vector DB.
        Returns a stats dict: {"num_docs": int, "num_chunks": int, "embed_model": str}
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self._top_k = top_k

        # 1. Load all documents
        all_documents = []
        num_docs = 0
        for path in file_paths:
            try:
                docs = self._load_document(path)
                all_documents.extend(docs)
                num_docs += 1
                logger.info(f"Loaded: {path} ({len(docs)} page(s))")
            except Exception as exc:
                logger.warning(f"Failed to load {path}: {exc}")

        if not all_documents:
            raise ValueError("No documents could be loaded from the provided paths.")

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(chunk_size // 10, 20),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_documents)
        logger.info(f"Split into {len(chunks)} chunks (chunk_size={chunk_size})")

        if not chunks:
            raise ValueError("Document splitting produced zero chunks.")

        # 3. Create embeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embed_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 4. Build Chroma vector store
        # Use a temporary in-memory store to avoid persistence issues
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name="rag_collection",
        )

        self._ready = True

        stats = {
            "num_docs": num_docs,
            "num_chunks": len(chunks),
            "embed_model": self._embed_model_name,
        }
        logger.info(f"Index built: {stats}")
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        model: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, dict]:
        """
        Retrieve relevant chunks, build a context-augmented prompt,
        and call the LLM.

        Returns:
            (response_text: str, metadata: dict)
        """
        if not self._ready:
            raise RuntimeError("Index is not built. Call build_index() first.")

        # 1. Retrieve relevant chunks
        docs = self._vectorstore.similarity_search(prompt, k=self._top_k)
        self._last_chunks = [doc.page_content for doc in docs]
        context = "\n\n---\n\n".join(self._last_chunks)

        # 2. Build augmented prompt
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the context provided below. If the context does not contain enough "
            "information to answer, say so clearly.\n\n"
            f"Context:\n{context}"
        )

        # 3. Route to appropriate LLM
        if model.startswith("claude"):
            response_text, metadata = self._call_anthropic(
                system_prompt, prompt, model, stream_callback
            )
        elif model.startswith("gpt"):
            response_text, metadata = self._call_openai(
                system_prompt, prompt, model, stream_callback
            )
        elif model.startswith("gemini"):
            response_text, metadata = self._call_gemini(
                system_prompt, prompt, model, stream_callback
            )
        elif model.startswith("ollama/"):
            ollama_model = model.split("/", 1)[1]
            response_text, metadata = self._call_ollama(
                system_prompt, prompt, ollama_model, stream_callback
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        metadata["model"] = model
        metadata["num_chunks_retrieved"] = len(self._last_chunks)
        return response_text, metadata

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Returns True if an index has been built and queries can be made."""
        return self._ready

    def last_retrieved_chunks(self) -> list[str]:
        """Returns the context chunks used in the most recent query."""
        return self._last_chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Document Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_document(self, path: str):
        """Load a single file and return a list of LangChain Document objects."""
        from langchain.schema import Document

        ext = Path(path).suffix.lower()

        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": path})]

        elif ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
            return loader.load()

        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(path)
                return loader.load()
            except ImportError:
                import docx
                doc = docx.Document(path)
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                return [Document(page_content=text, metadata={"source": path})]

        elif ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return [Document(page_content=text, metadata={"source": path})]

        elif ext == ".csv":
            rows = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(", ".join(row))
            text = "\n".join(rows)
            return [Document(page_content=text, metadata={"source": path})]

        elif ext == ".html":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # Simple HTML tag stripping
            import re
            clean = re.sub(r"<[^>]+>", " ", text)
            clean = re.sub(r"\s+", " ", clean).strip()
            return [Document(page_content=clean, metadata={"source": path})]

        else:
            # Try to read as plain text as fallback
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": path})]

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Providers
    # ─────────────────────────────────────────────────────────────────────────

    def _call_anthropic(self, system_prompt, user_prompt, model, stream_callback):
        """Call Anthropic Claude API with streaming."""
        import anthropic

        client = anthropic.Anthropic()
        full_response = ""
        total_tokens = 0

        if stream_callback:
            with client.messages.stream(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    stream_callback(text)
                final = stream.get_final_message()
                total_tokens = (final.usage.input_tokens + final.usage.output_tokens)
        else:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            full_response = response.content[0].text
            total_tokens = response.usage.input_tokens + response.usage.output_tokens

        metadata = {"total_tokens": total_tokens}
        return full_response, metadata

    def _call_openai(self, system_prompt, user_prompt, model, stream_callback):
        """Call OpenAI GPT API with streaming."""
        import openai

        client = openai.OpenAI()
        full_response = ""
        total_tokens = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if stream_callback:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    stream_callback(text)
                if chunk.usage:
                    total_tokens = chunk.usage.total_tokens
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
            )
            full_response = response.choices[0].message.content
            total_tokens = response.usage.total_tokens

        metadata = {"total_tokens": total_tokens}
        return full_response, metadata

    def _call_gemini(self, system_prompt, user_prompt, model, stream_callback):
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai

            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)

            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
            )

            full_response = ""
            if stream_callback:
                response = gen_model.generate_content(
                    user_prompt, stream=True
                )
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        stream_callback(chunk.text)
            else:
                response = gen_model.generate_content(user_prompt)
                full_response = response.text

            # Gemini doesn't always report token counts easily
            metadata = {"total_tokens": len(full_response.split())}
            return full_response, metadata

        except ImportError:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )

    def _call_ollama(self, system_prompt, user_prompt, model, stream_callback):
        """Call local Ollama API with streaming."""
        import requests

        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": bool(stream_callback),
        }

        full_response = ""
        total_tokens = 0

        if stream_callback:
            resp = requests.post(url, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        text = data["message"]["content"]
                        full_response += text
                        stream_callback(text)
                    if data.get("done", False):
                        total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        else:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            full_response = data["message"]["content"]
            total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        metadata = {"total_tokens": total_tokens}
        return full_response, metadata
