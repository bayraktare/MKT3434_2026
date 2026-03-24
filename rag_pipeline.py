"""rag_pipeline.py
Baseline RAG pipeline for the MKT3434 multi-agent term project.

Public interface preserved for the provided GUI:
    build_index(file_paths, chunk_size, top_k) -> dict
    query(prompt, model, stream_callback)      -> (str, dict)
    is_ready()                                 -> bool
    last_retrieved_chunks()                    -> list[str]
"""

from __future__ import annotations

import csv
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp_handler import MCPHandler
from orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self._ready = False
        self._top_k = 5
        self._chunk_size = 512
        self._last_chunks: list[str] = []
        self._last_chunk_records: list[dict[str, Any]] = []
        self._chunks: list[dict[str, Any]] = []
        self._domains: list[str] = []

        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None

        self._dense_model = None
        self._dense_matrix = None
        self._embed_model_name = "tfidf-baseline"
        self._reranker_name = "weighted-score-baseline"

        self._orchestrator: Orchestrator | None = None

    # ------------------------------------------------------------------
    # Public API used by main.py
    # ------------------------------------------------------------------
    def build_index(self, file_paths: list[str], chunk_size: int = 512, top_k: int = 5) -> dict:
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunks = []
        self._last_chunks = []
        self._last_chunk_records = []

        loaded_docs: list[dict[str, Any]] = []
        for path in file_paths:
            try:
                docs = self._load_document(path)
                loaded_docs.extend(docs)
                logger.info("Loaded %s (%d document units)", path, len(docs))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        if not loaded_docs:
            raise ValueError("No documents could be loaded from the selected sources.")

        chunk_counter = 0
        for doc in loaded_docs:
            for piece in self._chunk_text(doc["text"], self._chunk_size):
                if not piece.strip():
                    continue
                chunk_counter += 1
                self._chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter:05d}",
                        "text": piece,
                        "source": doc["source"],
                        "source_name": Path(doc["source"]).name,
                        "page": doc.get("page"),
                        "domain": doc.get("domain", "general"),
                    }
                )

        if not self._chunks:
            raise ValueError("Document loading succeeded but produced zero chunks.")

        texts = [chunk["text"] for chunk in self._chunks]
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        self._embed_model_name = "tfidf-baseline"

        try:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dense_matrix = self._dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._embed_model_name = "all-MiniLM-L6-v2 + tfidf"
        except Exception as exc:
            logger.info("Dense embeddings unavailable, continuing with TF-IDF only: %s", exc)
            self._dense_model = None
            self._dense_matrix = None

        self._domains = sorted({chunk["domain"] for chunk in self._chunks if chunk.get("domain")})
        if "all" not in self._domains:
            self._domains.insert(0, "all")
        self._ready = True
        self._ensure_orchestrator()

        return {
            "num_docs": len({doc["source"] for doc in loaded_docs}),
            "num_chunks": len(self._chunks),
            "embed_model": self._embed_model_name,
            "reranker": self._reranker_name,
            "domains": self._domains,
        }

    def query(
        self,
        prompt: str,
        model: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, dict]:
        if not self._ready:
            raise RuntimeError("Index is not built. Call build_index() first.")

        clean_prompt, rules = self._extract_rules_from_prompt(prompt)
        self._ensure_orchestrator()
        assert self._orchestrator is not None
        self._orchestrator.mcp_handler = MCPHandler.get_active_handler()
        answer, metadata = self._orchestrator.run(
            query=clean_prompt,
            model=model,
            rules=rules,
            stream_callback=stream_callback,
        )
        metadata["num_chunks_retrieved"] = len(self._last_chunks)
        return answer, metadata

    def is_ready(self) -> bool:
        return self._ready

    def last_retrieved_chunks(self) -> list[str]:
        return self._last_chunks

    # ------------------------------------------------------------------
    # Retrieval helpers used by agents
    # ------------------------------------------------------------------
    @property
    def top_k(self) -> int:
        return self._top_k

    def available_domains(self) -> list[str]:
        return list(self._domains) if self._domains else ["all"]

    def set_last_retrieved_chunks(self, chunk_records: list[dict[str, Any]]) -> None:
        self._last_chunk_records = list(chunk_records)
        self._last_chunks = [chunk.get("text", "") for chunk in chunk_records]

    def retrieve(
        self,
        query: str,
        domain: str = "all",
        top_k: int | None = None,
        use_rerank: bool = True,
        candidate_multiplier: int = 4,
    ) -> list[dict[str, Any]]:
        if not self._ready or self._tfidf is None or self._tfidf_matrix is None:
            raise RuntimeError("Index is not ready.")

        top_k = top_k or self._top_k
        allowed_indices = [
            idx for idx, chunk in enumerate(self._chunks)
            if domain in ("all", "", None) or chunk.get("domain") == domain
        ]
        if not allowed_indices:
            allowed_indices = list(range(len(self._chunks)))

        query_vec = self._tfidf.transform([query])
        sparse_scores = cosine_similarity(query_vec, self._tfidf_matrix[allowed_indices]).ravel()

        if self._dense_model is not None and self._dense_matrix is not None:
            dense_query = self._dense_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            dense_scores = np.dot(self._dense_matrix[allowed_indices], dense_query[0])
        else:
            dense_scores = np.zeros_like(sparse_scores)

        combined_scores = 0.65 * _minmax(dense_scores) + 0.35 * _minmax(sparse_scores)
        candidate_count = min(len(allowed_indices), max(top_k * candidate_multiplier, top_k))
        top_positions = np.argsort(combined_scores)[::-1][:candidate_count]

        candidates: list[dict[str, Any]] = []
        for pos in top_positions:
            original_idx = allowed_indices[int(pos)]
            item = dict(self._chunks[original_idx])
            item["score"] = float(combined_scores[int(pos)])
            item["dense_score"] = float(dense_scores[int(pos)]) if len(dense_scores) else 0.0
            item["sparse_score"] = float(sparse_scores[int(pos)]) if len(sparse_scores) else 0.0
            candidates.append(item)

        if use_rerank:
            candidates = self._rerank_candidates(query, candidates)

        return candidates[:top_k]

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, dict[str, Any]]:
        model_lower = model.lower()
        if model_lower.startswith("claude"):
            return self._call_anthropic(system_prompt, user_prompt, model, stream_callback)
        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            return self._call_openai(system_prompt, user_prompt, model, stream_callback)
        if model_lower.startswith("gemini"):
            return self._call_gemini(system_prompt, user_prompt, model, stream_callback)
        if model_lower.startswith("ollama/"):
            return self._call_ollama(system_prompt, user_prompt, model.split("/", 1)[1], stream_callback)
        raise ValueError(f"Unsupported model provider: {model}. Supported prefixes: gpt-, o1-, o3-, claude-, gemini-, ollama/")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_orchestrator(self) -> None:
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(self, mcp_handler=MCPHandler.get_active_handler())

    def _extract_rules_from_prompt(self, prompt: str) -> tuple[str, dict[str, Any] | None]:
        match = re.search(r"\[RULES\]\s*(\{.*?\})\s*\[/RULES\]", prompt, flags=re.DOTALL)
        if not match:
            return prompt, None
        try:
            rules = json.loads(match.group(1))
        except json.JSONDecodeError:
            return prompt, None
        cleaned = (prompt[: match.start()] + prompt[match.end() :]).strip()
        return cleaned, rules

    def _load_document(self, path: str) -> list[dict[str, Any]]:
        ext = Path(path).suffix.lower()
        domain = self._infer_domain(path)

        if ext in {".txt", ".md"}:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            return [{"text": text, "source": path, "page": None, "domain": domain}]

        if ext == ".pdf":
            return self._load_pdf(path, domain)

        if ext == ".docx":
            try:
                import docx

                doc = docx.Document(path)
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception as exc:
                raise RuntimeError(f"DOCX loading failed: {exc}") from exc
            return [{"text": text, "source": path, "page": None, "domain": domain}]

        if ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                data = json.load(handle)
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return [{"text": text, "source": path, "page": None, "domain": domain}]

        if ext == ".csv":
            rows: list[str] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    rows.append(", ".join(str(cell) for cell in row))
            return [{"text": "\n".join(rows), "source": path, "page": None, "domain": domain}]

        if ext == ".html":
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
            clean = re.sub(r"<[^>]+>", " ", html.unescape(raw))
            clean = re.sub(r"\s+", " ", clean).strip()
            return [{"text": clean, "source": path, "page": None, "domain": domain}]

        raw_text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return [{"text": raw_text, "source": path, "page": None, "domain": domain}]

    def _load_pdf(self, path: str, domain: str) -> list[dict[str, Any]]:
        texts: list[dict[str, Any]] = []
        reader = None
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
        except Exception:
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(path)
            except Exception as exc:
                raise RuntimeError(f"PDF loading failed: {exc}") from exc

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                texts.append({"text": text, "source": path, "page": page_idx, "domain": domain})
        return texts

    def _infer_domain(self, path: str) -> str:
        p = Path(path)
        parts = [part.lower() for part in p.parts]
        for candidate in ["health", "engineering", "sports", "politics"]:
            if candidate in parts:
                return candidate
        if p.parent.name.lower() not in {"", ".", "data"}:
            return p.parent.name.lower()
        return "general"

    def _chunk_text(self, text: str, chunk_size: int, overlap: int | None = None) -> list[str]:
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []
        overlap = overlap if overlap is not None else max(40, chunk_size // 8)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = current[-overlap:] if overlap < len(current) else current
                current = f"{tail}\n\n{para}".strip()
                if len(current) <= chunk_size:
                    continue
            pieces = self._split_long_text(para, chunk_size, overlap)
            if pieces:
                chunks.extend(pieces[:-1])
                current = pieces[-1]
        if current:
            chunks.append(current)
        return chunks

    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces: list[str] = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                    tail = current[-overlap:] if overlap < len(current) else current
                    current = f"{tail} {sent}".strip()
                else:
                    for idx in range(0, len(sent), max(1, chunk_size - overlap)):
                        pieces.append(sent[idx : idx + chunk_size])
                    current = ""
        if current:
            pieces.append(current)
        return pieces

    def _rerank_candidates(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_terms = set(_keywords(query))
        for item in candidates:
            text = item.get("text", "")
            t_terms = set(_keywords(text))
            coverage = len(q_terms & t_terms) / max(1, len(q_terms))
            exact_phrase_bonus = 0.15 if query.lower() in text.lower() else 0.0
            item["rerank_score"] = 0.7 * item.get("score", 0.0) + 0.25 * coverage + exact_phrase_bonus
        candidates.sort(key=lambda row: row.get("rerank_score", row.get("score", 0.0)), reverse=True)
        for item in candidates:
            item["score"] = float(item.get("rerank_score", item.get("score", 0.0)))
        return candidates

    # ------------------------------------------------------------------
    # Provider wrappers
    # ------------------------------------------------------------------
    def _call_anthropic(self, system_prompt, user_prompt, model, stream_callback=None):
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
                total_tokens = int(final.usage.input_tokens + final.usage.output_tokens)
        else:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            full_response = response.content[0].text
            total_tokens = int(response.usage.input_tokens + response.usage.output_tokens)

        return full_response, {"total_tokens": total_tokens}

    def _call_openai(self, system_prompt, user_prompt, model, stream_callback=None):
        from openai import OpenAI

        client = OpenAI()
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
                    token = chunk.choices[0].delta.content
                    full_response += token
                    stream_callback(token)
                if getattr(chunk, "usage", None):
                    total_tokens = int(chunk.usage.total_tokens)
        else:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=2048)
            full_response = response.choices[0].message.content or ""
            total_tokens = int(response.usage.total_tokens)

        return full_response, {"total_tokens": total_tokens}

    def _call_gemini(self, system_prompt, user_prompt, model, stream_callback=None):
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)

        full_response = ""
        if stream_callback:
            response = gen_model.generate_content(user_prompt, stream=True)
            for chunk in response:
                text = getattr(chunk, "text", "")
                if text:
                    full_response += text
                    stream_callback(text)
        else:
            response = gen_model.generate_content(user_prompt)
            full_response = getattr(response, "text", "") or ""
        return full_response, {"total_tokens": len(full_response.split())}

    def _call_ollama(self, system_prompt, user_prompt, model, stream_callback=None):
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
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                token = message.get("content", "")
                if token:
                    full_response += token
                    stream_callback(token)
                if data.get("done"):
                    total_tokens = int(data.get("eval_count", 0) + data.get("prompt_eval_count", 0))
        else:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            full_response = data.get("message", {}).get("content", "")
            total_tokens = int(data.get("eval_count", 0) + data.get("prompt_eval_count", 0))
        return full_response, {"total_tokens": total_tokens}


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------

def _minmax(values: np.ndarray) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.array([])
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if abs(max_v - min_v) < 1e-9:
        return np.ones_like(values) if max_v > 0 else np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)



def _keywords(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())
