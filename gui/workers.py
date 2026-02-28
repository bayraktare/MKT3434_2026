"""Background QThread workers for non-blocking indexing and LLM querying."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from rag.pipeline import RAGPipeline


class IndexingWorker(QThread):
    """Indexes documents in a background thread so the GUI stays responsive."""

    finished: Signal = Signal(object)
    error: Signal = Signal(str)

    def __init__(self, file_paths: list[str]) -> None:
        super().__init__()
        self.file_paths = file_paths

    def run(self) -> None:
        try:
            pipeline = RAGPipeline()
            pipeline.index_documents(self.file_paths)
            self.finished.emit(pipeline)
        except Exception as exc:
            self.error.emit(str(exc))


class QueryWorker(QThread):
    """Runs the RAG query + LLM call in a background thread."""

    result: Signal = Signal(object)
    error: Signal = Signal(str)

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        prompt: str,
        model: str,
        provider: str,
        api_key: str,
    ) -> None:
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.prompt = prompt
        self.model = model
        self.provider = provider
        self.api_key = api_key

    def run(self) -> None:
        try:
            result = self.rag_pipeline.query(
                prompt=self.prompt,
                model=self.model,
                provider=self.provider,
                api_key=self.api_key,
            )
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
