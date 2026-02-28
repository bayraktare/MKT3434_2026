"""
MKT3434 – Introduction to Machine Learning
Term Project GUI Application

Mechatronics Engineering Department, Yıldız Technical University
Instructor: Ertugrul Bayraktar

A PySide6-based graphical interface for:
  • Selecting an LLM model (OpenAI, Anthropic Claude, or local Ollama)
  • Selecting and indexing multi-format document sources (TXT/PDF/DOCX/JSON/CSV)
  • Querying those documents via RAG (Retrieval-Augmented Generation)
  • Displaying responses with timestamps
  • Monitoring all interactions via a built-in log viewer and app.log

Students should extend this starter application for their term project.
"""

import logging
import os
import sys
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mcp_client import MCPClient
from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging – messages go to both the console and app.log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class QueryWorker(QThread):
    """Runs the RAG query on a background thread to keep the UI responsive."""

    response_ready = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)

    def __init__(self, engine: RAGEngine, prompt: str, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._prompt = prompt

    def run(self) -> None:
        try:
            self.status_update.emit("Processing query …")
            response = self._engine.query(self._prompt)
            self.response_ready.emit(response)
            self.status_update.emit("Ready")
        except Exception as exc:
            logger.error("Query error: %s", exc)
            self.error_occurred.emit(str(exc))
            self.status_update.emit("Error – see log for details")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for the MKT3434 term project."""

    _SUPPORTED_FORMATS = (
        "Documents (*.txt *.pdf *.docx *.json *.csv *.md);;"
        "All Files (*)"
    )

    _AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "ollama/llama3.2",
        "ollama/mistral",
        "ollama/phi3",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._engine = RAGEngine()
        self._mcp = MCPClient()
        self._worker: QueryWorker | None = None
        self._source_paths: list[str] = []

        self._build_ui()
        logger.info("Application started")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("MKT3434 – RAG & MCP Term Project  |  YTÜ Mechatronics")
        self.setMinimumSize(920, 700)
        self.resize(1100, 820)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel(
            "MKT3434 – Introduction to Machine Learning  |  Term Project"
        )
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # ── Top section: model selection (left) + source selection (right)
        top_splitter = QSplitter(Qt.Horizontal)

        top_splitter.addWidget(self._build_model_group())
        top_splitter.addWidget(self._build_source_group())
        top_splitter.setSizes([360, 540])
        layout.addWidget(top_splitter)

        # ── Prompt input ───────────────────────────────────────────────
        layout.addWidget(self._build_prompt_group())

        # ── Response output ────────────────────────────────────────────
        layout.addWidget(self._build_output_group())

        # ── Log monitor ────────────────────────────────────────────────
        layout.addWidget(self._build_log_group())

        # ── Status bar ─────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready – add source documents and click Load & Index.")

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress)

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model Selection")
        vbox = QVBoxLayout(group)

        vbox.addWidget(QLabel("LLM Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(self._AVAILABLE_MODELS)
        vbox.addWidget(self._model_combo)

        vbox.addWidget(QLabel("API Key (leave blank for local Ollama models):"))
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("sk-… or empty for Ollama")
        self._api_key_input.setEchoMode(QLineEdit.Password)
        vbox.addWidget(self._api_key_input)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("MCP Tools available in this session:"))
        self._mcp_tool_list = QListWidget()
        self._mcp_tool_list.setMaximumHeight(80)
        self._refresh_mcp_tools()
        vbox.addWidget(self._mcp_tool_list)

        vbox.addStretch()
        return group

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source Selection")
        vbox = QVBoxLayout(group)

        vbox.addWidget(QLabel("Document sources (TXT / PDF / DOCX / JSON / CSV):"))
        self._source_list = QListWidget()
        self._source_list.setToolTip("Double-click to remove an entry")
        vbox.addWidget(self._source_list)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Source(s)")
        self._add_btn.clicked.connect(self._add_sources)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._remove_source)

        self._load_btn = QPushButton("Load & Index")
        self._load_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self._load_btn.clicked.connect(self._load_sources)

        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addWidget(self._load_btn)
        vbox.addLayout(btn_row)
        return group

    def _build_prompt_group(self) -> QGroupBox:
        group = QGroupBox("Prompt Input")
        vbox = QVBoxLayout(group)

        self._prompt_input = QTextEdit()
        self._prompt_input.setPlaceholderText("Enter your query or prompt here …")
        self._prompt_input.setMaximumHeight(90)
        vbox.addWidget(self._prompt_input)

        btn_row = QHBoxLayout()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_all)

        self._send_btn = QPushButton("Send Query")
        self._send_btn.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 5px 18px;"
        )
        self._send_btn.clicked.connect(self._send_query)

        btn_row.addStretch()
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._send_btn)
        vbox.addLayout(btn_row)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Response Output")
        vbox = QVBoxLayout(group)
        self._output_display = QTextEdit()
        self._output_display.setReadOnly(True)
        self._output_display.setPlaceholderText("Responses will appear here …")
        vbox.addWidget(self._output_display)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("LLM Interaction Monitor  (also saved to app.log)")
        vbox = QVBoxLayout(group)
        self._log_display = QTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setMaximumHeight(110)
        self._log_display.setStyleSheet("font-family: monospace; font-size: 11px;")
        vbox.addWidget(self._log_display)
        return group

    # ------------------------------------------------------------------
    # Slots / callbacks
    # ------------------------------------------------------------------

    def _refresh_mcp_tools(self) -> None:
        self._mcp_tool_list.clear()
        for tool in self._mcp.list_tools():
            self._mcp_tool_list.addItem(f"[MCP] {tool['name']} – {tool['description']}")

    def _add_sources(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Source Documents", "", self._SUPPORTED_FORMATS
        )
        for path in files:
            if path not in self._source_paths:
                self._source_paths.append(path)
                self._source_list.addItem(os.path.basename(path))
        if files:
            self._status_bar.showMessage(
                f"Added {len(files)} file(s). Click 'Load & Index' to process."
            )
            logger.info("Sources added: %s", files)

    def _remove_source(self) -> None:
        row = self._source_list.currentRow()
        if row >= 0:
            removed = self._source_paths.pop(row)
            self._source_list.takeItem(row)
            logger.info("Source removed: %s", removed)

    def _load_sources(self) -> None:
        if not self._source_paths:
            QMessageBox.warning(
                self, "No Sources", "Please add at least one source document first."
            )
            return

        self._load_btn.setEnabled(False)
        self._status_bar.showMessage("Indexing documents – please wait …")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate spinner

        try:
            api_key = self._api_key_input.text().strip() or None
            self._engine.load_documents(self._source_paths, api_key=api_key)
            msg = f"✓ {len(self._source_paths)} document(s) indexed and ready."
            self._status_bar.showMessage(msg)
            self._append_log(msg)
            logger.info("Documents indexed: %s", self._source_paths)
            QMessageBox.information(self, "Success", msg)
        except Exception as exc:
            logger.error("Failed to load documents: %s", exc)
            QMessageBox.critical(self, "Error", f"Failed to load documents:\n{exc}")
            self._status_bar.showMessage("Error loading documents – see app.log")
        finally:
            self._load_btn.setEnabled(True)
            self._progress.setVisible(False)

    def _send_query(self) -> None:
        prompt = self._prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt before sending.")
            return

        if not self._engine.is_ready():
            QMessageBox.warning(
                self,
                "Not Ready",
                "Please load and index source documents before querying.",
            )
            return

        model = self._model_combo.currentText()
        api_key = self._api_key_input.text().strip() or None
        self._engine.set_model(model, api_key=api_key)

        self._send_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._worker = QueryWorker(self._engine, prompt)
        self._worker.response_ready.connect(self._on_response)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.status_update.connect(self._status_bar.showMessage)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        logger.info("Query sent | model=%s | prompt=%s", model, prompt[:120])

    def _on_response(self, response: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        prompt_text = self._prompt_input.toPlainText().strip()

        self._output_display.append(
            f"<b>[{ts}] Query:</b> {prompt_text}"
        )
        self._output_display.append(f"<b>Response:</b> {response}")
        self._output_display.append("<hr>")

        # Scroll to the bottom
        cursor = self._output_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._output_display.setTextCursor(cursor)

        self._append_log(f"[{ts}] model={self._model_combo.currentText()} | response_len={len(response)}")
        logger.info("Response received (len=%d): %s …", len(response), response[:200])

    def _on_error(self, error: str) -> None:
        QMessageBox.critical(self, "Query Error", f"An error occurred:\n{error}")
        self._output_display.append(
            f"<span style='color:red;'>[ERROR] {error}</span><hr>"
        )
        self._append_log(f"[ERROR] {error}")

    def _on_worker_done(self) -> None:
        self._send_btn.setEnabled(True)
        self._progress.setVisible(False)

    def _clear_all(self) -> None:
        self._prompt_input.clear()
        self._output_display.clear()

    def _append_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_display.append(f"{ts}  {message}")
        cursor = self._log_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_display.setTextCursor(cursor)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
