"""PySide6 main window for the MKT3434 RAG & MCP Assistant."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.workers import IndexingWorker, QueryWorker
from monitor.response_monitor import ResponseMonitor

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".pdf", ".docx", ".json", ".csv", ".md"}
)

MODELS: dict[str, list[str]] = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "Ollama (Local)": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "phi3",
        "gemma2",
    ],
}


class MainWindow(QMainWindow):
    """Main application window for the MKT3434 RAG & MCP Assistant."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MKT3434 – RAG & MCP Assistant")
        self.setMinimumSize(1100, 700)

        self.monitor = ResponseMonitor()
        self.rag_pipeline = None
        self.indexing_worker: IndexingWorker | None = None
        self.query_worker: QueryWorker | None = None
        self._start_time: datetime = datetime.now()

        self._build_menu()
        self._build_ui()
        self._build_status_bar()

    # ------------------------------------------------------------------ #
    # Menu bar                                                             #
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        add_files_action = QAction("Add &Files…", self)
        add_files_action.setShortcut("Ctrl+O")
        add_files_action.triggered.connect(self._add_files)
        file_menu.addAction(add_files_action)

        add_folder_action = QAction("Add &Folder…", self)
        add_folder_action.triggered.connect(self._add_folder)
        file_menu.addAction(add_folder_action)

        file_menu.addSeparator()

        save_log_action = QAction("&Save Monitor Log…", self)
        save_log_action.setShortcut("Ctrl+S")
        save_log_action.triggered.connect(self._save_monitor_log)
        file_menu.addAction(save_log_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------ #
    # Central widget                                                        #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Horizontal splitter: sources (left) | interaction (right)
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.addWidget(self._build_sources_panel())
        h_splitter.addWidget(self._build_interaction_panel())
        h_splitter.setSizes([280, 820])

        main_layout.addWidget(h_splitter, stretch=3)
        main_layout.addWidget(self._build_monitor_panel(), stretch=1)

    # ------------------------------------------------------------------ #
    # Sources panel (left)                                                 #
    # ------------------------------------------------------------------ #

    def _build_sources_panel(self) -> QGroupBox:
        group = QGroupBox("📁  Document Sources")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_remove = QPushButton("Remove")
        for btn in (self.btn_add_files, self.btn_add_folder, self.btn_remove):
            btn.setFixedHeight(28)
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addWidget(self.btn_remove)
        layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setToolTip(
            "Supported: TXT · PDF · DOCX · JSON · CSV · MD"
        )
        layout.addWidget(self.file_list)

        # Index button + indeterminate progress bar
        self.btn_index = QPushButton("⚡  Index Documents")
        self.btn_index.setFixedHeight(32)
        self.btn_index.setEnabled(False)
        self.btn_index.clicked.connect(self._index_documents)
        layout.addWidget(self.btn_index)

        self.index_progress = QProgressBar()
        self.index_progress.setVisible(False)
        layout.addWidget(self.index_progress)

        fmt_label = QLabel("Formats: TXT · PDF · DOCX · JSON · CSV · MD")
        fmt_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(fmt_label)

        return group

    # ------------------------------------------------------------------ #
    # Interaction panel (right)                                            #
    # ------------------------------------------------------------------ #

    def _build_interaction_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── Settings row ────────────────────────────────────────────────
        settings_group = QGroupBox("⚙️  Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(MODELS.keys()))
        self.provider_combo.setFixedWidth(150)
        self.provider_combo.currentTextChanged.connect(self._update_model_list)
        settings_layout.addWidget(self.provider_combo)

        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(240)
        settings_layout.addWidget(self.model_combo)

        settings_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText(
            "Enter API key (or set env var OPENAI_API_KEY / ANTHROPIC_API_KEY)"
        )
        settings_layout.addWidget(self.api_key_input)

        self.btn_mcp = QPushButton("🔌  MCP: Off")
        self.btn_mcp.setCheckable(True)
        self.btn_mcp.setFixedWidth(110)
        self.btn_mcp.toggled.connect(self._toggle_mcp)
        settings_layout.addWidget(self.btn_mcp)

        layout.addWidget(settings_group)
        self._update_model_list(self.provider_combo.currentText())

        # ── Prompt area ─────────────────────────────────────────────────
        prompt_group = QGroupBox("💬  Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Enter your question here…\n\n"
            "RAG will retrieve relevant context from the indexed documents "
            "and pass it to the selected LLM."
        )
        self.prompt_input.setMaximumHeight(120)
        prompt_layout.addWidget(self.prompt_input)

        btn_row = QHBoxLayout()
        self.btn_submit = QPushButton("▶  Submit Query")
        self.btn_submit.setFixedHeight(34)
        self.btn_submit.clicked.connect(self._submit_query)
        self.btn_clear_prompt = QPushButton("Clear")
        self.btn_clear_prompt.setFixedHeight(34)
        self.btn_clear_prompt.clicked.connect(self.prompt_input.clear)
        btn_row.addWidget(self.btn_submit)
        btn_row.addWidget(self.btn_clear_prompt)
        btn_row.addStretch()
        prompt_layout.addLayout(btn_row)

        layout.addWidget(prompt_group)

        # ── Output tabs ─────────────────────────────────────────────────
        output_tabs = QTabWidget()

        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        self.response_output.setPlaceholderText("LLM response will appear here…")
        output_tabs.addTab(self.response_output, "📄  Response")

        self.context_output = QTextEdit()
        self.context_output.setReadOnly(True)
        self.context_output.setPlaceholderText(
            "Retrieved document chunks (RAG context) will appear here…"
        )
        output_tabs.addTab(self.context_output, "🔍  Retrieved Context")

        layout.addWidget(output_tabs, stretch=1)

        return widget

    # ------------------------------------------------------------------ #
    # Monitor panel (bottom)                                               #
    # ------------------------------------------------------------------ #

    def _build_monitor_panel(self) -> QGroupBox:
        group = QGroupBox("📊  Monitor Log")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)

        btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.setFixedHeight(24)
        btn_clear_log.clicked.connect(self._clear_monitor_log)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear_log)
        layout.addLayout(btn_row)

        self.monitor_log = QTextEdit()
        self.monitor_log.setReadOnly(True)
        self.monitor_log.setMaximumHeight(130)
        self.monitor_log.setFont(QFont("Courier New", 9))
        self.monitor_log.setPlaceholderText("Query events will be logged here…")
        layout.addWidget(self.monitor_log)

        return group

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ------------------------------------------------------------------ #
    # Source management                                                    #
    # ------------------------------------------------------------------ #

    def _add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Document Files",
            "",
            f"Documents ({exts});;All Files (*)",
        )
        self._add_paths(paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            paths = [
                str(p)
                for p in Path(folder).rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            self._add_paths(paths)

    def _add_paths(self, paths: list[str]) -> None:
        existing = {
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
        }
        added = 0
        for path in paths:
            if path not in existing:
                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.UserRole, path)
                item.setToolTip(path)
                self.file_list.addItem(item)
                added += 1
        if added:
            self.btn_index.setEnabled(True)
            self.status_bar.showMessage(
                f"Added {added} file(s). "
                f"Total: {self.file_list.count()} | "
                "Click 'Index Documents' to build the RAG index."
            )

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self.btn_index.setEnabled(self.file_list.count() > 0)
        self.status_bar.showMessage(f"Files remaining: {self.file_list.count()}")

    # ------------------------------------------------------------------ #
    # Document indexing                                                    #
    # ------------------------------------------------------------------ #

    def _index_documents(self) -> None:
        paths = [
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
        ]
        if not paths:
            return

        self.btn_index.setEnabled(False)
        self.index_progress.setVisible(True)
        self.index_progress.setRange(0, 0)  # indeterminate spinner
        self.status_bar.showMessage("Indexing documents…")

        self.indexing_worker = IndexingWorker(paths)
        self.indexing_worker.finished.connect(self._on_indexing_done)
        self.indexing_worker.error.connect(self._on_indexing_error)
        self.indexing_worker.start()

    @Slot(object)
    def _on_indexing_done(self, pipeline: object) -> None:
        self.rag_pipeline = pipeline
        self.index_progress.setVisible(False)
        self.btn_index.setEnabled(True)
        n = self.file_list.count()
        self.status_bar.showMessage(
            f"✅ Indexed {n} document(s). RAG pipeline ready."
        )
        self._log_to_monitor(f"Indexed {n} document(s) — RAG pipeline ready")

    @Slot(str)
    def _on_indexing_error(self, error: str) -> None:
        self.index_progress.setVisible(False)
        self.btn_index.setEnabled(True)
        self.status_bar.showMessage("❌ Indexing failed.")
        QMessageBox.critical(self, "Indexing Error", error)
        self._log_to_monitor(f"ERROR – Indexing: {error}", error=True)

    # ------------------------------------------------------------------ #
    # Query submission                                                     #
    # ------------------------------------------------------------------ #

    def _submit_query(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt.")
            return

        if self.rag_pipeline is None:
            QMessageBox.warning(
                self,
                "No Index",
                "Please add documents and click 'Index Documents' first.",
            )
            return

        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        api_key = self.api_key_input.text().strip()

        if not api_key:
            env_var = (
                "ANTHROPIC_API_KEY"
                if provider == "Anthropic"
                else "OPENAI_API_KEY"
            )
            api_key = os.environ.get(env_var, "")

        if not api_key and provider != "Ollama (Local)":
            QMessageBox.warning(
                self,
                "No API Key",
                f"Please enter an API key or set the environment variable.\n"
                f"For {provider}: OPENAI_API_KEY / ANTHROPIC_API_KEY",
            )
            return

        self.btn_submit.setEnabled(False)
        self.response_output.clear()
        self.context_output.clear()
        self.status_bar.showMessage("Querying LLM…")
        self._start_time = datetime.now()

        self.query_worker = QueryWorker(
            rag_pipeline=self.rag_pipeline,
            prompt=prompt,
            model=model,
            provider=provider,
            api_key=api_key,
        )
        self.query_worker.result.connect(self._on_query_result)
        self.query_worker.error.connect(self._on_query_error)
        self.query_worker.start()

    @Slot(object)
    def _on_query_result(self, result: dict) -> None:
        self.btn_submit.setEnabled(True)
        elapsed = (datetime.now() - self._start_time).total_seconds()

        self.response_output.setPlainText(result["answer"])

        ctx_text = "\n\n".join(
            f"--- Source {i + 1}: {chunk['source']} (p.{chunk['page']}) ---\n"
            f"{chunk['content']}"
            for i, chunk in enumerate(result.get("context", []))
        )
        self.context_output.setPlainText(ctx_text or "No context retrieved.")

        token_info = result.get("usage", {})
        self.status_bar.showMessage(
            f"✅ Done in {elapsed:.1f}s | "
            f"Tokens: {token_info.get('total_tokens', '?')} | "
            f"Model: {result.get('model', '?')}"
        )

        log_line = self.monitor.log_response(
            prompt=self.prompt_input.toPlainText().strip(),
            response=result["answer"],
            model=result.get("model", "?"),
            provider=result.get("provider", "?"),
            elapsed=elapsed,
            tokens=token_info,
            sources=[c["source"] for c in result.get("context", [])],
        )
        self._log_to_monitor(log_line)

    @Slot(str)
    def _on_query_error(self, error: str) -> None:
        self.btn_submit.setEnabled(True)
        self.status_bar.showMessage("❌ Query failed.")
        QMessageBox.critical(self, "Query Error", error)
        self._log_to_monitor(f"ERROR – Query: {error}", error=True)

    # ------------------------------------------------------------------ #
    # MCP toggle                                                           #
    # ------------------------------------------------------------------ #

    def _toggle_mcp(self, checked: bool) -> None:
        if checked:
            self.btn_mcp.setText("🔌  MCP: On")
            self.status_bar.showMessage(
                "MCP enabled — connecting to configured servers…"
            )
            if self.rag_pipeline:
                self.rag_pipeline.mcp_enabled = True
            self._log_to_monitor("MCP enabled")
        else:
            self.btn_mcp.setText("🔌  MCP: Off")
            self.status_bar.showMessage("MCP disabled.")
            if self.rag_pipeline:
                self.rag_pipeline.mcp_enabled = False
            self._log_to_monitor("MCP disabled")

    # ------------------------------------------------------------------ #
    # Model combo                                                          #
    # ------------------------------------------------------------------ #

    def _update_model_list(self, provider: str) -> None:
        self.model_combo.clear()
        self.model_combo.addItems(MODELS.get(provider, []))

    # ------------------------------------------------------------------ #
    # Monitor log helpers                                                  #
    # ------------------------------------------------------------------ #

    def _log_to_monitor(self, message: str, *, error: bool = False) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        cursor = self.monitor_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        if error:
            fmt.setForeground(QColor("red"))
        else:
            fmt.setForeground(self.monitor_log.palette().text().color())
        cursor.insertText(line + "\n", fmt)
        self.monitor_log.setTextCursor(cursor)
        self.monitor_log.ensureCursorVisible()

    def _clear_monitor_log(self) -> None:
        self.monitor_log.clear()
        self.monitor.clear()

    # ------------------------------------------------------------------ #
    # File actions                                                         #
    # ------------------------------------------------------------------ #

    def _save_monitor_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Monitor Log",
            "monitor_log.json",
            "JSON Files (*.json);;Text Files (*.txt)",
        )
        if path:
            self.monitor.save_log(path)
            self.status_bar.showMessage(f"Monitor log saved to {path}")

    # ------------------------------------------------------------------ #
    # About dialog                                                         #
    # ------------------------------------------------------------------ #

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About MKT3434 Assistant",
            "<h3>MKT3434 – RAG &amp; MCP Assistant</h3>"
            "<p>A teaching tool for the <em>Introduction to Machine Learning</em> "
            "course (MKT3434) at Yıldız Technical University, "
            "Mechatronics Engineering Dept.</p>"
            "<p><b>Features:</b> multi-format document ingestion (TXT · PDF · "
            "DOCX · JSON · CSV · MD), RAG pipeline with FAISS vector store, "
            "MCP (Model Context Protocol) integration, and LLM response "
            "monitoring.</p>"
            "<p><b>Supported models:</b> OpenAI · Anthropic · Ollama (local)</p>",
        )
