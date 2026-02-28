"""
MKT3434 - Introduction to Machine Learning taught by Assoc. Prof. Ertugrul Bayraktar
Term Project Starter GUI
Yildiz Technical University – Mechatronics Engineering Department

This file provides the base GUI that students will extend.
Students must NOT modify the submission/logging logic.
"""

import sys
import os
import json
import logging
import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QFileDialog,
    QListWidget, QListWidgetItem, QSplitter, QStatusBar,
    QGroupBox, QProgressBar, QTabWidget, QSizePolicy, QFrame,
    QScrollArea, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QIcon, QColor, QPalette, QTextCursor, QFontDatabase

# ── Local modules (students implement these) ──────────────────────────────────
from rag_pipeline import RAGPipeline
from llm_monitor import LLMMonitor
from mcp_handler import MCPHandler

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Worker thread for async LLM calls ────────────────────────────────────────
class LLMWorker(QThread):
    result_ready   = Signal(str)
    token_received = Signal(str)
    error_occurred = Signal(str)
    finished_query = Signal(dict)   # emits monitoring metadata

    def __init__(self, rag: RAGPipeline, monitor: LLMMonitor, prompt: str, model: str):
        super().__init__()
        self.rag     = rag
        self.monitor = monitor
        self.prompt  = prompt
        self.model   = model

    def run(self):
        try:
            logger.info(f"Query started | model={self.model} | prompt_len={len(self.prompt)}")
            start = datetime.datetime.now()

            response, metadata = self.rag.query(
                prompt=self.prompt,
                model=self.model,
                stream_callback=self.token_received.emit,
            )

            elapsed = (datetime.datetime.now() - start).total_seconds()
            metadata.update({"elapsed_sec": round(elapsed, 3), "model": self.model})

            self.monitor.record(self.prompt, response, metadata)
            self.result_ready.emit(response)
            self.finished_query.emit(metadata)
            logger.info(f"Query finished | elapsed={elapsed:.2f}s | tokens={metadata.get('total_tokens', 'N/A')}")

        except Exception as exc:
            logger.error(f"LLM error: {exc}", exc_info=True)
            self.error_occurred.emit(str(exc))


# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    # Available models – students may extend this list
    MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-1.5-pro",
        "ollama/llama3",
    ]

    SUPPORTED_EXTS = [".txt", ".pdf", ".docx", ".json", ".csv", ".md", ".html"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MKT3434 – ML Term Project | YTU Mechatronics taught by Assoc. Prof. Ertugrul Bayraktar")
        self.resize(1280, 820)
        self._apply_stylesheet()

        # Core components (students extend these)
        self.rag     = RAGPipeline()
        self.monitor = LLMMonitor()
        self.mcp     = MCPHandler()
        self.worker  = None

        self._build_ui()
        self._connect_signals()
        logger.info("Application started.")

    # ── Stylesheet ────────────────────────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
        /* ── Global ── */
        * { font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; }

        QMainWindow, QWidget {
            background-color: #0d1117;
            color: #e6edf3;
        }

        /* ── Group boxes ── */
        QGroupBox {
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-top: 12px;
            padding: 10px 8px 8px 8px;
            font-size: 11px;
            font-weight: bold;
            color: #8b949e;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            left: 10px;
            top: -1px;
        }

        /* ── ComboBox ── */
        QComboBox {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 6px 10px;
            color: #c9d1d9;
            font-size: 12px;
        }
        QComboBox:hover { border-color: #58a6ff; }
        QComboBox::drop-down { border: none; width: 24px; }
        QComboBox QAbstractItemView {
            background-color: #161b22;
            border: 1px solid #30363d;
            selection-background-color: #1f6feb;
            color: #c9d1d9;
        }

        /* ── Buttons ── */
        QPushButton {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 7px 14px;
            color: #c9d1d9;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #30363d;
            border-color: #8b949e;
        }
        QPushButton:pressed { background-color: #161b22; }

        QPushButton#btn_send {
            background-color: #1f6feb;
            border-color: #1f6feb;
            color: #ffffff;
            font-weight: bold;
            font-size: 13px;
        }
        QPushButton#btn_send:hover { background-color: #388bfd; border-color: #388bfd; }
        QPushButton#btn_send:disabled { background-color: #21262d; border-color: #30363d; color: #484f58; }

        QPushButton#btn_clear {
            background-color: transparent;
            border-color: #da3633;
            color: #f85149;
        }
        QPushButton#btn_clear:hover { background-color: #da363322; }

        /* ── Text areas ── */
        QTextEdit, QListWidget {
            background-color: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 13px;
            selection-background-color: #1f6feb44;
        }
        QTextEdit:focus, QListWidget:focus { border-color: #388bfd; }

        QListWidget::item { padding: 5px 8px; border-radius: 4px; }
        QListWidget::item:selected { background-color: #1f6feb33; color: #58a6ff; }
        QListWidget::item:hover { background-color: #21262d; }

        /* ── Line edit ── */
        QLineEdit {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 6px 10px;
            color: #c9d1d9;
            font-size: 12px;
        }
        QLineEdit:focus { border-color: #58a6ff; }

        /* ── Progress bar ── */
        QProgressBar {
            background-color: #21262d;
            border: none;
            border-radius: 3px;
            height: 4px;
            text-align: center;
            color: transparent;
        }
        QProgressBar::chunk { background-color: #1f6feb; border-radius: 3px; }

        /* ── Tabs ── */
        QTabWidget::pane { border: 1px solid #30363d; border-radius: 6px; }
        QTabBar::tab {
            background-color: transparent;
            border: none;
            padding: 8px 16px;
            color: #8b949e;
            font-size: 12px;
        }
        QTabBar::tab:selected { color: #e6edf3; border-bottom: 2px solid #1f6feb; }
        QTabBar::tab:hover { color: #c9d1d9; }

        /* ── Splitter ── */
        QSplitter::handle { background-color: #21262d; width: 2px; height: 2px; }

        /* ── Scrollbar ── */
        QScrollBar:vertical {
            background: transparent; width: 8px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #30363d; border-radius: 4px; min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #484f58; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        /* ── Status bar ── */
        QStatusBar { background-color: #161b22; border-top: 1px solid #21262d; font-size: 11px; color: #8b949e; }

        /* ── Labels ── */
        QLabel#lbl_title {
            font-size: 18px;
            font-weight: bold;
            color: #58a6ff;
            letter-spacing: 1px;
        }
        QLabel#lbl_subtitle {
            font-size: 11px;
            color: #8b949e;
            letter-spacing: 0.5px;
        }
        """)

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        # Header
        root.addWidget(self._build_header())

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Main splitter: left panel | right panel
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 940])
        root.addWidget(splitter, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._set_status("Ready", "neutral")

    def _build_header(self):
        frame = QFrame()
        frame.setStyleSheet("QFrame { border-bottom: 1px solid #21262d; padding-bottom: 8px; }")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 8)

        dot = QLabel("●")
        dot.setStyleSheet("color: #1f6feb; font-size: 20px;")
        layout.addWidget(dot)

        texts = QVBoxLayout()
        texts.setSpacing(1)
        title = QLabel("MKT3454  –  RAG / MCP Explorer")
        title.setObjectName("lbl_title")
        sub = QLabel("Yıldız Technical University · Mechatronics Engineering · Introduction to Machine Learning")
        sub.setObjectName("lbl_subtitle")
        texts.addWidget(title)
        texts.addWidget(sub)
        layout.addLayout(texts)
        layout.addStretch()

        # API key field (masked)
        key_layout = QVBoxLayout()
        key_layout.setSpacing(2)
        key_label = QLabel("API KEY")
        key_label.setStyleSheet("font-size: 10px; color: #484f58; letter-spacing: 1px;")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-... / anthropic...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setFixedWidth(230)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.api_key_input)
        layout.addLayout(key_layout)

        return frame

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        # ── Model Selection ──
        model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.MODELS)
        mg_layout.addWidget(self.model_combo)

        # MCP toggle
        self.btn_mcp = QPushButton("⚡  Connect MCP Server")
        self.btn_mcp.setCheckable(True)
        mg_layout.addWidget(self.btn_mcp)
        layout.addWidget(model_group)

        # ── Source Files ──
        src_group = QGroupBox("Data Sources")
        sg_layout = QVBoxLayout(src_group)

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("＋ Add Files")
        self.btn_add_dir   = QPushButton("📁 Add Folder")
        self.btn_remove    = QPushButton("✕")
        self.btn_remove.setFixedWidth(30)
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_dir)
        btn_row.addWidget(self.btn_remove)
        sg_layout.addLayout(btn_row)

        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(200)
        self.source_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.source_list.setToolTip("Select sources to include in RAG retrieval")
        sg_layout.addWidget(self.source_list)

        # Index button
        self.btn_index = QPushButton("⚙  Build / Refresh Index")
        self.btn_index.setStyleSheet("""
            QPushButton { border-color: #3fb950; color: #3fb950; }
            QPushButton:hover { background-color: #3fb95022; }
        """)
        sg_layout.addWidget(self.btn_index)

        # Source stats label
        self.lbl_src_stats = QLabel("No documents indexed.")
        self.lbl_src_stats.setStyleSheet("font-size: 11px; color: #8b949e;")
        self.lbl_src_stats.setWordWrap(True)
        sg_layout.addWidget(self.lbl_src_stats)

        layout.addWidget(src_group)

        # ── RAG Settings ──
        rag_group = QGroupBox("Retrieval Settings")
        rg_layout = QVBoxLayout(rag_group)

        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk size"))
        self.chunk_size = QComboBox()
        self.chunk_size.addItems(["256", "512", "1024", "2048"])
        self.chunk_size.setCurrentIndex(1)
        chunk_row.addWidget(self.chunk_size)
        rg_layout.addLayout(chunk_row)

        topk_row = QHBoxLayout()
        topk_row.addWidget(QLabel("Top-K"))
        self.top_k = QComboBox()
        self.top_k.addItems(["3", "5", "8", "10"])
        self.top_k.setCurrentIndex(1)
        topk_row.addWidget(self.top_k)
        rg_layout.addLayout(topk_row)

        layout.addWidget(rag_group)
        layout.addStretch()

        return panel

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(8)

        tabs = QTabWidget()

        # ── Tab 1: Chat ──
        chat_tab = QWidget()
        ct_layout = QVBoxLayout(chat_tab)
        ct_layout.setSpacing(8)

        # Output display
        out_group = QGroupBox("Response")
        og_layout = QVBoxLayout(out_group)
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setPlaceholderText("Model response will appear here…")
        self.output_display.setMinimumHeight(300)
        og_layout.addWidget(self.output_display)
        ct_layout.addWidget(out_group, stretch=1)

        # Retrieved chunks display
        ctx_group = QGroupBox("Retrieved Context Chunks")
        cg_layout = QVBoxLayout(ctx_group)
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setPlaceholderText("Retrieved document chunks will appear here…")
        self.context_display.setMaximumHeight(180)
        self.context_display.setStyleSheet("font-size: 11px; color: #8b949e;")
        cg_layout.addWidget(self.context_display)
        ct_layout.addWidget(ctx_group)

        # Prompt input area
        prompt_group = QGroupBox("Prompt")
        pg_layout = QVBoxLayout(prompt_group)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Type your question here…\n\n"
            "Tip: Your question will be answered using the indexed documents via RAG."
        )
        self.prompt_input.setMaximumHeight(110)
        pg_layout.addWidget(self.prompt_input)

        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("🗑  Clear")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_send  = QPushButton("▶  Send Query")
        self.btn_send.setObjectName("btn_send")
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_send)
        pg_layout.addLayout(btn_row)

        ct_layout.addWidget(prompt_group)
        tabs.addTab(chat_tab, "💬  Chat")

        # ── Tab 2: Monitor ──
        monitor_tab = QWidget()
        ml_layout = QVBoxLayout(monitor_tab)
        self.monitor_display = QTextEdit()
        self.monitor_display.setReadOnly(True)
        self.monitor_display.setPlaceholderText("LLM call metadata and token usage will be logged here…")
        self.monitor_display.setStyleSheet("font-size: 11px;")
        ml_layout.addWidget(self.monitor_display)

        btn_export = QPushButton("💾  Export Monitor Log (JSON)")
        btn_export.clicked.connect(self._export_monitor_log)
        ml_layout.addWidget(btn_export)
        tabs.addTab(monitor_tab, "📊  Monitor")

        # ── Tab 3: Session Log ──
        log_tab = QWidget()
        ll_layout = QVBoxLayout(log_tab)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-size: 11px; color: #8b949e;")
        ll_layout.addWidget(self.log_display)
        tabs.addTab(log_tab, "📝  Session Log")

        layout.addWidget(tabs, stretch=1)
        return panel

    # ── Signal Connections ────────────────────────────────────────────────────
    def _connect_signals(self):
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_dir.clicked.connect(self._add_directory)
        self.btn_remove.clicked.connect(self._remove_selected_sources)
        self.btn_index.clicked.connect(self._build_index)
        self.btn_send.clicked.connect(self._send_query)
        self.btn_clear.clicked.connect(self._clear_chat)
        self.btn_mcp.toggled.connect(self._toggle_mcp)
        self.api_key_input.editingFinished.connect(self._update_api_key)

    # ── Slots / Actions ───────────────────────────────────────────────────────
    def _add_files(self):
        ext_filter = "Supported Files ({})".format(
            " ".join(f"*{e}" for e in self.SUPPORTED_EXTS)
        )
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Data Source Files", "", f"{ext_filter};;All Files (*)"
        )
        for f in files:
            self._add_source_item(f)

    def _add_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if directory:
            for root, _, files in os.walk(directory):
                for fname in files:
                    if Path(fname).suffix.lower() in self.SUPPORTED_EXTS:
                        self._add_source_item(os.path.join(root, fname))

    def _add_source_item(self, path: str):
        # Avoid duplicates
        existing = [self.source_list.item(i).data(Qt.UserRole)
                    for i in range(self.source_list.count())]
        if path in existing:
            return
        ext  = Path(path).suffix.lower()
        icon = {"pdf": "📄", "docx": "📝", "txt": "📃",
                "json": "🔷", "csv": "📊", "md": "📋", "html": "🌐"}.get(ext[1:], "📁")
        item = QListWidgetItem(f"{icon}  {Path(path).name}")
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.source_list.addItem(item)
        self._log(f"Source added: {path}")

    def _remove_selected_sources(self):
        for item in self.source_list.selectedItems():
            self.source_list.takeItem(self.source_list.row(item))

    def _build_index(self):
        paths = [self.source_list.item(i).data(Qt.UserRole)
                 for i in range(self.source_list.count())]
        if not paths:
            self._set_status("No sources to index.", "warning")
            return

        self._set_status("Indexing documents…", "info")
        self.progress.setVisible(True)
        self.btn_index.setEnabled(False)

        try:
            stats = self.rag.build_index(
                file_paths=paths,
                chunk_size=int(self.chunk_size.currentText()),
                top_k=int(self.top_k.currentText()),
            )
            self.lbl_src_stats.setText(
                f"✓ {stats['num_chunks']} chunks · {stats['num_docs']} docs · "
                f"embed: {stats.get('embed_model', 'N/A')}"
            )
            self._set_status("Index built successfully.", "success")
            self._log(f"Index built: {stats}")
        except Exception as exc:
            self._set_status(f"Indexing failed: {exc}", "error")
            self._log(f"[ERROR] Indexing: {exc}")
        finally:
            self.progress.setVisible(False)
            self.btn_index.setEnabled(True)

    def _send_query(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self._set_status("Please enter a prompt.", "warning")
            return
        if not self.rag.is_ready():
            self._set_status("Please build the index first.", "warning")
            return

        self.btn_send.setEnabled(False)
        self.progress.setVisible(True)
        self.output_display.clear()
        self.context_display.clear()
        self._set_status("Querying model…", "info")

        model = self.model_combo.currentText()
        self.worker = LLMWorker(self.rag, self.monitor, prompt, model)
        self.worker.token_received.connect(self._append_token)
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_query.connect(self._on_query_finished)
        self.worker.start()

    def _append_token(self, token: str):
        self.output_display.moveCursor(QTextCursor.End)
        self.output_display.insertPlainText(token)
        self.output_display.moveCursor(QTextCursor.End)

    def _on_result_ready(self, response: str):
        # Show retrieved chunks
        chunks = self.rag.last_retrieved_chunks()
        if chunks:
            self.context_display.setPlainText(
                "\n\n─────────────────────\n\n".join(chunks)
            )

    def _on_query_finished(self, metadata: dict):
        self.btn_send.setEnabled(True)
        self.progress.setVisible(False)
        self._set_status(
            f"Done · {metadata.get('elapsed_sec', '?')}s · "
            f"{metadata.get('total_tokens', '?')} tokens",
            "success"
        )
        # Update monitor tab
        entry = json.dumps(metadata, indent=2, ensure_ascii=False)
        self.monitor_display.append(f"[{datetime.datetime.now().isoformat(timespec='seconds')}]\n{entry}\n{'─'*60}\n")
        self._log(f"Query metadata: {metadata}")

    def _on_error(self, message: str):
        self.btn_send.setEnabled(True)
        self.progress.setVisible(False)
        self.output_display.setPlainText(f"⚠ Error:\n\n{message}")
        self._set_status(f"Error: {message[:80]}", "error")

    def _clear_chat(self):
        self.output_display.clear()
        self.context_display.clear()
        self.prompt_input.clear()
        self._set_status("Cleared.", "neutral")

    def _toggle_mcp(self, checked: bool):
        if checked:
            try:
                self.mcp.connect()
                self.btn_mcp.setText("✅  MCP Connected")
                self.btn_mcp.setStyleSheet("QPushButton { border-color: #3fb950; color: #3fb950; }")
                self._log("MCP server connected.")
            except Exception as exc:
                self.btn_mcp.setChecked(False)
                self._set_status(f"MCP connection failed: {exc}", "error")
        else:
            self.mcp.disconnect()
            self.btn_mcp.setText("⚡  Connect MCP Server")
            self.btn_mcp.setStyleSheet("")
            self._log("MCP server disconnected.")

    def _update_api_key(self):
        key = self.api_key_input.text().strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            os.environ["OPENAI_API_KEY"]    = key
            self._set_status("API key updated.", "success")
            self._log("API key updated (hidden).")

    def _export_monitor_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Monitor Log", "monitor_log.json", "JSON (*.json)")
        if path:
            self.monitor.export_json(path)
            self._set_status(f"Monitor log saved: {path}", "success")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, message: str, level: str = "neutral"):
        colors = {
            "success": "#3fb950",
            "error":   "#f85149",
            "warning": "#d29922",
            "info":    "#58a6ff",
            "neutral": "#8b949e",
        }
        color = colors.get(level, "#8b949e")
        self.status.showMessage(message)
        self.status.setStyleSheet(
            f"QStatusBar {{ color: {color}; background-color: #161b22; border-top: 1px solid #21262d; font-size: 11px; }}"
        )

    def _log(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{ts}]  {message}")
        logger.info(message)

    def closeEvent(self, event):
        self.monitor.export_json(LOG_DIR / "final_monitor_log.json")
        logger.info("Application closed. Monitor log saved.")
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
