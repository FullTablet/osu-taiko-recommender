"""Run slow work in a background thread with a progress dialog."""
from __future__ import annotations

import time
import traceback

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout


class ProgressDialog(QDialog):
    """Modal progress window. (QProgressDialog re-enters the event loop from setValue,
    which can crash with frequent updates from a worker thread; this one doesn't.)"""
    canceled = Signal()

    def __init__(self, title, cancellable, parent):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self)
        self.title = title
        self.label = QLabel(title)
        self.label.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        if cancellable:
            btn = QPushButton("cancel")
            btn.clicked.connect(lambda: (btn.setEnabled(False), btn.setText("cancelling…"), self.canceled.emit()))
            lay.addWidget(btn)

    def setValue(self, v):
        self.bar.setValue(v)

    def setLabelText(self, text):
        self.label.setText(text)

    def reject(self):          # Esc does nothing while working
        pass


class _Worker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel_requested = False

    def run(self):
        try:
            result = self.fn(self._progress, lambda: self.cancel_requested)
        except Exception as e:                                       # noqa: BLE001 - shown to the user
            traceback.print_exc()
            self.failed.emit(str(e) or e.__class__.__name__)
        else:
            self.finished.emit(result)

    _last = 0.0

    def _progress(self, fraction, text=""):
        now = time.monotonic()
        value = int(max(0.0, min(1.0, float(fraction))) * 1000)
        if now - self._last >= 0.05 or value >= 1000:          # at most ~20 updates per second
            self._last = now
            self.progress.emit(value, str(text or ""))


class _Relay(QObject):
    """Lives in the GUI thread; worker signals reach it through queued connections, so the
    callbacks (which touch widgets) always run in the GUI thread."""

    def __init__(self, on_progress, on_done, on_failed):
        super().__init__()
        self._p, self._d, self._f = on_progress, on_done, on_failed

    @Slot(int, str)
    def progress(self, value, text):
        self._p(value, text)

    @Slot(object)
    def done(self, result):
        self._d(result)

    @Slot(str)
    def failed(self, msg):
        self._f(msg)


_running = []          # keep threads alive until they finish


def run_task(parent, title: str, fn, on_done=None, on_error=None, cancellable: bool = True):
    """fn(progress(fraction, text), cancelled() -> bool) runs in a thread; on_done(result)
    runs in the GUI thread afterwards. Shows a modal progress dialog meanwhile."""
    dialog = ProgressDialog(title, cancellable, parent)
    thread = QThread()
    worker = _Worker(fn)
    worker.moveToThread(thread)

    def update(value, text):
        dialog.setValue(value)
        if text:
            dialog.setLabelText(f"{title}\n{text}")

    def cleanup():
        assert QThread.currentThread() is not thread, "callback ran in the worker thread"
        dialog.hide()
        dialog.deleteLater()
        thread.quit()
        thread.wait()
        if (thread, worker, relay) in _running:
            _running.remove((thread, worker, relay))

    def done(result):
        cleanup()
        if on_done:
            on_done(result)

    def failed(msg):
        cleanup()
        if on_error:
            on_error(msg)
        else:
            QMessageBox.warning(parent, title, msg)

    relay = _Relay(update, done, failed)
    worker.progress.connect(relay.progress, Qt.QueuedConnection)
    worker.finished.connect(relay.done, Qt.QueuedConnection)
    worker.failed.connect(relay.failed, Qt.QueuedConnection)
    dialog.canceled.connect(lambda: setattr(worker, "cancel_requested", True))
    thread.started.connect(worker.run)
    _running.append((thread, worker, relay))
    thread.start()
    dialog.show()
    return thread
