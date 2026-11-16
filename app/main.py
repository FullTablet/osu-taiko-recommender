"""osu!taiko recommender: estimates your osu!taiko skill from your plays and recommends charts."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from taiko_core import packages
    from taiko_gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(packages.APP_NAME)
    app.setStyle("Fusion")

    log_path = packages.data_dir() / "error_log.txt"

    def excepthook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass
        QMessageBox.critical(None, "error", f"something went wrong:\n{value}\n\n(details saved in {log_path})")

    sys.excepthook = excepthook
    window = MainWindow()
    window.show()
    if not os.environ.get("OTR_NO_START"):
        QTimer.singleShot(0, window.start)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
