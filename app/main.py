"""osu!taiko recommender: estimates your osu!taiko skill from your plays and recommends charts."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


ICON_DIR = Path(__file__).resolve().parent / "assets"


def app_icon():
    """The window/taskbar icon (.ico has every size Windows needs; .png as a fallback)."""
    from PySide6.QtGui import QIcon

    for name in ("icon.ico", "icon.png"):
        if (ICON_DIR / name).exists():
            icon = QIcon(str(ICON_DIR / name))
            if not icon.isNull():
                return icon
    return QIcon()


def set_windows_app_id():
    """Without this, Windows groups the window under pythonw.exe and shows Python's icon
    in the taskbar instead of the app's."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("osu.taiko.recommender")
        except Exception:                                                   # noqa: BLE001
            pass


def main():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from taiko_core import packages
    from taiko_gui.main_window import MainWindow

    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(packages.APP_NAME)
    app.setWindowIcon(app_icon())
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
