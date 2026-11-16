"""Main window."""
from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QLabel, QMainWindow, QMessageBox, QTabWidget

from taiko_core import packages
from taiko_core.app_state import AppState

from .tabs import ChartsTab, PlaysTab, RecommendTab, UpdatesTab


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle(f"{packages.APP_NAME} {packages.app_config().get('app_version', '')}")
        self.resize(1000, 700)
        self.tabs = QTabWidget()
        self.recommend = RecommendTab(self)
        self.plays = PlaysTab(self)
        self.charts = ChartsTab(self)
        self.updates = UpdatesTab(self)
        self.tabs.addTab(self.recommend, "recommendations")
        self.tabs.addTab(self.plays, "my plays")
        self.tabs.addTab(self.charts, "my charts")
        self.tabs.addTab(self.updates, "downloads && updates")
        self.setCentralWidget(self.tabs)
        credit = QLabel(packages.ATTRIBUTION)
        credit.setStyleSheet("color: gray")
        self.statusBar().addPermanentWidget(credit)
        self.tabs.currentChanged.connect(lambda _i: self.refresh_all())
        self.refresh_all()

    def refresh_all(self):
        for t in (self.recommend, self.plays, self.charts, self.updates):
            t.refresh()
        has_base = self.state.has_base
        self.tabs.setTabEnabled(0, has_base)
        self.tabs.setTabEnabled(1, has_base)
        status = "ready" if has_base else "the base database isn't installed yet"
        self.statusBar().showMessage(status)

    def start(self):
        """Called once the window is shown: route to the right tab and ask for the pp rank if needed."""
        if not self.state.has_base:
            self.tabs.setCurrentWidget(self.updates)
            QMessageBox.information(self, "welcome", "to start, download the base database "
                                    "('check for updates', then 'download base database').")
            return
        self.tabs.setCurrentWidget(self.recommend)
        self.ask_rank()

    def after_package_change(self):
        if self.state.has_base:
            self.tabs.setCurrentWidget(self.recommend)
            self.ask_rank()

    def ask_rank(self, force: bool = False):
        if not self.state.has_base:
            return
        if self.state.user.current_skill() is not None and not force:
            return
        rank, ok = QInputDialog.getInt(
            self, "your rank", "what's your global osu!taiko pp rank?\n"
            "(used for your first skill estimate; your plays refine it afterwards)",
            int(self.state.user.get("pp_rank", 10000)), 1, 10_000_000, 1)
        if ok:
            self.state.set_skill_from_rank(rank)
            self.refresh_all()
