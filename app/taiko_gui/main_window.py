"""Main window."""
from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QLabel, QMainWindow, QMessageBox, QTabWidget

from taiko_core import packages
from taiko_core.app_state import AppState

from .tabs import ChartsTab, PlaysTab, RecommendTab, UpdatesTab


RANK_TEXT = (
    "what's your global osu!taiko pp rank?\n\n"
    "your pp rank only gives a rough estimate of your skill: it puts every skill dimension "
    "at the same percentile, while real players are stronger in some than in others.\n\n"
    "for a better estimate, add a few plays that you think represent your true skill in the "
    "\"my plays\" tab (\"add play…\"), or play the recommended charts and submit the accuracy "
    "you got. each play refines the estimate.")


def rank_dialog(parent, default: int) -> QInputDialog:
    dlg = QInputDialog(parent)
    dlg.setWindowTitle("your rank")
    dlg.setInputMode(QInputDialog.IntInput)
    dlg.setIntRange(1, 10_000_000)
    dlg.setIntStep(1)
    dlg.setIntValue(default)
    dlg.setLabelText(RANK_TEXT)
    for lab in dlg.findChildren(QLabel):                 # long text: wrap it in a wider dialog
        lab.setWordWrap(True)
    dlg.setMinimumWidth(520)
    return dlg


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
        self.tabs.currentChanged.connect(lambda _i: self.refresh_current())
        self.refresh_all()

    def refresh_current(self):
        """Switching tabs: only the tab being shown needs fresh contents."""
        tab = self.tabs.currentWidget()
        if hasattr(tab, "refresh"):
            tab.refresh()

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
        dlg = rank_dialog(self, int(self.state.user.get("pp_rank", 10000)))
        if dlg.exec():
            self.state.set_skill_from_rank(dlg.intValue())
            self.refresh_all()
