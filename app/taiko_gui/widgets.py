"""Small shared widgets: skill panel, chart picker, mod selector."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from taiko_core.app_state import chart_label
from taiko_core.mods import MOD_COMBINATIONS, mods_label

DIM_NAMES = ["α", "β", "γ", "δ", "ε"]


def pct(q: float, digits: int = 1) -> str:
    return f"{q * 100:.{digits}f}%"


class SkillPanel(QGroupBox):
    def __init__(self, state, parent=None):
        super().__init__("your skill (percentile among top-10,000 plays)", parent)
        self.state = state
        lay = QGridLayout(self)
        self.labels = []
        for i, name in enumerate(DIM_NAMES):
            head = QLabel(name)
            hf = head.font()
            hf.setPointSize(hf.pointSize() + 2)
            head.setFont(hf)
            lay.addWidget(head, 0, i, alignment=Qt.AlignCenter)
            lab = QLabel("-")
            lab.setAlignment(Qt.AlignCenter)
            f = lab.font()
            f.setPointSize(f.pointSize() + 4)
            f.setBold(True)
            lab.setFont(f)
            lay.addWidget(lab, 1, i)
            self.labels.append(lab)
        self.note = QLabel("")
        self.note.setStyleSheet("color: gray")
        lay.addWidget(self.note, 2, 0, 1, len(DIM_NAMES))
        self.refresh()

    def refresh(self):
        cur = self.state.current_skill() if self.state.has_base else None
        if cur is None:
            for lab in self.labels:
                lab.setText("-")
            self.note.setText("no skill estimate yet")
            return
        for lab, q in zip(self.labels, cur["quantiles"]):
            lab.setText(pct(q))
        how = "from your pp rank" if cur["method"] == "rank" else f"from your last {cur['n_plays']} plays"
        self.note.setText(f"estimated {how} ({cur['created_at']})")


class ModSelector(QWidget):
    """Checkboxes for every mod combination."""

    def __init__(self, selected, parent=None, columns: int = 6):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.boxes = {}
        for i, m in enumerate(MOD_COMBINATIONS):
            cb = QCheckBox(mods_label(m))
            cb.setChecked(m in selected)
            lay.addWidget(cb, i // columns, i % columns)
            self.boxes[m] = cb

    def selected(self) -> list[int]:
        return [m for m, cb in self.boxes.items() if cb.isChecked()]


class ChartPickerDialog(QDialog):
    """Search a chart (base: beatmap ID or words; imported: file hash or words), pick
    mods and enter the accuracy."""

    def __init__(self, state, parent=None, title="add a play"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 520)
        self.state = state
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("beatmap ID, file hash (imported charts) or words from artist/title/difficulty")
        btn = QPushButton("search")
        row.addWidget(self.query)
        row.addWidget(btn)
        lay.addLayout(row)
        self.results = QListWidget()
        self.results.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self.results)
        form = QFormLayout()
        self.mods = QComboBox()
        self.acc = QDoubleSpinBox()
        self.acc.setRange(0.0, 100.0)
        self.acc.setDecimals(2)
        self.acc.setSuffix(" %")
        self.acc.setValue(95.0)
        form.addRow("mods", self.mods)
        form.addRow("accuracy", self.acc)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(buttons)
        btn.clicked.connect(self.search)
        self.query.returnPressed.connect(self.search)
        self.results.currentItemChanged.connect(self.chart_changed)
        buttons.accepted.connect(self.accept_if_valid)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)

    def search(self):
        self.results.clear()
        for ref, info in self.state.search(self.query.text()):
            src = "ranked" if ref.source == "base" else "imported"
            ident = f"#{info['beatmap_id']}" if ref.source == "base" else (info.get("file_md5") or "")[:10]
            item = QListWidgetItem(f"{chart_label(info)}   ({src} {ident})")
            item.setData(Qt.UserRole, ref)
            self.results.addItem(item)
        if self.results.count() == 0:
            self.results.addItem(QListWidgetItem("no charts found"))

    def chart_changed(self, item, _prev=None):
        self.mods.clear()
        ref = item.data(Qt.UserRole) if item else None
        if ref is None:
            self.ok_button.setEnabled(False)
            return
        db = self.state.db(ref.source)
        for m in db.mods_of(ref.chart_id):
            self.mods.addItem(mods_label(m), m)
        self.ok_button.setEnabled(self.mods.count() > 0)

    def accept_if_valid(self):
        if self.selection() is not None:
            self.accept()

    def selection(self):
        item = self.results.currentItem()
        ref = item.data(Qt.UserRole) if item else None
        if ref is None or self.mods.currentData() is None:
            return None
        return ref, int(self.mods.currentData()), self.acc.value() / 100.0
