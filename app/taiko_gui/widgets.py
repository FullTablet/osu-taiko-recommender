"""Small shared widgets: skill panel, chart picker, mod selectors, table cells."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRegularExpression, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication, QPalette, QPen, QRegularExpressionValidator
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QStyle, QStyledItemDelegate, QToolTip,
                               QVBoxLayout, QWidget)

from taiko_core.app_state import chart_label
from taiko_core.mods import DT, EZ, FL, HD, HR, HT, MOD_COMBINATIONS, mods_label

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


class ModCheckRow(QWidget):
    """One checkbox per mod; the combination is the sum of the checked ones (none = NM).
    Checking one mod of an incompatible pair (EZ/HR, DT/HT) unchecks the other."""
    changed = Signal()
    MODS = ((EZ, "EZ"), (HD, "HD"), (HR, "HR"), (DT, "DT"), (HT, "HT"), (FL, "FL"))
    INCOMPATIBLE = ((EZ, HR), (DT, HT))

    def __init__(self, value: int = 0, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.boxes = {}
        for bit, name in self.MODS:
            cb = QCheckBox(name)
            cb.setChecked(bool(value & bit))
            cb.toggled.connect(lambda checked, b=bit: self._toggled(b, checked))
            lay.addWidget(cb)
            self.boxes[bit] = cb

    def _toggled(self, bit, checked):
        if checked:
            for a, b in self.INCOMPATIBLE:
                other = b if bit == a else a if bit == b else None
                if other is not None and self.boxes[other].isChecked():
                    self.boxes[other].blockSignals(True)
                    self.boxes[other].setChecked(False)
                    self.boxes[other].blockSignals(False)
        self.changed.emit()

    def value(self) -> int:
        return sum(bit for bit, cb in self.boxes.items() if cb.isChecked())

    def set_value(self, value: int):
        for bit, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(bool(value & bit))
            cb.blockSignals(False)


def link_button(text: str, tooltip: str, on_click, align_left: bool = False) -> QPushButton:
    """A small push button used inside table cells (links and copy-to-clipboard)."""
    b = QPushButton(text)
    b.setToolTip(tooltip)
    b.setCursor(Qt.PointingHandCursor)
    b.setAutoDefault(False)
    b.setStyleSheet("QPushButton { padding: 1px 8px; %s }" % ("text-align: left;" if align_left else ""))
    if on_click is not None:
        b.clicked.connect(on_click)
    return b


def open_url(url: str):
    QDesktopServices.openUrl(QUrl(url))


COPY_TIP = "copied! paste it (ctrl+v) into the search box of osu!'s song selection to find this chart"


def copy_with_tip(widget: QWidget, text: str):
    QGuiApplication.clipboard().setText(text)
    QToolTip.showText(widget.mapToGlobal(QPoint(0, widget.height())), COPY_TIP, widget, QRect(), 5000)


def format_accuracy_shorthand(text: str) -> str:
    """Accuracy typed without a dot: 2 to 4 digits become XX.XX ("97" -> "97.00",
    "974" -> "97.40", "7685" -> "76.85"); "100" and "1000" mean 100.00. Anything else
    is returned unchanged."""
    t = text.strip()
    if not (t.isascii() and t.isdigit() and 2 <= len(t) <= 4):
        return text
    if t in ("100", "1000"):
        return "100.00"
    return f"{t[:2]}.{t[2:].ljust(2, '0')}"


def check_typed_accuracy(text: str) -> tuple[float, str | None]:
    """Accuracy typed in the add-play window -> (value in %, suggestion). suggestion is
    the 'XX.XX' text to confirm with the player, or None if the value is used as typed:
    2-4 digits without a dot get the same shorthand as the recommendations table, and
    anything over 100 becomes 100.00. Raises ValueError for empty or invalid text."""
    t = text.strip().rstrip("%").strip().replace(",", ".")
    if not t:
        raise ValueError("type the accuracy you got")
    short = format_accuracy_shorthand(t)
    if short != t:
        return float(short), short
    v = float(t)
    if not v >= 0:
        raise ValueError("the accuracy can't be negative")
    if v > 100:
        return 100.0, "100.00"
    return v, None


class EditableCellDelegate(QStyledItemDelegate):
    """Draws a cell as an input box (lighter fill, frame) with a 'type here!' placeholder
    when empty; the placeholder is only painted, never part of the cell's text."""
    PLACEHOLDER = "type here!"

    @staticmethod
    def fill_color(palette: QPalette) -> QColor:
        base = palette.color(QPalette.Base)
        if base.lightness() < 128:                       # dark theme: a lighter grey
            return base.lighter(250)
        return QColor(255, 250, 225)                     # light theme: a light cream, unlike the white rows

    def paint(self, painter, option, index):
        style = option.widget.style() if option.widget else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter, option.widget)   # selection background
        pal = option.palette
        r = option.rect.adjusted(3, 2, -3, -2)
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)
        painter.setBrush(self.fill_color(pal))
        painter.setPen(QPen(pal.color(QPalette.Highlight), 1))
        painter.drawRoundedRect(r, 3, 3)
        text = index.data(Qt.DisplayRole) or ""
        f = painter.font()
        fill = self.fill_color(pal)
        if text:
            painter.setPen(QColor(20, 20, 20) if fill.lightness() >= 128 else pal.color(QPalette.Text))
        else:
            f.setItalic(True)
            painter.setFont(f)
            painter.setPen(QColor(150, 150, 150))
            text = self.PLACEHOLDER
        painter.drawText(r, Qt.AlignCenter, text)
        painter.restore()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_editor: QLineEdit | None = None
        self._watcher = _OutsideClickWatcher(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._watcher)

    def createEditor(self, parent, option, index):
        ed = super().createEditor(parent, option, index)
        if isinstance(ed, QLineEdit):
            ed.setPlaceholderText(self.PLACEHOLDER)
            ed.setAlignment(Qt.AlignCenter)
            self.open_editor = ed
            ed.destroyed.connect(lambda _o=None: setattr(self, "open_editor", None))
        return ed

    def finish_editing(self):
        """Store (and convert) what was typed and close the box."""
        ed = self.open_editor
        if ed is not None:
            self.open_editor = None
            self.commitData.emit(ed)
            self.closeEditor.emit(ed, QStyledItemDelegate.NoHint)


    def setModelData(self, editor, model, index):
        if isinstance(editor, QLineEdit):
            model.setData(index, format_accuracy_shorthand(editor.text()), Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class _OutsideClickWatcher(QObject):
    """Application-wide filter: a click anywhere outside the open accuracy box finishes
    editing, even on parts of the window that don't take keyboard focus (where Qt alone
    would leave the box open). It only looks at mouse presses and never consumes them."""

    def __init__(self, delegate: "EditableCellDelegate"):
        super().__init__(delegate)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        ed = self.delegate.open_editor
        if ed is not None and event.type() == QEvent.MouseButtonPress and isinstance(obj, QWidget):
            try:
                inside = obj is ed or ed.isAncestorOf(obj)
            except RuntimeError:                  # editor already deleted
                self.delegate.open_editor = None
                return False
            if not inside:
                self.delegate.finish_editing()
        return False



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
        self.mod_row = ModCheckRow(0)
        self.mods_note = QLabel("")
        self.mods_note.setStyleSheet("color: gray")
        self.available: list[int] = []
        self.acc = QLineEdit()
        self.acc.setPlaceholderText("e.g. 97.25")
        self.acc.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\s*\d{0,9}([.,]\d{0,6})?\s*%?\s*$")))
        self.acc.setMaximumWidth(160)
        self.accepted_accuracy: float | None = None          # 0..1, set when OK is accepted
        mods_box = QVBoxLayout()
        mods_box.addWidget(self.mod_row)
        mods_box.addWidget(self.mods_note)
        form.addRow("mods", mods_box)
        form.addRow("accuracy", self.acc)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(buttons)
        btn.clicked.connect(self.search)
        self.query.returnPressed.connect(self.search)
        self.results.currentItemChanged.connect(self.chart_changed)
        self.mod_row.changed.connect(self.update_ok)
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

    def current_ref(self):
        item = self.results.currentItem()
        return item.data(Qt.UserRole) if item else None

    def chart_changed(self, item, _prev=None):
        ref = item.data(Qt.UserRole) if item else None
        self.available = list(self.state.db(ref.source).mods_of(ref.chart_id)) if ref is not None else []
        if self.available and self.mod_row.value() not in self.available:
            self.mod_row.set_value(0 if 0 in self.available else self.available[0])
        self.update_ok()

    def update_ok(self):
        ref = self.current_ref()
        mods = self.mod_row.value()
        avail = ", ".join(mods_label(m) for m in self.available)
        if ref is None:
            self.mods_note.setText("pick a chart first")
            ok = False
        elif mods in self.available:
            self.mods_note.setText(f"available for this chart: {avail}")
            ok = True
        else:
            self.mods_note.setText(f"{mods_label(mods)} isn't available for this chart (available: {avail or 'none'})")
            ok = False
        self.ok_button.setEnabled(ok)

    def confirm_accuracy(self, suggestion: str) -> bool:
        """'did you mean to type XX.XX%?' -> True for 'yes, add', False for 'no, cancel'."""
        box = QMessageBox(self)
        box.setWindowTitle("check the accuracy")
        box.setIcon(QMessageBox.Question)
        box.setText(f"did you mean to type {suggestion}%?")
        yes = box.addButton(f"yes, add {suggestion}%", QMessageBox.AcceptRole)
        box.addButton("no, cancel", QMessageBox.RejectRole)
        box.setDefaultButton(yes)
        box.exec()
        return box.clickedButton() is yes

    def accept_if_valid(self):
        if self.current_ref() is None or self.mod_row.value() not in self.available:
            return
        try:
            value, suggestion = check_typed_accuracy(self.acc.text())
        except ValueError as e:
            QMessageBox.warning(self, "accuracy", str(e) if "accuracy" in str(e) else "that isn't a valid accuracy")
            self.acc.setFocus()
            return
        if suggestion is not None:
            if not self.confirm_accuracy(suggestion):
                self.acc.setFocus()
                self.acc.selectAll()
                return                                       # back to the window to fix it
            self.acc.setText(suggestion)
        self.accepted_accuracy = round(value / 100.0, 8)
        self.accept()

    def selection(self):
        ref = self.current_ref()
        mods = self.mod_row.value()
        if ref is None or mods not in self.available:
            return None
        return ref, mods, self.accepted_accuracy
