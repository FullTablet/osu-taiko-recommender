"""Chart details window: header, per-note expected accuracy curve, and expected accuracy
and skill priorities for every mod combination of the chart."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHeaderView, QLabel, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QToolTip, QVBoxLayout, QWidget)

from taiko_core.curve import colors, curve_stats
from taiko_core.recommend import DIMENSIONS

from .widgets import link_button


def _pct(q: float, digits: int = 2) -> str:
    return f"{q * 100:.{digits}f}%"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


class NoteCurve(QWidget):
    """Smoothed per-note expected accuracy, coloured by distance to the mean."""
    MARGIN = (64, 14, 14, 36)            # left, top, right, bottom

    def __init__(self, per_note, parent=None):
        super().__init__(parent)
        self.set_data(per_note)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_data(self, per_note):
        self.raw = np.asarray(per_note, np.float64)          # what the tooltip shows
        st = curve_stats(per_note)
        self.y = st["smoothed"]
        self.mean, self.sd = st["mean"], st["sd"]
        self.cols = [QColor(*c) for c in colors(st["z"])]
        self.n = len(self.y)
        lo, hi = (float(self.y.min()), float(self.y.max())) if self.n else (0.0, 1.0)
        lo, hi = min(lo, self.mean - 2 * self.sd), max(hi, self.mean + 2 * self.sd)
        pad = max((hi - lo) * 0.06, 1e-4)
        self.lo, self.hi = max(0.0, lo - pad), min(1.0, hi + pad)
        if not self.hi > self.lo:
            self.lo, self.hi = max(0.0, self.lo - 0.01), min(1.0, self.hi + 0.01)
        self.update()

    def plot_rect(self) -> QRectF:
        l, t, r, b = self.MARGIN
        return QRectF(l, t, max(self.width() - l - r, 10), max(self.height() - t - b, 10))

    def to_xy(self, i: float, v: float, rect: QRectF) -> QPointF:
        x = rect.left() + (i / max(self.n - 1, 1)) * rect.width()
        y = rect.bottom() - (v - self.lo) / (self.hi - self.lo) * rect.height()
        return QPointF(x, y)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = self.palette()
        rect = self.plot_rect()
        p.fillRect(rect, pal.color(QPalette.Base))
        text = pal.color(QPalette.Text)
        grid = QColor(text)
        grid.setAlpha(40)
        f = p.font()
        f.setPointSizeF(f.pointSizeF() * 0.9)
        p.setFont(f)
        # y axis: expected accuracy
        ticks = nice_ticks(self.lo, self.hi, 6)
        step = (ticks[1] - ticks[0]) if len(ticks) > 1 else 0.01
        digits = max(1, math.ceil(-math.log10(step * 100) - 1e-9)) if step > 0 else 1
        for v in ticks:
            pt = self.to_xy(0, v, rect)
            p.setPen(QPen(grid, 1))
            p.drawLine(QPointF(rect.left(), pt.y()), QPointF(rect.right(), pt.y()))
            p.setPen(text)
            p.drawText(QRectF(0, pt.y() - 9, rect.left() - 6, 18), Qt.AlignRight | Qt.AlignVCenter, _pct(v, digits))
        # x axis: note number
        if self.n > 1:
            for v in nice_ticks(1, self.n, 8):
                pt = self.to_xy(v - 1, self.lo, rect)
                p.setPen(QPen(grid, 1))
                p.drawLine(QPointF(pt.x(), rect.top()), QPointF(pt.x(), rect.bottom()))
                p.setPen(text)
                p.drawText(QRectF(pt.x() - 40, rect.bottom() + 3, 80, 16), Qt.AlignHCenter | Qt.AlignTop, f"{v:,.0f}")
        p.drawText(QRectF(rect.left(), rect.bottom() + 18, rect.width(), 16), Qt.AlignHCenter | Qt.AlignTop, "note")
        # mean and +-2 sd
        for v, c in ((self.mean - 2 * self.sd, QColor(220, 40, 40)), (self.mean, QColor(text)),
                     (self.mean + 2 * self.sd, QColor(40, 90, 230))):
            if self.lo <= v <= self.hi:
                c.setAlpha(150)
                pen = QPen(c, 1, Qt.DashLine)
                p.setPen(pen)
                y = self.to_xy(0, v, rect).y()
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        # the curve: one segment per pair of notes, coloured by its z-score (smooth gradient)
        p.setClipRect(rect)
        if self.n == 1:
            p.setPen(QPen(self.cols[0], 6, Qt.SolidLine, Qt.RoundCap))
            p.drawPoint(self.to_xy(0, self.y[0], rect))
        pts = [self.to_xy(i, v, rect) for i, v in enumerate(self.y)]
        for i in range(self.n - 1):
            p.setPen(QPen(self.cols[i], 2.2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pts[i], pts[i + 1])
        p.setClipping(False)
        p.setPen(QPen(grid, 1))
        p.drawRect(rect)
        p.end()

    def mouseMoveEvent(self, event):
        rect = self.plot_rect()
        if self.n == 0 or not rect.contains(event.position()):
            QToolTip.hideText()
            return
        i = int(round((event.position().x() - rect.left()) / rect.width() * (self.n - 1)))
        i = min(max(i, 0), self.n - 1)
        QToolTip.showText(event.globalPosition().toPoint(),
                          f"note {i + 1:,}: {_pct(self.raw[i])} expected", self)


class ChartDetailsDialog(QDialog):
    def __init__(self, details: dict, parent=None, on_add=None):
        """on_add(mods): called by the 'add to list' buttons (the window then closes)."""
        super().__init__(parent)
        self.on_add = on_add
        self.setWindowTitle("chart details")
        self.details = details
        lay = QVBoxLayout(self)
        self.head = QLabel()
        hf = self.head.font()
        hf.setPointSizeF(hf.pointSizeF() * 1.3)
        hf.setBold(True)
        self.head.setFont(hf)
        self.head.setWordWrap(True)
        self.head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.head)

        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setStyleSheet("color: gray")
        lay.addWidget(self.sub)
        self.curve = NoteCurve(details["per_note"])
        lay.addWidget(self.curve, 3)

        lay.addWidget(QLabel(f"every mod combination in the database: expected accuracy with your current skill, "
                             f"and how much each skill matters for reaching {_pct(details['target'])} "
                             "(the target accuracy in your settings)"))
        rows = details["rows"]
        self.table = QTableWidget(len(rows), 4 + len(DIMENSIONS))
        self.table.setHorizontalHeaderLabels(["mods", "expected accuracy"] + DIMENSIONS + ["plot", "add to list"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, r in enumerate(rows):
            cells = [r["mods_label"], _pct(r["expected"])] + [_pct(x, 1) for x in r["priorities"]]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, it)
            btn = link_button("click here", f"show the plot for {r['mods_label']}",
                              lambda _c=False, m=r["mods"]: self.show_mods(m))
            self.table.setCellWidget(i, len(cells), btn)
            add = link_button("add", f"add this chart with {r['mods_label']} at the bottom of the recommendations list",
                              lambda _c=False, m=r["mods"]: self.add_to_list(m))
            add.setEnabled(on_add is not None)
            self.table.setCellWidget(i, len(cells) + 1, add)
        self.table.resizeRowsToContents()
        h = (self.table.horizontalHeader().sizeHint().height() + 2 * self.table.frameWidth()
             + sum(self.table.rowHeight(i) for i in range(len(rows))))
        self.table.setFixedHeight(min(h, 420))           # every row visible (scrolls past ~14 rows)
        lay.addWidget(self.table)
        self.resize(980, min(320 + 120 + self.table.height(), 900))
        self.shown_mods = None
        self.show_mods(details["mods"])

    def add_to_list(self, mods: int):
        if self.on_add is not None:
            self.on_add(mods)
            self.close()

    def show_mods(self, mods: int):
        """Switch the header and the plot to another mod combination (bold row in the table)."""
        row = next((r for r in self.details["rows"] if r["mods"] == mods), None)
        per_note = row["per_note"] if row is not None else self.details["per_note"]
        header = row["header"] if row is not None else self.details["header"]
        self.shown_mods = mods
        self.head.setText(header)
        self.curve.set_data(per_note)
        self.sub.setText(f"expected accuracy on each note with your current skill ({len(per_note):,} notes, "
                         f"lightly smoothed). mean {_pct(self.curve.mean)}; colour: red = 2 standard deviations "
                         f"or more below the mean, yellow = near the mean, blue = 2 or more above "
                         f"(1 standard deviation = {self.curve.sd * 100:.2f} percentage points).")
        for i, r in enumerate(self.details["rows"]):
            for c in range(2 + len(DIMENSIONS)):
                it = self.table.item(i, c)
                f = QFont(self.table.font())
                f.setBold(r["mods"] == mods)
                it.setFont(f)
                it.setToolTip("shown in the plot" if r["mods"] == mods else "")


__all__ = ["ChartDetailsDialog", "NoteCurve", "nice_ticks"]
