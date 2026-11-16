"""The four tabs: recommendations (main), my plays, my charts, downloads & updates."""
from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from taiko_core import packages
from taiko_core.app_state import COMP_FILE, chart_label
from taiko_core.chartdb import ChartDB, IncompatibleDatabase
from taiko_core.importer import import_files, recompute_all
from taiko_core.mods import DEFAULT_IMPORT_MODS, DT, EZ, FL, HD, HR, HT, mods_label
from taiko_core.recommend import DIMENSION_NAMES, DIMENSIONS

from .widgets import ChartPickerDialog, ModSelector, SkillPanel, pct
from .worker import run_task


def _item(text, editable=False, align=None):
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if align is not None:
        it.setTextAlignment(align)
    return it


def _table(headers, stretch_col=0):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    return t


def parse_percent(text: str):
    """'97.5', '97.5%', '97,5' -> 0.975; '' -> None; raises ValueError otherwise."""
    t = text.strip().replace("%", "").replace(",", ".")
    if not t:
        return None
    v = float(t)
    if not 0.0 <= v <= 100.0:
        raise ValueError("accuracy must be between 0 and 100")
    return v / 100.0


# =============================================================================== recommendations
class RecommendTab(QWidget):
    PRIORITY_ITEMS = [("none", None)] + [(f"{g}  {n}", i) for i, (g, n) in enumerate(zip(DIMENSIONS, DIMENSION_NAMES))]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state
        self.recs = []
        lay = QVBoxLayout(self)
        self.skill = SkillPanel(self.state)
        lay.addWidget(self.skill)

        box = QGroupBox("recommendation settings")
        grid = QGridLayout(box)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 1_000_000)
        self.k_spin.setSingleStep(100)
        self.s_spin = QSpinBox()
        self.s_spin.setRange(1, 1000)
        self.priority = QComboBox()
        for text, val in self.PRIORITY_ITEMS:
            self.priority.addItem(text, val)
        self.target = QDoubleSpinBox()
        self.target.setDecimals(2)
        self.target.setRange(0.01, 99.99)            # strictly between 0% and 100%
        self.target.setSuffix(" %")
        self.target.setSingleStep(0.5)
        grid.addWidget(QLabel("charts to search (K)"), 0, 0)
        grid.addWidget(self.k_spin, 0, 1)
        grid.addWidget(QLabel("charts to show (S)"), 0, 2)
        grid.addWidget(self.s_spin, 0, 3)
        grid.addWidget(QLabel("skill priority"), 1, 0)
        grid.addWidget(self.priority, 1, 1)
        grid.addWidget(QLabel("target accuracy"), 1, 2)
        grid.addWidget(self.target, 1, 3)
        mrow = QHBoxLayout()
        self.mod_boxes = {}
        for bit, name in ((EZ, "EZ"), (HD, "HD"), (HR, "HR"), (DT, "DT"), (HT, "HT"), (FL, "FL")):
            cb = QCheckBox(name)
            self.mod_boxes[bit] = cb
            mrow.addWidget(cb)
        self.mods_info = QLabel("")
        self.mods_info.setStyleSheet("color: gray")
        mrow.addSpacing(12)
        mrow.addWidget(self.mods_info)
        mrow.addStretch()
        grid.addWidget(QLabel("mods"), 2, 0)
        grid.addLayout(mrow, 2, 1, 1, 3)
        lay.addWidget(box)

        row = QHBoxLayout()
        self.get_btn = QPushButton("get recommendations")
        self.submit_btn = QPushButton("submit plays and update skill")
        self.submit_btn.setEnabled(False)
        row.addWidget(self.get_btn)
        self.result_info = QLabel("")
        self.result_info.setStyleSheet("color: gray")
        row.addWidget(self.result_info)
        row.addStretch()
        row.addWidget(self.submit_btn)
        lay.addLayout(row)
        hint = QLabel("play the charts, then type the accuracy you got (leave empty if you skipped one); the last "
                      "row counts as your most recent play. α–ε: how much each skill matters for reaching the "
                      "target accuracy on that chart.")
        hint.setStyleSheet("color: gray")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.table = _table(["chart", "mods", "expected accuracy"] + DIMENSIONS + ["your accuracy (%)"])
        lay.addWidget(self.table)

        self.load_settings()
        self.get_btn.clicked.connect(self.get_recommendations)
        self.submit_btn.clicked.connect(self.submit)
        for w in (self.k_spin, self.s_spin):
            w.valueChanged.connect(lambda _v: self.save_settings())
        self.target.valueChanged.connect(lambda _v: self.save_settings())
        self.priority.currentIndexChanged.connect(lambda _i: self.save_settings())
        for bit, cb in self.mod_boxes.items():
            cb.toggled.connect(lambda checked, b=bit: self.mod_toggled(b, checked))

    # ---- settings
    def selected_mods(self) -> int:
        return sum(bit for bit, cb in self.mod_boxes.items() if cb.isChecked())

    def mod_toggled(self, bit, checked):
        if checked:
            for a_, b_ in ((EZ, HR), (DT, HT)):                    # incompatible pairs
                other = b_ if bit == a_ else a_ if bit == b_ else None
                if other is not None and self.mod_boxes[other].isChecked():
                    self.mod_boxes[other].setChecked(False)
        self.save_settings()

    def load_settings(self):
        st = self.state.recommendation_settings()
        widgets = [self.k_spin, self.s_spin, self.target, self.priority] + list(self.mod_boxes.values())
        for w in widgets:
            w.blockSignals(True)
        self.k_spin.setValue(st["K"])
        self.s_spin.setValue(st["S"])
        self.target.setValue(round(st["target"] * 100, 2))
        self.priority.setCurrentIndex(max(0, self.priority.findData(st["priority"])))
        for bit, cb in self.mod_boxes.items():
            cb.setChecked(bool(st["mods"] & bit))
        for w in widgets:
            w.blockSignals(False)
        self.update_mods_info()

    def current_settings(self) -> dict:
        return {"K": self.k_spin.value(), "S": self.s_spin.value(), "priority": self.priority.currentData(),
                "target": self.target.value() / 100.0, "mods": self.selected_mods()}

    def save_settings(self):
        self.state.save_recommendation_settings(self.current_settings())
        self.update_mods_info()

    def update_mods_info(self):
        mods = self.selected_mods()
        n = self.state.count_variants(mods) if self.state.has_base else 0
        self.mods_info.setText(f"→ {mods_label(mods)}: {n:,} charts available")

    def refresh(self):
        self.skill.refresh()
        ok = self.state.has_base and self.state.current_skill() is not None
        self.get_btn.setEnabled(ok)
        if self.state.has_base:
            self.update_mods_info()

    # ---- actions
    def get_recommendations(self):
        settings = self.current_settings()
        if self.state.count_variants(settings["mods"]) == 0:
            QMessageBox.information(self, "no charts", f"there are no charts with {mods_label(settings['mods'])} "
                                    "yet. the ranked database has NM, HD, HR, DT, HDHR, HDDT, HT and EZ; other "
                                    "combinations come from charts you import with that combination checked.")
            return

        def done(res):
            self.recs = res["charts"]
            self.table.setRowCount(len(self.recs))
            for i, r in enumerate(self.recs):
                self.table.setItem(i, 0, _item(r["label"]))
                self.table.setItem(i, 1, _item(r["mods_label"], align=Qt.AlignCenter))
                exp = _item(pct(r["expected"], 2), align=Qt.AlignCenter)
                if not r["in_window"]:
                    exp.setForeground(Qt.gray)
                    exp.setToolTip("outside target ± 1%")
                self.table.setItem(i, 2, exp)
                for k, p in enumerate(r["priorities"]):
                    self.table.setItem(i, 3 + k, _item(pct(p, 1), align=Qt.AlignCenter))
                self.table.setItem(i, 8, _item("", editable=True, align=Qt.AlignCenter))
            self.result_info.setText(f"{res['in_window']} of {res['candidates']:,} charts searched are within "
                                     f"target ± 1%")
            self.submit_btn.setEnabled(bool(self.recs))

        run_task(self, "finding charts for you",
                 lambda p, c: self.state.recommend(settings, progress=p, cancelled=c), done)

    def submit(self):
        rows = []
        try:
            for i, r in enumerate(self.recs):
                it = self.table.item(i, 8)
                acc = parse_percent(it.text() if it else "")
                if acc is not None:
                    rows.append((r["ref"], r["mods"], acc))
        except ValueError as e:
            QMessageBox.warning(self, "accuracy", f"row {i + 1}: {e}")
            return
        if not rows:
            QMessageBox.information(self, "no plays", "type the accuracy of at least one chart first.")
            return
        self.state.record_plays(rows)
        self.submit_btn.setEnabled(False)

        def done(info):
            self.main.refresh_all()
            if info.get("missing"):
                QMessageBox.information(self, "skill updated",
                                        f"{info['missing']} of your plays are on charts that aren't available "
                                        "(removed or not yet re-imported) and were left out.")
            self.get_recommendations()

        run_task(self, "updating your skill", lambda p, c: self.state.update_skill(progress=p), done,
                 cancellable=False)


# =============================================================================== my plays
class PlaysTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.add_btn = QPushButton("add play…")
        self.edit_btn = QPushButton("change accuracy…")
        self.del_btn = QPushButton("delete selected")
        self.recalc_btn = QPushButton("recalculate skill")
        self.rank_btn = QPushButton("restart from pp rank…")
        for b in (self.add_btn, self.edit_btn, self.del_btn):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(self.rank_btn)
        row.addWidget(self.recalc_btn)
        lay.addLayout(row)
        self.info = QLabel("")
        self.info.setStyleSheet("color: gray")
        lay.addWidget(self.info)
        self.table = _table(["#", "chart", "mods", "accuracy", "entered"], stretch_col=1)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table)
        self.add_btn.clicked.connect(self.add_play)
        self.edit_btn.clicked.connect(self.edit_play)
        self.del_btn.clicked.connect(self.delete_plays)
        self.recalc_btn.clicked.connect(self.recalculate)
        self.rank_btn.clicked.connect(lambda: self.main.ask_rank(force=True))
        self.table.doubleClicked.connect(lambda _i: self.edit_play())

    def refresh(self):
        plays = self.state.user.plays(newest_first=True)
        self.table.setRowCount(len(plays))
        n = len(plays)
        for i, p in enumerate(plays):
            self.table.setItem(i, 0, _item(n - i, align=Qt.AlignCenter))
            label = p["label"] or (f"#{p['beatmap_id']}" if p["source"] == "base" else p["file_md5"])
            if p["source"] == "comp":
                label += "  (imported)"
            it = _item(label)
            it.setData(Qt.UserRole, p["play_id"])
            self.table.setItem(i, 1, it)
            self.table.setItem(i, 2, _item(mods_label(p["mods"]), align=Qt.AlignCenter))
            self.table.setItem(i, 3, _item(pct(p["accuracy"], 2), align=Qt.AlignCenter))
            self.table.setItem(i, 4, _item(p["played_at"] or ""))
        ok = self.state.has_base
        for b in (self.add_btn, self.recalc_btn, self.rank_btn):
            b.setEnabled(ok)
        self.info.setText(f"{n} plays (newest first). only the most recent 1,000 are used, "
                          "each weighted 0.99× the one after it.")

    def _selected_ids(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 1).data(Qt.UserRole) for r in rows]

    def add_play(self):
        dlg = ChartPickerDialog(self.state, self)
        if dlg.exec() and dlg.selection():
            ref, mods, acc = dlg.selection()
            self.state.record_plays([(ref, mods, acc)])
            self.refresh()

    def edit_play(self):
        ids = self._selected_ids()
        if len(ids) != 1:
            return
        play = next(p for p in self.state.user.plays() if p["play_id"] == ids[0])
        val, ok = QInputDialog.getDouble(self, "change accuracy", f"{play['label']}\naccuracy (%)",
                                         play["accuracy"] * 100, 0.0, 100.0, 2)
        if ok:
            self.state.user.update_play(ids[0], accuracy=val / 100.0)
            self.refresh()

    def delete_plays(self):
        ids = self._selected_ids()
        if ids and QMessageBox.question(self, "delete plays", f"delete {len(ids)} play(s)?") == QMessageBox.Yes:
            self.state.user.delete_plays(ids)
            self.refresh()

    def recalculate(self):
        def done(info):
            self.main.refresh_all()
            msg = f"skill re-estimated from {info['plays']} plays."
            if info.get("missing"):
                msg += f" {info['missing']} plays on unavailable charts were left out."
            QMessageBox.information(self, "skill updated", msg)

        run_task(self, "updating your skill", lambda p, c: self.state.update_skill(progress=p), done,
                 cancellable=False)


# =============================================================================== my charts
class ChartsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state
        lay = QVBoxLayout(self)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        box = QGroupBox("mod combinations to compute when importing")
        bl = QVBoxLayout(box)
        self.mods = ModSelector(self.state.user.get("import_mods", DEFAULT_IMPORT_MODS))
        bl.addWidget(self.mods)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.import_btn = QPushButton("import .osu / .osz files…")
        self.delete_btn = QPushButton("delete selected")
        self.recompute_btn = QPushButton("recompute for the installed model")
        self.export_btn = QPushButton("export database…")
        self.merge_btn = QPushButton("import database…")
        for b in (self.import_btn, self.delete_btn, self.recompute_btn):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(self.export_btn)
        row.addWidget(self.merge_btn)
        lay.addLayout(row)
        srow = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter by file hash or words")
        srow.addWidget(QLabel("search"))
        srow.addWidget(self.filter)
        lay.addLayout(srow)
        self.table = _table(["chart", "notes", "mods", "file hash", "added"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table)
        self.import_btn.clicked.connect(self.import_charts)
        self.delete_btn.clicked.connect(self.delete_charts)
        self.recompute_btn.clicked.connect(self.recompute)
        self.export_btn.clicked.connect(self.export_db)
        self.merge_btn.clicked.connect(self.merge_db)
        self.filter.textChanged.connect(lambda _t: self.refresh_table())
        for cb in self.mods.boxes.values():
            cb.toggled.connect(lambda _c: self.state.user.set("import_mods", self.mods.selected()))

    def refresh(self):
        s = self.state
        if not s.exp_manifest:
            msg = "the expansion pack isn't installed: download it in the 'downloads & updates' tab to import charts."
        elif s.expansion_mismatch:
            msg = "the expansion pack belongs to a different model than the base database: download the matching one."
        else:
            msg = "imported charts are stored in your own chart database and used together with the ranked charts."
        if s.comp is not None and not s.comp_valid and s.has_base:
            msg += ("\nyour imported charts were computed with a different model and can't be used until they are "
                    "recomputed" + (" (button below)." if s.has_expansion else " (needs the matching expansion pack)."))
        self.status.setText(msg)
        self.import_btn.setEnabled(s.has_expansion)
        self.recompute_btn.setEnabled(s.has_expansion and s.comp is not None and not s.comp_valid)
        self.export_btn.setEnabled(s.comp is not None)
        self.merge_btn.setEnabled(s.has_base or s.has_expansion)
        self.refresh_table()

    def refresh_table(self):
        db = self.state.comp
        rows = db.search(self.filter.text(), limit=5000) if db else []
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            it = _item(chart_label(r))
            it.setData(Qt.UserRole, r["chart_id"])
            self.table.setItem(i, 0, it)
            self.table.setItem(i, 1, _item(r["n_notes"], align=Qt.AlignCenter))
            self.table.setItem(i, 2, _item(len(db.mods_of(r["chart_id"])), align=Qt.AlignCenter))
            self.table.setItem(i, 3, _item(r["file_md5"]))
            self.table.setItem(i, 4, _item(r["added_at"] or ""))

    def import_charts(self):
        files, _ = QFileDialog.getOpenFileNames(self, "import charts", "", "osu! charts (*.osu *.osz)")
        if not files:
            return
        mods = self.mods.selected()
        if not mods:
            QMessageBox.warning(self, "import", "select at least one mod combination.")
            return
        db = self.state.ensure_comp()
        model = self.state.model()

        def done(summary):
            self.main.refresh_all()
            lines = [f"{summary.added} chart(s) imported."]
            for r in summary.results:
                if r.status != "added":
                    lines.append(f"{r.source}: {r.message}")
            QMessageBox.information(self, "import", "\n".join(lines[:40]) + ("\n…" if len(lines) > 40 else ""))

        run_task(self, "importing charts",
                 lambda p, c: import_files(files, db, model, mods, progress=p, cancelled=c), done)

    def _selected_ids(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows]

    def delete_charts(self):
        ids = self._selected_ids()
        if ids and QMessageBox.question(self, "delete charts", f"delete {len(ids)} chart(s) from your database?\n"
                                        "plays on them stay in your list but are ignored.") == QMessageBox.Yes:
            self.state.comp.delete_charts(ids)
            self.main.refresh_all()

    def recompute(self):
        def done(res):
            ok, failed = res
            self.main.refresh_all()
            QMessageBox.information(self, "recompute", f"{ok} chart(s) recomputed, {failed} failed.")

        run_task(self, "recomputing your charts",
                 lambda p, c: recompute_all(self.state.comp, self.state.model(), progress=p, cancelled=c), done)

    def export_db(self):
        path, _ = QFileDialog.getSaveFileName(self, "export chart database", "my_taiko_charts.sqlite",
                                              "chart database (*.sqlite)")
        if path:
            self.state.comp.commit()
            shutil.copyfile(self.state.comp.path, path)
            QMessageBox.information(self, "export", f"saved {Path(path).name}")

    def merge_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "import chart database", "", "chart database (*.sqlite)")
        if not path:
            return
        db = self.state.ensure_comp()

        def work(p, c):
            return db.merge_from(path, progress=lambda f: p(f, ""))

        def done(res):
            self.main.refresh_all()
            QMessageBox.information(self, "import database", f"{res[0]} chart(s) added, {res[1]} already present.")

        def failed(msg):
            QMessageBox.warning(self, "import database", f"can't import that database: {msg}")

        run_task(self, "importing chart database", work, done, failed, cancellable=False)


# =============================================================================== downloads & updates
class UpdatesTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state
        self.remote = None
        lay = QVBoxLayout(self)
        self.installed = QLabel("")
        self.installed.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.installed)
        row = QHBoxLayout()
        self.check_btn = QPushButton("check for updates")
        row.addWidget(self.check_btn)
        row.addStretch()
        lay.addLayout(row)
        self.remote_label = QLabel("")
        self.remote_label.setWordWrap(True)
        lay.addWidget(self.remote_label)
        row2 = QHBoxLayout()
        self.base_btn = QPushButton("download base database")
        self.exp_btn = QPushButton("download expansion pack")
        self.app_btn = QPushButton("open the new app version's page")
        for b in (self.base_btn, self.exp_btn, self.app_btn):
            b.setEnabled(False)
            row2.addWidget(b)
        row2.addStretch()
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        self.file_base_btn = QPushButton("install base database from file…")
        self.file_exp_btn = QPushButton("install expansion pack from file…")
        row3.addWidget(self.file_base_btn)
        row3.addWidget(self.file_exp_btn)
        row3.addStretch()
        lay.addLayout(row3)
        lay.addStretch()
        self.check_btn.clicked.connect(self.check)
        self.base_btn.clicked.connect(lambda: self.download("base"))
        self.exp_btn.clicked.connect(lambda: self.download("expansion"))
        self.app_btn.clicked.connect(lambda: packages.open_url(self.remote["app"]["url"]))
        self.file_base_btn.clicked.connect(lambda: self.install_file("base"))
        self.file_exp_btn.clicked.connect(lambda: self.install_file("expansion"))

    def refresh(self):
        s = self.state
        cfg = packages.app_config()
        b, e = s.base_manifest, s.exp_manifest
        lines = [f"app version: {cfg.get('app_version', '?')}",
                 f"base database: {b['version'] + ' (model ' + b['model_id'] + ')' if b else 'not installed'}",
                 f"expansion pack: {e['version'] + ' (model ' + e['model_id'] + ')' if e else 'not installed'}"]
        if s.comp is not None:
            lines.append(f"your imported charts: {s.comp.count()} (model {s.comp.model_id or '?'}"
                         f"{'' if s.comp_valid or not s.has_base else ', needs recomputing'})")
        lines.append(f"data folder: {packages.data_dir()}")
        if packages._token():
            lines.append("developer mode: downloading through the GitHub API with OTR_GITHUB_TOKEN")
        lines.append("")
        lines.append(packages.ATTRIBUTION)
        self.installed.setText("\n".join(lines))

    def check(self):
        def done(remote):
            self.remote = remote
            s = self.state
            parts = []
            app_v = remote.get("app", {}).get("version")
            cur_v = packages.app_config().get("app_version", "0")
            newer_app = app_v and packages.version_tuple(app_v) > packages.version_tuple(cur_v)
            parts.append(f"app: latest {app_v}" + (" (newer than yours)" if newer_app else " (you're up to date)"))
            for kind, label in (("base", "base database"), ("expansion", "expansion pack")):
                ent = remote.get(kind)
                mine = s.base_manifest if kind == "base" else s.exp_manifest
                if not ent:
                    parts.append(f"{label}: not available")
                    continue
                state = "not installed" if not mine else (
                    "up to date" if mine["version"] == ent["version"] else f"you have {mine['version']}")
                parts.append(f"{label}: {ent['version']}, {ent['size'] / 2**20:,.0f} MB ({state})")
            self.remote_label.setText("\n".join(parts))
            self.base_btn.setEnabled("base" in remote)
            self.exp_btn.setEnabled("expansion" in remote)
            self.app_btn.setEnabled(bool(newer_app))

        run_task(self, "checking for updates", lambda p, c: packages.fetch_versions(), done, cancellable=False)

    def _confirm_base(self, new_model_id: str | None) -> bool:
        s = self.state
        if s.comp is None or s.comp.count() == 0:
            return True
        changes = new_model_id is None or new_model_id != (s.comp.model_id or "")
        if not changes:
            return True
        return QMessageBox.question(
            self, "new base database",
            "this base database comes from a different model version, so your imported charts will have to be "
            "imported again (or recomputed with the matching expansion pack). continue?") == QMessageBox.Yes

    def download(self, kind):
        ent = self.remote[kind]
        if kind == "base" and not self._confirm_base(ent.get("model_id")):
            return
        if kind == "expansion" and self.state.model_id and ent.get("model_id") != self.state.model_id:
            if QMessageBox.question(self, "expansion pack", "this expansion pack doesn't match your installed base "
                                    "database (different model). download it anyway?") != QMessageBox.Yes:
                return

        def done(_m):
            self.state.reload()
            self.main.refresh_all()
            self.main.after_package_change()

        run_task(self, f"downloading the {'base database' if kind == 'base' else 'expansion pack'}",
                 lambda p, c: packages.download_package(ent, kind, progress=p, cancelled=c), done)

    def install_file(self, kind):
        path, _ = QFileDialog.getOpenFileName(self, "install package", "", "package (*.zip)")
        if not path:
            return
        if kind == "base" and not self._confirm_base(None):
            return

        def done(_m):
            self.state.reload()
            self.main.refresh_all()
            self.main.after_package_change()

        run_task(self, "installing", lambda p, c: packages.install_zip(path, kind, progress=lambda f: p(f, "")), done,
                 cancellable=False)


__all__ = ["RecommendTab", "PlaysTab", "ChartsTab", "UpdatesTab", "COMP_FILE", "ChartDB", "IncompatibleDatabase"]
