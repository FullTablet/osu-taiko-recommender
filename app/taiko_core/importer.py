"""Import .osu / .osz files into the complementary chart database (expansion pack).

.osu -> BO values (fixed parser) -> congregal note features (intended strainer
formula) -> normalized with the expansion pack's mean/std -> NumPy note subnet
-> per-note slope/bias per mod combination -> ChartDB.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chartdb import ChartDB
from .osu.congregal import chart_parts
from .osu.osu_reader import parse_osu_bytes
from .osu.taiko_bo import OsuParseError, compute_bo_all, get_mode, parse_taiko_chart

PIPELINE_VERSION = "taiko-congregal-1"      # fixed BO parser + intended strainers, congregal column order
MSG_CONVERT = "autoconverts are not implemented yet, sorry!"
MSG_NOT_TAIKO = "this is not a taiko chart"
MSG_NO_TAIKO_IN_OSZ = "no taiko charts in this file"


@dataclass
class ImportResult:
    source: str                     # file path (and entry name for .osz)
    status: str                     # 'added', 'duplicate', 'skipped', 'error'
    message: str = ""
    chart_id: int | None = None


@dataclass
class ImportSummary:
    results: list[ImportResult] = field(default_factory=list)

    @property
    def added(self) -> int:
        return sum(r.status == "added" for r in self.results)


def chart_info(chart, osu) -> dict:
    """Metadata row for an imported chart."""
    md = chart.metadata
    diff = osu.key_values("Difficulty")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    times = chart.circle_times
    length = float((times.max() - times.min()) / 1000.0) if len(times) > 1 else 0.0
    bpm = None
    unin = [(t, bl) for t, bl in chart.timing_points if bl > 0]
    if unin:
        unin.sort()
        end = float(times.max()) if len(times) else unin[-1][0]
        spans = {}
        for i, (t, bl) in enumerate(unin):
            t_next = unin[i + 1][0] if i + 1 < len(unin) else max(end, t)
            spans[round(60000.0 / bl, 3)] = spans.get(round(60000.0 / bl, 3), 0.0) + max(t_next - t, 0.0)
        bpm = max(spans, key=spans.get)                       # BPM covering the most time
    bid = chart.beatmap_id_in_file

    def to_int(v):
        try:
            return int(float(v)) if v not in (None, "") and float(v) > 0 else None
        except ValueError:
            return None

    return dict(beatmap_id=bid, beatmapset_id=to_int(md.get("BeatmapSetID")), file_md5=chart.file_md5,
                artist=md.get("Artist"), title=md.get("Title"), artist_unicode=md.get("ArtistUnicode"),
                title_unicode=md.get("TitleUnicode"), creator=md.get("Creator"), version=md.get("Version"),
                source=md.get("Source"), tags=md.get("Tags"), status=None, length_s=length, bpm=bpm,
                od=num(diff.get("OverallDifficulty")), hp=num(diff.get("HPDrainRate")), n_notes=int(len(times)),
                file_name=chart.file_name)


class ExpansionModel:
    """note_net.npz of the expansion pack: weights + normalization + ids."""

    def __init__(self, path):
        from .notenet import NoteNet

        self.net = NoteNet(path)
        cfg = self.net.cfg
        self.mean = np.asarray(cfg["feature_mean"], np.float64)
        self.std = np.asarray(cfg["feature_std"], np.float64)
        self.model_id = cfg["model_id"]
        self.pipeline_version = cfg.get("pipeline_version", PIPELINE_VERSION)

    def compute_variants(self, osu_bytes: bytes, file_name: str, mods_list, batch: int = 6):
        """-> (chart, osu, {mods: (slope, bias)}). Raises NotTaikoError etc."""
        osu = parse_osu_bytes(osu_bytes, file_name)
        chart = parse_taiko_chart(osu)
        mods_list = sorted(set(int(m) for m in mods_list))
        bo = compute_bo_all(chart, legacy=False, mods_list=mods_list)
        parts = chart_parts(bo)
        feats = {m: ((parts.features(m).astype(np.float64) - self.mean) / self.std).astype(np.float32)
                 for m in mods_list}
        out = {}
        for i in range(0, len(mods_list), batch):
            group = mods_list[i:i + batch]
            slope, bias = self.net.forward(np.stack([feats[m] for m in group]))
            for j, m in enumerate(group):
                out[m] = (slope[j], bias[j])
        return chart, osu, out


def _iter_osu_payloads(path: Path):
    """Yield (display name, bytes, file name) for a .osu or each .osu inside a .osz."""
    if path.suffix.lower() == ".osz":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".osu") and not info.is_dir():
                    yield f"{path.name} / {info.filename}", z.read(info), Path(info.filename).name
    else:
        yield path.name, path.read_bytes(), path.name


def import_files(paths, db: ChartDB, model: ExpansionModel, mods_list, progress=None,
                 cancelled=lambda: False) -> ImportSummary:
    """Import .osu/.osz files. progress(fraction, text) is called as files are processed."""
    summary = ImportSummary()
    have = db.md5_set()
    paths = [Path(p) for p in paths]
    for fi, path in enumerate(paths):
        if cancelled():
            break
        entries = []
        try:
            entries = list(_iter_osu_payloads(path))
        except (zipfile.BadZipFile, OSError) as e:
            summary.results.append(ImportResult(str(path), "error", f"can't read file: {e}"))
            continue
        modes, taiko_found = [], False
        file_results = []
        for ei, (name, raw, fname) in enumerate(entries):
            if cancelled():
                break
            if progress:
                progress((fi + ei / max(len(entries), 1)) / len(paths), name)
            try:
                osu = parse_osu_bytes(raw, fname)
                mode = get_mode(osu)
                modes.append(mode)
                if mode != 1:
                    continue
                taiko_found = True
                if osu.file_md5 in have:
                    file_results.append(ImportResult(name, "duplicate", "already in your chart database"))
                    continue
                chart, osu, variants = model.compute_variants(raw, fname, mods_list)
                if chart.circle_times.size == 0:
                    file_results.append(ImportResult(name, "skipped", "the chart has no notes"))
                    continue
                cid = db.add_chart(chart_info(chart, osu), variants, raw)
                have.add(chart.file_md5)
                file_results.append(ImportResult(name, "added", f"{len(variants)} mod combinations", cid))
            except (OsuParseError, ValueError, IndexError) as e:
                file_results.append(ImportResult(name, "error", f"can't read chart: {e}"))
        if not taiko_found and entries:
            msg = MSG_CONVERT if 0 in modes else (MSG_NOT_TAIKO if path.suffix.lower() == ".osu" else MSG_NO_TAIKO_IN_OSZ)
            summary.results.append(ImportResult(str(path.name), "skipped", msg))
        elif not entries:
            summary.results.append(ImportResult(str(path.name), "skipped", "no .osu files inside"))
        summary.results.extend(file_results)
    if progress:
        progress(1.0, "done")
    return summary


def recompute_all(db: ChartDB, model: ExpansionModel, progress=None, cancelled=lambda: False) -> tuple[int, int]:
    """Recompute every imported chart from its stored .osu with the installed model
    (after a model change). Keeps each chart's mod selection."""
    ids = [r[0] for r in db.conn.execute("SELECT chart_id FROM charts ORDER BY chart_id")]
    ok = failed = 0
    for i, cid in enumerate(ids):
        if cancelled():
            break
        raw = db.osu_bytes(cid)
        mods = db.mods_of(cid)
        info = db.chart(cid)
        try:
            if raw is None:
                raise ValueError("no stored .osu")
            _, _, variants = model.compute_variants(raw, info.get("file_name") or "chart.osu", mods)
            db.conn.execute("DELETE FROM variants WHERE chart_id = ?", (cid,))
            db.put_variants(cid, variants, commit=False)
            ok += 1
        except Exception:                                                  # noqa: BLE001 - report and continue
            failed += 1
        if progress:
            progress((i + 1) / max(len(ids), 1), info.get("title") or "")
    db.set_meta(model_id=model.model_id, pipeline_version=model.pipeline_version)
    db.commit()
    return ok, failed
