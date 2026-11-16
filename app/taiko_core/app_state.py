"""Everything the GUI needs, without any GUI code (so it can be tested headless)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import packages
from . import recommend as R
from .chartdb import ChartDB, IncompatibleDatabase
from .mods import mods_label
from .quantiles import QuantileTable
from .skill import PlayData, estimate_skill, expected_accuracy, quantiles_for_accuracy, rank_to_quantile
from .userdata import UserData

COMP_FILE = "complementary.sqlite"

DEFAULT_REC_SETTINGS = {"K": 1000, "S": 10, "priority": None, "target": 0.97, "mods": 0}


def validate_rec_settings(settings: dict) -> dict:
    from .mods import MOD_COMBINATIONS

    d = dict(DEFAULT_REC_SETTINGS)
    d.update(settings or {})
    k, s = int(d["K"]), int(d["S"])
    target = float(d["target"])
    pr = d.get("priority")
    mods = int(d["mods"])
    if k < 1 or s < 1:
        raise ValueError("K and S must be at least 1")
    if not 0.0 < target < 1.0:
        raise ValueError("the target accuracy must be above 0% and below 100%")
    if pr is not None and pr not in range(5):
        raise ValueError("unknown skill priority")
    if mods not in MOD_COMBINATIONS:
        raise ValueError("invalid mod combination")
    return {"K": k, "S": s, "priority": None if pr is None else int(pr), "target": target, "mods": mods}


@dataclass(frozen=True)
class ChartRef:
    source: str          # 'base' or 'comp'
    chart_id: int


def chart_label(info: dict) -> str:
    return f"{info.get('artist') or '?'} - {info.get('title') or '?'} [{info.get('version') or '?'}]"


class AppState:
    def __init__(self):
        self.user = UserData(packages.user_dir())
        self.base: ChartDB | None = None
        self.quantiles: QuantileTable | None = None
        self.base_manifest: dict | None = None
        self.exp_manifest: dict | None = None
        self.comp: ChartDB | None = None
        self._model = None
        self.reload()

    # ------------------------------------------------------------------ packages
    def reload(self):
        for db in (self.base, self.comp):
            if db is not None:
                db.close()
        self.base = self.comp = self.quantiles = None
        self._model = None
        self.base_manifest = packages.read_manifest("base")
        self.exp_manifest = packages.read_manifest("expansion")
        if self.base_manifest:
            d = packages.package_dir("base")
            try:
                self.base = ChartDB(d / "charts.sqlite", readonly=True)
                self.quantiles = QuantileTable.from_json((d / "skill_quantiles.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, IncompatibleDatabase):
                self.base = self.quantiles = None
                self.base_manifest = None
        comp_path = packages.user_dir() / COMP_FILE
        if comp_path.exists():
            try:
                self.comp = ChartDB(comp_path)
            except IncompatibleDatabase:
                self.comp = None

    @property
    def has_base(self) -> bool:
        return self.base is not None and self.quantiles is not None

    @property
    def model_id(self) -> str:
        return (self.base_manifest or {}).get("model_id", "")

    @property
    def has_expansion(self) -> bool:
        return self.exp_manifest is not None and (not self.model_id or self.exp_manifest.get("model_id") == self.model_id)

    @property
    def expansion_mismatch(self) -> bool:
        return self.exp_manifest is not None and bool(self.model_id) and self.exp_manifest.get("model_id") != self.model_id

    @property
    def comp_valid(self) -> bool:
        """Imported charts are usable only if computed with the installed model."""
        return self.comp is not None and self.comp.model_id == self.model_id and bool(self.model_id)

    def model(self):
        if self._model is None:
            from .importer import ExpansionModel

            self._model = ExpansionModel(packages.package_dir("expansion") / "note_net.npz")
        return self._model

    def ensure_comp(self) -> ChartDB:
        if self.comp is None:
            mid = self.model_id or (self.exp_manifest or {}).get("model_id", "")
            self.comp = ChartDB(packages.user_dir() / COMP_FILE,
                                create_meta={"kind": "complementary", "model_id": mid, "bits": 16})
        return self.comp

    # ------------------------------------------------------------------ charts
    def db(self, source: str) -> ChartDB | None:
        return self.base if source == "base" else (self.comp if self.comp_valid else None)

    def chart_info(self, ref: ChartRef) -> dict | None:
        db = self.db(ref.source)
        return db.chart(ref.chart_id) if db else None

    def variant(self, ref: ChartRef, mods: int):
        db = self.db(ref.source)
        return db.variant(ref.chart_id, mods) if db else None

    def resolve_play(self, play: dict) -> ChartRef | None:
        if play["source"] == "base" and self.base is not None:
            ids = self.base.find(beatmap_id=play["beatmap_id"])
        elif play["source"] == "comp" and self.comp_valid:
            ids = self.comp.find(file_md5=play["file_md5"])
        else:
            ids = []
        return ChartRef(play["source"], ids[0]) if ids else None

    def search(self, text: str, limit: int = 300) -> list[tuple[ChartRef, dict]]:
        """Base charts are searched by beatmap ID (or words); imported charts by file hash (or words)."""
        out = []
        if self.base is not None:
            out += [(ChartRef("base", r["chart_id"]), r) for r in self.base.search(text, limit)]
        if self.comp_valid:
            out += [(ChartRef("comp", r["chart_id"]), r) for r in self.comp.search(text, limit)]
        return out

    # ------------------------------------------------------------------ skill
    def current_skill(self) -> dict | None:
        cur = self.user.current_skill()
        if cur is None or not self.has_base:
            return cur
        cur["quantiles"] = self.quantiles.quantiles_of(cur["skill"])        # always w.r.t. the installed table
        return cur

    def set_skill_from_rank(self, rank: int) -> np.ndarray:
        q = rank_to_quantile(rank)
        skill = self.quantiles.values_at(q)
        self.user.save_skill(skill, self.quantiles.quantiles_of(skill), 0, "rank", self.model_id)
        self.user.set("pp_rank", int(rank))
        return skill

    def play_data(self, max_plays: int = 1000, progress=None) -> tuple[list[PlayData], int]:
        """Recent plays (oldest first) with their per-note values; returns (data, n_unusable)."""
        plays = self.user.recent_plays(max_plays)
        data, missing = [], 0
        for i, p in enumerate(plays):
            ref = self.resolve_play(p)
            v = self.variant(ref, p["mods"]) if ref else None
            if v is None:
                missing += 1
            else:
                data.append(PlayData(v[0], v[1], p["accuracy"]))
            if progress and (i % 20 == 0 or i + 1 == len(plays)):
                progress(0.8 * (i + 1) / max(len(plays), 1), f"loading plays {i + 1}/{len(plays)}")
        return data, missing

    def update_skill(self, progress=None) -> dict:
        cur = self.user.current_skill()
        if cur is None:
            raise RuntimeError("no starting skill: enter your pp rank first")
        data, missing = self.play_data(progress=progress)
        if progress:
            progress(0.85, "estimating skill")
        skill, info = estimate_skill(data, prior_mean=cur["skill"],
                                     progress=(lambda f: progress(0.85 + 0.15 * f, "estimating skill")) if progress else None)
        q = self.quantiles.quantiles_of(skill)
        self.user.save_skill(skill, q, len(data), "plays", self.model_id)
        info.update(missing=missing, skill=skill, quantiles=q)
        return info

    def expected_accuracy(self, ref: ChartRef, mods: int, skill=None) -> float | None:
        v = self.variant(ref, mods)
        if v is None:
            return None
        s = self.current_skill()["skill"] if skill is None else skill
        return expected_accuracy(v[0], v[1], s)

    def quantiles_needed(self, ref: ChartRef, mods: int, direction, target: float):
        v = self.variant(ref, mods)
        if v is None:
            return None
        return quantiles_for_accuracy(v[0], v[1], self.quantiles, direction, target)

    # ------------------------------------------------------------------ recommendations
    def recommendation_settings(self) -> dict:
        d = dict(DEFAULT_REC_SETTINGS)
        d.update(self.user.get("recommendation", {}) or {})
        return d

    def save_recommendation_settings(self, settings: dict):
        clean = validate_rec_settings(settings)
        self.user.set("recommendation", clean)
        return clean

    def count_variants(self, mods: int) -> int:
        total = 0
        for _, db in self._pools():
            total += db.conn.execute("SELECT COUNT(*) FROM variants WHERE mods = ?", (int(mods),)).fetchone()[0]
        return total

    def _pools(self):
        pools = []
        if self.base is not None:
            pools.append(("base", self.base))
        if self.comp_valid:
            pools.append(("comp", self.comp))
        return pools

    def random_variants(self, k: int, mods: int, rng=random) -> list[ChartRef]:
        """k random charts (without repetition) that have the given mod combination,
        drawn uniformly from the base and imported charts together."""
        pools = self._pools()
        counts = [db.conn.execute("SELECT COUNT(*) FROM variants WHERE mods = ?", (int(mods),)).fetchone()[0]
                  for _, db in pools]
        total = sum(counts)
        picks = rng.sample(range(total), min(k, total)) if total else []
        out, start = [], 0
        for (src, db), c in zip(pools, counts):
            kk = sum(1 for i in picks if start <= i < start + c)
            start += c
            if kk:
                rows = db.conn.execute("SELECT chart_id FROM variants WHERE mods = ? ORDER BY RANDOM() LIMIT ?",
                                       (int(mods), kk)).fetchall()
                out += [ChartRef(src, r[0]) for r in rows]
        rng.shuffle(out)
        return out

    def recommend(self, settings: dict | None = None, progress=None, cancelled=lambda: False) -> dict:
        """Draw K random charts with the selected mods, compute each fit error, return the S
        lowest (see recommend.py), with the priority of every skill dimension for those S."""
        st = validate_rec_settings(settings or self.recommendation_settings())
        target, mods, pdim = st["target"], st["mods"], st["priority"]
        skill = self.current_skill()["skill"]
        cands = self.random_variants(st["K"], mods)
        scored = []
        for i, ref in enumerate(cands):
            if cancelled():
                break
            v = self.variant(ref, mods)
            if v is not None and len(v[0]):
                cm = R.ChartModel(v[0], v[1], self.quantiles)
                ev = R.evaluate(cm, skill, target, pdim)
                scored.append((ev["fiterror"], ref, cm, ev))
            if progress and (i % 5 == 0 or i + 1 == len(cands)):
                progress(0.9 * (i + 1) / max(len(cands), 1), f"checking chart {i + 1:,}/{len(cands):,}")
        scored.sort(key=lambda t: t[0])
        chosen = scored[:st["S"]]
        out = []
        for j, (fit, ref, cm, ev) in enumerate(chosen):
            pri = R.all_priorities(cm, target, ev["priorities"], ev["x_c"])
            info = self.chart_info(ref)
            out.append(dict(ref=ref, mods=mods, expected=ev["ha"], hapower=ev["hapower"], fiterror=fit,
                            in_window=ev["in_window"], priorities=pri, info=info, label=chart_label(info),
                            mods_label=mods_label(mods)))
            if progress:
                progress(0.9 + 0.1 * (j + 1) / max(len(chosen), 1), "computing skill priorities")
        return dict(charts=out, candidates=len(scored), in_window=sum(1 for s in scored if s[3]["in_window"]),
                    settings=st, available=self.count_variants(mods))

    def record_plays(self, rows: list[tuple[ChartRef, int, float]]):
        """rows in play order (the last becomes the most recent); accuracy 0..1."""
        for ref, mods, acc in rows:
            info = self.chart_info(ref)
            if ref.source == "base":
                self.user.add_play("base", mods, acc, beatmap_id=info["beatmap_id"], label=chart_label(info))
            else:
                self.user.add_play("comp", mods, acc, file_md5=info["file_md5"], label=chart_label(info))

    def close(self):
        for db in (self.base, self.comp):
            if db is not None:
                db.close()
        self.user.close()


def default_rank_prompt_needed(state: AppState) -> bool:
    return state.has_base and state.user.current_skill() is None


def file_size_text(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 2**20:,.1f} MB"
    except OSError:
        return "-"
