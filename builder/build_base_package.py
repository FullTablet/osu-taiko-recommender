r"""Build the base package: ranked/approved taiko charts with per-note difficulty
values for the common mod combinations, and the skill quantile table.

    python build_base_package.py ^
        --checkpoint <dataset>\checkpoint_last.pt --dataset_dir <dataset> ^
        --congregal_db <congregals.sqlite> --plays_db <taiko_plays.sqlite> ^
        --dump_dir <data.ppy.sh dump folder> --osu_dir <data.ppy.sh osu_files folder> ^
        --version 2026.11.1 --out ..\dist\base

Charts: every beatmap with playmode 1 and approved 1 or 2 (taiko_plays.sqlite's
beatmaps.eligible) that has congregal values. Metadata comes from the dump's
osu_beatmaps.sql / osu_beatmapsets.sql; the length (first to last note, breaks
included) is read from the chart's .osu file in --osu_dir. Per-note slope/bias come from the
checkpoint's note subnet (PyTorch, GPU if available) for NM, HD, HR, DT, HDHR,
HDDT, HT, EZ, stored quantized (--bits).

Skill quantiles: for every play in taiko_plays.sqlite whose player has a skill
row, the player's skill at the time of the play (per-player time axis from
player.db, interpolated between keyframes exactly like the model) -> 1000
keypoints per dimension, tail slopes over the outer 1%.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

import build_common as C

FILES = ["charts.sqlite", "skill_quantiles.json"]


# --------------------------------------------------------------------------- metadata
def load_metadata(dump_dir: Path, wanted: set[int]) -> dict[int, dict]:
    import plays_db as P

    def rows(path, table, cols_wanted):
        cols = P.read_columns(path, table)
        idx = {c: cols.index(c) for c in cols_wanted}
        rx = P.row_regex(len(cols))
        for payload, _ in P.iter_insert_payloads(path, table):
            for g in P.iter_rows(payload, rx):
                yield {c: g[i] for c, i in idx.items()}

    maps = {}
    for r in rows(dump_dir / "osu_beatmaps.sql", "osu_beatmaps",
                  ["beatmap_id", "beatmapset_id", "version", "diff_drain",
                   "diff_overall", "bpm", "approved", "checksum"]):
        bid = int(r["beatmap_id"])
        if bid in wanted:
            maps[bid] = dict(beatmap_id=bid, beatmapset_id=P.sql_int(r["beatmapset_id"]),
                             version=P.sql_str(r["version"]),
                             hp=float(r["diff_drain"]), od=float(r["diff_overall"]), bpm=float(r["bpm"]),
                             status=int(r["approved"]), file_md5=P.sql_str(r["checksum"]))
    sets = {m["beatmapset_id"] for m in maps.values()}
    set_info = {}
    for r in rows(dump_dir / "osu_beatmapsets.sql", "osu_beatmapsets",
                  ["beatmapset_id", "artist", "artist_unicode", "title", "title_unicode", "creator", "source", "tags"]):
        sid = int(r["beatmapset_id"])
        if sid in sets:
            set_info[sid] = {k: P.sql_str(r[k]) for k in
                             ("artist", "artist_unicode", "title", "title_unicode", "creator", "source", "tags")}
    for m in maps.values():
        m.update(set_info.get(m["beatmapset_id"], {}))
    return maps


# --------------------------------------------------------------------------- lengths
class OsuFiles:
    """Finds a beatmap's .osu file: <osu_dir>/<beatmap_id>.osu (data.ppy.sh's osu_files
    layout), else any file with that name below osu_dir."""

    def __init__(self, osu_dir: Path | None):
        self.dir = Path(osu_dir) if osu_dir else None
        self._index: dict[str, Path] | None = None

    def path(self, bid: int) -> Path | None:
        if self.dir is None:
            return None
        p = self.dir / f"{bid}.osu"
        if p.is_file():
            return p
        if self._index is None:
            self._index = {f.name.lower(): f for f in self.dir.rglob("*.osu")}
        return self._index.get(f"{bid}.osu")


def note_span(path: Path) -> tuple[float, str] | None:
    """(seconds from first to last note, file md5) with the app's own .osu parser."""
    from taiko_core.importer import note_span_seconds
    from taiko_core.osu.osu_reader import read_osu_file
    from taiko_core.osu.taiko_bo import parse_taiko_chart

    try:
        chart = parse_taiko_chart(read_osu_file(path))
    except Exception:                                                       # noqa: BLE001
        return None
    return note_span_seconds(chart.circle_times), chart.file_md5


# --------------------------------------------------------------------------- skill per play
def play_skills(plays_db: Path, dataset_dir: Path, table: np.ndarray, log) -> np.ndarray:
    roster = {int(k): v for k, v in json.loads((dataset_dir / "player_roster.json").read_text())["mapping"].items()}
    pconn = sqlite3.connect(dataset_dir / "player.db")
    span = {u: (f, s) for u, f, s in pconn.execute("SELECT user_id, first_play_unix, span_s FROM players")}
    pconn.close()
    conn = sqlite3.connect(f"file:{plays_db.as_posix()}?mode=ro", uri=True)
    n = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
    users = np.empty(n, np.int64)
    ended = np.empty(n, np.int64)
    cur = conn.execute("SELECT user_id, ended_at FROM plays")
    i = 0
    while True:
        rows = cur.fetchmany(500_000)
        if not rows:
            break
        a = np.array(rows, np.int64)
        users[i:i + len(a)], ended[i:i + len(a)] = a[:, 0], a[:, 1]
        i += len(a)
    conn.close()
    uu, inv = np.unique(users, return_inverse=True)
    idx = np.array([roster.get(int(u), -1) for u in uu])
    first = np.array([span.get(int(u), (0, 0))[0] for u in uu], np.float64)
    sp = np.array([span.get(int(u), (0, 0))[1] for u in uu], np.float64)
    ok_user = (idx >= 0) & np.array([int(u) in span for u in uu])
    keep = ok_user[inv]
    log(f"plays: {n:,}; with a skill row: {keep.sum():,} ({(~ok_user).sum():,} players without one skipped)")
    pi = inv[keep]
    t = (ended[keep] - first[pi]) / np.where(sp[pi] > 0, sp[pi], 1.0)
    t = np.where(sp[pi] > 0, t, 0.0).clip(0.0, 1.0)                     # the model clamps to [0, 1] too
    k = table.shape[1]
    pos = t * (k - 1)
    f0 = np.floor(pos).astype(np.int64)
    f1 = np.minimum(f0 + 1, k - 1)
    frac = (pos - f0)[:, None]
    rows = idx[pi]
    return (1.0 - frac) * table[rows, f0] + frac * table[rows, f1]


# --------------------------------------------------------------------------- main
def build(args):
    import torch

    from build_training_data import CongregalReader, MODS_RADIX
    from taiko_core.chartdb import ChartDB
    from taiko_core.mods import BASE_MODS
    from taiko_core.quantiles import QuantileTable
    from taiko_core.skill import expected_accuracy

    t0 = time.time()
    report = []

    def log(msg):
        print(msg, flush=True)
        report.append(msg)

    dataset = Path(args.dataset_dir)
    meta = C.load_dataset_meta(dataset)
    mean = np.asarray(meta["object_feature_mean"], np.float64)
    std = np.asarray(meta["object_feature_std"], np.float64)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = C.load_checkpoint_model(args.checkpoint, device)
    model_id = C.compute_model_id(ckpt, mean, std)
    out = C.prepare_out(args.out, args.overwrite, FILES)
    log(f"model id {model_id}, device {device}")

    # ---- charts ----------------------------------------------------------------------------
    conn = sqlite3.connect(f"file:{Path(args.plays_db).as_posix()}?mode=ro", uri=True)
    eligible = {r[0] for r in conn.execute("SELECT beatmap_id FROM beatmaps WHERE eligible = 1")}
    conn.close()
    reader = CongregalReader(args.congregal_db)
    cindex = reader.index()
    by_bid = {}
    for key, (md5, n, legacy) in cindex.items():
        bid, mods = key // MODS_RADIX, key % MODS_RADIX
        if bid in eligible and n > 0:
            by_bid.setdefault(bid, {})[mods] = md5
    bids = sorted(b for b, m in by_bid.items() if all(x in m for x in BASE_MODS))
    log(f"eligible beatmaps: {len(eligible):,}; with congregal values for all base mods: {len(bids):,} "
        f"({len(eligible) - len(bids):,} missing)")
    if args.limit:
        bids = bids[:args.limit]
    metadata = load_metadata(Path(args.dump_dir), set(bids))
    log(f"metadata found for {len(metadata):,} beatmaps")
    osu_files = OsuFiles(args.osu_dir)
    n_len = n_len_missing = n_len_other_md5 = 0

    db = ChartDB(out / "charts.sqlite", create_meta={"kind": "base", "model_id": model_id, "version": args.version,
                                                     "bits": args.bits, "pipeline_version": C.PIPELINE_VERSION})
    rng = np.random.default_rng(0)
    err_samples = []
    last = 0.0
    with torch.no_grad():
        for i, bid in enumerate(bids):
            md5 = by_bid[bid][0]
            feats = reader.chart_features(md5, BASE_MODS, "congregal")
            x = np.stack([((feats[m].astype(np.float64) - mean) / std).astype(np.float32) for m in BASE_MODS])
            o = model.object_subnet(torch.from_numpy(x).to(device))
            slope, bias = o["slope"].float().cpu().numpy(), o["bias"].float().cpu().numpy()
            variants = {m: (slope[j], bias[j]) for j, m in enumerate(BASE_MODS)}
            info = dict(metadata.get(bid, {"beatmap_id": bid}))
            info.setdefault("file_md5", md5)
            p = osu_files.path(bid)
            span = note_span(p) if p is not None else None
            if span is None:
                info["length_s"] = None
                n_len_missing += 1
            else:
                info["length_s"] = span[0]
                n_len += 1
                n_len_other_md5 += span[1] != md5
            info["n_notes"] = int(slope.shape[1])
            cid = db.add_chart(info, variants, commit=False)
            if rng.random() < args.error_sample:
                err_samples.append((cid, variants))
            if (i + 1) % 500 == 0:
                db.commit()
            now = time.time()
            if now - last > 10 or i + 1 == len(bids):
                last = now
                print(f"  charts: {i + 1:,}/{len(bids):,} ({now - t0:.0f}s)", flush=True)
    db.commit()
    log(f"lengths (first to last note) from .osu files: {n_len:,}; without a readable .osu file: "
        f"{n_len_missing:,} (shown blank)" + (f"; {n_len_other_md5:,} files differ from the version the "
                                               "congregals were computed from" if n_len_other_md5 else ""))
    if args.osu_dir is None:
        log("WARNING: no --osu_dir given, so no chart has a length")

    # ---- skill quantiles -------------------------------------------------------------------
    table = ckpt["model"]["skill_subnet.table"].float().numpy()
    samples = play_skills(Path(args.plays_db), dataset, table, log)
    qt = QuantileTable.from_samples(samples, keypoints=args.keypoints, tail=args.tail)
    qt.meta.update(model_id=model_id, version=args.version)
    (out / "skill_quantiles.json").write_text(qt.to_json(), encoding="utf-8")
    for d in range(qt.D):
        v = qt.values[d]
        log(f"skill dim {d + 1}: q0 {v[0]:.3f}  q0.5 {v[len(v) // 2]:.3f}  q1 {v[-1]:.3f}  "
            f"tail slopes {qt.slope_low[d]:.3f} / {qt.slope_high[d]:.3f}")

    # ---- quantization error ----------------------------------------------------------------
    if err_samples:
        diffs = []
        for cid, variants in err_samples:
            for m, (s_true, b_true) in variants.items():
                s_q, b_q = db.variant(cid, m)
                for q in (0.1, 0.5, 0.9):
                    sk = qt.values_at(q)
                    diffs.append(abs(expected_accuracy(s_true, b_true, sk) - expected_accuracy(s_q, b_q, sk)))
        diffs = np.array(diffs)
        log(f"{args.bits}-bit storage: expected-accuracy error on {len(err_samples)} charts: "
            f"mean {diffs.mean() * 100:.4f}%, max {diffs.max() * 100:.4f}% (percentage points)")
    db.set_meta(n_charts=db.count())
    db.conn.execute("VACUUM")
    db.close()
    manifest = C.write_manifest(out, "base", args.version, model_id, FILES)
    size = sum(f["size"] for f in manifest["files"].values())
    log(f"done in {time.time() - t0:.0f}s: {len(bids):,} charts, package {size / 2**20:,.0f} MB -> {out}")
    (out / "build_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_dir", required=True, help="dataset folder: meta.json, player_roster.json, player.db")
    ap.add_argument("--congregal_db", required=True)
    ap.add_argument("--plays_db", required=True, help="taiko_plays.sqlite")
    ap.add_argument("--dump_dir", required=True, help="folder with osu_beatmaps.sql and osu_beatmapsets.sql")
    ap.add_argument("--osu_dir", default=None,
                    help="data.ppy.sh osu_files folder (<beatmap_id>.osu), for the chart lengths")
    ap.add_argument("--version", required=True, help="package version, e.g. 2026.11.1")
    ap.add_argument("--out", default=str(C.HERE.parent / "dist" / "base"))
    ap.add_argument("--bits", type=int, choices=[8, 16], default=8, help="per-note value precision")
    ap.add_argument("--keypoints", type=int, default=1000)
    ap.add_argument("--tail", type=float, default=0.01, help="fraction used for the tail slopes")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--error_sample", type=float, default=0.01, help="fraction of charts used for the error report")
    ap.add_argument("--limit", type=int, help="only the first N charts (testing)")
    ap.add_argument("--overwrite", action="store_true")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
