"""BO values -> Congregal values (the "objects" database the network reads).

Pipeline:  .osu -> BO (taiko_bo.py, .npz) -> **Congregal** (this file) -> NN difficulty.

Each Congregal row has 41 values:

    0  BID
    1  Mods
    2  note index (0-based)
    3  tanh(dt / 5000)            dt = BO[1] (delta time, ms)
    4  BO[5]                      vision shift
    5  BO[6]                      scroll rate
    6..35   PreCongregal strainers: for i = 1..15, (don strain_i, kat strain_i)
    36..38  PreCongregal OD part: BO[2], BO[3], BO[4]   (timing windows / 20)
    39..40  PreCongregal StrainImpact: (is_don, is_kat)

Columns 3..40 are the 38 object features `difficulty_nn` expects.

PreCongregal = Mathematica's PreTryBMCalc net as intended (NOTE: the old net as
actually built swapped time and impact in the fold -- see old_net_strainers_multi;
build_training_data.py uses that version for compatibility with the trained net):
    impact_k = (BO[0]==0, BO[0]==1)
    T_k      = arctanh(min(dt_k / 500, 0.99))
    S_i[k]   = impact_k + S_i[k-1] * exp(-T_k / i),   S_i[-1] = 0,  i = 1..15

Speed-ups (results are bit-identical to computing all 38 features per mod):
  * Of the 38 features, tanh(dt) and the 30 strainers depend only on the note
    times, i.e. only on DT/HT -> computed once per distinct dt sequence
    (3 per chart: NM / DT / HT), not 36 times.
  * Timing windows and vision shift are constant per (chart, mods), and the
    impact columns are the note colours: stored as scalars / one byte per note.
  * The recurrence is solved as a blocked closed-form scan, vectorised over
    all speed variants and strainers at once.
  * Files are processed by a pool of worker processes; the main process only
    writes to SQLite, in batched transactions.

Database layout (congregals.sqlite) -- ~10x smaller than storing 38 floats per
note per mod; `CongregalDB.load()` / `.iter_charts()` / `--export_csv` rebuild
the full 38 features:

    congregal_charts  (file_md5 PK, beatmap_id, legacy, n_notes, colors BLOB uint8)
    congregal_speeds  (file_md5, speed_id, features BLOB float32 n x 31
                       = [tanh(dt/5000), 30 strainers])
    congregals        (file_md5, mods, beatmap_id, n_notes, speed_id,
                       w300, w100, wmiss, vision, rates BLOB float32 n)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N_STRAINERS = 15
STRAIN_FACTORS = np.arange(1, N_STRAINERS + 1, dtype=np.float64)  # StrainFactors[n] = n
_INV_FACTORS = 1.0 / STRAIN_FACTORS
N_FEATURES = 38
N_COLUMNS = 41
N_SPEED_FEATURES = 1 + 2 * N_STRAINERS  # tanh(dt) + 30 strainers
_MAX_BLOCK_EXPONENT = 600.0  # exp(600) ~ 1e260 < float64 max


# --------------------------------------------------------------------------- math
def object_time(dt: np.ndarray) -> np.ndarray:
    return np.arctanh(np.minimum(dt / 500.0, 0.99))


def strainers_multi(impact: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fold S_i[k] = impact_k + S_i[k-1]*exp(-t_k/i), i = 1..15, for V time sequences.

    impact: (n, 2), t: (n, V)  ->  (n, V, 15, 2)
    """
    n, v = t.shape
    out = np.empty((n, v, N_STRAINERS, 2), dtype=np.float64)
    if n == 0:
        return out
    tmax = max(float(np.max(np.abs(t))), 1e-12)
    block = max(1, int(_MAX_BLOCK_EXPONENT / tmax))  # keeps exp(c) finite inside a block
    state = np.zeros((v, N_STRAINERS, 2))
    imp = impact[:, None, None, :]
    for s in range(0, n, block):
        e = min(n, s + block)
        c = np.cumsum(t[s:e], axis=0)[:, :, None] * _INV_FACTORS        # (L, V, 15)
        ec = np.exp(c)[..., None]                                       # (L, V, 15, 1)
        acc = np.cumsum(imp[s:e] * ec, axis=0)
        acc += state
        acc /= ec
        out[s:e] = acc
        state = acc[-1]
    return out


def old_net_strainers_multi(impact: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The strainers the old Mathematica net (PreTryBMCalc) actually computed.

    In that NetGraph the fold's FunctionLayer #Impact + #State Exp[-#Time/i]
    received the object time in its Impact slot and the impact in its Time slot,
    so the values the old network was trained on are

        S_i[k] = T_k + S_i[k-1] * exp(-impact_k / i),   S_i[-1] = 0

    (checked against the old dataset: 18 no-mod/DT/HT charts reproduced to
    float32 precision). Same shapes as strainers_multi: impact (n, 2),
    t (n, V) -> (n, V, 15, 2).
    """
    n, v = t.shape
    out = np.empty((n, v, N_STRAINERS, 2), dtype=np.float64)
    if n == 0:
        return out
    logs = -impact[:, None, :] * _INV_FACTORS[None, :, None]           # (n, 15, 2), >= -1 per step
    block = int(_MAX_BLOCK_EXPONENT)                                    # |cumsum| <= block inside a block
    state = np.zeros((v, N_STRAINERS, 2))
    for s in range(0, n, block):
        e = min(n, s + block)
        c = np.cumsum(logs[s:e], axis=0)                                # (L, 15, 2), <= 0
        acc = np.cumsum(t[s:e, :, None, None] * np.exp(-c)[:, None], axis=0)   # (L, V, 15, 2)
        acc += state
        acc *= np.exp(c)[:, None]
        out[s:e] = acc
        state = acc[-1]
    return out


def old_net_speed_features(colors: np.ndarray, tanh_dt: np.ndarray) -> np.ndarray:
    """Rebuild the 31 speed features [tanh(dt/5000), 30 old-net strainers] from the
    stored tanh(dt/5000) column and the note colours. float32 (n, 31)."""
    x = np.asarray(tanh_dt, dtype=np.float64)
    dt = 5000.0 * np.arctanh(np.clip(x, 0.0, 1.0 - 1e-12))              # tanh saturated -> very long dt
    s = old_net_strainers_multi(_impact(colors), object_time(dt)[:, None])[:, 0]
    out = np.empty((len(x), N_SPEED_FEATURES), dtype=np.float32)
    out[:, 0] = x
    out[:, 1:] = s.reshape(len(x), 2 * N_STRAINERS)
    return out


def strainers(impact: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Single time sequence: impact (n, 2), t (n,) -> (n, 15, 2)."""
    return strainers_multi(impact, t[:, None])[:, 0]


def _impact(colors: np.ndarray) -> np.ndarray:
    return np.stack([(colors == 0), (colors == 1)], axis=1).astype(np.float64)


def pre_congregal(bo: np.ndarray) -> np.ndarray:
    """PreTryBMCalc: BO (n, 7) -> (n, 35)."""
    bo = np.asarray(bo, dtype=np.float64)
    impact = _impact(bo[:, 0])
    s = strainers(impact, object_time(bo[:, 1]))
    return np.concatenate([s.reshape(len(bo), 2 * N_STRAINERS), bo[:, 2:5], impact], axis=1)


def congregal_features(bo: np.ndarray) -> np.ndarray:
    """The 38 object features (Congregal columns 3..40) of one BO array, float32 (n, 38).
    Reference/single-chart path; batch processing uses chart_parts()."""
    bo = np.asarray(bo, dtype=np.float64)
    head = np.stack([np.tanh(bo[:, 1] / 5000.0), bo[:, 5], bo[:, 6]], axis=1)
    return np.concatenate([head, pre_congregal(bo)], axis=1).astype(np.float32)


def congregal(bo: np.ndarray, bid: int, mods: int) -> np.ndarray:
    """Full 41-value Congregal rows, float64 (n, 41) (float64 keeps BID exact)."""
    f = congregal_features(bo).astype(np.float64)
    n = len(f)
    ids = np.column_stack([np.full(n, bid), np.full(n, mods), np.arange(n)]).astype(np.float64)
    return np.concatenate([ids, f], axis=1)


# --------------------------------------------------------------------------- compact form
class ChartParts:
    """All mod combinations of one chart, without redundancy.

    colors   uint8 (n,)
    speeds   list of float32 (n, 31) arrays: [tanh(dt/5000), 30 strainers]
    per_mod  {mods: (speed_id, (w300, w100, wmiss), vision, rates float32 (n,))}
    """

    __slots__ = ("colors", "speeds", "per_mod")

    def __init__(self, colors, speeds, per_mod):
        self.colors, self.speeds, self.per_mod = colors, speeds, per_mod

    @property
    def n_notes(self) -> int:
        return len(self.colors)

    def features(self, mods: int) -> np.ndarray:
        """Rebuild the 38 features (n, 38) float32 for one mod combination."""
        sid, windows, vision, rates = self.per_mod[mods]
        return assemble_features(self.colors, self.speeds[sid], windows, vision, rates)


def assemble_features(colors, speed_feats, windows, vision, rates) -> np.ndarray:
    n = len(colors)
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    out[:, 0] = speed_feats[:, 0]
    out[:, 1] = vision
    out[:, 2] = rates
    out[:, 3:33] = speed_feats[:, 1:]
    out[:, 33:36] = np.asarray(windows, dtype=np.float32)
    out[:, 36] = colors == 0
    out[:, 37] = colors == 1
    return out


def chart_parts(bo_all: dict[int, np.ndarray]) -> ChartParts:
    """Compute the congregal values of all mod combinations of one chart."""
    mods_sorted = sorted(bo_all)
    first = np.asarray(bo_all[mods_sorted[0]])
    colors = first[:, 0].astype(np.uint8)
    groups: dict[bytes, int] = {}
    dts: list[np.ndarray] = []
    per_mod = {}
    for m in mods_sorted:
        bo = np.asarray(bo_all[m])
        if len(bo) != len(colors) or not np.array_equal(bo[:, 0], first[:, 0]):
            raise ValueError(f"mods {m}: note count / colours differ between mod combinations")
        if len(bo):
            w, vis = bo[0, 2:5], bo[0, 5]
            if not (np.all(bo[:, 2:5] == w) and np.all(bo[:, 5] == vis)):
                raise ValueError(f"mods {m}: timing windows / vision not constant within the chart")
        else:
            w, vis = np.zeros(3, np.float32), np.float32(0)
        dt = np.ascontiguousarray(bo[:, 1])
        key = dt.tobytes()
        if key not in groups:
            groups[key] = len(dts)
            dts.append(dt.astype(np.float64))
        per_mod[m] = (groups[key], tuple(float(x) for x in w), float(vis),
                      np.ascontiguousarray(bo[:, 6], dtype=np.float32))

    n = len(colors)
    dt_mat = np.stack(dts, axis=1) if n else np.zeros((0, len(dts)))       # (n, V)
    s = strainers_multi(_impact(colors), object_time(dt_mat))               # (n, V, 15, 2)
    speeds = []
    for v in range(len(dts)):
        sf = np.empty((n, N_SPEED_FEATURES), dtype=np.float32)
        sf[:, 0] = np.tanh(dt_mat[:, v] / 5000.0)
        sf[:, 1:] = s[:, v].reshape(n, 2 * N_STRAINERS)
        speeds.append(sf)
    return ChartParts(colors, speeds, per_mod)


def write_objects_csv(fh, bid: int, mods: int, features: np.ndarray) -> None:
    """Append headerless rows: BID, Mods, index, 38 features."""
    n = len(features)
    if n == 0:
        return
    idx = np.column_stack([np.full(n, bid, np.int64), np.full(n, mods, np.int64), np.arange(n)])
    fmt = "%d,%d,%d," + ",".join(["%.9g"] * N_FEATURES)
    np.savetxt(fh, np.column_stack([idx, features.astype(np.float64)]), fmt=fmt)


# --------------------------------------------------------------------------- database
class CongregalDB:
    """SQLite store of congregal values (see module docstring for the layout).

    Query examples:
        SELECT beatmap_id, mods, n_notes FROM congregals WHERE mods & 64 AND n_notes > 1000;
        SELECT COUNT(*), SUM(n_notes) FROM congregal_charts;

    Charts are keyed by file_md5. Writing a chart again replaces it; writing a
    new version of a beatmap (same BID, different file) replaces the old one.
    """

    SCHEMA_VERSION = 2
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS congregal_charts (
        file_md5   TEXT PRIMARY KEY,
        beatmap_id INTEGER,
        legacy     INTEGER NOT NULL,
        n_notes    INTEGER NOT NULL,
        colors     BLOB    NOT NULL              -- uint8 (n_notes,), 0 don / 1 kat
    );
    CREATE INDEX IF NOT EXISTS idx_cc_bid ON congregal_charts(beatmap_id);
    CREATE TABLE IF NOT EXISTS congregal_speeds (
        file_md5   TEXT    NOT NULL,
        speed_id   INTEGER NOT NULL,
        features   BLOB    NOT NULL,             -- float32 LE (n_notes, 31)
        PRIMARY KEY (file_md5, speed_id)
    );
    CREATE TABLE IF NOT EXISTS congregals (
        file_md5   TEXT    NOT NULL,
        mods       INTEGER NOT NULL,
        beatmap_id INTEGER,
        n_notes    INTEGER NOT NULL,
        speed_id   INTEGER NOT NULL,
        w300       REAL    NOT NULL,             -- timing windows / 20
        w100       REAL    NOT NULL,
        wmiss      REAL    NOT NULL,
        vision     REAL    NOT NULL,
        rates      BLOB    NOT NULL,             -- float32 LE (n_notes,)
        PRIMARY KEY (file_md5, mods)
    );
    CREATE INDEX IF NOT EXISTS idx_congregals_bid ON congregals(beatmap_id, mods);
    """

    def __init__(self, path: str, commit_every: int = 200):
        import sqlite3

        self.path = path
        self.conn = sqlite3.connect(path)
        old = self.conn.execute(
            "SELECT 1 FROM pragma_table_info('congregals') WHERE name = 'features'").fetchone()
        if old:
            self.conn.close()
            raise RuntimeError(
                f"{path} was created by an older congregal.py (one 38-feature blob per mod). "
                "Delete it (or pass a different --congregal_db) and re-run; it is fully rebuildable.")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-200000")  # ~200 MB page cache
        self.conn.executescript(self._SCHEMA)
        self.conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        self.conn.commit()
        self.commit_every = commit_every
        self._pending = 0

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ write
    def write_chart(self, bid, file_md5: str, legacy: bool, parts: ChartParts) -> None:
        """Store one chart (all its mod combinations). Commits every `commit_every` charts."""
        c = self.conn
        if bid is not None:
            stale = [r[0] for r in c.execute(
                "SELECT file_md5 FROM congregal_charts WHERE beatmap_id = ? AND file_md5 != ?",
                (bid, file_md5))]
            for md5 in stale:
                self._delete(md5)
        self._delete(file_md5)
        n = parts.n_notes
        c.execute("INSERT INTO congregal_charts VALUES (?,?,?,?,?)",
                  (file_md5, bid, int(legacy), n, parts.colors.tobytes()))
        c.executemany("INSERT INTO congregal_speeds VALUES (?,?,?)",
                      [(file_md5, i, sf.astype("<f4", copy=False).tobytes())
                       for i, sf in enumerate(parts.speeds)])
        c.executemany("INSERT INTO congregals VALUES (?,?,?,?,?,?,?,?,?,?)",
                      [(file_md5, m, bid, n, sid, w[0], w[1], w[2], vis,
                        rates.astype("<f4", copy=False).tobytes())
                       for m, (sid, w, vis, rates) in parts.per_mod.items()])
        self._pending += 1
        if self._pending >= self.commit_every:
            c.commit()
            self._pending = 0

    def _delete(self, md5: str) -> None:
        for table in ("congregal_charts", "congregal_speeds", "congregals"):
            self.conn.execute(f"DELETE FROM {table} WHERE file_md5 = ?", (md5,))

    # ------------------------------------------------------------------ read
    def load(self, bid: int, mods: int) -> np.ndarray | None:
        """Features (n, 38) float32 for one beatmap/mod combination, or None."""
        row = self.conn.execute(
            "SELECT file_md5 FROM congregals WHERE beatmap_id = ? AND mods = ? LIMIT 1",
            (bid, mods)).fetchone()
        if row is None:
            return None
        for _, _, f in self.iter_charts("g.file_md5 = ? AND g.mods = ?", (row[0], mods)):
            return f
        return None

    def iter_charts(self, where: str = "", params=()):
        """Yield (beatmap_id, mods, features (n, 38)) for every stored chart/mod.

        `where` is an SQL condition on table alias g (congregals), e.g. "g.mods = 0"
        or "g.beatmap_id IN (1, 2, 3)".
        """
        sql = """SELECT g.beatmap_id, g.mods, g.n_notes, g.w300, g.w100, g.wmiss, g.vision, g.rates,
                        c.colors, s.features
                 FROM congregals g
                 JOIN congregal_charts c ON c.file_md5 = g.file_md5
                 JOIN congregal_speeds s ON s.file_md5 = g.file_md5 AND s.speed_id = g.speed_id"""
        if where:
            sql += " WHERE " + where
        sql += " ORDER BY g.beatmap_id, g.file_md5, g.mods"
        for bid, mods, n, w0, w1, w2, vis, rates, colors, sfeat in self.conn.execute(sql, params):
            yield bid, mods, assemble_features(
                np.frombuffer(colors, np.uint8),
                np.frombuffer(sfeat, "<f4").reshape(n, N_SPEED_FEATURES),
                (w0, w1, w2), vis, np.frombuffer(rates, "<f4"))

    def export_csv(self, out_path: str, where: str = "", params=()) -> int:
        """Write the stored charts as objects.csv (BID, Mods, index, 38 features). Returns rows."""
        rows = 0
        with open(out_path, "w", newline="") as fh:
            for bid, mods, f in self.iter_charts(where, params):
                write_objects_csv(fh, -1 if bid is None else bid, mods, f)
                rows += len(f)
        return rows


# --------------------------------------------------------------------------- per-file work
def _load_bo_npz(path: str, mods_list):
    with np.load(path) as z:
        mods = [int(m) for m in z["mods"]]
        if mods_list:
            missing = set(mods_list) - set(mods)
            if missing:
                raise ValueError(f"mods {sorted(missing)} not in {path}")
            mods = list(mods_list)
        bo_all = {m: z[f"mods_{m}"] for m in mods}
        bid = int(z["beatmap_id"])
        return bo_all, (None if bid < 0 else bid), str(z["file_md5"]), bool(z["legacy"])


def process_file(task):
    """Worker: one input file -> result dict (runs in a child process).

    No database access here; the main process resolves the last BID fallback,
    records metadata and writes the congregals.
    """
    path, legacy, mods_list, bid_override = task
    try:
        if path.lower().endswith(".npz"):
            bo_all, bid, md5, file_legacy = _load_bo_npz(path, mods_list)
            return {"status": "ok", "path": path, "bid": bid, "md5": md5,
                    "legacy": file_legacy, "parts": chart_parts(bo_all), "chart": None}
        from .osu_reader import read_osu_file
        from .taiko_bo import NotTaikoError, compute_bo_all, parse_taiko_chart, resolve_bid_offline

        try:
            chart = parse_taiko_chart(read_osu_file(path))
        except NotTaikoError as e:
            return {"status": "skip", "path": path, "msg": str(e)}
        bid, source = resolve_bid_offline(chart, bid_override)
        # Drop the big arrays before sending the chart back to the main process.
        meta_chart = chart.__class__(**{**chart.__dict__, "circle_times": np.empty(0),
                                        "circle_colors": np.empty(0, np.int8),
                                        "timing_points": [], "timing_points_legacy": []})
        result = {"status": "ok", "path": path, "bid": bid, "bid_source": source,
                  "md5": chart.file_md5, "legacy": legacy, "chart": meta_chart}
        try:
            result["parts"] = chart_parts(compute_bo_all(chart, legacy=legacy, mods_list=mods_list))
        except Exception as e:  # metadata is still recorded, like process_osu_file
            result.update(status="fail", msg=f"{type(e).__name__}: {e}")
        return result
    except Exception as e:
        return {"status": "fail", "path": path, "msg": f"{type(e).__name__}: {e}", "chart": None}


def _iter_results(tasks, workers: int):
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            yield process_file(t)
        return
    chunk = max(1, min(32, len(tasks) // (workers * 8)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(process_file, tasks, chunksize=chunk)


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    from beatmap_db import BeatmapDB
    from taiko_bo import _collect_inputs, _read_list_file, record_metadata, validate_mods, write_failed_list

    ap = argparse.ArgumentParser(description="BO .npz files (or .osu files) -> congregal database")
    ap.add_argument("inputs", nargs="*",
                    help="BO .npz / .osu files, folders, and/or .txt lists of paths (e.g. failed_files.txt)")
    ap.add_argument("--congregal_db", default="congregals.sqlite", help="congregal database (default output)")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="also write the processed charts to this objects.csv (headerless)")
    ap.add_argument("--no_db", action="store_true", help="with --csv: write only the CSV")
    ap.add_argument("--append", action="store_true", help="append to --csv instead of overwriting")
    ap.add_argument("--export_csv", default=None, metavar="PATH",
                    help="export everything already in --congregal_db to this CSV (no inputs needed)")
    ap.add_argument("--db", default="beatmaps.sqlite", help="metadata DB (used for .osu inputs)")
    ap.add_argument("--legacy", action="store_true", help="legacy BO mode (for .osu inputs)")
    ap.add_argument("--bid", type=int, default=None, help="BeatmapID override (single .osu only)")
    ap.add_argument("--mods", type=int, nargs="*", default=None, help="subset of mod values")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes (default: CPU cores - 1; 1 = no parallelism)")
    ap.add_argument("--failed_list", default="failed_files.txt",
                    help="where to list files that failed (written only if any fail)")
    args = ap.parse_args(argv)

    if args.no_db and not args.csv:
        ap.error("--no_db needs --csv")
    if not args.inputs:
        if not args.export_csv:
            ap.error("give inputs, or --export_csv to export the existing database")
        with CongregalDB(args.congregal_db) as cdb:
            n = cdb.export_csv(args.export_csv)
        print(f"exported {n} rows from {args.congregal_db} to {args.export_csv}")
        return 0

    files = []
    for p in args.inputs:
        if p.lower().endswith(".txt") and os.path.isfile(p):
            files += _read_list_file(p)
        elif os.path.isdir(p):
            files += _collect_inputs([p], args.recursive)
            if args.recursive:
                for root, _, names in os.walk(p):
                    files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".npz")]
            else:
                files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(".npz")]
        else:
            files.append(p)
    if args.bid is not None and len(files) != 1:
        ap.error("--bid can only be used with exactly one input file")
    for m in args.mods or []:
        validate_mods(m)

    tasks = [(f, args.legacy, args.mods, args.bid) for f in files]
    cdb = None if args.no_db else CongregalDB(args.congregal_db)
    fh = open(args.csv, "a" if args.append else "w", newline="") if args.csv else None
    meta_db = BeatmapDB(args.db) if any(not f.lower().endswith(".npz") for f in files) else None
    ok = skipped = failed = rows = 0
    failed_files = []
    t0 = last = time.time()
    try:
        for i, r in enumerate(_iter_results(tasks, args.workers), 1):
            name = os.path.basename(r["path"])
            if r["status"] == "skip":
                print(f"SKIP  {r['msg']}")
                skipped += 1
                continue
            bid = r.get("bid")
            if r.get("chart") is not None:  # .osu input: last BID fallback + metadata
                bid, _ = record_metadata(r["chart"], meta_db, bid, r.get("bid_source"))
            if r["status"] == "fail":
                print(f"FAIL  {r['path']}: {r['msg']}")
                failed += 1
                failed_files.append(r["path"])
                continue
            if bid is None:
                print(f"WARN  {name}: no BeatmapID (file, name or override)")
            parts = r["parts"]
            if cdb is not None:
                cdb.write_chart(bid, r["md5"], r["legacy"], parts)
            if fh is not None:
                for m in sorted(parts.per_mod):
                    write_objects_csv(fh, -1 if bid is None else bid, m, parts.features(m))
            rows += parts.n_notes * len(parts.per_mod)
            ok += 1
            now = time.time()
            if now - last > 5:
                rate = i / (now - t0)
                print(f"  {i}/{len(tasks)} files ({rate:.1f}/s, ~{(len(tasks) - i) / rate:.0f}s left)")
                last = now
    finally:
        if cdb is not None:
            cdb.close()
        if fh is not None:
            fh.close()
        if meta_db is not None:
            meta_db.close()
    dest = " + ".join(x for x in ((None if args.no_db else args.congregal_db), args.csv) if x)
    print(f"done in {time.time() - t0:.1f}s: {ok} files -> {rows} note rows in {dest}; "
          f"{skipped} not taiko, {failed} failed")
    if failed_files:
        write_failed_list(args.failed_list, failed_files)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
