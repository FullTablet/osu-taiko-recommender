"""Chart databases (SQLite): the base package's ranked charts and the user's
complementary (imported) charts share one format.

Per chart: metadata. Per (chart, mods) "variant": the note subnet's per-note
slope (5) and bias (5), stored compactly:
  * each of the 10 columns is quantized to `bits` (8 or 16) bits between its own
    min and max for that variant (stored as float32 lo/step),
  * stored column-major (all slope_1 values, then slope_2, ...) and zlib-compressed.
With 8 bits the error per value is at most (max-min)/510; build_base_package.py
reports its effect on expected accuracies.

Imported charts also keep their zlib-compressed .osu file, so they can be
recomputed when a new model is installed.
"""
from __future__ import annotations

import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np

FORMAT_NAME = "taiko-chart-db"
FORMAT_VERSION = 1
N_VALUES = 10                     # slope (5) + bias (5)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS charts (
    chart_id        INTEGER PRIMARY KEY,
    beatmap_id      INTEGER,
    beatmapset_id   INTEGER,
    file_md5        TEXT,
    artist          TEXT,
    title           TEXT,
    artist_unicode  TEXT,
    title_unicode   TEXT,
    creator         TEXT,
    version         TEXT,
    source          TEXT,
    tags            TEXT,
    status          INTEGER,          -- osu! approved value (1 ranked, 2 approved, ...); NULL if unknown
    length_s        REAL,
    bpm             REAL,
    od              REAL,
    hp              REAL,
    n_notes         INTEGER,
    file_name       TEXT,
    added_at        TEXT,
    osu_blob        BLOB              -- zlib .osu (imported charts only)
);
CREATE INDEX IF NOT EXISTS idx_charts_bid ON charts(beatmap_id);
CREATE INDEX IF NOT EXISTS idx_charts_md5 ON charts(file_md5);
CREATE TABLE IF NOT EXISTS variants (
    chart_id  INTEGER NOT NULL,
    mods      INTEGER NOT NULL,
    n_notes   INTEGER NOT NULL,
    lo        BLOB NOT NULL,          -- float32 x 10
    step      BLOB NOT NULL,          -- float32 x 10
    data      BLOB NOT NULL,          -- zlib(uint8/uint16 10 x n, column-major)
    PRIMARY KEY (chart_id, mods)
);
"""

CHART_COLUMNS = ["beatmap_id", "beatmapset_id", "file_md5", "artist", "title", "artist_unicode",
                 "title_unicode", "creator", "version", "source", "tags", "status", "length_s", "bpm",
                 "od", "hp", "n_notes", "file_name", "added_at"]


class IncompatibleDatabase(Exception):
    pass


def encode_values(slope: np.ndarray, bias: np.ndarray, bits: int = 8) -> tuple[bytes, bytes, bytes]:
    x = np.concatenate([np.asarray(slope, np.float32), np.asarray(bias, np.float32)], axis=1)   # (n, 10)
    if x.shape[1] != N_VALUES:
        raise ValueError("slope and bias must be (n, 5)")
    levels = (1 << bits) - 1
    lo = x.min(axis=0) if len(x) else np.zeros(N_VALUES, np.float32)
    hi = x.max(axis=0) if len(x) else np.zeros(N_VALUES, np.float32)
    step = np.where(hi > lo, (hi - lo) / levels, 1.0).astype(np.float32)
    q = np.rint((x - lo) / step).clip(0, levels).astype(np.uint8 if bits == 8 else np.uint16)
    data = zlib.compress(np.ascontiguousarray(q.T).tobytes(), 9)
    return lo.astype(np.float32).tobytes(), step.tobytes(), data


def decode_values(n: int, lo: bytes, step: bytes, data: bytes, bits: int) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.uint8 if bits == 8 else np.uint16
    q = np.frombuffer(zlib.decompress(data), dtype).reshape(N_VALUES, n).T
    x = q.astype(np.float32) * np.frombuffer(step, np.float32) + np.frombuffer(lo, np.float32)
    return x[:, :5], x[:, 5:]


class ChartDB:
    """One chart database file."""

    def __init__(self, path, create_meta: dict | None = None, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self.conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.executescript(SCHEMA)
        self.readonly = readonly
        meta = self.meta()
        if not meta:
            if create_meta is None:
                raise IncompatibleDatabase(f"{self.path} is not a chart database")
            m = {"format": FORMAT_NAME, "format_version": str(FORMAT_VERSION), "bits": "8",
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            m.update({k: str(v) for k, v in create_meta.items()})
            self.conn.executemany("INSERT INTO meta VALUES (?,?)", m.items())
            self.conn.commit()
            meta = m
        if meta.get("format") != FORMAT_NAME or int(meta.get("format_version", 0)) > FORMAT_VERSION:
            raise IncompatibleDatabase(f"{self.path}: unsupported chart database format")
        self.bits = int(meta.get("bits", 8))

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict:
        try:
            return dict(self.conn.execute("SELECT key, value FROM meta"))
        except sqlite3.OperationalError:
            return {}

    def set_meta(self, **kv):
        self.conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", [(k, str(v)) for k, v in kv.items()])
        self.conn.commit()

    @property
    def model_id(self) -> str:
        return self.meta().get("model_id", "")

    # ------------------------------------------------------------------ write
    def add_chart(self, info: dict, variants: dict[int, tuple[np.ndarray, np.ndarray]],
                  osu_bytes: bytes | None = None, commit: bool = True) -> int:
        row = [info.get(c) for c in CHART_COLUMNS]
        if row[CHART_COLUMNS.index("added_at")] is None:
            row[CHART_COLUMNS.index("added_at")] = time.strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.execute(
            f"INSERT INTO charts ({', '.join(CHART_COLUMNS)}, osu_blob) VALUES ({', '.join('?' * (len(CHART_COLUMNS) + 1))})",
            row + [zlib.compress(osu_bytes, 9) if osu_bytes else None])
        cid = cur.lastrowid
        self.put_variants(cid, variants, commit=False)
        if commit:
            self.conn.commit()
        return cid

    def put_variants(self, chart_id: int, variants: dict, commit: bool = True):
        rows = []
        for mods, (slope, bias) in variants.items():
            lo, step, data = encode_values(slope, bias, self.bits)
            rows.append((chart_id, int(mods), len(slope), lo, step, data))
        self.conn.executemany("INSERT OR REPLACE INTO variants VALUES (?,?,?,?,?,?)", rows)
        if commit:
            self.conn.commit()

    def delete_charts(self, chart_ids):
        ids = [(int(c),) for c in chart_ids]
        self.conn.executemany("DELETE FROM variants WHERE chart_id = ?", ids)
        self.conn.executemany("DELETE FROM charts WHERE chart_id = ?", ids)
        self.conn.commit()

    def commit(self):
        self.conn.commit()

    # ------------------------------------------------------------------ read
    def variant(self, chart_id: int, mods: int):
        row = self.conn.execute("SELECT n_notes, lo, step, data FROM variants WHERE chart_id = ? AND mods = ?",
                                (int(chart_id), int(mods))).fetchone()
        if row is None:
            return None
        return decode_values(row[0], row[1], row[2], row[3], self.bits)

    def mods_of(self, chart_id: int) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT mods FROM variants WHERE chart_id = ? ORDER BY mods",
                                                (int(chart_id),))]

    def chart(self, chart_id: int) -> dict | None:
        cur = self.conn.execute(f"SELECT chart_id, {', '.join(CHART_COLUMNS)} FROM charts WHERE chart_id = ?",
                                (int(chart_id),))
        row = cur.fetchone()
        return None if row is None else dict(zip(["chart_id"] + CHART_COLUMNS, row))

    def osu_bytes(self, chart_id: int) -> bytes | None:
        row = self.conn.execute("SELECT osu_blob FROM charts WHERE chart_id = ?", (int(chart_id),)).fetchone()
        return zlib.decompress(row[0]) if row and row[0] else None

    def find(self, beatmap_id: int | None = None, file_md5: str | None = None) -> list[int]:
        if beatmap_id is not None:
            return [r[0] for r in self.conn.execute("SELECT chart_id FROM charts WHERE beatmap_id = ?", (int(beatmap_id),))]
        if file_md5 is not None:
            return [r[0] for r in self.conn.execute("SELECT chart_id FROM charts WHERE file_md5 = ?", (file_md5.lower(),))]
        return []

    def search(self, text: str = "", limit: int = 500) -> list[dict]:
        """Charts whose artist/title/version/creator contain `text` (all words), or whose
        beatmap ID / file hash equals it."""
        cols = ["chart_id"] + CHART_COLUMNS
        sel = f"SELECT {', '.join(cols)} FROM charts"
        text = text.strip()
        if not text:
            rows = self.conn.execute(sel + " ORDER BY chart_id DESC LIMIT ?", (limit,)).fetchall()
        elif text.isdigit():
            rows = self.conn.execute(sel + " WHERE beatmap_id = ? OR beatmapset_id = ? LIMIT ?",
                                     (int(text), int(text), limit)).fetchall()
        elif len(text) == 32 and all(ch in "0123456789abcdefABCDEF" for ch in text):
            rows = self.conn.execute(sel + " WHERE file_md5 = ? LIMIT ?", (text.lower(), limit)).fetchall()
        else:
            words = text.lower().split()
            cond = " AND ".join(["lower(coalesce(artist,'') || ' ' || coalesce(title,'') || ' ' || "
                                 "coalesce(version,'') || ' ' || coalesce(creator,'')) LIKE ?"] * len(words))
            rows = self.conn.execute(sel + f" WHERE {cond} LIMIT ?", [f"%{w}%" for w in words] + [limit]).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM charts").fetchone()[0]

    def all_variants(self) -> list[tuple[int, int]]:
        return self.conn.execute("SELECT chart_id, mods FROM variants").fetchall()

    def md5_set(self) -> set:
        return {r[0] for r in self.conn.execute("SELECT file_md5 FROM charts WHERE file_md5 IS NOT NULL")}

    # ------------------------------------------------------------------ merge
    def merge_from(self, other_path, progress=None) -> tuple[int, int]:
        """Append the charts of another complementary database, skipping file hashes
        already present. Raises IncompatibleDatabase if the model differs."""
        other = ChartDB(other_path, readonly=True)
        try:
            if other.model_id != self.model_id:
                raise IncompatibleDatabase(
                    f"that database was made with model {other.model_id or '?'}; yours uses {self.model_id or '?'}")
            have = self.md5_set()
            ids = [r[0] for r in other.conn.execute("SELECT chart_id FROM charts ORDER BY chart_id")]
            added = skipped = 0
            for i, cid in enumerate(ids):
                info = other.chart(cid)
                if info["file_md5"] in have:
                    skipped += 1
                    continue
                osu = other.osu_bytes(cid)
                variants = {m: other.variant(cid, m) for m in other.mods_of(cid)}
                self.add_chart(info, variants, osu, commit=False)
                have.add(info["file_md5"])
                added += 1
                if progress:
                    progress((i + 1) / len(ids))
            self.conn.commit()
            return added, skipped
        finally:
            other.close()
