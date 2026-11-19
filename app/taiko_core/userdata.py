"""The player's own data: plays, skill estimates, settings."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS plays (
    play_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    position    REAL NOT NULL,           -- order of play: larger = more recent
    source      TEXT NOT NULL,           -- 'base' or 'comp'
    beatmap_id  INTEGER,                 -- base charts are referenced by beatmap ID
    file_md5    TEXT,                    -- imported charts by file hash
    mods        INTEGER NOT NULL,
    accuracy    REAL NOT NULL,           -- 0..1
    played_at   TEXT,
    label       TEXT                     -- 'Artist - Title [Version]' for display
);
CREATE TABLE IF NOT EXISTS skill_estimates (
    estimate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    skill       TEXT NOT NULL,           -- JSON list (skill values)
    quantiles   TEXT,                    -- JSON list
    n_plays     INTEGER,
    method      TEXT,                    -- 'rank' (initial) or 'plays'
    model_id    TEXT
);
"""


class UserData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.folder / "user.db", check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.settings_path = self.folder / "settings.json"
        try:
            self.settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.settings = {}

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ settings
    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        tmp = self.settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        tmp.replace(self.settings_path)

    # ------------------------------------------------------------------ plays
    def add_play(self, source: str, mods: int, accuracy: float, beatmap_id=None, file_md5=None,
                 label: str = "", played_at: str | None = None) -> int:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be between 0 and 1")
        pos = self.conn.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM plays").fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO plays (position, source, beatmap_id, file_md5, mods, accuracy, played_at, label) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (pos, source, beatmap_id, file_md5, int(mods), float(accuracy),
             played_at or time.strftime("%Y-%m-%d %H:%M:%S"), label))
        self.conn.commit()
        return cur.lastrowid

    def plays(self, newest_first: bool = False, limit: int | None = None) -> list[dict]:
        cols = ["play_id", "position", "source", "beatmap_id", "file_md5", "mods", "accuracy", "played_at", "label"]
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT {', '.join(cols)} FROM plays ORDER BY position {order}"
        rows = self.conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def recent_plays(self, n: int) -> list[dict]:
        """The n most recent plays, oldest first."""
        return list(reversed(self.plays(newest_first=True, limit=n)))

    def update_play(self, play_id: int, accuracy: float | None = None, mods: int | None = None):
        if accuracy is not None:
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError("accuracy must be between 0 and 1")
            self.conn.execute("UPDATE plays SET accuracy = ? WHERE play_id = ?", (float(accuracy), play_id))
        if mods is not None:
            self.conn.execute("UPDATE plays SET mods = ? WHERE play_id = ?", (int(mods), play_id))
        self.conn.commit()

    def delete_plays(self, play_ids):
        self.conn.executemany("DELETE FROM plays WHERE play_id = ?", [(int(p),) for p in play_ids])
        self.conn.commit()

    def count_plays(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]

    # ------------------------------------------------------------------ skill
    def save_skill(self, skill, quantiles, n_plays: int, method: str, model_id: str = ""):
        self.conn.execute(
            "INSERT INTO skill_estimates (created_at, skill, quantiles, n_plays, method, model_id) VALUES (?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps([float(x) for x in skill]),
             json.dumps([float(x) for x in quantiles]), int(n_plays), method, model_id))
        self.conn.commit()

    def current_skill(self) -> dict | None:
        row = self.conn.execute("SELECT created_at, skill, quantiles, n_plays, method, model_id "
                                "FROM skill_estimates ORDER BY estimate_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return dict(created_at=row[0], skill=np.array(json.loads(row[1])), quantiles=np.array(json.loads(row[2] or "[]")),
                    n_plays=row[3], method=row[4], model_id=row[5])

    def previous_skill(self) -> dict | None:
        """The estimate before the current one (None if there is only one)."""
        row = self.conn.execute("SELECT created_at, skill, quantiles, n_plays, method, model_id "
                                "FROM skill_estimates ORDER BY estimate_id DESC LIMIT 1 OFFSET 1").fetchone()
        if row is None:
            return None
        return dict(created_at=row[0], skill=np.array(json.loads(row[1])), quantiles=np.array(json.loads(row[2] or "[]")),
                    n_plays=row[3], method=row[4], model_id=row[5])

    def reset_skill(self):
        self.conn.execute("DELETE FROM skill_estimates")
        self.conn.commit()
