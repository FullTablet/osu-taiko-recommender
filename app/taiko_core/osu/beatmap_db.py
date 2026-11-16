"""SQLite database: beatmap ID -> [Metadata] fields (+ file name and MD5).

One row per beatmap. Rows are matched, in this order, by:
  * beatmap_id, when known (from the file, or a --bid override);
  * otherwise file_md5, stored with beatmap_id = NULL.
If a file previously stored with a NULL ID is later read with a known ID
(e.g. with --bid), the NULL row is merged into the ID row.

Re-reading a beatmap updates its row (metadata, file name, hash, last_seen);
first_seen is kept from the first time it was read.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

# [Metadata] key in the .osu file -> column name.
METADATA_COLUMNS = {
    "Title": "title",
    "TitleUnicode": "title_unicode",
    "Artist": "artist",
    "ArtistUnicode": "artist_unicode",
    "Creator": "creator",
    "Version": "version",
    "Source": "source",
    "Tags": "tags",
    "BeatmapSetID": "beatmap_set_id",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS beatmaps (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    beatmap_id      INTEGER UNIQUE,          -- NULL when unknown
    beatmap_set_id  INTEGER,
    title           TEXT,
    title_unicode   TEXT,
    artist          TEXT,
    artist_unicode  TEXT,
    creator         TEXT,
    version         TEXT,
    source          TEXT,
    tags            TEXT,
    extra_metadata  TEXT,                    -- JSON of any other [Metadata] keys
    file_name       TEXT NOT NULL,
    file_md5        TEXT NOT NULL,
    file_encoding   TEXT,
    format_version  INTEGER,
    bid_source      TEXT,                    -- 'file', 'override', 'filename', 'db_md5' or NULL
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_beatmaps_md5 ON beatmaps(file_md5);
"""


def _to_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


class BeatmapDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ lookup
    def bid_for_md5(self, file_md5: str) -> int | None:
        row = self.conn.execute(
            "SELECT beatmap_id FROM beatmaps WHERE file_md5 = ? AND beatmap_id IS NOT NULL "
            "ORDER BY last_seen DESC LIMIT 1",
            (file_md5,),
        ).fetchone()
        return None if row is None else row[0]

    def get(self, beatmap_id: int) -> dict | None:
        cur = self.conn.execute("SELECT * FROM beatmaps WHERE beatmap_id = ?", (beatmap_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    # ------------------------------------------------------------------ upsert
    def upsert(
        self,
        metadata: dict[str, str],
        beatmap_id: int | None,
        bid_source: str | None,
        file_name: str,
        file_md5: str,
        file_encoding: str | None = None,
        format_version: int | None = None,
    ) -> int:
        """Insert or update one beatmap. Returns the row_id."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        values = {col: metadata.get(key) for key, col in METADATA_COLUMNS.items()}
        values["beatmap_set_id"] = _to_int(values["beatmap_set_id"])
        extra = {k: v for k, v in metadata.items()
                 if k not in METADATA_COLUMNS and k != "BeatmapID"}
        values.update(
            beatmap_id=beatmap_id,
            extra_metadata=json.dumps(extra, ensure_ascii=False) if extra else None,
            file_name=file_name,
            file_md5=file_md5,
            file_encoding=file_encoding,
            format_version=format_version,
            bid_source=bid_source,
            last_seen=now,
        )

        with self.conn:  # one transaction
            first_seen = now
            if beatmap_id is not None:
                # Fold in any earlier NULL-ID row for this exact file.
                old = self.conn.execute(
                    "SELECT first_seen FROM beatmaps WHERE file_md5 = ? AND beatmap_id IS NULL",
                    (file_md5,),
                ).fetchone()
                if old is not None:
                    first_seen = old[0]
                    self.conn.execute(
                        "DELETE FROM beatmaps WHERE file_md5 = ? AND beatmap_id IS NULL",
                        (file_md5,),
                    )
                existing = self.conn.execute(
                    "SELECT row_id, first_seen FROM beatmaps WHERE beatmap_id = ?",
                    (beatmap_id,),
                ).fetchone()
            else:
                existing = self.conn.execute(
                    "SELECT row_id, first_seen FROM beatmaps WHERE file_md5 = ? AND beatmap_id IS NULL",
                    (file_md5,),
                ).fetchone()

            cols = list(values)
            if existing is None:
                values["first_seen"] = first_seen
                cols = list(values)
                cur = self.conn.execute(
                    f"INSERT INTO beatmaps ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [values[c] for c in cols],
                )
                return cur.lastrowid
            row_id, _ = existing
            self.conn.execute(
                f"UPDATE beatmaps SET {', '.join(c + ' = ?' for c in cols)} WHERE row_id = ?",
                [values[c] for c in cols] + [row_id],
            )
            return row_id
