"""Robust reading of .osu files: encoding detection + section splitting.

.osu files are usually UTF-8 (with or without BOM), but older or hand-edited
files show up as UTF-16 (with or without BOM), Shift-JIS/CP932, GBK, EUC-KR,
CP1252, ... Only the [Metadata] section realistically contains non-ASCII text;
every numeric section is plain ASCII, so a wrong guess can at worst garble a
title/tag string, never a BO value.

Detection order:
  1. Byte-order marks (UTF-8-sig, UTF-32 LE/BE, UTF-16 LE/BE).
  2. BOM-less UTF-16 (lots of NUL bytes in alternating positions).
  3. Strict UTF-8 (the overwhelmingly common case).
  4. charset_normalizer, if installed (it ships with `requests`).
  5. Strict CP932 -> GBK -> EUC-KR -> CP1252, then Latin-1 (never fails).
"""
from __future__ import annotations

import codecs
import hashlib
from dataclasses import dataclass, field

_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

_FALLBACK_ENCODINGS = ("cp932", "gbk", "euc_kr", "cp1252")
_DETECTION_CANDIDATES = ("cp932", "gb18030", "big5", "cp949", "cp1252", "cp1251", "cp1250")


def _strip_bom(raw: bytes) -> tuple[bytes, str | None]:
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return raw[len(bom):], enc
    return raw, None


def _bomless_utf16(raw: bytes) -> str | None:
    sample = raw[:4096]
    if len(sample) < 4:
        return None
    even_nuls = sample[0::2].count(0)
    odd_nuls = sample[1::2].count(0)
    half = len(sample) / 2
    if odd_nuls > 0.3 * half and even_nuls < 0.05 * half:
        return "utf-16-le"
    if even_nuls > 0.3 * half and odd_nuls < 0.05 * half:
        return "utf-16-be"
    return None


def decode_osu_bytes(raw: bytes) -> tuple[str, str]:
    """Decode raw .osu bytes. Returns (text, encoding_name_used)."""
    body, enc = _strip_bom(raw)
    if enc is not None:
        return body.decode(enc, errors="replace"), enc

    enc = _bomless_utf16(raw)
    if enc is not None:
        return raw.decode(enc, errors="replace"), enc

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        from charset_normalizer import from_bytes  # optional dependency

        # Restrict to code pages osu! files realistically use; unrestricted
        # detection on short, mostly-ASCII text picks exotic Baltic/Greek pages.
        best = from_bytes(raw, cp_isolation=list(_DETECTION_CANDIDATES)).best()
        if best is not None and best.encoding:
            return str(best), best.encoding
    except Exception:  # library missing or failed -> fall through
        pass

    for enc in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


@dataclass
class OsuFile:
    """A parsed-into-sections .osu file (no game logic here)."""

    file_name: str
    file_md5: str
    encoding: str
    format_version: int | None
    # Section name -> raw lines (blank lines and '//' comments kept, line
    # terminators and trailing whitespace stripped). Keeping blanks lets the
    # legacy BO mode reproduce Mathematica's "stop at first blank line" rule.
    sections: dict[str, list[str]] = field(default_factory=dict)

    def key_values(self, section: str) -> dict[str, str]:
        """'Key: value' / 'Key:value' pairs of a section (first occurrence wins)."""
        out: dict[str, str] = {}
        for line in self.sections.get(section, []):
            if not line or line.startswith("//") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            out.setdefault(key.strip(), value.strip())
        return out

    def data_lines(self, section: str) -> list[str]:
        """Non-blank, non-comment lines of a section."""
        return [
            ln for ln in self.sections.get(section, [])
            if ln.strip() and not ln.lstrip().startswith("//")
        ]


def parse_osu_bytes(raw: bytes, file_name: str = "") -> OsuFile:
    text, encoding = decode_osu_bytes(raw)
    text = text.replace("\x00", "")
    # splitlines() handles \r\n, \r and \n (and stray Unicode separators).
    lines = [ln.rstrip() for ln in text.splitlines()]

    format_version = None
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for ln in lines:
        stripped = ln.strip()
        if current is None and stripped.lower().startswith("osu file format v"):
            try:
                format_version = int(stripped[len("osu file format v"):].strip())
            except ValueError:
                pass
            continue
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2:
            current = stripped[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(ln)

    return OsuFile(
        file_name=file_name,
        file_md5=hashlib.md5(raw).hexdigest(),
        encoding=encoding,
        format_version=format_version,
        sections=sections,
    )


READ_RETRY_DELAYS = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)  # seconds between attempts


def read_bytes_with_retry(path, delays=READ_RETRY_DELAYS) -> bytes:
    """Read a whole file, retrying transient OS errors.

    Cloud-synced folders (Box Drive, OneDrive, Dropbox "online-only" files)
    keep files as placeholders that are downloaded on first open; when many
    processes open placeholders at once, Windows can fail the open/read with
    errors like `[Errno 22] Invalid argument` that succeed on a later attempt.
    FileNotFoundError / IsADirectoryError are not retried.
    """
    import time

    for attempt in range(len(delays) + 1):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise
        except OSError as e:
            if attempt == len(delays):
                raise OSError(e.errno, f"{e.strerror} (still failing after {attempt + 1} attempts; "
                                       "if this folder is cloud-synced, make it available offline "
                                       "or copy it to a local disk)") from e
            time.sleep(delays[attempt])
    raise AssertionError("unreachable")


def read_osu_file(path) -> OsuFile:
    import os

    raw = read_bytes_with_retry(path)
    return parse_osu_bytes(raw, file_name=os.path.basename(str(path)))
