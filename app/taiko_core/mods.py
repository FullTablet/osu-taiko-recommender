"""Mod combinations (same bitmask as the training pipeline)."""
from __future__ import annotations

from .osu.taiko_bo import EZ, FL, HD, HR, HT, DT, MOD_COMBINATIONS, MOD_NAMES, mods_to_string  # noqa: F401

# The base package ships these combinations: NM, HD, HR, DT, HDHR, HDDT, HT, EZ.
BASE_MODS = [0, HD, HR, DT, HD | HR, HD | DT, HT, EZ]

# Default selection for manual imports: every combination without FL.
DEFAULT_IMPORT_MODS = [m for m in MOD_COMBINATIONS if not m & FL]


def mods_label(mods: int) -> str:
    """'NM', 'HD', 'HDHR', 'HDDT', ... (no separators, osu! style)."""
    if mods == 0:
        return "NM"
    order = [(EZ, "EZ"), (HD, "HD"), (HR, "HR"), (DT, "DT"), (HT, "HT"), (FL, "FL")]
    return "".join(name for bit, name in order if mods & bit)


def parse_mods_label(text: str) -> int:
    """Inverse of mods_label; accepts 'NM', 'HDDT', 'HD+DT', 'hd dt', 'NC' (= DT), 'DC' (= HT)."""
    t = text.upper().replace("+", "").replace(" ", "").replace(",", "")
    t = t.replace("NC", "DT").replace("DC", "HT")
    if t in ("", "NM", "NOMOD"):
        return 0
    names = {"EZ": EZ, "HD": HD, "HR": HR, "DT": DT, "HT": HT, "FL": FL}
    bits = 0
    for i in range(0, len(t), 2):
        part = t[i:i + 2]
        if part not in names:
            raise ValueError(f"unknown mod {part!r} in {text!r}")
        bits |= names[part]
    if bits not in MOD_COMBINATIONS:
        raise ValueError(f"{text!r} is not a valid combination")
    return bits
