"""Taiko .osu  ->  BO values (one n x 7 float32 array per mod combination).

Pipeline position:  .osu  ->  BO values  ->  Congregal values  ->  NN difficulty.

Each BO row is one hit circle (drumrolls and spinners are skipped), sorted by
time then colour:

    col 0  colour            0 = don (red), 1 = kat (blue: whistle or clap bit)
    col 1  delta time (ms)   time since previous circle, after DT/HT rate change;
                             the first circle gets  time + 1e11
    col 2  300 window / 20   (ms / 20, after OD mods and DT/HT)
    col 3  100 window / 20
    col 4  miss window / 20
    col 5  vision shift      HD +1 (HDHR +1.5), FL +2
    col 6  scroll rate       1000/beatLength * SV/1.4 * SVmultiplier * mod factor
                             (0.1 before the first timing point)

Two behaviours:

  default ("fixed")   corrected version of the Mathematica reference code.
  legacy=True         reproduces the Mathematica reference code exactly,
                      including its known quirks (for data the existing network
                      was trained on).

Differences, legacy -> fixed:
  1. HT timing windows: legacy re-checks the DT bit (DT windows get /1.5/0.75,
     HT windows unchanged); fixed divides by 1.5 for DT and 0.75 for HT.
  2. Mod effect on timing points: legacy multiplies timing-point TIMES by
     1.5 (DT), 1.4*4/3 (HR), 0.75 (HT), 0.8 (EZ) and leaves rates alone.
     Fixed: DT/HT divide timing-point times by 1.5/0.75 (same as the notes)
     AND multiply scroll rates by 1.5/0.75; HR/EZ multiply scroll rates by
     1.4*4/3 / 0.8 and leave times alone.
  3. Inherited points: legacy omits the SliderMultiplier/1.4 factor; fixed
     includes it.
  4. Base beat length for inherited points: legacy uses the last uninherited
     point in FILE order; fixed uses the last one in CHRONOLOGICAL order
     (uninherited first on equal times), so file order never matters.
  Same in both: on equal timestamps the larger scroll rate wins (timing points
  sorted by (time, rate)), as intended.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np

from .beatmap_db import BeatmapDB
from .osu_reader import OsuFile, read_osu_file

# --------------------------------------------------------------------------- mods
EZ, HD, HR, DT, HT, FL = 2, 8, 16, 64, 256, 1024
MOD_NAMES = {EZ: "EZ", HD: "HD", HR: "HR", DT: "DT", HT: "HT", FL: "FL"}
_INCOMPATIBLE = ((EZ, HR), (DT, HT))


def _all_mod_combinations() -> list[int]:
    mods = list(MOD_NAMES)
    combos = set()
    for r in range(len(mods) + 1):
        for subset in itertools.combinations(mods, r):
            if any(a in subset and b in subset for a, b in _INCOMPATIBLE):
                continue
            combos.add(sum(subset))
    return sorted(combos)


MOD_COMBINATIONS: list[int] = _all_mod_combinations()  # 36 values, 0 .. 1306


def mods_to_string(mods: int) -> str:
    names = [name for bit, name in MOD_NAMES.items() if mods & bit]
    return "+".join(names) if names else "NM"


def validate_mods(mods: int) -> None:
    if mods & ~sum(MOD_NAMES):
        raise ValueError(f"mods={mods} contains unsupported bits")
    for a, b in _INCOMPATIBLE:
        if mods & a and mods & b:
            raise ValueError(f"mods={mods}: {MOD_NAMES[a]} and {MOD_NAMES[b]} cannot be combined")


# --------------------------------------------------------------------------- errors
class NotTaikoError(Exception):
    """The file's [General] Mode is not 1."""


class OsuParseError(Exception):
    """The file is taiko but could not be turned into BO values."""


# --------------------------------------------------------------------------- chart
@dataclass
class TaikoChart:
    file_name: str
    file_md5: str
    encoding: str
    format_version: int | None
    metadata: dict[str, str]
    beatmap_id_in_file: int | None
    od: float | None
    slider_multiplier: float | None
    # Timing points as (time, beatLength), file order, comments/blank lines removed.
    timing_points: list[tuple[float, float]]
    # Same, but cut at the first blank line, as the Mathematica reader did.
    timing_points_legacy: list[tuple[float, float]]
    circle_times: np.ndarray   # float64, file order
    circle_colors: np.ndarray  # int8, file order (0 don, 1 kat)


def _float_or_none(s: str | None) -> float | None:
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def get_mode(osu: OsuFile) -> int:
    value = osu.key_values("General").get("Mode")
    if value is None:
        return 0  # osu! default
    try:
        return int(float(value))
    except ValueError:
        raise OsuParseError(f"unreadable Mode value {value!r}")


def _parse_timing_line(line: str) -> tuple[float, float]:
    f = line.split(",")
    return float(f[0]), float(f[1])


def parse_taiko_chart(osu: OsuFile) -> TaikoChart:
    """Check Mode: 1 and extract everything BO generation needs."""
    mode = get_mode(osu)
    if mode != 1:
        raise NotTaikoError(f"{osu.file_name}: Mode: {mode} (not taiko), skipped")

    metadata = osu.key_values("Metadata")
    bid = _float_or_none(metadata.get("BeatmapID"))
    bid = int(bid) if bid is not None and bid > 0 else None

    diff = osu.key_values("Difficulty")
    od = _float_or_none(diff.get("OverallDifficulty"))
    sv = _float_or_none(diff.get("SliderMultiplier"))

    tp_all: list[tuple[float, float]] = []
    tp_legacy: list[tuple[float, float]] = []
    legacy_open = True
    for line in osu.sections.get("TimingPoints", []):
        if line.strip() == "":
            legacy_open = False  # Mathematica stopped reading here
            continue
        if line.lstrip().startswith("//"):
            continue
        try:
            tp = _parse_timing_line(line)
        except (ValueError, IndexError):
            raise OsuParseError(f"{osu.file_name}: bad timing point line {line!r}")
        tp_all.append(tp)
        if legacy_open:
            tp_legacy.append(tp)

    times, colors = [], []
    for line in osu.data_lines("HitObjects"):
        f = line.split(",")
        try:
            t = float(f[2])
            obj_type = int(float(f[3]))
            hitsound = int(float(f[4]))
        except (ValueError, IndexError):
            raise OsuParseError(f"{osu.file_name}: bad hit object line {line!r}")
        if obj_type & 1:  # hit circle; sliders (drumrolls) and spinners skipped
            times.append(t)
            colors.append(1 if hitsound & (2 | 8) else 0)  # whistle or clap -> kat

    return TaikoChart(
        file_name=osu.file_name,
        file_md5=osu.file_md5,
        encoding=osu.encoding,
        format_version=osu.format_version,
        metadata=metadata,
        beatmap_id_in_file=bid,
        od=od,
        slider_multiplier=sv,
        timing_points=tp_all,
        timing_points_legacy=tp_legacy,
        circle_times=np.asarray(times, dtype=np.float64),
        circle_colors=np.asarray(colors, dtype=np.int8),
    )


# --------------------------------------------------------------------------- BO
FIRST_DELTA_ORIGIN = -10.0e10  # Mathematica's -10.0*^10
DEFAULT_RATE = 0.1


def timing_windows(od: float) -> np.ndarray:
    if od < 5:
        return np.array([50 - 3 * od, 120 - 8 * od, 135 - 8 * od], dtype=np.float64)
    return np.array([50 - 3 * od, 110 - 6 * od, 120 - 5 * od], dtype=np.float64)


def _rate_objects_legacy(chart: TaikoChart, sv: float) -> tuple[np.ndarray, np.ndarray]:
    rates, times = [], []
    unrate = None
    for t, bl in chart.timing_points_legacy:  # file order
        if bl > 0:
            unrate = bl
            r = 1000 / unrate * (sv / 1.4)
        else:
            if unrate is None or bl == 0:
                raise OsuParseError(
                    f"{chart.file_name}: inherited timing point at {t} with no preceding "
                    "uninherited point (or zero beat length); Mathematica code fails here")
            r = 1000 / unrate * (-100.0 / bl)  # (no SV/1.4 factor: legacy quirk 3)
        rates.append(r)
        times.append(t)
    return np.asarray(rates, dtype=np.float64), np.asarray(times, dtype=np.float64)


def _rate_objects_fixed(chart: TaikoChart, sv: float) -> tuple[np.ndarray, np.ndarray]:
    # Chronological, uninherited before inherited on equal times; stable otherwise.
    tps = sorted(chart.timing_points, key=lambda p: (p[0], 0 if p[1] > 0 else 1))
    rates, times = [], []
    unrate = None
    for t, bl in tps:
        if not np.isfinite(bl) or bl == 0:
            continue
        if bl > 0:
            unrate = bl
            r = 1000 / unrate * (sv / 1.4)
        else:
            if unrate is None:
                continue  # inherited point before any BPM is defined: no effect
            r = 1000 / unrate * (sv / 1.4) * (-100.0 / bl)
        rates.append(r)
        times.append(t)
    return np.asarray(rates, dtype=np.float64), np.asarray(times, dtype=np.float64)


def compute_bo(chart: TaikoChart, mods: int, legacy: bool = False) -> np.ndarray:
    """BO values of one chart for one mod combination: float32 array (n_circles, 7)."""
    validate_mods(mods)
    ez, hd, hr, dt, ht, fl = (bool(mods & m) for m in (EZ, HD, HR, DT, HT, FL))

    od, sv = chart.od, chart.slider_multiplier
    if od is None or sv is None:
        if legacy:
            raise OsuParseError(f"{chart.file_name}: missing OverallDifficulty/SliderMultiplier")
        od = 5.0 if od is None else od    # osu! defaults
        sv = 1.4 if sv is None else sv

    # --- per-chart constants -----------------------------------------------
    if ez:
        od = od * 0.5
    if hr:
        od = min(od * 1.4, 10.0)
    vision = 0.0
    if hd:
        vision += 1.5 if hr else 1.0
    if fl:
        vision += 2.0

    windows = timing_windows(od)
    if dt:
        windows = windows / 1.5
    if legacy:
        if dt:                      # quirk 1: HT line re-checks the DT bit
            windows = windows / 0.75
    elif ht:
        windows = windows / 0.75
    windows = windows / 20.0

    # --- notes: sort by (time, colour), apply DT/HT to times ----------------
    order = np.lexsort((chart.circle_colors, chart.circle_times))
    times = chart.circle_times[order]
    colors = chart.circle_colors[order].astype(np.float64)
    if dt:
        times = times / 1.5
    if ht:
        times = times / 0.75

    # --- timing points ------------------------------------------------------
    if legacy:
        r_rates, r_times = _rate_objects_legacy(chart, sv)
    else:
        r_rates, r_times = _rate_objects_fixed(chart, sv)
    o = np.lexsort((r_rates, r_times))  # sort by time, then rate (larger rate wins ties)
    r_rates, r_times = r_rates[o], r_times[o]

    if legacy:                       # quirk 2: times scaled, rates untouched
        if dt:
            r_times = r_times * 1.5
        if hr:
            r_times = r_times * 1.4 * 4 / 3
        if ht:
            r_times = r_times * 0.75
        if ez:
            r_times = r_times * 0.80
    else:
        if dt:
            r_times = r_times / 1.5
            r_rates = r_rates * 1.5
        if ht:
            r_times = r_times / 0.75
            r_rates = r_rates * 0.75
        if hr:
            r_rates = r_rates * (1.4 * 4 / 3)
        if ez:
            r_rates = r_rates * 0.80

    # Rate in effect at each note = last timing point with time <= note time.
    idx = np.searchsorted(r_times, times, side="right") - 1
    if len(r_rates):
        rate = np.where(idx >= 0, r_rates[np.maximum(idx, 0)], DEFAULT_RATE)
    else:
        rate = np.full(len(times), DEFAULT_RATE)

    n = len(times)
    out = np.empty((n, 7), dtype=np.float64)
    out[:, 0] = colors
    out[:, 1] = np.diff(np.concatenate(([FIRST_DELTA_ORIGIN], times)))
    out[:, 2:5] = windows
    out[:, 5] = vision
    out[:, 6] = rate
    return out.astype(np.float32)


def compute_bo_all(chart: TaikoChart, legacy: bool = False,
                   mods_list: list[int] | None = None) -> dict[int, np.ndarray]:
    """{mods: BO array} for every requested mod combination (default: all 36)."""
    return {m: compute_bo(chart, m, legacy) for m in (mods_list or MOD_COMBINATIONS)}


# --------------------------------------------------------------------------- file-level API
@dataclass
class ProcessedFile:
    path: str
    beatmap_id: int | None
    bid_source: str | None
    file_md5: str
    bo: dict[int, np.ndarray]

    @property
    def output_stem(self) -> str:
        return str(self.beatmap_id) if self.beatmap_id is not None else f"md5_{self.file_md5}"


def resolve_bid_offline(chart: TaikoChart, bid_override: int | None = None):
    """BID from override > file > numeric file name (no database needed).
    Returns (bid or None, source or None)."""
    bid, source = chart.beatmap_id_in_file, ("file" if chart.beatmap_id_in_file else None)
    if bid_override is not None:
        if bid is not None and bid != bid_override:
            warnings.warn(f"{chart.file_name}: file says BeatmapID {bid}, using override {bid_override}")
        return bid_override, "override"
    if bid is None:
        stem = os.path.splitext(chart.file_name)[0].strip()
        if stem.isdigit() and int(stem) > 0:          # e.g. "1000101.osu" -> 1000101
            return int(stem), "filename"
    return bid, source


def record_metadata(chart: TaikoChart, db: BeatmapDB | None, bid, source):
    """Last BID fallback (earlier DB row with the same MD5), then upsert the metadata."""
    if db is None:
        return bid, source
    if bid is None:
        known = db.bid_for_md5(chart.file_md5)
        if known is not None:
            bid, source = known, "db_md5"
    db.upsert(chart.metadata, bid, source, chart.file_name, chart.file_md5,
              chart.encoding, chart.format_version)
    return bid, source


def process_osu_file(path: str, db: BeatmapDB | None = None, bid_override: int | None = None,
                     legacy: bool = False, mods_list: list[int] | None = None) -> ProcessedFile:
    """Read one .osu file: check Mode 1, record metadata in the DB, compute BO values.

    BeatmapID priority: bid_override > BeatmapID in the file > numeric file name
    ("1000101.osu" -> 1000101) > earlier DB row with the same file MD5 > None.

    Raises NotTaikoError for non-taiko files (nothing is written to the DB).
    """
    osu = read_osu_file(path)
    chart = parse_taiko_chart(osu)  # raises NotTaikoError
    bid, source = resolve_bid_offline(chart, bid_override)
    bid, source = record_metadata(chart, db, bid, source)
    bo = compute_bo_all(chart, legacy=legacy, mods_list=mods_list)
    return ProcessedFile(path=str(path), beatmap_id=bid, bid_source=source,
                         file_md5=chart.file_md5, bo=bo)


def save_npz(result: ProcessedFile, out_dir: str, legacy: bool) -> str:
    path = os.path.join(out_dir, result.output_stem + ".npz")
    arrays = {f"mods_{m}": a for m, a in result.bo.items()}
    np.savez_compressed(
        path,
        mods=np.asarray(sorted(result.bo), dtype=np.int32),
        beatmap_id=np.int64(result.beatmap_id if result.beatmap_id is not None else -1),
        file_md5=np.str_(result.file_md5),
        legacy=np.bool_(legacy),
        **arrays,
    )
    return path


def save_csv(result: ProcessedFile, out_dir: str) -> str:
    """Headerless CSV, one row per (mods, circle): mods, then the 7 BO values."""
    path = os.path.join(out_dir, result.output_stem + ".csv")
    with open(path, "w", newline="") as fh:
        for m in sorted(result.bo):
            a = result.bo[m]
            if len(a) == 0:
                continue
            block = np.column_stack([np.full(len(a), m, dtype=np.float64), a.astype(np.float64)])
            np.savetxt(fh, block, delimiter=",", fmt=["%d"] + ["%.9g"] * 7)
    return path


def load_npz(path: str) -> dict[int, np.ndarray]:
    with np.load(path) as z:
        return {int(m): z[f"mods_{int(m)}"] for m in z["mods"]}


# --------------------------------------------------------------------------- CLI
def _read_list_file(path: str) -> list[str]:
    """A .txt input is a list of file paths, one per line (e.g. failed_files.txt)."""
    with open(path, encoding="utf-8-sig") as fh:
        return [ln.strip().strip('"') for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def write_failed_list(path: str, failed: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(f + "\n" for f in failed))
    print(f"{len(failed)} failed files listed in {path} (pass it as an input to retry just those)")


def _collect_inputs(inputs: list[str], recursive: bool) -> list[str]:
    files = []
    for p in inputs:
        if p.lower().endswith(".txt") and os.path.isfile(p):
            files += _read_list_file(p)
        elif os.path.isdir(p):
            if recursive:
                for root, _, names in os.walk(p):
                    files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".osu")]
            else:
                files += [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.lower().endswith(".osu")]
        else:
            files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compute taiko BO values from .osu files.")
    ap.add_argument("inputs", nargs="+",
                    help=".osu files, folders containing them, and/or .txt lists of paths")
    ap.add_argument("--db", default="beatmaps.sqlite", help="SQLite metadata database (created if missing)")
    ap.add_argument("--out_dir", default="bo_values", help="where BO files are written")
    ap.add_argument("--format", choices=["npz", "csv", "both"], default="npz")
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the Mathematica reference code exactly (incl. its quirks)")
    ap.add_argument("--bid", type=int, default=None,
                    help="BeatmapID override (only with a single input file)")
    ap.add_argument("--mods", type=int, nargs="*", default=None,
                    help="only these mod values (default: all 36 combinations)")
    ap.add_argument("--recursive", action="store_true", help="search folders recursively")
    ap.add_argument("--failed_list", default="failed_files.txt",
                    help="where to list files that failed (written only if any fail)")
    args = ap.parse_args(argv)

    files = _collect_inputs(args.inputs, args.recursive)
    if args.bid is not None and len(files) != 1:
        ap.error("--bid can only be used with exactly one .osu file")
    if args.mods:
        for m in args.mods:
            validate_mods(m)
    os.makedirs(args.out_dir, exist_ok=True)

    ok = skipped = failed = 0
    failed_files = []
    with BeatmapDB(args.db) as db:
        for f in files:
            try:
                res = process_osu_file(f, db, args.bid, args.legacy, args.mods)
            except NotTaikoError as e:
                print(f"SKIP  {e}")
                skipped += 1
                continue
            except Exception as e:  # keep going on bad files
                print(f"FAIL  {f}: {type(e).__name__}: {e}")
                failed += 1
                failed_files.append(f)
                continue
            if args.format in ("npz", "both"):
                save_npz(res, args.out_dir, args.legacy)
            if args.format in ("csv", "both"):
                save_csv(res, args.out_dir)
            n = len(next(iter(res.bo.values()))) if res.bo else 0
            bid_txt = res.beatmap_id if res.beatmap_id is not None else "None (md5 " + res.file_md5 + ")"
            print(f"OK    {os.path.basename(f)}: BID {bid_txt}, {n} notes, {len(res.bo)} mod combos")
            ok += 1
    print(f"done: {ok} processed, {skipped} not taiko, {failed} failed")
    if failed_files:
        write_failed_list(args.failed_list, failed_files)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
