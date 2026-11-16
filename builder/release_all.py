r"""Build a complete release, step by step, and print the gh commands to publish it.

    cd <your copy of taiko_app>\builder
    python release_all.py --app_version 0.1.0 --data_version 2026.11.1

Your input paths are read from builder\local_paths.json (private, ignored by git; start
from local_paths.example.json).

Steps (all by default; pick some with --steps, e.g. --steps base,release):
  check      inputs exist, the dataset is the new one, PyTorch works, free disk space
  snapshot   copy checkpoint_last.pt to dist\work\checkpoint_snapshot.pt (safe while
             train.py keeps overwriting it) and check the copy loads and matches the dataset
  base       build_base_package.py   -> dist\base
  expansion  build_expansion_pack.py -> dist\expansion (verified against PyTorch)
  release    make_release.py          -> dist\release, then the gh commands to publish

For a quick trial with a partially trained net, add --limit 200 (only the first 200
charts in the base package) and use a test tag, e.g. --tag v0.0.1-test.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import build_common as C

# ---------------------------------------------------------------------------- your paths
# Read from builder/local_paths.json (private, ignored by git): no local path is ever
# written into the published source code.
_LOCAL = C.HERE / "local_paths.json"


def _load_paths():
    try:
        cfg = json.loads(_LOCAL.read_text(encoding="utf-8"))["paths"]
    except (OSError, ValueError, KeyError):
        return None
    return cfg


_P = _load_paths() or {}
DATASET_DIR = Path(_P["dataset_dir"]) if "dataset_dir" in _P else None
CHECKPOINT = Path(_P["checkpoint"]) if "checkpoint" in _P else None
CONGREGAL_DB = Path(_P["congregal_db"]) if "congregal_db" in _P else None
PLAYS_DB = Path(_P["plays_db"]) if "plays_db" in _P else None
DUMP_DIR = Path(_P["dump_dir"]) if "dump_dir" in _P else None
DIST = C.HERE.parent / "dist"
REPO = "FullTablet/osu-taiko-recommender"
STEPS = ["check", "snapshot", "base", "expansion", "release"]


def banner(i, n, text):
    print(f"\n{'-' * 78}\n[{i}/{n}] {text}\n{'-' * 78}", flush=True)


def run(script: str, *args):
    cmd = [sys.executable, str(C.HERE / script), *map(str, args)]
    t = time.time()
    r = subprocess.run(cmd, cwd=C.HERE)
    if r.returncode:
        raise SystemExit(f"\n{script} failed (exit code {r.returncode}); fix the problem above and rerun "
                         f"with --steps starting from this step.")
    print(f"({script} took {time.time() - t:.0f} s)")


def step_check(args):
    if None in (DATASET_DIR, CHECKPOINT, CONGREGAL_DB, PLAYS_DB, DUMP_DIR):
        raise SystemExit(f"{_LOCAL} is missing or incomplete: copy local_paths.example.json to local_paths.json "
                         "and fill in your paths")
    missing = [p for p in (CHECKPOINT, DATASET_DIR / "meta.json", DATASET_DIR / "player_roster.json",
                           DATASET_DIR / "player.db", CONGREGAL_DB, PLAYS_DB,
                           DUMP_DIR / "osu_beatmaps.sql", DUMP_DIR / "osu_beatmapsets.sql") if not p.exists()]
    if missing:
        raise SystemExit("missing input files:\n  " + "\n  ".join(map(str, missing)))
    meta = C.load_dataset_meta(DATASET_DIR)              # refuses the old dataset format
    print(f"dataset: {meta['num_charts']:,} charts, {meta['total_plays']:,} plays, roster {meta['num_players']:,}")
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not available in this Python; run this with the python you use for train.py")
    print(f"PyTorch {torch.__version__}, CUDA {'available' if torch.cuda.is_available() else 'NOT available (slow)'}")
    DIST.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DIST).free / 2**30
    print(f"free space where dist is: {free:,.1f} GB")
    if free < 10:
        print("WARNING: less than 10 GB free; the base package and its release zip can take several GB")
    import git_setup

    probs = git_setup.problems(git_setup.git_state(), REPO)
    if probs:
        print("NOTE: the git repository isn't ready for publishing yet (building continues):\n  - "
              + "\n  - ".join(probs) + "\n  fix it any time before publishing with: python git_setup.py --apply")
    else:
        print("git repository: ready")


def step_snapshot(args):
    import torch

    work = DIST / "work"
    work.mkdir(parents=True, exist_ok=True)
    snap = work / "checkpoint_snapshot.pt"
    for attempt in range(1, 7):
        shutil.copyfile(CHECKPOINT, snap)
        try:
            ckpt = torch.load(snap, map_location="cpu", weights_only=False)
            break
        except Exception as e:                                          # noqa: BLE001 - caught mid-write
            print(f"  the copy doesn't load ({e.__class__.__name__}); train.py may be writing it. retry in 30 s")
            time.sleep(30)
    else:
        raise SystemExit("couldn't get a complete copy of the checkpoint")
    meta = json.loads((DATASET_DIR / "meta.json").read_text())
    if ckpt["config"]["num_players"] != meta["num_players"]:
        raise SystemExit(f"checkpoint has {ckpt['config']['num_players']} players but the dataset roster has "
                         f"{meta['num_players']}: wrong checkpoint/dataset pair")
    info = {"source": str(CHECKPOINT), "step": ckpt.get("step"), "skill_only_iters": ckpt.get("skill_only_iters"),
            "copied": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work / "checkpoint_snapshot.json").write_text(json.dumps(info, indent=2))
    print(f"snapshot of step {info['step']:,} -> {snap}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app_version", required=True, help="e.g. 0.1.0")
    ap.add_argument("--data_version", required=True, help="version of the base/expansion packages, e.g. 2026.11.1")
    ap.add_argument("--tag", help="release tag (default v<app_version>)")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of " + ",".join(STEPS))
    ap.add_argument("--limit", type=int, help="only the first N charts in the base package (trial runs)")
    ap.add_argument("--previous", help="previous versions.json (when rebuilding only some packages)")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        ap.error(f"unknown step(s) {bad}")
    tag = args.tag or f"v{args.app_version}"
    snap = DIST / "work" / "checkpoint_snapshot.pt"
    t0 = time.time()
    for i, step in enumerate(steps, 1):
        if step == "check":
            banner(i, len(steps), "checking inputs")
            step_check(args)
        elif step == "snapshot":
            banner(i, len(steps), f"copying the checkpoint ({CHECKPOINT.name})")
            step_snapshot(args)
        elif step == "base":
            banner(i, len(steps), "building the base package (charts + skill quantiles)")
            extra = ["--limit", args.limit] if args.limit else []
            run("build_base_package.py", "--checkpoint", snap, "--dataset_dir", DATASET_DIR,
                "--congregal_db", CONGREGAL_DB, "--plays_db", PLAYS_DB, "--dump_dir", DUMP_DIR,
                "--version", args.data_version, "--out", DIST / "base", "--overwrite", *extra)
        elif step == "expansion":
            banner(i, len(steps), "building the expansion pack (note subnet for NumPy)")
            run("build_expansion_pack.py", "--checkpoint", snap, "--dataset_dir", DATASET_DIR,
                "--version", args.data_version, "--out", DIST / "expansion", "--congregal_db", CONGREGAL_DB,
                "--overwrite")
        elif step == "release":
            banner(i, len(steps), "packaging the release")
            extra = ["--previous", args.previous] if args.previous else []
            base = ["--base", DIST / "base"] if (DIST / "base" / "manifest.json").exists() else []
            exp = ["--expansion", DIST / "expansion"] if (DIST / "expansion" / "manifest.json").exists() else []
            run("make_release.py", "--repo", REPO, "--tag", tag, "--app_version", args.app_version,
                "--out", DIST / "release", *base, *exp, *extra)
    print(f"\nall steps done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
