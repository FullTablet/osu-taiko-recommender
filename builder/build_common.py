"""Shared helpers for the builder scripts (run on the training machine, need PyTorch)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "app"
CLAUDE_DIR = HERE.parent.parent
for p in (APP_DIR, CLAUDE_DIR / "difficulty_nn", CLAUDE_DIR / "taiko_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from taiko_core.importer import PIPELINE_VERSION  # noqa: E402
from taiko_core.packages import ATTRIBUTION  # noqa: E402,F401

SUBNET_PREFIX = "object_subnet."


def load_checkpoint_model(path, device="cpu"):
    """-> (DifficultyModel in eval mode, checkpoint dict)."""
    import torch
    from model import DifficultyModel, ModelConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = DifficultyModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt


def subnet_state(ckpt) -> dict:
    return {k[len(SUBNET_PREFIX):]: v for k, v in ckpt["model"].items() if k.startswith(SUBNET_PREFIX)}


def compute_model_id(ckpt, feature_mean, feature_std) -> str:
    """Identifies the note subnet + feature normalization + pipeline: every package and
    chart database made from them carries this id, and the app refuses to mix ids."""
    import numpy as np

    h = hashlib.sha256()
    for k, v in sorted(subnet_state(ckpt).items()):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    h.update(np.asarray(feature_mean, np.float64).tobytes())
    h.update(np.asarray(feature_std, np.float64).tobytes())
    h.update(PIPELINE_VERSION.encode())
    return h.hexdigest()[:16]


def load_dataset_meta(dataset_dir) -> dict:
    meta = json.loads((Path(dataset_dir) / "meta.json").read_text())
    if "strainer_formula" not in meta or meta.get("time_axis") != "per_player":
        raise SystemExit(f"error: {Path(dataset_dir) / 'meta.json'} was not written by the current "
                         "build_training_data.py (no strainer_formula / per-player time axis): this looks like "
                         "the old dataset. Build the packages from the new dataset and its checkpoint.")
    formula = meta["strainer_formula"]
    order = meta.get("feature_column_order", "congregal")
    if formula != "congregal" or order != "congregal":
        raise SystemExit(f"error: the dataset uses strainer formula {formula!r} / order {order!r}; the app "
                         "pipeline computes the intended formula in congregal order. Use a dataset built with "
                         "build_training_data.py defaults.")
    return meta


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(folder: Path, package: str, version: str, model_id: str, files: list[str], extra=None) -> dict:
    m = {"package": package, "version": version, "model_id": model_id, "pipeline_version": PIPELINE_VERSION,
         "created": time.strftime("%Y-%m-%d %H:%M:%S"),
         "files": {f: {"size": (folder / f).stat().st_size, "sha256": sha256_file(folder / f)} for f in files}}
    if extra:
        m.update(extra)
    (folder / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def prepare_out(folder, overwrite: bool, files: list[str]) -> Path:
    folder = Path(folder)
    existing = [f for f in files + ["manifest.json"] if (folder / f).exists()]
    if existing and not overwrite:
        raise SystemExit(f"error: {folder} already has {existing} (pass --overwrite)")
    folder.mkdir(parents=True, exist_ok=True)
    for f in existing:
        (folder / f).unlink()
    return folder
