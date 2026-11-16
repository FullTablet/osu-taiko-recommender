r"""Build the expansion pack: the note subnet as a NumPy weights file.

    python build_expansion_pack.py --checkpoint <dataset>\checkpoint_last.pt --dataset_dir <dataset> ^
        --version 2026.11.1 --out ..\dist\expansion [--congregal_db congregals.sqlite]

Writes note_net.npz (weights + feature normalization + ids) and manifest.json,
then checks the NumPy forward pass against PyTorch: on random inputs, and on
real charts if --congregal_db is given.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import build_common as C


def export(args):
    import torch

    meta = C.load_dataset_meta(args.dataset_dir)
    model, ckpt = C.load_checkpoint_model(args.checkpoint)
    mean, std = meta["object_feature_mean"], meta["object_feature_std"]
    model_id = C.compute_model_id(ckpt, mean, std)
    cfg = ckpt["config"]
    out = C.prepare_out(args.out, args.overwrite, ["note_net.npz"])
    dtype = np.float16 if args.dtype == "float16" else np.float32
    params = {k: v.detach().cpu().float().numpy().astype(dtype) for k, v in C.subnet_state(ckpt).items()}
    config = dict(window=cfg["window"], num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
                  skill_dim=cfg["skill_dim"], aa_dim=cfg["aa_dim"], object_feature_dim=cfg["object_feature_dim"],
                  feature_mean=list(map(float, mean)), feature_std=list(map(float, std)),
                  model_id=model_id, pipeline_version=C.PIPELINE_VERSION, version=args.version)
    np.savez(out / "note_net.npz", config_json=np.array(json.dumps(config)), **params)
    manifest = C.write_manifest(out, "expansion", args.version, model_id, ["note_net.npz"])
    print(f"wrote {out / 'note_net.npz'} ({(out / 'note_net.npz').stat().st_size / 2**20:.1f} MB), model id {model_id}")

    # ---- verification ---------------------------------------------------------------------
    from taiko_core.notenet import NoteNet

    net = NoteNet(out / "note_net.npz")
    rng = np.random.default_rng(0)
    worst = 0.0
    tests = [rng.normal(size=(2, n, cfg["object_feature_dim"])).astype(np.float32) for n in (1, 63, 64, 65, 300)]
    if args.congregal_db:
        from build_training_data import CongregalReader
        from taiko_core.mods import BASE_MODS

        reader = CongregalReader(args.congregal_db)
        idx = list(reader.index().items())
        for j in rng.choice(len(idx), size=min(args.verify_charts, len(idx)), replace=False):
            md5 = idx[j][1][0]
            feats = reader.chart_features(md5, BASE_MODS, "congregal")
            arr = [((feats[m].astype(np.float64) - mean) / std).astype(np.float32) for m in BASE_MODS if m in feats]
            if arr and len(arr[0]):
                tests.append(np.stack(arr))
    with torch.no_grad():
        for x in tests:
            t_out = model.object_subnet(torch.from_numpy(x))
            s_np, b_np = net.forward(x)
            d = max(np.abs(t_out["slope"].numpy() - s_np).max(), np.abs(t_out["bias"].numpy() - b_np).max())
            worst = max(worst, float(d))
    print(f"NumPy vs PyTorch on {len(tests)} inputs: max |difference| in slope/bias = {worst:.2e}")
    tol = 1e-3 if args.dtype == "float32" else 5e-2
    if worst > tol:
        raise SystemExit(f"error: the NumPy note subnet differs from PyTorch by {worst:.2e} (> {tol})")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_dir", required=True, help="dataset folder (meta.json with the feature normalization)")
    ap.add_argument("--version", required=True, help="package version, e.g. 2026.11.1")
    ap.add_argument("--out", default=str(C.HERE.parent / "dist" / "expansion"))
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="weight precision (float32 matches the base database exactly; float16 halves the size)")
    ap.add_argument("--congregal_db", help="also verify on real charts from this congregal database")
    ap.add_argument("--verify_charts", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    export(ap.parse_args(argv))


if __name__ == "__main__":
    main()
