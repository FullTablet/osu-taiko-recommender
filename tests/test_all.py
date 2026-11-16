"""Self-checks for the osu!taiko recommender app and builders.

    python tests/test_all.py

Needs numpy; the builder / NumPy-vs-PyTorch checks also need torch and the
sibling difficulty_nn + taiko_pipeline folders; the GUI check needs PySide6.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent           # taiko_app
CLAUDE = ROOT.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "builder"))
sys.path.insert(0, str(CLAUDE / "taiko_pipeline"))
sys.path.insert(0, str(CLAUDE / "difficulty_nn"))

FAILS = []
OSU_DIR = Path(os.environ.get("TAIKO_TEST_OSU_DIR", ""))


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------------------ unit checks
def test_quantiles():
    from taiko_core.quantiles import QuantileTable

    rng = np.random.default_rng(0)
    samples = np.stack([rng.normal(0, 1, 200_000), rng.gamma(2, 1, 200_000), rng.uniform(-3, 5, 200_000),
                        rng.normal(5, 0.1, 200_000), rng.standard_t(3, 200_000)], axis=1)
    qt = QuantileTable.from_samples(samples, keypoints=1000, tail=0.01)
    qt = QuantileTable.from_json(qt.to_json())
    ok = True
    for d in range(5):
        qs = np.linspace(-0.5, 1.5, 4001)
        v = qt.value(d, qs)
        back = qt.quantile(d, v)
        ok &= np.allclose(back, qs, atol=1e-6) and np.all(np.diff(v) > 0)
    check(ok, "quantile table: reverse(direct(q)) == q on [-0.5, 1.5], strictly increasing")
    eps = 1e-9
    cont = all(abs(qt.value(d, 1 + eps) - qt.value(d, 1 - eps)) < 1e-4 and
               abs(qt.value(d, -eps) - qt.value(d, eps)) < 1e-4 and
               abs(qt.quantile(d, qt.values[d, -1] + eps) - 1) < 1e-6 and abs(qt.quantile(d, qt.values[d, 0] - eps)) < 1e-6
               for d in range(5))
    check(cont, "quantile table: continuous at quantile 0 and 1")
    d = 0
    s_hi = (qt.values[d, -1] - np.quantile(samples[:, d], 0.99)) / 0.01
    check(abs(qt.value(d, 1.1) - (qt.values[d, -1] + 0.1 * s_hi)) < 1e-9,
          "quantile table: linear continuation with the top-1% secant slope")
    check(abs(qt.value(0, 0.5) - np.median(samples[:, 0])) < 0.01 and abs(qt.quantile(0, 0.0) - 0.5) < 0.01,
          "quantile table: median of a normal sample")
    ties = QuantileTable.from_samples(np.r_[np.zeros(500), np.arange(1, 501)][:, None].astype(float), 101)
    check(abs(ties.quantile(0, 0.0) - 0.25) < 0.02, "quantile table: a value in a run of ties maps to the run's middle")


def test_skill():
    from taiko_core.quantiles import QuantileTable
    from taiko_core.skill import (PlayData, estimate_skill, expected_accuracy, quantiles_for_accuracy,
                                  rank_to_quantile, recency_weights)

    rng = np.random.default_rng(1)
    true = np.array([1.0, -0.5, 0.3, 2.0, 0.0])
    plays = []
    for _ in range(300):
        n = rng.integers(100, 600)
        slope = np.abs(rng.normal(0.6, 0.3, (n, 5)))
        bias = rng.normal(0, 1.5, (n, 5)) + rng.normal(0, 1, 5)
        plays.append(PlayData(slope, bias, expected_accuracy(slope, bias, true)))
    est, info = estimate_skill(plays, prior_mean=np.zeros(5), prior_precision=0.0)
    # the estimate must be the maximum of the recency-weighted likelihood ...
    from taiko_core.skill import sigmoid
    w = recency_weights(len(plays))
    grad = sum(wi * (p.slope.T @ (p.accuracy - sigmoid(p.slope @ est - (p.slope * p.bias).sum(1))))
               for wi, p in zip(w, plays))
    scale = sum(wi * len(p.slope) for wi, p in zip(w, plays))
    check(np.abs(grad).max() / scale < 1e-8, f"skill estimate is the likelihood maximum ({info['iterations']} Newton steps)")
    # ... and reproduce the observed accuracies about as well as the true skill does
    err_est = np.mean([abs(expected_accuracy(p.slope, p.bias, est) - p.accuracy) for p in plays])
    check(err_est < 0.01, f"estimated skill reproduces the play accuracies (mean error {err_est * 100:.2f}%)")
    perfect = [PlayData(p.slope, p.bias, 1.0) for p in plays[:3]]
    est2, _ = estimate_skill(perfect, prior_mean=np.zeros(5))
    check(np.all(np.isfinite(est2)), "skill estimate stays finite when every play is 100%")
    w = recency_weights(4)
    check(np.allclose(w, [0.99 ** 3, 0.99 ** 2, 0.99, 1.0]), "recency weights 1, 0.99, 0.99^2 ... (newest = 1)")
    many = [plays[0]] * 1200
    _, info3 = estimate_skill(many, prior_mean=np.zeros(5))
    check(info3["plays"] == 1000, "only the 1000 most recent plays are used")

    check(abs(rank_to_quantile(1) - 1.0) < 1e-12 and abs(rank_to_quantile(10_000) - 0.3) < 1e-12 and
          abs(rank_to_quantile(100_000) - 0.01) < 1e-12, "pp rank -> quantile through (1, 1), (10k, 0.3), (100k, 0.01)")
    check(rank_to_quantile(1_000_000) < 0.01 and rank_to_quantile(1000) > 0.3, "pp rank extrapolates beyond 100k")

    samples = rng.normal(0, 1, (50_000, 5))
    qt = QuantileTable.from_samples(samples, 1000)
    slope, bias = plays[0].slope, plays[0].bias
    for direction in ([1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [0, 1, 0, 0, 0]):
        q = quantiles_for_accuracy(slope, bias, qt, direction, 0.9)
        acc = expected_accuracy(slope, bias, qt.values_at(q))
        ratio_ok = np.allclose(q / max(q.max(), 1e-12) * max(direction), direction) if q.max() > 0 else True
        check(q is not None and abs(acc - 0.9) < 1e-5 and ratio_ok,
              f"quantiles for 90% along {direction}: {np.round(q, 3).tolist()} -> {acc * 100:.3f}%")
    easy = PlayData(slope, bias - 20, 1.0)
    q_easy = quantiles_for_accuracy(easy.slope, easy.bias, qt, [1, 1, 1, 1, 1], 0.6)
    check(q_easy is not None and q_easy[0] < 0, "a very easy chart needs quantiles below 0 (extrapolated)")


def test_recommend_math():
    from taiko_core import recommend as R
    from taiko_core.quantiles import QuantileTable

    rng = np.random.default_rng(0)
    qt = QuantileTable.from_samples(rng.normal(0, 1, (200_000, 5)), 1000)

    def chart(bias_mean, n=400, seed=0, dims=None):
        r = np.random.default_rng(seed)
        slope = np.abs(r.normal(1.0, 0.3, (n, 5)))
        if dims is not None:
            slope[:, [k for k in range(5) if k not in dims]] *= 1e-3
        return R.ChartModel(slope, r.normal(bias_mean, 0.5, (n, 5)), qt)

    def brute_x(cm, target, k=None):                    # plain bisection on the user's paths
        lo, hi = -1e6, 50.0
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            q = np.full(5, R.softplus(mid) + mid / 1e4)
            if k is not None:
                q[k] = 2 * R.softplus(mid) + mid / 1e4
            if cm.accuracy_at_quantiles(q) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # 1. agreement with a direct evaluation of the formulas on a normal chart
    cm = chart(-0.5)
    ok = True
    for target in (0.9, 0.97):
        xc_b = brute_x(cm, target)
        for k in range(5):
            xp_b = brute_x(cm, target, k)
            want = R.softplus(xc_b) / R.softplus(xp_b) - 1
            got = R.priority_from_x(R.solve_x(cm, target), R.solve_x(cm, target, k))
            ok &= abs(got - want) < 1e-7 * max(1, want)
            q = np.full(5, R.path_a(R.solve_x(cm, target, k)))
            q[k] = R.path_b(R.solve_x(cm, target, k))
            ok &= abs(cm.accuracy_at_quantiles(q) - target) < 1e-10
    check(ok, "priority = softplus(x_c)/softplus(x_p) - 1 matches direct bisection; roots hit the target")

    # 2. the counterexample charts: never negative (was -0.85 .. -1.06 and -2e-14 with a(x) in the powers)
    easy = chart(-6.0)
    worst = min(R.priority_from_x(R.solve_x(easy, t), R.solve_x(easy, t, k))
                for t in (0.9, 0.95, 0.97, 0.99885, 0.9989, 0.999) for k in range(5))
    check(worst >= R.PRIORITY_FLOOR, f"easy chart: every priority >= floor (min {worst:.3g})")
    ultra = chart(-60.0)                                  # x underflows: softplus(x) = 0 in float64
    pr = [R.priority_from_x(R.solve_x(ultra, 0.97), R.solve_x(ultra, 0.97, k)) for k in range(5)]
    ev = R.evaluate(ultra, qt.values_at(0.3), 0.97, 0)
    check(all(np.isfinite(pr)) and min(pr) >= R.PRIORITY_FLOOR and np.isfinite(ev["fiterror"]) and ev["fiterror"] > 0,
          "ultra-easy chart (softplus underflow): finite, positive priorities and fit error")

    # 2b. termination / below-quantile-0 searches / saturated accuracies
    calls = {"n": 0}
    orig = R.ChartModel.accuracy_at_quantiles

    def counting(self, q):
        calls["n"] += 1
        return orig(self, q)

    R.ChartModel.accuracy_at_quantiles = counting
    try:
        xs = R.solve_x(ultra, 0.97)
        below = R.path_a(xs) < 0 and abs(ultra.accuracy_at_quantiles(np.full(5, R.path_a(xs))) - 0.97) < 1e-9
        calls["n"] = 0
        degenerate = R.ChartModel(np.full((50, 5), 1e-9), np.zeros((50, 5)), qt)      # ~50% at any skill
        x_deg = R.solve_x(degenerate, 0.97)
        n_deg = calls["n"]
    finally:
        R.ChartModel.accuracy_at_quantiles = orig
    check(below, "ultra-easy chart: the search goes below quantile 0 and hits the target exactly")
    check(x_deg == 1e4 and n_deg < 60, f"unreachable target: stops at the bracket limit ({n_deg} evaluations)")
    hp1 = R.hapower(ultra, 1.0)
    hp_easier = R.hapower(chart(-80.0), 1.0)
    check(np.isfinite(hp1) and hp_easier < hp1 and
          abs(ultra.accuracy_at_quantiles(np.full(5, hp1)) - (1 - R.HA_CLAMP)) < 1e-13,
          "ha rounding to exactly 100%: hapower well defined (accuracy 1 - 1e-12) and lower for an easier chart")

    # 3. hapower: the balanced quantile reproducing ha
    skill = qt.values_at([0.8, 0.2, 0.5, 0.6, 0.4])
    ha = cm.accuracy_at_skill(skill)
    hp = R.hapower(cm, ha)
    check(abs(cm.accuracy_at_quantiles(np.full(5, hp)) - ha) < 1e-10, f"hapower {hp:.4f} reproduces ha {ha * 100:.2f}%")

    # 4. a chart that only uses dimension 0 has a large alpha priority, ~0 for the others
    only0 = chart(-0.5, dims=[0])
    pr0 = R.all_priorities(only0, 0.97)
    check(pr0[0] > 0.3 and max(pr0[1:]) < 0.01, f"alpha-only chart: priorities {[round(p, 4) for p in pr0]}")
    reuse = R.all_priorities(only0, 0.97, known={0: 123.0})
    check(reuse[0] == 123.0, "all_priorities reuses an already computed priority")

    # 5. fit error: inside the window beats outside; lower hapower wins inside
    target = 0.97
    f_in = R.softplus(0.9) + R.window_penalty(0.975, target)
    f_out = R.softplus(-5.0) + R.window_penalty(0.9805, target)
    check(f_in < f_out and R.window_penalty(0.98, target) == 0 and R.window_penalty(0.96, target) == 0,
          "fit error: window target ± 1% (inclusive) dominates hapower")


def test_chartdb(tmp: Path):
    from taiko_core.chartdb import ChartDB, IncompatibleDatabase

    rng = np.random.default_rng(2)
    slope = np.abs(rng.normal(1, 0.5, (700, 5))).astype(np.float32)
    bias = rng.normal(0, 2, (700, 5)).astype(np.float32)
    for bits, tol in ((8, 1 / 510), (16, 1 / 131070)):
        db = ChartDB(tmp / f"c{bits}.sqlite", create_meta={"kind": "complementary", "model_id": "m1", "bits": bits})
        cid = db.add_chart({"file_md5": "a" * 32, "artist": "Art", "title": "Tit", "version": "Oni"}, {0: (slope, bias)}, b"osu")
        s2, b2 = db.variant(cid, 0)
        x = np.concatenate([slope, bias], 1)
        rngs = x.max(0) - x.min(0)
        err = np.abs(np.concatenate([s2, b2], 1) - x) / rngs
        check(err.max() <= tol * 1.01, f"{bits}-bit storage error within (max-min)/{2 * (2**bits - 1)}")
        db.close()
    db = ChartDB(tmp / "c8.sqlite")
    check(db.search("tit oni")[0]["title"] == "Tit" and db.search("a" * 32) and db.osu_bytes(1) == b"osu",
          "chart search by words and by file hash; stored .osu")
    other = ChartDB(tmp / "o.sqlite", create_meta={"kind": "complementary", "model_id": "m1"})
    other.add_chart({"file_md5": "a" * 32}, {0: (slope, bias)})
    other.add_chart({"file_md5": "b" * 32}, {0: (slope, bias), 8: (slope, bias)})
    other.close()
    added, skipped = db.merge_from(tmp / "o.sqlite")
    check((added, skipped) == (1, 1) and db.count() == 2 and db.mods_of(db.find(file_md5="b" * 32)[0]) == [0, 8],
          "merging a database appends new charts and skips duplicate file hashes")
    bad = ChartDB(tmp / "bad.sqlite", create_meta={"kind": "complementary", "model_id": "other"})
    bad.close()
    try:
        db.merge_from(tmp / "bad.sqlite")
        check(False, "merge refuses a database from another model")
    except IncompatibleDatabase:
        check(True, "merge refuses a database from another model")
    blob = db.conn.execute("SELECT length(data) FROM variants WHERE chart_id = 1").fetchone()[0]
    check(blob <= 700 * 10, f"8-bit storage: {blob:,} bytes for 700 notes x 10 values (<= 1 byte each)")


def test_mods():
    from taiko_core.mods import BASE_MODS, DEFAULT_IMPORT_MODS, mods_label, parse_mods_label

    check([mods_label(m) for m in BASE_MODS] == ["NM", "HD", "HR", "DT", "HDHR", "HDDT", "HT", "EZ"],
          "base mods: NM HD HR DT HDHR HDDT HT EZ")
    check(len(DEFAULT_IMPORT_MODS) == 18 and all(not m & 1024 for m in DEFAULT_IMPORT_MODS),
          "default import selection: the 18 combinations without FL")
    check(parse_mods_label("HD+NC") == 72 and parse_mods_label("nm") == 0, "mod label parsing")


# ------------------------------------------------------------------------------ torch-based checks
def test_notenet_vs_torch():
    import torch
    from model import DifficultyModel, ModelConfig
    from taiko_core.notenet import NoteNet

    torch.manual_seed(0)
    cfg = ModelConfig(num_players=3, aa_dim=32, num_layers=3, num_heads=4, ff_mult=2, window=8)
    model = DifficultyModel(cfg).eval()
    tmp = Path(tempfile.mkdtemp())
    params = {k[len("object_subnet."):]: v.numpy() for k, v in model.state_dict().items() if k.startswith("object_subnet.")}
    conf = dict(window=8, num_heads=4, num_layers=3, skill_dim=5, aa_dim=32, object_feature_dim=38)
    np.savez(tmp / "n.npz", config_json=np.array(json.dumps(conf)), **params)
    net = NoteNet(tmp / "n.npz")
    worst = 0
    with torch.no_grad():
        for n in (1, 7, 8, 9, 31, 100):
            x = np.random.default_rng(n).normal(size=(3, n, 38)).astype(np.float32)
            o = model.object_subnet(torch.from_numpy(x))
            s, b = net.forward(x)
            worst = max(worst, np.abs(o["slope"].numpy() - s).max(), np.abs(o["bias"].numpy() - b).max())
    shutil.rmtree(tmp)
    check(worst < 1e-4, f"NumPy note subnet == PyTorch (max diff {worst:.1e}; lengths around the window size)")


def write_dump(dump: Path, bids):
    dump.mkdir(parents=True, exist_ok=True)
    rows = ",".join(f"({b},{100 + b},'Diff (it\\'s) {b}',120,110,5,{4 + b % 3},180.5,1,1,'md5_{b}')" for b in bids)
    (dump / "osu_beatmaps.sql").write_text(
        "CREATE TABLE `osu_beatmaps` (\n  `beatmap_id` int,\n  `beatmapset_id` int,\n  `version` varchar(80),\n"
        "  `total_length` int,\n  `hit_length` int,\n  `diff_drain` float,\n  `diff_overall` float,\n  `bpm` float,\n"
        "  `playmode` int,\n  `approved` int,\n  `checksum` varchar(32),\n  PRIMARY KEY (`beatmap_id`)\n);\n"
        f"INSERT INTO `osu_beatmaps` VALUES {rows};\n", encoding="utf-8")
    srows = ",".join(f"({100 + b},'Artist {b}','Title {b}','Mapper','tag',_binary 'x\\0y',NULL,NULL,'')" for b in bids)
    (dump / "osu_beatmapsets.sql").write_text(
        "CREATE TABLE `osu_beatmapsets` (\n  `beatmapset_id` int,\n  `artist` varchar(80),\n  `title` varchar(80),\n"
        "  `creator` varchar(80),\n  `tags` varchar(80),\n  `body_hash` binary(16),\n  `artist_unicode` varchar(80),\n"
        "  `title_unicode` varchar(80),\n  `source` varchar(80)\n);\n"
        f"INSERT INTO `osu_beatmapsets` VALUES {srows};\n",
        encoding="utf-8")


def test_end_to_end(tmp: Path):
    import test_build_training_data as T
    from taiko_core.mods import BASE_MODS

    T.MODS = sorted(set(BASE_MODS))                    # synthetic congregal DB with the base mods
    d = tmp / "inputs"
    d.mkdir()
    inp = T.make_inputs(d)
    ckpt, _ = T.make_checkpoint(d / "old_ckpt.pt", CLAUDE / "difficulty_nn")
    import build_training_data as BTD

    BTD.main(["--plays_db", str(d / "plays.sqlite"), "--congregal_db", str(d / "congregals.sqlite"),
              "--old_data_dir", str(inp["old"]), "--out_dir", str(d / "dataset"), "--old_checkpoint", str(ckpt)])
    write_dump(d / "dump", range(1, 12))
    import make_release as MR
    import release_all as RA

    dist = tmp / "dist"
    RA.DATASET_DIR = d / "dataset"
    RA.CHECKPOINT = d / "dataset" / "checkpoint_last.pt"
    RA.CONGREGAL_DB = d / "congregals.sqlite"
    RA.PLAYS_DB = d / "plays.sqlite"
    RA.DUMP_DIR = d / "dump"
    RA.DIST = dist
    RA.REPO = "someone/otr"
    RA.main(["--app_version", "0.1.0", "--data_version", "2026.11.1", "--tag", "v1"])
    snap = json.loads((dist / "work" / "checkpoint_snapshot.json").read_text())
    check((dist / "work" / "checkpoint_snapshot.pt").exists() and snap["step"] == 1234,
          "release_all: checkpoint snapshot taken and checked (step 1234)")
    check((dist / "expansion" / "note_net.npz").exists(), "release_all: expansion pack built and verified against PyTorch")
    try:
        old_meta = tmp / "old_meta"
        old_meta.mkdir()
        (old_meta / "meta.json").write_text(json.dumps({"num_charts": 1}))
        import build_common as BC
        BC.load_dataset_meta(old_meta)
        check(False, "builders refuse the old dataset format")
    except SystemExit as e:
        check("old dataset" in str(e), "builders refuse the old dataset format (no strainer_formula / time axis)")
    bm = json.loads((dist / "base" / "manifest.json").read_text())
    em = json.loads((dist / "expansion" / "manifest.json").read_text())
    check(bm["model_id"] == em["model_id"], "base and expansion packages carry the same model id")
    from taiko_core.chartdb import ChartDB

    cdb = ChartDB(dist / "base" / "charts.sqlite", readonly=True)
    info = cdb.chart(cdb.find(beatmap_id=4)[0])
    check(cdb.count() == 8 and info["title"] == "Title 4" and info["version"] == "Diff (it's) 4" and
          cdb.mods_of(info["chart_id"]) == sorted(BASE_MODS) and info["status"] == 1,
          "base charts: the 8 eligible beatmaps with metadata and the 8 base mod combinations")
    cdb.close()
    qj = json.loads((dist / "base" / "skill_quantiles.json").read_text())
    check(len(qj["values"]) == 5 and len(qj["values"][0]) == 1000 and qj["meta"]["n_samples"] > 100,
          "skill quantile table: 5 dimensions x 1000 keypoints from the plays")

    rel_full = dist / "release"
    notes = (rel_full / "release_notes.md").read_text(encoding="utf-8")
    check("chart data and statistics derived from osu! data (data.ppy.sh)" in notes and "Install.bat" in notes,
          "release: release notes with install steps and the data credit")
    with zipfile.ZipFile(rel_full / "osu-taiko-recommender-0.1.0.zip") as z:
        names = z.namelist()
        cfg = json.loads(z.read("osu!taiko recommender/app/config.json"))
        scripts = [z.read(n) for n in names if n.lower().endswith((".bat", ".ps1"))]
    check(len(scripts) >= 2 and all(b.count(b"\n") == b.count(b"\r\n") > 0 for b in scripts),
          "release: Windows scripts in the app zip have CRLF line endings")
    check("osu!taiko recommender/Install.bat" in names and "osu!taiko recommender/app/main.py" in names and
          cfg["github_repo"] == "someone/otr" and not any("__pycache__" in n for n in names),
          "release: app zip 'osu!taiko recommender/' with Install.bat and the configured repo")
    MR.main(["--repo", "someone/otr", "--tag", "v1", "--app_version", "0.1.0", "--base", str(dist / "base"),
             "--expansion", str(dist / "expansion"), "--out", str(dist / "release_split"), "--part_mb", "1"])
    rel = dist / "release_split"
    versions = json.loads((rel / "versions.json").read_text())
    check(len(versions["base"]["parts"]) >= 1 and versions["base"]["parts"][0]["url"].startswith(
        "https://github.com/someone/otr/releases/download/v1/"), "release: versions.json with download URLs (split parts)")
    return d, rel, versions


def join_parts(rel: Path, entry) -> Path:
    out = rel / "joined.zip"
    with open(out, "wb") as f:
        for p in entry["parts"]:
            f.write((rel / p["url"].rsplit("/", 1)[1]).read_bytes())
    return out


def test_app_state(tmp: Path, rel: Path, versions: dict):
    os.environ["OTR_DATA_DIR"] = str(tmp / "appdata")
    from taiko_core import packages
    from taiko_core.app_state import AppState, ChartRef
    from taiko_core.importer import ExpansionModel, import_files

    for kind in ("base", "expansion"):
        packages.install_zip(join_parts(rel, versions[kind]), kind)
    state = AppState()
    check(state.has_base and state.has_expansion and state.user.current_skill() is None,
          "app: packages installed, no skill estimate yet (the GUI asks for the pp rank)")
    s0 = state.set_skill_from_rank(10_000)
    check(np.allclose(state.current_skill()["quantiles"], 0.3, atol=1e-6), "rank 10,000 -> every skill at quantile 0.3")
    res = state.recommend({"K": 30, "S": 5, "priority": 1, "target": 0.97, "mods": 0})
    recs = res["charts"]
    fits = [r["fiterror"] for r in recs]
    check(len(recs) == 5 and all(0 <= r["expected"] <= 1 for r in recs) and fits == sorted(fits) and
          all(len(r["priorities"]) == 5 and min(r["priorities"]) > 0 for r in recs) and
          all(r["mods"] == 0 for r in recs), "recommendations: S lowest fit errors, NM only, 5 priorities each")
    hd = state.recommend({"K": 1000, "S": 3, "priority": None, "target": 0.9, "mods": 72})
    check(hd["candidates"] == hd["available"] and all(r["mods_label"] == "HDDT" for r in hd["charts"]),
          "recommendations: HD+DT searches only HDDT charts (K capped at what's available)")
    state.save_recommendation_settings({"K": 777, "S": 7, "priority": 2, "target": 0.955, "mods": 24})
    st2 = AppState()
    check(st2.recommendation_settings() == {"K": 777, "S": 7, "priority": 2, "target": 0.955, "mods": 24},
          "recommendation settings are remembered between sessions")
    st2.save_recommendation_settings({"K": 1000, "S": 10, "priority": None, "target": 0.97, "mods": 0})
    st2.close()
    for bad in ({"target": 1.0}, {"target": 0.0}, {"mods": 18}):
        try:
            state.save_recommendation_settings(bad)
            check(False, f"invalid settings rejected: {bad}")
        except ValueError:
            pass
    state.record_plays([(r["ref"], r["mods"], 0.97) for r in recs])
    info = state.update_skill()
    check(info["plays"] == 5 and not np.allclose(info["skill"], s0), "skill updated from the recorded plays")
    order = [p["label"] for p in state.user.recent_plays(10)]
    check(order == [r["label"] for r in recs], "plays are stored in table order (last row = most recent)")

    # importing real .osu / .osz files with the expansion pack
    taiko = OSU_DIR / "3343408.osu"
    mania = OSU_DIR / "1159137.osu"
    if taiko.exists() and mania.exists():
        std = tmp / "std.osu"
        std.write_bytes(taiko.read_bytes().replace(b"Mode: 1", b"Mode: 0"))
        osz_mixed = tmp / "mixed.osz"
        with zipfile.ZipFile(osz_mixed, "w") as z:
            z.write(taiko, "a.osu")
            z.write(std, "b.osu")
        osz_std = tmp / "std_only.osz"
        with zipfile.ZipFile(osz_std, "w") as z:
            z.write(std, "b.osu")
        db = state.ensure_comp()
        model = state.model()
        summ = import_files([taiko, mania, std, osz_mixed, osz_std], db, model, [0, 8, 64])
        msgs = {r.source: (r.status, r.message) for r in summ.results}
        check(summ.added == 1 and db.count() == 1, "import: the taiko chart is added once")
        check(msgs.get("1159137.osu", ("", ""))[1] == "this is not a taiko chart", "import: a mania chart is rejected")
        check(msgs.get("std.osu", ("", ""))[1] == "autoconverts are not implemented yet, sorry!",
              "import: an osu!standard chart gets the autoconvert message")
        check(msgs.get("std_only.osz", ("", ""))[1] == "autoconverts are not implemented yet, sorry!" and
              "mixed.osz" not in msgs and msgs.get("mixed.osz / a.osu", ("", ""))[0] == "duplicate",
              "import: .osz messages only when no taiko chart inside; duplicates by file hash skipped")
        state.reload()
        comp_refs = [r for r, i in state.search("chaosmaid") if r.source == "comp"]
        check(state.comp_valid and comp_refs and state.db("comp").mods_of(comp_refs[0].chart_id) == [0, 8, 64],
              "imported chart is searchable and usable (same model id)")
        state.record_plays([(comp_refs[0], 64, 0.9)])
        info2 = state.update_skill()
        check(info2["plays"] == 6 and info2["missing"] == 0, "skill estimation uses imported charts too")
        q = state.quantiles_needed(comp_refs[0], 0, [2, 1, 1, 1, 1], 0.95)
        check(q is not None and abs(q[0] - 2 * q[1]) < 1e-9, "quantiles needed for 95% along [2,1,1,1,1]")
    else:
        print("(skipping .osu import checks: set TAIKO_TEST_OSU_DIR)")
    state.close()
    return tmp / "appdata"


def test_privacy(tmp: Path):
    import sqlite3 as _sq

    import make_release as MR
    import privacy_check as PC

    secret = "SecretUserName"
    at, colon = chr(64), chr(58)                       # built at runtime: the test source itself stays clean
    drive_path = "I" + colon + "\\Some Drive\\folder"
    mail = "someone" + at + "example.com"
    root = tmp / "privacy"
    src = root / "src"
    (src / "app").mkdir(parents=True)
    (src / "app" / "ok.py").write_text("x = 1  # nothing private\n")
    (src / "README.md").write_text("install with <your folder>\\Install.bat\n")
    (src / "dist").mkdir()
    (src / "dist" / "leak.txt").write_text(secret)                        # ignored folder
    check(PC.check(None, [], src, strings=[secret]) == [], "privacy: a clean source tree passes")
    (src / "app" / "bad.py").write_text(f'DATA = r"{drive_path}"\nMAIL = "{mail}"\n')
    (src / "notes.md").write_text(f"made by {secret}\n")
    found = PC.check(None, [], src, strings=[secret])
    check(any("absolute Windows path" in f for f in found) and any("e-mail" in f for f in found) and
          any("notes.md" in f and "private string" in f for f in found) and not any("dist/" in f for f in found),
          "privacy: drive paths, e-mail addresses and private strings in source are found (ignored folders skipped)")

    pkg = root / "pkg"
    pkg.mkdir()
    db = _sq.connect(pkg / "charts.sqlite")
    db.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('built_by', ?)", (f"C:/Users/{secret}/x",))
    db.commit()
    db.close()
    np.savez(pkg / "note_net.npz", config_json=np.array(json.dumps({"path": f"D:/{secret}/data"})))
    (pkg / "manifest.json").write_text(json.dumps({"files": {"charts.sqlite": {}, "note_net.npz": {}}}))
    found = PC.check(None, [pkg], None, strings=[secret])
    check(any("charts.sqlite" in f for f in found) and any("note_net.npz" in f for f in found),
          "privacy: private strings inside the chart database and the weights (.npz, UTF-32) are found")
    rel = root / "rel"
    rel.mkdir()
    with zipfile.ZipFile(rel / "osu-taiko-recommender-0.1.0.zip", "w") as z:
        z.writestr("osu!taiko recommender/app/config.json", json.dumps({"github_repo": "a/b", "note": mail}))
    found = PC.check(rel, [], None, strings=[secret])
    check(any("e-mail" in f and "config.json" in f for f in found), "privacy: the app zip's files are checked too")

    # make_release refuses to print the publish commands
    leaky = root / "leaky_base"
    shutil.copytree(tmp / "dist" / "base", leaky)
    m = json.loads((leaky / "manifest.json").read_text())
    m["note"] = "built in E" + colon + "\\private\\folder"
    (leaky / "manifest.json").write_text(json.dumps(m))
    try:
        MR.main(["--repo", "someone/otr", "--tag", "v1", "--app_version", "0.1.0", "--base", str(leaky),
                 "--out", str(root / "out"), "--skip_source_check"])
        check(False, "privacy: make_release refuses a release with private data")
    except SystemExit as e:
        check(e.code == 1, "privacy: make_release refuses a release with private data (no publish commands)")


def test_git_setup(tmp: Path):
    import subprocess

    import git_setup as G

    if not shutil.which("git"):
        print("(git not installed: git setup check skipped)")
        return
    repo = tmp / "gitrepo"
    (repo / "builder").mkdir(parents=True)
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    shutil.copyfile(ROOT / ".gitattributes", repo / ".gitattributes")
    (repo / "Install.bat").write_bytes(b"@echo off\r\necho hi\r\n")
    (repo / "builder" / "local_paths.json").write_text('{"private_strings": ["SecretFolderName"]}')
    (repo / "README.md").write_text("x")
    (repo / "app").mkdir()
    (repo / "app" / "main.py").write_text("print('hi')\n")
    for f in ("git_hooks.py", "privacy_check.py"):
        shutil.copyfile(ROOT / "builder" / f, repo / "builder" / f)
    old_root, old_id = G.REPO_ROOT, G.noreply_identity
    G.REPO_ROOT = repo
    G.noreply_identity = lambda: ("FullTablet", "123+FullTablet" + chr(64) + "users.noreply.github.com")
    env_keep = {k: os.environ.get(k) for k in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM")}
    os.environ["GIT_CONFIG_GLOBAL"] = str(tmp / "empty_gitconfig")      # ignore the machine's own identity
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        s = G.git_state(repo)
        check(not s["initialized"] and any("not a git repository" in p for p in G.problems(s)),
              "git setup: uninitialized folder reported")
        lines = "\n".join(G.instructions(tag="v9.9.9"))
        check(lines.index("git init -b main") < lines.index("git add -A"),
              "git setup: instructions put git init before git add -A")
        check(G.apply() == 0 and G.problems(G.git_state(repo)) == [], "git setup: --apply makes the repository ready")
        s = G.git_state(repo)
        check(s["email"].endswith(G.NOREPLY) and s["remote"].endswith("osu-taiko-recommender.git"),
              "git setup: no-reply identity and remote set")
        lines = G.instructions(tag="v9.9.9")
        check(not any("git init" in x for x in lines), "git setup: no setup commands once ready")
        r = subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=true", "add", "-A"], check=True,
                           capture_output=True, text=True)
        check("CRLF" not in r.stderr and "LF will be replaced" not in r.stderr,
              "git setup: no line-ending warnings with Windows' autocrlf (.gitattributes)")
        check(not G.git_state(repo)["private_tracked"], "git setup: git add -A leaves local_paths.json out")
        subprocess.run(["git", "-C", str(repo), "add", "-f", "builder/local_paths.json"], check=True, capture_output=True)
        check(any("tracked" in p for p in G.problems(G.git_state(repo))), "git setup: tracked local_paths.json reported")
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "me" + chr(64) + "example.com"], check=True)
        check(any("no-reply" in p for p in G.problems(G.git_state(repo))), "git setup: real e-mail reported")
        real_fix = G.ownership_fix
        G.ownership_fix = lambda root=None: 'git config --global --add safe.directory "X:/repo"'
        try:
            s = G.git_state(repo)
            check(any("dubious ownership" in p for p in G.problems(s))
                  and G.setup_commands(s)[0].startswith("git config --global --add safe.directory"),
                  "git setup: 'dubious ownership' reported with the safe.directory command first")
        finally:
            G.ownership_fix = real_fix
        check(G.ownership_fix(repo) is None, "git setup: own folder isn't reported as dubious")

        # safety hooks
        def gitc(*a, **kw):
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, **kw)

        gitc("config", "user.email", "123+FullTablet" + chr(64) + "users.noreply.github.com")
        check(G.git_state(repo)["hooks"], "git hooks: installed by --apply")
        r = gitc("commit", "-m", "with private file")
        check(r.returncode != 0 and "local_paths.json" in r.stderr and "REFUSED" in r.stderr,
              "git hooks: pre-commit refuses builder/local_paths.json")
        gitc("rm", "--cached", "-q", "builder/local_paths.json")
        (repo / "dist").mkdir()
        (repo / "dist" / "big.bin").write_bytes(b"x")
        gitc("add", "-f", "dist/big.bin")
        r = gitc("commit", "-m", "with dist")
        check(r.returncode != 0 and "dist/big.bin" in r.stderr, "git hooks: pre-commit refuses dist/ even if forced")
        gitc("rm", "--cached", "-q", "dist/big.bin")
        (repo / "notes.txt").write_text("my folder is SecretFolderName\n")
        gitc("add", "notes.txt")
        r = gitc("commit", "-m", "private text")
        check(r.returncode != 0 and "SecretFolderName" in r.stderr, "git hooks: pre-commit refuses private strings")
        gitc("rm", "--cached", "-q", "notes.txt")
        r = gitc("commit", "-m", "clean")
        check(r.returncode == 0, "git hooks: a clean commit goes through" + ("" if r.returncode == 0 else ": " + r.stderr))
        bare = tmp / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        gitc("remote", "set-url", "origin", str(bare))
        r = gitc("push", "-q", "origin", "main")
        check(r.returncode == 0, "git hooks: clean push goes through" + ("" if r.returncode == 0 else ": " + r.stderr))
        gitc("add", "-f", "builder/local_paths.json")
        gitc("commit", "-q", "--no-verify", "-m", "sneak in")
        gitc("rm", "--cached", "-q", "builder/local_paths.json")
        gitc("commit", "-q", "-m", "and remove again")
        r = gitc("push", "-q", "origin", "main")
        check(r.returncode != 0 and "local_paths.json" in r.stderr,
              "git hooks: pre-push refuses a private file committed and deleted in a later commit")
        shutil.move(str(repo / "builder" / "git_hooks.py"), str(tmp / "git_hooks_moved.py"))
        r = gitc("commit", "--allow-empty", "-m", "no source")
        check(r.returncode != 0 and "missing" in r.stderr, "git hooks: commits refused when the source is missing")
        check(any("missing" in p for p in G.problems(G.git_state(repo))), "git setup: missing source files reported")
    finally:
        G.REPO_ROOT, G.noreply_identity = old_root, old_id
        for k, v in env_keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_private_downloads(tmp: Path, rel: Path, versions: dict):
    """Token mode (private repository) against a local fake GitHub: API JSON, asset download
    with Accept: octet-stream, redirect to a storage host that must NOT receive the token."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from taiko_core import packages

    files = {p.name: p.read_bytes() for p in rel.iterdir()}
    names = sorted(files)
    seen = {"storage_auth": [], "api_auth": []}

    class Storage(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["storage_auth"].append(self.headers.get("Authorization"))
            data = files[names[int(self.path.rsplit("/", 1)[1])]]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    storage = HTTPServer(("127.0.0.1", 0), Storage)

    class Api(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["api_auth"].append(self.headers.get("Authorization"))
            if self.headers.get("Authorization") != "Bearer secret":
                self.send_response(404)
                self.end_headers()
                return
            base = f"http://127.0.0.1:{self.server.server_port}"
            assets = [{"name": n, "url": f"{base}/assets/{i}"} for i, n in enumerate(names)]
            if self.path in ("/repos/someone/otr/releases/latest", "/repos/someone/otr/releases/tags/v1"):
                body = json.dumps({"tag_name": "v1", "assets": assets}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/assets/") and self.headers.get("Accept") == "application/octet-stream":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{storage.server_port}/blob/{self.path.rsplit('/', 1)[1]}")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    api = HTTPServer(("127.0.0.1", 0), Api)
    for srv in (storage, api):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    old_base, old_dir = packages.API_BASE, os.environ.get("OTR_DATA_DIR")
    packages.API_BASE = f"http://127.0.0.1:{api.server_port}"
    os.environ["OTR_DATA_DIR"] = str(tmp / "appdata_private")
    os.environ["OTR_GITHUB_TOKEN"] = "secret"
    try:
        remote = packages.fetch_versions("someone/otr")
        check(remote["base"]["sha256"] == versions["base"]["sha256"], "private repo: versions.json through the API")
        m = packages.download_package(remote["base"], "base")
        check(m["package"] == "base" and packages.read_manifest("base") is not None,
              f"private repo: base package downloaded ({len(remote['base']['parts'])} parts) and installed")
        check(seen["storage_auth"] and all(a is None for a in seen["storage_auth"]) and
              all(a == "Bearer secret" for a in seen["api_auth"]),
              "private repo: token sent to the API only, dropped on the redirect to storage")
    finally:
        packages.API_BASE = old_base
        os.environ.pop("OTR_GITHUB_TOKEN", None)
        if old_dir:
            os.environ["OTR_DATA_DIR"] = old_dir
        api.shutdown()
        storage.shutdown()


def test_gui(tmp: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("(PySide6 not installed: GUI check skipped)")
        return
    import time as _time

    from taiko_gui import worker
    from taiko_gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()

    def wait_tasks(timeout=60):
        t0 = _time.time()
        while worker._running and _time.time() - t0 < timeout:
            QCoreApplication.processEvents()
            _time.sleep(0.01)
        QCoreApplication.processEvents()

    rt = win.recommend
    rt.get_recommendations()
    wait_tasks()
    n = rt.table.rowCount()
    check(n > 0, f"GUI: recommendations table filled ({n} rows)")
    rt.mod_boxes[64].setChecked(True)
    rt.mod_boxes[256].setChecked(True)                       # DT and HT are incompatible
    check(not rt.mod_boxes[64].isChecked() and rt.selected_mods() == 256, "GUI: checking HT unchecks DT")
    rt.mod_boxes[256].setChecked(False)
    check(rt.target.minimum() > 0 and rt.target.maximum() < 100, "GUI: target accuracy strictly inside (0%, 100%)")
    before = win.state.user.count_plays()
    check(rt.table.columnCount() == 9 and rt.table.horizontalHeaderItem(3).text() == "α",
          "GUI: priority columns α–ε next to the expected accuracy")
    rt.table.item(0, 8).setText("98.5")
    rt.table.item(min(2, n - 1), 8).setText("91%")
    rt.submit()
    wait_tasks()
    after = win.state.user.count_plays()
    check(after == before + (2 if n > 2 else 1), "GUI: submitting stores the filled rows as plays")
    check(rt.table.rowCount() > 0 and win.state.current_skill()["method"] == "plays",
          "GUI: skill re-estimated and new recommendations shown")
    win.plays.refresh()
    check(win.plays.table.rowCount() == after, "GUI: plays tab lists every play")
    win.charts.refresh()
    win.updates.refresh()
    check("base database: 2026.11.1" in win.updates.installed.text() and
          "data.ppy.sh" in win.updates.installed.text(), "GUI: updates tab shows installed versions and the data credit")
    check(win.windowTitle().startswith("osu!taiko recommender"), "GUI: window title 'osu!taiko recommender'")
    win.close()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="taikoskill_test_"))
    try:
        test_mods()
        test_quantiles()
        test_skill()
        test_recommend_math()
        test_chartdb(tmp)
        test_git_setup(tmp)
        try:
            import torch  # noqa: F401
        except ImportError:
            print("(torch not installed: builder checks skipped)")
        else:
            test_notenet_vs_torch()
            _, rel, versions = test_end_to_end(tmp)
            test_app_state(tmp, rel, versions)
            test_private_downloads(tmp, rel, versions)
            test_privacy(tmp)
            test_gui(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
