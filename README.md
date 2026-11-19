# osu!taiko recommender

Estimates your osu!taiko skill in five dimensions (α, β, γ, δ, ε) from the accuracy of your
plays, and recommends charts that expose your weaknesses at the accuracy you're aiming for.
The difficulty model is a neural network trained on the plays of the top 10,000 taiko players.

The five skill dimensions aren't defined by hand: they come out of the training, which looks for
whatever set of skills best predicts the accuracy players actually get. So they don't necessarily
match the categories a person would come up with. From my own testing, this is how I'd describe
them (better descriptions from more skilled players are welcome):

- **α (alpha):** pattern complexity and density.
- **β (beta):** raw speed and stamina.
- **γ (gamma):** finger control at high speeds.
- **δ (delta):** reading, especially slider velocity and rhythm changes.
- **ε (epsilon):** rhythm complexity, keeping accuracy at lower speeds and high OD (overall difficulty).

_chart data and statistics derived from osu! data (data.ppy.sh)_

## Install (Windows 10 or later)

1. Download `osu-taiko-recommender-<version>.zip` from the latest release and unzip it anywhere.
2. Run `Install.bat` in the unzipped folder. It downloads a private copy of Python and the
   libraries the app needs into that folder (nothing is installed or written anywhere else).
3. Start `osu!taiko recommender.bat`. On first start the app downloads the chart database
   (tab "downloads & updates") and asks for your pp rank.

To update, unzip the new version over the old folder and run `Install.bat` again; your data
(the `data` folder) is kept. To uninstall, delete the folder.

---

# For developers

```
taiko_app/
  builder/   scripts run on the training machine (need PyTorch) to make the downloads
             release_all.py runs them all step by step; see RELEASING.md
  app/       the end-user application (its own Python, NumPy + PySide6)
  dist/      builder output (not in the repository)
  tests/     test_all.py: maths, storage, importer, builders end to end, GUI (offscreen)
```

## What the user downloads

| Package | Contents | Needed for |
|---|---|---|
| App (`osu-taiko-recommender-<ver>.zip`) | `Install.bat` + `app/` | everything |
| Base (`taiko_base_<ver>.zip`) | `charts.sqlite`: every ranked + approved native taiko chart (metadata + per-note slope/bias for NM, HD, HR, DT, HDHR, HDDT, HT, EZ, 8-bit quantized); `skill_quantiles.json`: 1,000 keypoints per skill dimension | recommendations, skill estimation |
| Expansion (`taiko_expansion_<ver>.zip`) | `note_net.npz`: the note subnet (no skill subnet) + feature normalization | importing `.osu`/`.osz` files, rarer mod combinations |

Every package and chart database carries a **model id** (a hash of the note-subnet weights, the feature normalization and the feature pipeline version). The app only combines files with the same id:
- An expansion pack from another model is refused.
- Imported charts from another model are set aside until they're recomputed, which works from each chart's stored `.osu` with the matching expansion pack.

## Building a release (your machine)

See **RELEASING.md**. In short:
```
cd builder
python release_all.py --app_version 0.1.0 --data_version 2026.11.1
```
`release_all.py` reads your input paths from `builder/local_paths.json` (private, ignored by git;
start from `local_paths.example.json`) and runs, step by step: input checks, a verified
snapshot of the checkpoint, `build_base_package.py`, `build_expansion_pack.py` and
`make_release.py`. The last one runs `privacy_check.py` over the source and every release file,
then prints the `gh` commands to publish the release.
Before the first release, set up the repository once with `python builder\git_setup.py --apply`
(git init, GitHub no-reply identity, remote); RELEASING.md has the details.

**`build_base_package.py`:**
- **Charts:** every beatmap with `eligible = 1` in `taiko_plays.sqlite` (playmode 1, approved 1/2) that has congregal values. Per-note values are computed on the GPU with the checkpoint's note subnet.
- **Skill quantiles:** for each play in `taiko_plays.sqlite`, the player's skill at that moment, on their own time axis from `player.db`, interpolated like the model. Those values become 1,000 keypoints per dimension, with tail slopes over the outer 1%.
- **Report:** `build_report.txt` includes the 8-bit storage error measured on expected accuracies (`--bits 16` doubles the size).

**`build_expansion_pack.py`:** checks the NumPy forward pass against PyTorch on random inputs and on real charts. It stops if they differ by more than 1e-3.

**`make_release.py`:**
- Zips the app and the packages, splitting zips over 1.9 GB into parts (GitHub's per-file limit is 2 GiB), and writes `versions.json` and `release_notes.md`.
- Prints the `gh release create ... --latest` command for everything in `dist\release`. The app reads `https://github.com/<repo>/releases/latest/download/versions.json`, or the GitHub API with `OTR_GITHUB_TOKEN` while the repository is private.
- To publish only a new app version without re-uploading the databases, pass `--previous <old versions.json>` and leave out `--base`/`--expansion`. Their entries keep pointing at the older release.

## The app

**Install:** `Install.bat` downloads the embeddable Python 3.12 into `runtime\` and installs numpy / PySide6 / charset-normalizer there (pip without cache). It creates the launcher `osu!taiko recommender.bat` in the same folder: no shortcuts, nothing outside the installation folder. The app keeps everything it writes in `<installation folder>\data` (packages, `user\user.db`, `user\complementary.sqlite`, `user\settings.json`, `error_log.txt`).

**First start:**
- Without the base database, the app opens **downloads & updates**.
- Otherwise it asks for the player's taiko pp rank. The rank becomes a quantile: piecewise linear in log10(rank) through rank 1 → 1.0, 10,000 → 0.3 and 100,000 → 0.01, continued beyond 100,000. Each skill dimension starts at that quantile.

**Tabs:**
- **recommendations** (main):
  - **Settings, remembered between sessions:**
    - K, charts to search (default 1,000);
    - S, charts to show (default 10);
    - skill priority (none, α alpha, β beta, γ gamma, δ delta, ε epsilon);
    - target accuracy (strictly between 0% and 100%, default 97%);
    - mods: the exact combination. None checked = NM; HD + DT = HDDT; EZ/HR and DT/HT exclude each other.
  - "get recommendations" draws K random charts with that combination from the ranked and imported charts, computes each fit error (`taiko_core/recommend.py`) and shows the S lowest.
  - **Columns:** expected accuracy (grey when outside target ± 1%) and the priority of every skill dimension, α–ε.
  - The player types their accuracies (empty = skipped) and submits. Rows are saved in table order, so the last row becomes the most recent play. The skill is re-estimated and a new list appears.
  - **Fit error:**
    - ha = expected accuracy; hapower = the balanced quantile (same quantile in all dimensions) giving ha.
    - fiterror = softplus(hapower) + 25·10¹² × (distance of ha outside target ± 1%)². So charts inside the window come first, and among them the ones that expose the player's weaknesses (lowest hapower).
  - **With a priority k:**
    - Skill paths a(x) = softplus(x) + x/10⁴ for the other dimensions and b(x) = 2·softplus(x) + x/10⁴ for dimension k.
    - x_c is where the all-a path reaches the target; x_p where the path with k at b(x) does.
    - priority_k = softplus(x_c)/softplus(x_p) − 1, and the fit error is divided by priority_k⁵.
    - The powers use softplus rather than a(x), so the priority is always ≥ 0. With a(x), very easy charts could get a negative priority and jump to the top (shown 2026-11-16).
    - The value runs from about 0% (the skill doesn't matter) to about 100% (the only skill that matters).
  - **Safeguards:**
    - The priority is computed as expm1(log softplus(x_c) − log softplus(x_p)), stable for nearly equal values and when softplus underflows.
    - It's floored at 10⁻⁹, so the division stays finite.
    - Roots are bracketed, with Illinois regula falsi plus a bisection fallback.
  - **Timing:** K = 1,000 takes about 1 s without a priority and about 3 s with one (sandbox CPU, average 600 notes).
- **my plays:** list, add (search base charts by beatmap ID, imported charts by file hash, or by words), change accuracy, delete, recalculate, restart from pp rank.
- **my charts:**
  - Import `.osu`/`.osz` files. All 18 mod combinations without FL are checked by default, and the checkboxes are remembered.
  - Delete charts, recompute after a model change, export your chart database, and import (merge) another one: duplicates by file hash are skipped, and other models are refused.
  - Only native taiko charts are imported. osu!standard charts get "autoconverts are not implemented yet, sorry!" and other modes "this is not a taiko chart". In an `.osz`, these messages appear only if it has no taiko chart.
- **downloads & updates:** installed versions, "check for updates" (only when pressed), download base / expansion (with progress, SHA-256 checks and multi-part joining), install either package from a local zip, and a link to a newer app version. Before installing a base database from a different model, the app warns that imported charts must be imported again or recomputed.

**Maths** (`app/taiko_core`, NumPy only):
- **`skill.py`:**
  - Per-note logit `x = Σ slope_k (s_k − bias_k)`; chart expected accuracy = mean of sigmoid(x). The per-note confidence weight isn't used.
  - **Skill estimate:** maximizes `Σ_plays Σ_notes w·[acc_p·x − softplus(x)]` with a weight per note, `w = 0.99^((m−1)/650)`, where `m = 1` is the last note of the newest play and `m` counts notes backwards through the plays (newest 1,000 plays), minus `150 · Var(z₁, …, z₅)`, the sample variance of the five normal-equivalent scores `z_k = (s_k − median_k)/σ_k` (`σ_k` = interquartile range / 1.349, from the quantile table; weight `VARIANCE_WEIGHT` in `skill.py`), which pulls the estimate towards a balanced skill profile. `z` is affine in the skill, so the loss is convex and its minimum is global; solved by Newton's method (a few steps, well under a second).
  - **Weak prior:** a very weak Gaussian prior (precision 1) around the previous estimate keeps it finite if every play is 100%.
  - **`quantiles_for_accuracy`:** the quantiles `t·direction` from quantile 0 at which a chart's expected accuracy reaches a target, for example `[1,1,1,1,1]` or `[2,1,1,1,1]`. Bisection, extrapolating below 0 and above 1 when needed.
- **`quantiles.py`:** direct (quantile → value) and reverse (value → quantile) lookups, linear between keypoints. Outside [0, 1], the line continues with the secant slope of the outer 1%, anchored at the end keypoint, so both directions are continuous at 0 and 1. Displayed percentiles can go below 0% or above 100%.
- **`notenet.py`:** the note subnet's forward pass (windowed causal attention, GELU, LayerNorm, softplus), matching PyTorch to about 1e-7.
- **`importer.py`:** `.osu` → BO values (fixed parser) → congregal features (intended strainers) → normalization → note subnet → 16-bit storage in the user's chart database. The `.osu` file is kept for later recomputation.
- **Not shown in the GUI yet:**
  - per-note probabilities (`note_probabilities`), kept for a future choke-point graph;
  - `quantiles_needed` in `AppState`.

`app/taiko_core/osu/` holds copies of `osu_reader.py`, `taiko_bo.py`, `beatmap_db.py` and `congregal.py` from `taiko_pipeline` (imports made package-relative), so the app is self-contained. If you change the parser there, copy it here and change `PIPELINE_VERSION` in `importer.py`. That changes the model id, so old imported charts get recomputed.

## Tests

```
python tests\test_all.py
```

Set `TAIKO_TEST_OSU_DIR` to a folder with `3343408.osu` (taiko) and `1159137.osu` (mania) to include the import checks.

- **Without PyTorch:** only the maths and storage checks run.
- **With PyTorch and the sibling `difficulty_nn` / `taiko_pipeline` folders:** also the whole build → release → install → app flow on synthetic data.
- **With PySide6:** also an offscreen GUI run (recommendations, submitting plays, re-estimation).
