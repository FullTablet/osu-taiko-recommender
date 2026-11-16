# Releasing osu!taiko recommender (step by step)

The repository holds the source code (this folder). Users download the app and the
data packages from **GitHub releases**, never from the source tree.

## One-time setup

1. Install **git** (https://git-scm.com/download/win) and **GitHub CLI** (https://cli.github.com/).
2. Log in once (opens the browser):
   ```
   gh auth login
   ```
   Choose GitHub.com → HTTPS → "Login with a web browser".
3. Turn this folder into a git repository. **This must be done once, before the first release**;
   otherwise `git add -A` fails with "fatal: not a git repository". The easiest way:
   ```
   cd /d "<your copy of taiko_app>"
   python builder\git_setup.py --apply
   ```
   It runs `git init -b main`, sets your commit identity to your GitHub name and GitHub's private
   no-reply address (read from your `gh` login, so your real e-mail never appears in a commit) and
   adds the `origin` remote. It doesn't commit or push. Run `python builder\git_setup.py` (without
   `--apply`) at any time to see what is set up and which commands are still needed.

   The same by hand:
   ```
   git init -b main
   git config user.name "FullTablet"
   git config user.email "<id>+FullTablet@users.noreply.github.com"
   git remote add origin https://github.com/FullTablet/osu-taiko-recommender.git
   ```
   (the address with your numeric id is at GitHub → Settings → Emails). On that page, also tick
   "Keep my email addresses private" and "Block command line pushes that expose my email".

4. First commit and push:
   ```
   git add -A
   git status
   git commit -m "first version"
   git push -u origin main
   ```
   `git status` must not list `builder/local_paths.json`, `dist/` or `data/` (`.gitignore` keeps
   them out). Check the identity afterwards with `git log -1 --format="%an <%ae>"`.
   If the push is rejected ("fetch first") because the repository on GitHub already has commits
   (a README or license created on the website, for example), merge them in and push again:
   ```
   git pull origin main --allow-unrelated-histories --no-edit -X ours
   git push -u origin main
   ```
   Files that exist only on GitHub are kept; where both sides have the same file, yours wins.

   `.gitattributes` fixes the line endings (LF for everything, CRLF for `.bat`/`.ps1`), so git
   shows no "LF will be replaced by CRLF" warnings. Those warnings were harmless anyway.

## If a commit contains something it shouldn't

`git_setup.py --apply` installs two git hooks that refuse, on `git commit` and on `git push`:
build output (`dist/`, `data/`, `runtime/`), compiled files, `builder/local_paths.json`,
data/package files (`.zip`, `.pt`, `.npz`, `.sqlite`, ...), any file over 20 MB, and text with your
private paths, names or e-mail addresses. They work even if `.gitignore` is missing, and the
push hook checks every commit being pushed, so a file committed and deleted later is still caught.

If a bad commit exists locally but was never pushed, undo it while keeping your files:
```
git reset origin/main
```
(or `git reset HEAD~1` to undo just the last commit), then `git add -A` and commit again.
To also free the disk space the big files took inside `.git`:
```
git reflog expire --expire=now --all
git gc --prune=now
```
Never use `git rm -r .` to "start fresh": it deletes the files from the folder, not only from git.

## Your paths stay private

Your input paths live in `builder\local_paths.json`, which git ignores; the repository only has
`builder\local_paths.example.json` with placeholders. Before printing the publish commands,
`make_release.py` runs `privacy_check.py` over everything that would be published: the
repository source and every release file, including the chart database, weights and app zip.
It looks for:
- every path and string in `local_paths.json` (add more under `private_strings`);
- your Windows user name and home folder;
- in text files, any absolute path or e-mail address.

If it finds anything, it stops and lists it. You can also run it on its own:
`python builder\privacy_check.py`. Run it before every `git push` too.

## Every release

All builder steps run with the same `python` you use for `train.py` (it needs PyTorch).
`train.py` can keep running: the script builds from a verified copy of `checkpoint_last.pt`.

1. Build everything:
   ```
   cd /d "<your copy of taiko_app>\builder"
   python release_all.py --app_version 0.1.0 --data_version 2026.11.1
   ```
   It runs, in order, and stops at the first problem:

   | step | what it does |
   |---|---|
   | check | input files exist, the dataset is the new one, PyTorch/CUDA, free disk space |
   | snapshot | copies `checkpoint_last.pt` to `dist\work\` and checks the copy loads and matches the dataset |
   | base | builds `dist\base` (charts + skill quantiles); read `dist\base\build_report.txt` |
   | expansion | builds `dist\expansion` and checks the NumPy note subnet against PyTorch |
   | release | zips everything into `dist\release`, writes `versions.json` and `release_notes.md`, runs the privacy check, and prints the `gh` commands |

   To redo only some steps: `--steps base,release`. For a quick trial with a partially trained
   net: `--limit 200 --tag v0.0.1-test` (only 200 charts in the base package).

2. Run the commands printed at the end. They are:
   1. `git add -A`, `git commit`, `git push` (the source code of this version). If the one-time
      git setup isn't done yet, the printout lists its commands first (the `check` step also
      warns about it at the start);
   2. one `gh release create <tag> ... --latest` uploading every file in `dist\release`;
   3. `gh release view <tag>` to check.

   If an upload is interrupted, `gh release upload <tag> --clobber <files>` adds the missing files.

3. The app finds releases through "latest", so the newest release must be marked latest (the
   printed command does that).

**Versions:** app `0.1.0`, `0.1.1`, … (tag `v0.1.0`). Base/expansion packages `2026.11.1`
(data month + build number). If you publish only a new app version, keep the old packages
without re-uploading them:
```
python release_all.py --app_version 0.1.1 --data_version 2026.11.1 --steps release --previous "<old versions.json>"
```
after deleting `dist\base` and `dist\expansion` (or the packages in them are uploaded again).

## Testing while the repository is private

Anonymous downloads from a private repository fail. For your own tests, the app downloads
through the GitHub API when the environment variable `OTR_GITHUB_TOKEN` is set; the token from
your `gh` login works:
```
cd /d "<installation folder>"
for /f %t in ('gh auth token') do set OTR_GITHUB_TOKEN=%t
"osu!taiko recommender.bat"
```
(in a `.bat` file write `%%t` instead of `%t`). The "downloads & updates" tab then says
"developer mode". Users of the public repository never need this.

To try the app without the installer, from this folder: `python app\main.py` (with numpy and
PySide6 installed in that Python). It keeps its data in `taiko_app\data` (ignored by git).
