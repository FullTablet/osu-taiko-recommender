r"""Git hooks that keep private or huge files out of the repository.

Installed into .git/hooks by `python builder\git_setup.py --apply`. They run by themselves:

    pre-commit   refuses a commit whose staged files include build output, data, compiled
                 files, builder/local_paths.json, files over the size limit, or text with
                 your private paths/names/e-mail addresses (same rules as privacy_check.py)
    pre-push     the same checks over every file in every commit about to be pushed, so a
                 private file committed and deleted later still can't reach GitHub

The rules don't depend on .gitignore, so they hold even if .gitignore is missing.
Run by hand: python builder\git_hooks.py check   (checks the staged files)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_check as PC  # noqa: E402

MAX_BYTES = 20 * 2**20              # source files are small; anything bigger is a mistake
FORBIDDEN = [
    (re.compile(r"^(dist[^/]*|data|runtime)/"), "build output / app data"),
    (re.compile(r"(^|/)__pycache__/|\.py[co]$"), "compiled Python file"),
    (re.compile(r"^builder/local_paths\.json$"), "your private local paths"),
    (re.compile(r"\.(zip|zip\.\d{3}|pt|pth|npz|npy|sqlite|sqlite3|db|osz)$", re.I), "data/package file"),
    (re.compile(r"(^|/)osu!taiko recommender\.bat$"), "launcher created by the installer"),
]
ZERO = "0" * 40


def git(*args, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=input_bytes, capture_output=True, check=True).stdout


def path_problems(path: str) -> list[str]:
    return [f"{path}: {why}" for rx, why in FORBIDDEN if rx.search(path)]


def check_blobs(entries: list[tuple[str, str]]) -> list[str]:
    """entries: (path, blob sha). Returns the problems found."""
    findings = []
    for path, _ in entries:
        findings += path_problems(path)
    if not entries:
        return findings
    # sizes in one batch
    req = "".join(f"{sha}\n" for _, sha in entries).encode()
    sizes = {}
    for line in git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input_bytes=req).decode().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            sizes[parts[0]] = int(parts[2])
    checker = PC.Checker(PC.private_strings())
    for path, sha in entries:
        size = sizes.get(sha, 0)
        if size > MAX_BYTES:
            findings.append(f"{path}: {size / 2**20:,.1f} MB (limit {MAX_BYTES / 2**20:.0f} MB)")
            continue
        data = git("cat-file", "blob", sha)
        checker.text(path + " (file name)", path)
        if PC.is_text_name(path):
            checker.text(path, data.decode("utf-8", errors="replace"))
        else:
            import io
            checker.binary_stream(path, io.BytesIO(data))
    return findings + checker.findings


def staged_entries() -> list[tuple[str, str]]:
    """Added/changed files in the index: (path, blob sha)."""
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode()
    paths = [p for p in out.split("\0") if p]
    if not paths:
        return []
    entries = []
    for line in git("ls-files", "-s", "-z", "--", *paths).decode().split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            entries.append((path, meta.split()[1]))
    return entries


def pushed_entries(stdin_text: str) -> list[tuple[str, str]]:
    """Every blob (with its path) in the commits being pushed."""
    entries, seen = [], set()
    for line in stdin_text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:          # branch deletion
            continue
        local_sha, remote_sha = parts[1], parts[3]
        rng = [local_sha, "--not", "--remotes"] if remote_sha == ZERO else [local_sha, f"^{remote_sha}"]
        try:
            out = git("rev-list", "--objects", *rng).decode()
        except subprocess.CalledProcessError:              # remote commit unknown locally
            out = git("rev-list", "--objects", local_sha, "--not", "--remotes").decode()
        objs = [(l.split(" ", 1)[0], l.split(" ", 1)[1]) for l in out.splitlines() if " " in l]
        if not objs:
            continue
        req = "".join(f"{sha}\n" for sha, _ in objs).encode()
        types = git("cat-file", "--batch-check=%(objectname) %(objecttype)", input_bytes=req).decode().split("\n")
        blob = {t.split()[0] for t in types if t.endswith(" blob")}
        for sha, path in objs:
            if sha in blob and (path, sha) not in seen:
                seen.add((path, sha))
                entries.append((path, sha))
    return entries


def report(what: str, findings: list[str]) -> int:
    if not findings:
        return 0
    print(f"\n{what} REFUSED: these files must not go into the repository:", file=sys.stderr)
    for f in findings[:40]:
        print("  " + f, file=sys.stderr)
    if len(findings) > 40:
        print(f"  ... and {len(findings) - 40} more", file=sys.stderr)
    print("\nUnstage them (git restore --staged <file>) or, for commits already made, see RELEASING.md\n"
          "(\"if a commit contains something it shouldn't\").", file=sys.stderr)
    return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv else "check"
    if mode in ("pre-commit", "check"):
        return report("COMMIT", check_blobs(staged_entries()))
    if mode == "pre-push":
        return report("PUSH", check_blobs(pushed_entries(sys.stdin.read())))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
