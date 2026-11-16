r"""Check (and optionally do) the one-time git setup of the source repository.

    python git_setup.py            report what is set up and print the commands still needed
    python git_setup.py --apply    also do the local part: git init, your GitHub identity
                                   (with GitHub's private no-reply e-mail) and the remote

Nothing is committed or pushed by this script; it prints those commands for you.
make_release.py and release_all.py use the same checks, so their printed instructions
always match the state of your repository.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "FullTablet/osu-taiko-recommender"
NOREPLY = "@users.noreply.github.com"
PRIVATE_FILE = "builder/local_paths.json"
REQUIRED_FILES = [".gitignore", ".gitattributes", "app/main.py", "builder/git_hooks.py", "builder/privacy_check.py"]
HOOK_MARK = "# installed by builder/git_setup.py"
HOOKS = ("pre-commit", "pre-push")


def hook_text(name: str) -> str:
    return f"""#!/bin/sh
{HOOK_MARK}: refuses commits/pushes with private or huge files (see builder/git_hooks.py)
if [ ! -f builder/git_hooks.py ]; then
  echo "{name} REFUSED: builder/git_hooks.py is missing - the source code isn't in this folder." >&2
  exit 1
fi
if command -v python >/dev/null 2>&1; then PY=python; else PY="py -3"; fi
exec $PY builder/git_hooks.py {name} "$@"
"""


def hooks_installed(repo_root: Path | None = None) -> bool:
    repo_root = REPO_ROOT if repo_root is None else repo_root
    for h in HOOKS:
        f = repo_root / ".git" / "hooks" / h
        if not f.exists() or HOOK_MARK not in f.read_text(errors="replace"):
            return False
    return True


def install_hooks(repo_root: Path | None = None):
    repo_root = REPO_ROOT if repo_root is None else repo_root
    d = repo_root / ".git" / "hooks"
    d.mkdir(parents=True, exist_ok=True)
    for h in HOOKS:
        f = d / h
        f.write_bytes(hook_text(h).encode())         # LF line endings: run by git's sh
        f.chmod(0o755)


def _run(args, cwd=None):
    """Run a command; return its stripped stdout, or None if it fails or isn't installed."""
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def ownership_fix(repo_root: Path | None = None) -> str | None:
    """The command that makes git trust this folder, if git refuses it for "dubious ownership"
    (the folder belongs to another Windows account, common in synced folders); else None."""
    repo_root = REPO_ROOT if repo_root is None else repo_root
    try:
        r = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--git-dir"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0 or "dubious ownership" not in r.stderr:
        return None
    m = re.search(r"safe\.directory\s+(.+)", r.stderr)
    target = m.group(1).strip().strip("'\"") if m else repo_root.as_posix()
    return f'git config --global --add safe.directory "{target}"'


def noreply_identity():
    """(login, no-reply e-mail) of the gh login, or (None, None) if gh isn't available/logged in."""
    if not shutil.which("gh"):
        return None, None
    out = _run(["gh", "api", "user"])
    if not out:
        return None, None
    try:
        user = json.loads(out)
        return user["login"], f"{user['id']}+{user['login']}{NOREPLY}"
    except (ValueError, KeyError):
        return None, None


def git_state(repo_root: Path | None = None) -> dict:
    repo_root = REPO_ROOT if repo_root is None else repo_root
    s = {"git": bool(shutil.which("git")), "gh": bool(shutil.which("gh")),
         "initialized": (repo_root / ".git").exists(), "email": None, "name": None,
         "remote": None, "branch": None, "private_tracked": False, "private_ignored": None, "commits": False,
         "untrusted": None, "hooks": False,
         "missing": [f for f in REQUIRED_FILES if not (repo_root / f).exists()]}
    if not (s["git"] and s["initialized"]):
        return s
    s["untrusted"] = ownership_fix(repo_root)
    if s["untrusted"]:
        return s
    g = lambda *a: _run(["git", "-C", str(repo_root), *a])          # noqa: E731
    s["email"] = g("config", "user.email")
    s["name"] = g("config", "user.name")
    s["remote"] = g("remote", "get-url", "origin")
    s["branch"] = g("symbolic-ref", "--short", "HEAD")
    s["commits"] = g("rev-parse", "--verify", "HEAD") is not None
    s["hooks"] = hooks_installed(repo_root)
    s["private_tracked"] = bool(g("ls-files", PRIVATE_FILE))
    s["private_ignored"] = g("check-ignore", "-q", PRIVATE_FILE) is not None
    return s


def problems(s: dict, repo: str = DEFAULT_REPO) -> list[str]:
    """What still needs doing before `git add -A / commit / push` is safe; empty when all is fine."""
    p = []
    if s["missing"]:
        p.append("source files are missing from this folder: " + ", ".join(s["missing"])
                 + " - restore them before committing anything")
    if not s["git"]:
        p.append("git is not installed (https://git-scm.com/download/win)")
    if not s["gh"]:
        p.append("GitHub CLI is not installed (https://cli.github.com/)")
    if not s["initialized"]:
        p.append("this folder is not a git repository yet (one-time setup not done)")
        return p
    if s["untrusted"]:
        p.append("git refuses to use this folder ('dubious ownership'): the folder belongs to another "
                 "Windows account than yours, which is common in synced folders. It must be added to "
                 "git's safe.directory list (your user-wide git settings)")
        return p
    if not s["email"] or not s["email"].endswith(NOREPLY):
        p.append(f"git user.email is {s['email']!r}, not your GitHub no-reply address: "
                 "commits would publish it")
    if not s["remote"]:
        p.append("no 'origin' remote")
    elif repo.lower() not in s["remote"].lower():
        p.append(f"'origin' points to {s['remote']}, not {repo}")
    if not s["hooks"]:
        p.append("the safety hooks (refuse private or huge files on commit/push) aren't installed")
    if s["private_tracked"]:
        p.append(f"{PRIVATE_FILE} is tracked by git (it holds your paths)")
    elif s["private_ignored"] is False:
        p.append(f"{PRIVATE_FILE} is not ignored by git (check .gitignore)")
    return p


def setup_commands(s: dict, repo: str = DEFAULT_REPO, email: str | None = None,
                   login: str | None = None) -> list[str]:
    """The commands still needed for the local setup, in order (cmd.exe syntax)."""
    email = email or f"<id>+{login or 'FullTablet'}{NOREPLY}"
    cmds = []
    if not s["initialized"]:
        cmds.append("git init -b main")
    if s["untrusted"]:
        cmds.append(s["untrusted"])
    if not s["initialized"] or not s["email"] or not s["email"].endswith(NOREPLY):
        cmds += [f'git config user.name "{login or "FullTablet"}"', f'git config user.email "{email}"']
    url = f"https://github.com/{repo}.git"
    if not s["initialized"] or not s["remote"]:
        cmds.append(f"git remote add origin {url}")
    elif repo.lower() not in s["remote"].lower():
        cmds.append(f"git remote set-url origin {url}")
    if s["private_tracked"]:
        cmds.append(f"git rm --cached {PRIVATE_FILE}")
    if not s["hooks"]:
        cmds.append("python builder\\git_setup.py --apply   (installs the safety hooks)")
    return cmds


def publish_commands(tag: str | None = None) -> list[str]:
    msg = f"release {tag}" if tag else "first version"
    return ["git add -A", "git status", f'git commit -m "{msg}"', "git push -u origin main"]


def instructions(repo: str = DEFAULT_REPO, tag: str | None = None, indent: str = "     ") -> list[str]:
    """Printable lines: setup commands still needed (if any) followed by add/commit/push."""
    s = git_state()
    login, email = noreply_identity() if s["gh"] else (None, None)
    lines = []
    probs = problems(s, repo)
    if probs:
        lines.append("   Not ready yet:")
        lines += [f"     - {x}" for x in probs]
    if not s["git"] or not s["gh"]:
        lines.append("   Install the missing program(s), open a NEW command prompt, run `gh auth login`, then run this again.")
        return lines
    setup = setup_commands(s, repo, email, login)
    lines.append(f"{indent}cd /d \"{REPO_ROOT}\"")
    if setup:
        lines.append("   one-time setup (or run: python builder\\git_setup.py --apply):")
        lines += [indent + c for c in setup]
        if email is None:
            lines.append("   (replace <id> with the number shown at GitHub -> Settings -> Emails, or run "
                         "`gh auth login` first so the address can be filled in for you)")
    lines.append("   then:")
    lines += [indent + c for c in publish_commands(tag)]
    lines.append("   (`git status` must NOT list builder/local_paths.json, dist/ or data/; "
                 "if commit says 'nothing to commit', the code is already up to date: just continue.")
    lines.append("    If push is rejected because GitHub already has commits (a README or license made on the "
                 "website), run `git pull origin main --allow-unrelated-histories --no-edit -X ours` and push "
                 "again: files only on GitHub are kept, and where both have the same file, yours wins.)")
    return lines


def apply(repo: str = DEFAULT_REPO, trust_folder: bool = False) -> int:
    s = git_state()
    if not s["git"]:
        print("git is not installed: https://git-scm.com/download/win")
        return 1
    def g(*a):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"git {' '.join(a)} failed:\n{(r.stderr or r.stdout).strip()}")
        return r.stdout.strip()

    if not s["initialized"]:
        g("init", "-b", "main")
        print("created the git repository")
    fix = ownership_fix()
    if fix:
        print("git refuses to use this folder because it belongs to another Windows account than yours\n"
              "('dubious ownership'; common in synced folders such as Box, Dropbox or OneDrive). The fix adds this folder,\n"
              "and only this folder, to the safe.directory list in your user-wide git settings:\n"
              f"    {fix}")
        if not trust_folder:
            try:
                ok = input("run it now? [y/N] ").strip().lower() in ("y", "yes")
            except EOFError:
                ok = False
            if not ok:
                print("not changed. Run the command above yourself, then run this script again.")
                return 1
        target = fix.split('"')[1]
        r = subprocess.run(["git", "config", "--global", "--add", "safe.directory", target],
                           capture_output=True, text=True)
        if r.returncode != 0 or ownership_fix():
            print(f"that didn't work:\n{(r.stderr or r.stdout).strip()}\nrun the command above yourself, "
                  "then run this script again.")
            return 1
        print("git now trusts this folder")
    s = git_state()
    if not s["email"] or not s["email"].endswith(NOREPLY):
        login, email = noreply_identity()
        if email is None:
            print("couldn't get your GitHub no-reply address: install GitHub CLI and run `gh auth login` first.")
            return 1
        g("config", "user.name", login)
        g("config", "user.email", email)
        print(f"commit identity: {login} <GitHub no-reply address>")
    url = f"https://github.com/{repo}.git"
    if not s["remote"]:
        g("remote", "add", "origin", url)
        print(f"remote origin: {url}")
    elif repo.lower() not in s["remote"].lower():
        g("remote", "set-url", "origin", url)
        print(f"remote origin changed to {url}")
    if s["private_tracked"]:
        g("rm", "--cached", PRIVATE_FILE)
        print(f"{PRIVATE_FILE} is no longer tracked")
    install_hooks()
    print("safety hooks installed (pre-commit, pre-push)")
    if s["missing"]:
        print("WARNING: source files are missing from this folder: " + ", ".join(s["missing"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--apply", action="store_true", help="do the local setup (no commit, no push)")
    ap.add_argument("--trust_folder", action="store_true",
                    help="with --apply: add the folder to git's safe.directory without asking, if needed")
    args = ap.parse_args(argv)
    if args.apply:
        rc = apply(args.repo, args.trust_folder)
        if rc:
            raise SystemExit(rc)
        print("local setup done.")
    s = git_state()
    print(f"repository folder: {REPO_ROOT}")
    print(f"git repository: {'yes' if s['initialized'] else 'no'}; identity: {s['name']} <{s['email']}>; "
          f"remote: {s['remote']}; branch: {s['branch']}; has commits: {s['commits']}")
    probs = problems(s, args.repo)
    print("ready to commit and push." if not probs else "")
    print("\n".join(instructions(args.repo)))


if __name__ == "__main__":
    main()
