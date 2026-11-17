r"""Check that nothing private ends up in the published source code or in a release.

    python privacy_check.py                 # repository source + dist\release (+ dist\base, dist\expansion)

What counts as private:
  * every path and string in builder\local_paths.json ("paths" values and "private_strings"),
    plus your Windows user name and home folder;
  * in text files (source, docs, json, release notes): any absolute Windows path (like
    X:\<folder>), any home-folder path (like /home/<name>), and any e-mail address.

Binary files (chart database, weights) are searched for the private strings only (UTF-8,
UTF-16 and UTF-32 encodings; strings shorter than 7 characters are skipped there, because
random compressed bytes could match them by chance).

make_release.py runs this automatically and refuses to print the publish commands if
anything is found.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LOCAL_CONFIG = HERE / "local_paths.json"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".bat", ".ps1", ".cfg", ".toml", ".ini", ".yml", ".yaml",
                 ".gitignore", ".csv", ".html"}
ALWAYS_SKIP_DIRS = {".git", "__pycache__", "dist", "data", "runtime"}
MIN_BINARY_LEN = 7
_GENERIC_TEXT = [
    ("absolute Windows path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?=[^\s\"'`<>|*?]{2})")),
    ("home-folder path", re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9_.-]+")),
    ("e-mail address", re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
]


def load_local_config() -> dict:
    try:
        return json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def private_strings(cfg: dict | None = None) -> list[str]:
    cfg = load_local_config() if cfg is None else cfg
    out = set()
    for v in (cfg.get("paths") or {}).values():
        if isinstance(v, str) and len(v) >= 4:
            out.add(v.rstrip("\\/"))
    for v in cfg.get("private_strings") or []:
        if isinstance(v, str) and len(v) >= 4:
            out.add(v)
    for env in ("USERNAME", "USERPROFILE"):
        v = os.environ.get(env, "")
        if len(v) >= 4 and v.lower() not in ("root", "user", "admin", "administrator"):
            out.add(v)
    variants = set()
    for s in out:
        variants |= {s, s.replace("\\", "/"), s.replace("/", "\\")}
    return sorted(variants, key=len, reverse=True)


class Checker:
    def __init__(self, strings: list[str]):
        self.strings = strings
        self.findings: list[str] = []
        self._text_re = re.compile("|".join(re.escape(s) for s in strings), re.IGNORECASE) if strings else None
        long_ = [s for s in strings if len(s) >= MIN_BINARY_LEN]
        pats = []
        for s in long_:
            for enc in ("utf-8", "utf-16-le", "utf-32-le"):
                pats.append(re.escape(s.encode(enc)))
        self._bin_re = re.compile(b"|".join(pats), re.IGNORECASE) if pats else None
        self._overlap = max((len(s) for s in long_), default=1) * 4

    def text(self, name: str, text: str):
        if self._text_re:
            for m in self._text_re.finditer(text):
                self._report(name, text, m, "private string")
        for label, rx in _GENERIC_TEXT:
            for m in rx.finditer(text):
                self._report(name, text, m, label)

    def _report(self, name, text, m, label):
        line = text.count("\n", 0, m.start()) + 1
        self.findings.append(f"{name}:{line}: {label}: {m.group(0)!r}")

    def binary_stream(self, name: str, fh, chunk: int = 64 << 20):
        if self._bin_re is None:
            return
        tail = b""
        offset = 0
        while True:
            block = fh.read(chunk)
            if not block:
                break
            data = tail + block
            m = self._bin_re.search(data)
            if m:
                self.findings.append(f"{name}: private string in binary data near byte {offset - len(tail) + m.start():,}")
                return
            tail = data[-self._overlap:]
            offset += len(block)

    def path(self, p: Path, display: str | None = None):
        display = display or str(p.name)
        if is_text_name(p.name):
            self.text(display, p.read_text(encoding="utf-8", errors="replace"))
        else:
            with open(p, "rb") as fh:
                self.binary_stream(display, fh)

    def zip(self, p: Path):
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = f"{p.name} / {info.filename}"
                self.text(name + " (entry name)", info.filename)
                if is_text_name(info.filename):
                    self.text(name, z.read(info).decode("utf-8", errors="replace"))
                else:
                    with z.open(info) as fh:
                        self.binary_stream(name, fh)


def is_text_name(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(s) for s in TEXT_SUFFIXES) or "." not in n.rsplit("/", 1)[-1]


def repository_files(root: Path = REPO_ROOT) -> list[Path]:
    """Files git would publish (tracked + untracked-not-ignored); without git, a walk that
    skips the usual ignored folders and the private local config."""
    if (root / ".git").exists():
        try:
            out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                                 capture_output=True, text=True, check=True).stdout
            return [root / line for line in out.splitlines() if line and (root / line).is_file()]
        except (OSError, subprocess.CalledProcessError):
            pass
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or ALWAYS_SKIP_DIRS & set(rel.parts) or rel.parts[0].startswith("dist"):
            continue
        if p == LOCAL_CONFIG or p.suffix == ".zip" or re.search(r"\.zip\.\d{3}$", p.name) or p.suffix == ".pyc":
            continue
        files.append(p)
    return files


def check(release_dir: Path | None = None, package_dirs=(), source_root: Path | None = REPO_ROOT,
          strings: list[str] | None = None) -> list[str]:
    c = Checker(private_strings() if strings is None else strings)
    if source_root is not None:
        for p in repository_files(source_root):
            c.path(p, f"source/{p.relative_to(source_root).as_posix()}")
    if release_dir is not None and Path(release_dir).exists():
        for p in sorted(Path(release_dir).iterdir()):
            if p.suffix == ".zip" and not p.name.startswith("taiko_"):
                c.zip(p)                                   # the app zip (packages are checked unzipped below)
            elif p.suffix in (".json", ".md"):
                c.path(p, f"release/{p.name}")
    for d in package_dirs:
        d = Path(d)
        if (d / "manifest.json").exists():
            c.path(d / "manifest.json", f"{d.name}/manifest.json")
            for name in json.loads((d / "manifest.json").read_text(encoding="utf-8"))["files"]:
                c.path(d / name, f"{d.name}/{name}")
    return c.findings


def main(argv=None):
    dist = REPO_ROOT / "dist"
    findings = check(dist / "release", [dist / "base", dist / "expansion"])
    if findings:
        print("PRIVATE INFORMATION FOUND:\n  " + "\n  ".join(findings))
        sys.exit(1)
    print("privacy check passed: no private paths, names or e-mail addresses in the source or the release files")


if __name__ == "__main__":
    main()
