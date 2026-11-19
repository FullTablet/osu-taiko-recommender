"""Installed packages (base / expansion), versions, and downloads from GitHub.

Package zip layout (written by builder/make_release.py):
    base:       manifest.json, charts.sqlite, skill_quantiles.json
    expansion:  manifest.json, note_net.npz
manifest.json: {"package", "version", "model_id", "pipeline_version", "created",
                "files": {name: {"size", "sha256"}}}

versions.json (a release asset; the app reads the one of the LATEST release via
https://github.com/<repo>/releases/latest/download/versions.json):
    {"format": 1,
     "app":       {"version", "url"},
     "base":      {"version", "model_id", "pipeline_version", "size", "sha256",
                   "parts": [{"url", "size", "sha256"}, ...]},
     "expansion": {... same ...}}
Large zips are split into parts (GitHub's per-asset limit is 2 GiB); the app
downloads the parts, joins them and checks every hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

APP_NAME = "osu!taiko recommender"
PACKAGE_FILES = {"base": ["charts.sqlite", "skill_quantiles.json"], "expansion": ["note_net.npz"]}
USER_AGENT = "osu-taiko-recommender"
ATTRIBUTION = "chart data and statistics derived from osu! data (data.ppy.sh)"
TOKEN_ENV = "OTR_GITHUB_TOKEN"            # developer use: download from a private repository
API_BASE = os.environ.get("OTR_GITHUB_API", "https://api.github.com")


class PackageError(Exception):
    pass


def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def app_config() -> dict:
    try:
        return json.loads((app_root() / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"app_version": "0.0.0", "github_repo": ""}


def install_root() -> Path:
    """The installation folder (the one holding app/, runtime/ and the launcher)."""
    return app_root().parent


def data_dir() -> Path:
    """Everything the app writes lives in <installation folder>/data (nothing outside
    the installation folder). OTR_DATA_DIR overrides it (tests)."""
    env = os.environ.get("OTR_DATA_DIR")
    return Path(env) if env else install_root() / "data"


def package_dir(kind: str) -> Path:
    return data_dir() / kind


def user_dir() -> Path:
    return data_dir() / "user"


def read_manifest(kind: str) -> dict | None:
    p = package_dir(kind) / "manifest.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if all((package_dir(kind) / f).exists() for f in PACKAGE_FILES[kind]):
        return m
    return None


def version_tuple(v: str) -> tuple:
    out = []
    for part in str(v).replace("-", ".").split("."):
        out.append(int(part) if part.isdigit() else 0)
    return tuple(out)


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- install
def install_zip(zip_path, kind: str, expected_model_id: str | None = None, progress=None) -> dict:
    """Validate and install a package zip. Returns its manifest."""
    if kind not in PACKAGE_FILES:
        raise PackageError(f"unknown package kind {kind!r}")
    dest = package_dir(kind)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{kind}_", dir=dest.parent))
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if "manifest.json" not in names:
                raise PackageError(f"not an {APP_NAME} package (no manifest.json)")
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("package") != kind:
                raise PackageError(f"this is a {manifest.get('package')!r} package, not {kind!r}")
            members = ["manifest.json"] + list(manifest.get("files", {}))
            for i, name in enumerate(members):
                if name not in names or "/" in name or "\\" in name or name.startswith("."):
                    raise PackageError(f"package is missing or has an invalid file: {name}")
                z.extract(name, tmp)
                if progress:
                    progress(0.5 * (i + 1) / len(members))
        for i, (name, info) in enumerate(manifest["files"].items()):
            if sha256_file(tmp / name) != info["sha256"]:
                raise PackageError(f"{name} is damaged (checksum mismatch); download it again")
            if progress:
                progress(0.5 + 0.5 * (i + 1) / len(manifest["files"]))
        missing = [f for f in PACKAGE_FILES[kind] if f not in manifest["files"]]
        if missing:
            raise PackageError(f"package lacks {missing}")
        if expected_model_id and manifest.get("model_id") != expected_model_id:
            raise PackageError("package model does not match the expected one")
        replace_dir(tmp, dest, kind)
        return manifest
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


IN_USE_HELP = ("a file in it is in use. close any other copy of the app and programs that may be reading "
               "that folder (an antivirus scan, a sync or backup tool), then try again")


def _retry(action, what: str, attempts: int = 8, delay: float = 0.5):
    """Windows refuses to rename a folder while a file in it is open; antivirus scanners and
    sync tools hold files briefly, so retry for a few seconds before giving up."""
    for i in range(attempts):
        try:
            return action()
        except PermissionError as e:
            if i == attempts - 1:
                raise PackageError(f"couldn't {what}: {IN_USE_HELP}. ({e})") from e
            time.sleep(delay)


def replace_dir(new: Path, dest: Path, kind: str = "package", attempts: int = 8, delay: float = 0.5):
    """Swap the freshly unpacked folder `new` into place at `dest`, keeping the old one if
    anything fails. The caller must have closed every file it had open in `dest`."""
    for stale in dest.parent.glob(dest.name + ".old*"):           # leftovers of earlier attempts
        shutil.rmtree(stale, ignore_errors=True)
    old = dest.with_name(f"{dest.name}.old-{int(time.time() * 1000)}")
    if dest.exists():
        _retry(lambda: dest.rename(old), f"replace the installed {kind} package", attempts, delay)
    try:
        _retry(lambda: new.rename(dest), f"install the new {kind} package", attempts, delay)
    except BaseException:
        if old.exists() and not dest.exists():
            old.rename(dest)                                      # put the previous version back
        raise
    shutil.rmtree(old, ignore_errors=True)


def uninstall(kind: str):
    shutil.rmtree(package_dir(kind), ignore_errors=True)


# --------------------------------------------------------------------------- remote
class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """GitHub serves release assets by redirecting to a storage host that rejects an
    Authorization header; drop it when the redirect leaves the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for h in ("Authorization", "authorization"):
                new.headers.pop(h, None)
                new.unredirected_hdrs.pop(h, None)
        return new


def ssl_context() -> ssl.SSLContext:
    """Certificate checking for downloads. Python's own check on Windows only sees the root
    certificates Windows has already cached, and Windows adds roots lazily, so a site whose
    root it hasn't needed yet fails with 'unable to get issuer certificate'. truststore asks
    Windows itself to verify (it fetches missing roots, and honours company/antivirus roots);
    without it, the default store plus certifi's root list is used."""
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:                                                       # noqa: BLE001
        pass
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:                                                       # noqa: BLE001
        pass
    return ctx


_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context()), _DropAuthOnRedirect)


def _token() -> str:
    return os.environ.get(TOKEN_ENV, "").strip()


def _open(url: str, timeout: float = 30, accept: str | None = None, auth: bool = False):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    if auth and _token():
        headers["Authorization"] = f"Bearer {_token()}"
    return _opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def _api_json(path: str) -> dict:
    with _open(f"{API_BASE}{path}", accept="application/vnd.github+json", auth=True) as r:
        return json.loads(r.read().decode("utf-8"))


_release_cache: dict = {}


def _asset_api_url(browser_url: str) -> str:
    """https://github.com/<owner>/<repo>/releases/download/<tag>/<name> -> API asset URL
    (token mode only)."""
    parts = urllib.parse.urlsplit(browser_url).path.strip("/").split("/")
    if len(parts) < 6 or parts[2:4] != ["releases", "download"]:
        raise PackageError(f"unexpected download URL {browser_url}")
    owner, repo, tag, name = parts[0], parts[1], urllib.parse.unquote(parts[4]), urllib.parse.unquote(parts[5])
    key = (owner, repo, tag)
    if key not in _release_cache:
        _release_cache[key] = _api_json(f"/repos/{owner}/{repo}/releases/tags/{urllib.parse.quote(tag)}")
    for asset in _release_cache[key].get("assets", []):
        if asset["name"] == name:
            return asset["url"]
    raise PackageError(f"{name} is not in release {tag}")


def _open_download(browser_url: str, timeout: float = 60):
    if _token():
        return _open(_asset_api_url(browser_url), timeout, accept="application/octet-stream", auth=True)
    return _open(browser_url, timeout)


def _hint(e: Exception) -> str:
    if "CERTIFICATE_VERIFY_FAILED" in str(e) or isinstance(getattr(e, "reason", None), ssl.SSLError):
        return (" (the secure connection couldn't be verified: run Install.bat again to update the app's "
                "certificate support, and check the computer's date and time)")
    return "" if _token() else " (if the repository is private, set OTR_GITHUB_TOKEN)"


def fetch_versions(repo: str | None = None) -> dict:
    repo = repo or app_config().get("github_repo", "")
    if not repo or "/" not in repo or repo.startswith("OWNER"):
        raise PackageError("no GitHub repository is configured in config.json")
    try:
        if _token():                                   # private repository (developer testing)
            _release_cache.clear()
            rel = _api_json(f"/repos/{repo}/releases/latest")
            asset = next((a for a in rel.get("assets", []) if a["name"] == "versions.json"), None)
            if asset is None:
                raise PackageError("the latest release has no versions.json")
            with _open(asset["url"], accept="application/octet-stream", auth=True) as r:
                return json.loads(r.read().decode("utf-8"))
        with _open(f"https://github.com/{repo}/releases/latest/download/versions.json") as r:
            return json.loads(r.read().decode("utf-8"))
    except PackageError:
        raise
    except Exception as e:                                                  # noqa: BLE001
        raise PackageError(f"couldn't check for updates: {e}{_hint(e)}") from e


def download_package(entry: dict, kind: str, progress=None, cancelled=lambda: False) -> dict:
    """Download (all parts), verify, and install. progress(fraction, text)."""
    total = int(entry.get("size") or sum(p["size"] for p in entry["parts"]))
    work = data_dir() / "downloads"
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / f"{kind}.zip"
    try:
        try:
            _download_parts(entry, zip_path, total, progress, cancelled)
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError) as e:
            raise PackageError(f"download failed: {getattr(e, 'reason', e)}{_hint(e)}") from e
        if progress:
            progress(0.9, "installing")
        return install_zip(zip_path, kind, entry.get("model_id"),
                           progress=(lambda f: progress(0.9 + 0.1 * f, "installing")) if progress else None)
    finally:
        try:
            zip_path.unlink()
        except OSError:
            pass


def _download_parts(entry: dict, zip_path: Path, total: int, progress, cancelled):
    done = 0
    h = hashlib.sha256()
    with open(zip_path, "wb") as out:
        for pi, part in enumerate(entry["parts"]):
            ph = hashlib.sha256()
            with _open_download(part["url"]) as r:
                while True:
                    if cancelled():
                        raise PackageError("download cancelled")
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    ph.update(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(0.9 * done / max(total, 1),
                                 f"downloading part {pi + 1}/{len(entry['parts'])}: "
                                 f"{done / 2**20:,.0f} / {total / 2**20:,.0f} MB")
            if part.get("sha256") and ph.hexdigest() != part["sha256"]:
                raise PackageError(f"part {pi + 1} is damaged; try again")
    if entry.get("sha256") and h.hexdigest() != entry["sha256"]:
        raise PackageError("download is damaged (checksum mismatch); try again")


def open_url(url: str):
    """Open a web page in the default browser."""
    import webbrowser

    webbrowser.open(url)


def python_info() -> str:
    return sys.version.split()[0]
