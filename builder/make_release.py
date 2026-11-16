r"""Package everything for a GitHub release and print the gh commands to publish it.

    python make_release.py --tag v0.1.0 --app_version 0.1.0 ^
        [--base ..\dist\base] [--expansion ..\dist\expansion] [--previous old_versions.json]

Creates in --out (default ..\dist\release):
    osu-taiko-recommender-<app_version>.zip   the application (unzip, run Install.bat)
    taiko_base_<version>.zip[.001 ...]        base package (split into parts if > --part_mb)
    taiko_expansion_<version>.zip[...]        expansion pack
    versions.json                             what the app reads to find downloads
    release_notes.md                          text for the GitHub release page

A package you didn't rebuild can be carried over from the previous release's
versions.json with --previous (its URLs keep pointing to the older release, so it
needn't be uploaded again).
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import build_common as C
import git_setup

APP_NAME = "osu!taiko recommender"
APP_FOLDER = "osu!taiko recommender"            # top folder inside the app zip
APP_ASSET = "osu-taiko-recommender"             # asset file names avoid '!' and spaces
DEFAULT_REPO = "FullTablet/osu-taiko-recommender"


def zip_package(folder: Path, out_zip: Path):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(folder / "manifest.json", "manifest.json", compress_type=zipfile.ZIP_DEFLATED)
        for name in manifest["files"]:
            comp = zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            z.write(folder / name, name, compress_type=comp)
    return manifest


def split(path: Path, part_bytes: int) -> list[Path]:
    if path.stat().st_size <= part_bytes:
        return [path]
    parts = []
    with open(path, "rb") as f:
        i = 1
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = path.with_name(f"{path.name}.{i:03d}")
            p.write_bytes(chunk)
            parts.append(p)
            i += 1
    path.unlink()
    return parts


def package_entry(folder, kind, out, repo, tag, part_bytes):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    zip_path = out / f"taiko_{kind}_{manifest['version']}.zip"
    zip_package(folder, zip_path)
    whole_sha, size = C.sha256_file(zip_path), zip_path.stat().st_size
    parts = split(zip_path, part_bytes)
    url = f"https://github.com/{repo}/releases/download/{tag}/"
    return {"version": manifest["version"], "model_id": manifest["model_id"],
            "pipeline_version": manifest.get("pipeline_version"), "size": size, "sha256": whole_sha,
            "parts": [{"url": url + p.name, "size": p.stat().st_size, "sha256": C.sha256_file(p)} for p in parts]}


def crlf(data: bytes) -> bytes:
    """Windows scripts go into the zip with CRLF line endings, whatever the checkout has."""
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def zip_app(out: Path, app_version: str, repo: str) -> Path:
    app = C.APP_DIR
    target = out / f"{APP_ASSET}-{app_version}.zip"
    cfg = json.loads((app / "config.json").read_text(encoding="utf-8"))
    cfg.update(app_version=app_version, github_repo=repo)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(app.rglob("*")):
            rel = p.relative_to(app)
            if p.is_dir() or "__pycache__" in rel.parts or p.suffix == ".pyc":
                continue
            if rel.as_posix() == "config.json":
                z.writestr(f"{APP_FOLDER}/app/config.json", json.dumps(cfg, indent=2))
            elif rel.as_posix() == "setup/Install.bat":
                z.writestr(f"{APP_FOLDER}/Install.bat", crlf(p.read_bytes()))
            elif p.suffix.lower() in (".bat", ".cmd", ".ps1"):
                z.writestr(f"{APP_FOLDER}/app/{rel.as_posix()}", crlf(p.read_bytes()))
            else:
                z.write(p, f"{APP_FOLDER}/app/{rel.as_posix()}")
    return target


def release_notes(versions: dict, app_zip: Path) -> str:
    lines = [f"## {APP_NAME} {versions['app']['version']}", "",
             f"**Install:** download `{app_zip.name}`, unzip it anywhere, and run `Install.bat` inside the "
             "unzipped folder. It sets up a private copy of Python in that folder (nothing is installed "
             "elsewhere). Then start `osu!taiko recommender.bat`. The app downloads the chart database "
             "itself from the 'downloads & updates' tab.", "",
             "**Update:** unzip the new version over the old folder and run `Install.bat` again. "
             "Your data (the `data` folder) is kept.", ""]
    for kind, label in (("base", "base database"), ("expansion", "expansion pack (optional, for importing charts)")):
        if kind in versions:
            e = versions[kind]
            lines.append(f"- {label}: version {e['version']}, {e['size'] / 2**20:,.0f} MB, model `{e['model_id']}`")
    lines += ["", f"_{C.ATTRIBUTION}_", ""]
    return "\n".join(lines)


def gh_instructions(out: Path, repo: str, tag: str, app_version: str, files: list[Path]) -> str:
    q = lambda s: f'"{s}"'                                    # noqa: E731
    lines = [
        "",
        "=" * 78,
        "PUBLISH THIS RELEASE (Windows command prompt, with git and gh installed):",
        "",
        "1. Commit and push the source code (from the repository folder):",
        *git_setup.instructions(repo, tag),
        "",
        "2. Create the release and upload the files (one command, may take a while for large files):",
        f"     cd /d {q(out)}",
        f"     gh release create {tag} --repo {repo} --title {q(f'{APP_NAME} {app_version}')} "
        f"--notes-file release_notes.md --latest ^",
    ]
    for i, f in enumerate(files):
        lines.append(f"       {q(f.name)}" + (" ^" if i < len(files) - 1 else ""))
    lines += [
        "",
        "   If the upload is interrupted, add the missing files to the existing release with:",
        f"     gh release upload {tag} --repo {repo} --clobber <file> [<file> ...]",
        "",
        "3. Check it:",
        f"     gh release view {tag} --repo {repo}",
        "   The app finds the release through 'releases/latest', so this release must be the latest one.",
        "   While the repository is private, test the app's downloads with a token (see RELEASING.md):",
        "     for /f %t in ('gh auth token') do set OTR_GITHUB_TOKEN=%t",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository, owner/name")
    ap.add_argument("--tag", required=True, help="release tag you will create, e.g. v0.1.0")
    ap.add_argument("--app_version", required=True)
    ap.add_argument("--base", help="built base package folder")
    ap.add_argument("--expansion", help="built expansion pack folder")
    ap.add_argument("--previous", help="previous versions.json, to carry over packages not rebuilt")
    ap.add_argument("--out", default=str(C.HERE.parent / "dist" / "release"))
    ap.add_argument("--part_mb", type=int, default=1900, help="split zips larger than this (GitHub limit: 2 GiB)")
    ap.add_argument("--skip_source_check", action="store_true", help=argparse.SUPPRESS)   # tests only
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    prev = json.loads(Path(args.previous).read_text(encoding="utf-8")) if args.previous else {}
    versions = {"format": 1,
                "app": {"version": args.app_version, "url": f"https://github.com/{args.repo}/releases/tag/{args.tag}"}}
    part_bytes = args.part_mb * 2**20
    for kind, folder in (("base", args.base), ("expansion", args.expansion)):
        if folder:
            versions[kind] = package_entry(folder, kind, out, args.repo, args.tag, part_bytes)
        elif kind in prev:
            versions[kind] = prev[kind]
    if "base" in versions and "expansion" in versions and versions["base"]["model_id"] != versions["expansion"]["model_id"]:
        raise SystemExit("error: base and expansion packages come from different models")
    app_zip = zip_app(out, args.app_version, args.repo)
    (out / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (out / "release_notes.md").write_text(release_notes(versions, app_zip), encoding="utf-8")
    uploads = sorted(p for p in out.iterdir() if p.name != "release_notes.md")
    import privacy_check

    print("checking that nothing private is in the source code or the release files ...")
    pkg_dirs = [d for d in (args.base, args.expansion) if d]
    findings = privacy_check.check(out, pkg_dirs, None if args.skip_source_check else privacy_check.REPO_ROOT)
    if findings:
        print("\nPRIVATE INFORMATION FOUND - do not publish this release:\n  " + "\n  ".join(findings[:50]))
        print("\nRemove it (or, for a harmless false positive, adjust builder/local_paths.json) and run again.")
        raise SystemExit(1)
    print("privacy check passed.")
    print(f"release files in {out}:")
    for p in uploads:
        print(f"  {p.name}  ({p.stat().st_size / 2**20:,.1f} MB)")
    print(gh_instructions(out, args.repo, args.tag, args.app_version, uploads))
    return app_zip


if __name__ == "__main__":
    main()
