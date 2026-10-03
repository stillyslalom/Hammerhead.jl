"""Acquire/audit VSJ301 privately. Python 3.11+ standard library only.

No redistribution license is inferred. A first-use hash is a local identity,
not a publisher-authenticated checksum. Partial failed caches are preserved;
retry in a fresh directory. Outside concurrent writers are unsupported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html.parser
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tomllib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.vsj.jp/~pivstd/"
SCHEMA = "hammerhead-vsj301-cache-1"
ARCHIVES = {"301_raw.zip": 6292551, "301_ptc.zip": 11100328}
TERMS = {"usage.html": BASE, "dataset.html": BASE + "image3d/image301.html",
         "particle-format.html": BASE + "image3d/ptc.html"}
PERMISSION = "Anybody can download and analyse the Standard images for any purpose."
CITATION = "Okamoto K., Nishio S., Kobayashi T., Saga T., Takehara K. (2000), Evaluation of the 3D-PIV standard images (PIV-STD project), DOI:10.1007/BF03182404"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path, limit=12000000):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular cache file: {path}")
    size = path.stat().st_size
    if size > limit:
        raise ValueError("cache file exceeds bounded size")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = limit
        while remaining:
            block = stream.read(min(65536, remaining))
            if not block:
                break
            remaining -= len(block); hasher.update(block)
        if stream.read(1):
            raise ValueError("cache file grew beyond bounded size")
    return {"bytes": size, "sha256": hasher.hexdigest()}


class Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def check_terms(data):
    p = Text(); p.feed(data.decode("utf-8", errors="replace"))
    if PERMISSION not in " ".join(" ".join(p.parts).split()):
        raise ValueError("publisher permission statement changed/missing; review before acquisition")


def fetch(url, limit):
    # A redirect must stay on the primary HTTPS host; validate before body read.
    req = urllib.request.Request(url, headers={"User-Agent": "Hammerhead-VSJ301-research/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        if not response.url.startswith(BASE) or response.status != 200:
            raise ValueError("unexpected publisher redirect/status")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError("publisher response exceeds declared bound")
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("publisher response exceeds declared bound")
        if length is not None and len(data) != int(length):
            raise ValueError("incomplete publisher response")
        headers = {k: response.headers.get(k, "") for k in ("ETag", "Last-Modified", "Content-Type")}
        return data, response.url, headers


def archive_members(data, kind):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        extension, prefix = ("raw", "img") if kind == "301_raw.zip" else ("dat", "ptc")
        expected = {f"{prefix}{i:03d}.{extension}" for i in range(145)}
        names = [entry.filename for entry in entries]
        if len(names) != 145 or set(names) != expected:
            raise ValueError("unexpected/duplicate/traversing archive members")
        for entry in entries:
            mode = entry.external_attr >> 16
            if entry.flag_bits & 1 or stat.S_ISLNK(mode) or entry.is_dir():
                raise ValueError("encrypted/link/directory member refused")
            bound = 65536 if extension == "raw" else 300000
            if entry.file_size <= 0 or entry.file_size > bound or entry.compress_type not in (0, 8):
                raise ValueError("unsupported member size/compression")
            if extension == "raw" and entry.file_size != 65536:
                raise ValueError("RAW shape changed")
        selected = []
        for i in range(8):
            name = f"{prefix}{i:03d}.{extension}"
            entry = archive.getinfo(name)
            body = archive.read(entry)  # zipfile verifies CRC; uncompressed size bounded above.
            if len(body) != entry.file_size:
                raise ValueError("incomplete member")
            selected.append((name, body, entry.CRC))
        return selected


def canonical(path):
    value = str(Path(path).resolve())
    if os.name == "nt":
        value = "\\".join(part.rstrip(" .") for part in value.split("\\"))
    return os.path.normcase(value)


def check_new_cache(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("acquisition requires a fresh cache directory; use --offline for completed cache")
    canon = canonical(path); repo = canonical(ROOT); ignored = canonical(ROOT / "bench/profile-output")
    def within(child, parent):
        try:
            return os.path.commonpath((child, parent)) == parent
        except ValueError:  # distinct Windows volumes are outside, not malformed.
            return False
    if within(canon, repo) and not within(canon, ignored):
        raise ValueError("repository cache must stay under ignored bench/profile-output")
    if canon == ignored:
        raise ValueError("use a dedicated cache beneath bench/profile-output")
    return path


def q(value):
    return json.dumps(value, ensure_ascii=True)


def manifest_text(rows):
    text = [f"schema_version = {q(SCHEMA)}", "source_frames = [0, 1, 2, 3, 4, 5, 6, 7]",
            f"acquired_utc = {q(datetime.now(timezone.utc).isoformat())}",
            f"preparer_sha256 = {q(digest(Path(__file__).read_bytes()))}",
            f"python_version = {q(sys.version)}",
            f"citation = {q(CITATION)}", f"permission_statement = {q(PERMISSION)}",
            'redistribution_license = "not established; private cache only"',
            'hash_authenticity = "first acquisition local SHA256; no publisher checksum found"']
    for section, records in rows.items():
        for record in records:
            text.append(f"\n[[{section}]]")
            for key, value in record.items():
                text.append(f"{key} = {value if type(value) is int else q(value)}")
    return "\n".join(text) + "\n"


def write_new(path, data):
    if path.is_symlink():
        raise ValueError("symlink cache destination refused")
    with path.open("xb") as stream:
        stream.write(data)


def acquire(path):
    path = check_new_cache(path)
    pages = {name: fetch(url, 128000) for name, url in TERMS.items()}
    check_terms(pages["usage.html"][0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()  # exclusive creator; nothing existing is deleted or replaced.
    rows = {"pages": [], "archives": [], "members": []}
    for name, (data, final_url, headers) in pages.items():
        write_new(path / name, data)
        rows["pages"].append({"path": name, "url": TERMS[name], "final_url": final_url,
                              "bytes": len(data), "sha256": digest(data),
                              "etag": headers["ETag"], "last_modified": headers["Last-Modified"]})
    for name, size in ARCHIVES.items():
        url = BASE + "image3d/" + name
        data, final_url, headers = fetch(url, size)
        if len(data) != size:
            raise ValueError("archive size changed; review publisher update rather than silently repinning")
        selected = archive_members(data, name)
        write_new(path / name, data)
        rows["archives"].append({"path": name, "url": url, "final_url": final_url,
                                 "bytes": size, "sha256": digest(data),
                                 "etag": headers["ETag"], "last_modified": headers["Last-Modified"]})
        for member, body, crc in selected:
            write_new(path / member, body)
            rows["members"].append({"path": member, "archive": name, "bytes": len(body),
                                    "sha256": digest(body), "crc32": crc})
    # Last publication indicates completed acquisition. No power-loss guarantee.
    write_new(path / "cache.toml", manifest_text(rows).encode())
    return audit(path)


def audit(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("cache must be a regular directory")
    manifest = path / "cache.toml"; file_identity(manifest, 128000)
    data = tomllib.loads(manifest.read_text())
    expected = {"schema_version", "source_frames", "citation", "permission_statement",
                "redistribution_license", "hash_authenticity", "pages", "archives", "members",
                "acquired_utc", "preparer_sha256", "python_version"}
    if set(data) != expected or data["schema_version"] != SCHEMA or data["source_frames"] != list(range(8)) or any(type(i) is not int for i in data["source_frames"]):
        raise ValueError("unsupported/malformed cache schema/frame selection")
    if data["permission_statement"] != PERMISSION or data["citation"] != CITATION or not re.fullmatch(r"[0-9a-f]{64}", data["preparer_sha256"]):
        raise ValueError("cache attribution/terms changed")
    expected_files = {"cache.toml"}; seen = set()
    for section in ("pages", "archives", "members"):
        for row in data[section]:
            keys = {"path", "archive", "bytes", "sha256", "crc32"} if section == "members" else {"path", "url", "final_url", "bytes", "sha256", "etag", "last_modified"}
            if type(row) is not dict or set(row) != keys:
                raise ValueError("malformed cache records")
            name = row["path"]
            if Path(name).name != name or name in seen or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("unsafe/duplicate cache path")
            seen.add(name); expected_files.add(name)
            bound = 128000 if section == "pages" else 300000 if section == "members" else 12000000
            if type(row["bytes"]) is not int or not 0 <= row["bytes"] <= bound or (path / name).stat().st_size != row["bytes"]:
                raise ValueError("cache size changed/exceeds bound")
            identity = file_identity(path / name, bound)
            if identity != {k: row[k] for k in ("bytes", "sha256")}:
                raise ValueError("changed cache file")
    if {p.name for p in path.iterdir()} != expected_files:
        raise ValueError("unknown cache files refused")
    if len(data["pages"]) != 3 or {r["path"]: r["url"] for r in data["pages"]} != TERMS or any(r["url"] != r["final_url"] for r in data["pages"] + data["archives"]):
        raise ValueError("publisher page provenance changed")
    check_terms((path / "usage.html").read_bytes())
    if len(data["archives"]) != 2 or {r["path"]: r["bytes"] for r in data["archives"]} != ARCHIVES:
        raise ValueError("archive provenance changed")
    expected_members = {}
    for row in data["archives"]:
        if row["url"] != BASE + "image3d/" + row["path"]:
            raise ValueError("archive URL changed")
        for name, body, crc in archive_members((path / row["path"]).read_bytes(), row["path"]):
            expected_members[name] = {"archive": row["path"], "crc32": crc, "bytes": len(body), "sha256": digest(body)}
    if len(data["members"]) != 16 or {r["path"]: {k: r[k] for k in ("archive", "crc32", "bytes", "sha256")} for r in data["members"]} != expected_members:
        raise ValueError("member/archive binding changed")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "bench/profile-output/vsj301-cache")
    parser.add_argument("--offline", action="store_true", help="audit pinned completed cache; never access network")
    args = parser.parse_args()
    result = audit(args.cache) if args.offline else acquire(args.cache)
    print(f"Verified {len(result['members'])} selected members: {args.cache / 'cache.toml'}")


if __name__ == "__main__":
    main()
