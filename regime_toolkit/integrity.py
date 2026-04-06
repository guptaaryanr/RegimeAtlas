from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import hashlib
import json


DEFAULT_MANIFEST_FORMAT = "v1"


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute a stable SHA256 for a file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()



def describe_file(path: str | Path, *, relative_to: str | Path | None = None) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    if relative_to is None:
        relative_path = path.name
    else:
        relative_path = str(path.relative_to(Path(relative_to)))
    return {
        "name": path.name,
        "relative_path": relative_path,
        "bytes": int(stat.st_size),
        "sha256": file_sha256(path),
    }



def build_manifest_payload(
    *,
    files: Mapping[str, str | Path | None],
    base_dir: str | Path,
    format_version: str = DEFAULT_MANIFEST_FORMAT,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    entries: Dict[str, Any] = {}
    for key, path in files.items():
        if path is None:
            entries[key] = None
            continue
        entries[str(key)] = describe_file(path, relative_to=base_dir)
    return {
        "format_version": str(format_version),
        "files": entries,
    }



def write_manifest(
    *,
    files: Mapping[str, str | Path | None],
    base_dir: str | Path,
    manifest_name: str = "manifest.json",
    format_version: str = DEFAULT_MANIFEST_FORMAT,
) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = build_manifest_payload(
        files=files,
        base_dir=base_dir,
        format_version=format_version,
    )
    manifest_path = base_dir / manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return manifest_path



def validate_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> Dict[str, Any]:
    """
    Validate a saved artifact manifest against the current filesystem.

    Expected manifest shape:
        {
          "format_version": "v1",
          "files": {
             "results_json": {"relative_path": "results.json", "sha256": "...", "bytes": 123},
             ...
          }
        }
    """
    base_dir = Path(base_dir)
    file_entries = dict(manifest.get("files", {}))
    per_file: Dict[str, Any] = {}
    passed = True

    for key, info in file_entries.items():
        if info is None:
            per_file[key] = {"present": False, "passed": True, "reason": "optional file absent"}
            continue
        relative_path = info.get("relative_path") or info.get("name")
        expected_sha = info.get("sha256")
        expected_bytes = info.get("bytes")
        path = base_dir / str(relative_path)
        if not path.exists():
            per_file[key] = {
                "present": False,
                "passed": False,
                "reason": "missing file",
                "expected_relative_path": relative_path,
            }
            passed = False
            continue
        actual = describe_file(path, relative_to=base_dir)
        sha_ok = (expected_sha is None) or (actual["sha256"] == expected_sha)
        bytes_ok = (expected_bytes is None) or (actual["bytes"] == expected_bytes)
        file_passed = bool(sha_ok and bytes_ok)
        passed = passed and file_passed
        per_file[key] = {
            "present": True,
            "passed": file_passed,
            "relative_path": actual["relative_path"],
            "expected_sha256": expected_sha,
            "actual_sha256": actual["sha256"],
            "expected_bytes": expected_bytes,
            "actual_bytes": actual["bytes"],
            "sha256_ok": bool(sha_ok),
            "bytes_ok": bool(bytes_ok),
        }

    return {
        "format_version": manifest.get("format_version"),
        "n_files": len(file_entries),
        "per_file": per_file,
        "passed": bool(passed),
    }



def validate_manifest_file(manifest_path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return validate_manifest_payload(manifest, base_dir=manifest_path.parent)
