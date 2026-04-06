from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np

from .provenance import capture_provenance
from .integrity import build_manifest_payload

# Stable artifact contract:
# <outdir>/results.json
# <outdir>/config.json
# <outdir>/trajectories.npz
# <outdir>/manifest.json
# <outdir>/provenance.json
# <outdir>/figures/



def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)



def save_json(payload: Dict[str, Any], filepath: str | Path) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2, sort_keys=True)
    return filepath



def save_results_json(results: Dict[str, Any], filepath: str | Path) -> Path:
    filepath = Path(filepath)
    payload = dict(results)
    payload.pop("trajectories", None)
    return save_json(payload, filepath)



def save_trajectories_npz(results: Dict[str, Any], filepath: str | Path) -> Optional[Path]:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    trajectories = results.get("trajectories", {})
    if not trajectories:
        return None

    arrays: Dict[str, Any] = {}
    for idx, (param_value, traj) in enumerate(sorted(trajectories.items())):
        prefix = f"traj_{idx:03d}"
        arrays[f"{prefix}_param"] = np.array([float(param_value)], dtype=float)
        arrays[f"{prefix}_t"] = np.asarray(traj["t"], dtype=float)
        arrays[f"{prefix}_x"] = np.asarray(traj["x"], dtype=float)
        arrays[f"{prefix}_meta_json"] = np.array(
            json.dumps(_to_serializable(traj.get("meta", {})), sort_keys=True)
        )
    arrays["n_trajectories"] = np.array([len(trajectories)], dtype=int)
    np.savez_compressed(filepath, **arrays)
    return filepath



def save_sweep_bundle(
    results: Dict[str, Any],
    outdir: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Path]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = save_results_json(results, outdir / "results.json")
    trajectories_path = save_trajectories_npz(results, outdir / "trajectories.npz")

    config_path: Optional[Path] = None
    if config is not None:
        config_path = outdir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(config), f, indent=2, sort_keys=True)

    provenance = capture_provenance(config=config)
    provenance_path = save_json(provenance, outdir / "provenance.json")

    manifest = build_manifest_payload(
        files={
            "results_json": results_path,
            "trajectories_npz": trajectories_path,
            "config_json": config_path,
            "provenance_json": provenance_path,
        },
        base_dir=outdir,
        format_version="v1",
    )
    manifest_path = outdir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return {
        "results_json": results_path,
        "trajectories_npz": trajectories_path,
        "config_json": config_path,
        "provenance_json": provenance_path,
        "manifest_json": manifest_path,
    }
