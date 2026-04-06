from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import hashlib
import json
import platform
import sys

import matplotlib
import numpy as np
import scipy


def stable_hash_payload(payload: Any) -> str:
    """
    Stable SHA256 hash for JSON-like payloads.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def capture_provenance(*, config: Optional[Mapping[str, Any]] = None, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Capture lightweight environment and configuration provenance.

    This is intentionally dependency-light and deterministic enough for paper bundles.
    """
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    if config is not None:
        config_dict = dict(config)
        payload["config_sha256"] = stable_hash_payload(config_dict)
        payload["config_keys"] = sorted(config_dict.keys())
    if extra is not None:
        payload["extra"] = dict(extra)
    return payload
