"""Lightweight experiment logging + manifest writer."""
from __future__ import annotations

import datetime
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str = "experiment", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Write a JSON manifest describing an experiment run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
    }
    base.update(manifest)
    path = out_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(base, f, indent=2, default=str)
    return path
