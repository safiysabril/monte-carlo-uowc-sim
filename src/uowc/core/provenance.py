"""Provenance capture: git commit, library versions, platform, config content-hash,
ISO-8601 UTC timestamp.

Centralizes what :class:`~uowc.core.results.RunMetadata` needs to make a result
reconstructable from storage alone (data.md's reproducibility metadata; the
reproducibility checklist in research-methodology.md). Reads the ambient environment
(git, installed packages, platform) at call time; has no side effects beyond a single
``git`` subprocess invocation.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy

__all__ = [
    "utc_timestamp",
    "code_version",
    "library_versions",
    "platform_signature",
    "content_hash",
]

_GIT_TIMEOUT_S = 5.0
_UNKNOWN = "unknown"


def utc_timestamp() -> str:
    """Current time as an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=True,
    )
    return result.stdout.strip()


def code_version(*, repo_path: Path | str | None = None) -> str:
    """Git commit hash, suffixed ``-dirty`` if the working tree has uncommitted
    changes (data.md: "record the git commit, and whether the tree was dirty").

    A dirty-tree result is not reproducible from the commit alone, so the suffix must
    survive into any report built from it - never silently trim it back to a bare
    hash. Falls back to ``"unknown"`` if this is not a git checkout, or ``git`` is not
    on the path; a run outside version control is still storable, just not traceable
    to a commit.
    """
    cwd = Path(repo_path) if repo_path is not None else Path.cwd()
    try:
        commit = _run_git(["rev-parse", "HEAD"], cwd)
        dirty = _run_git(["status", "--porcelain"], cwd) != ""
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN
    return f"{commit}-dirty" if dirty else commit


def library_versions() -> dict[str, str]:
    """Versions of the scientific-computing stack this framework depends on
    (python.md: numpy, scipy, pandas, pyarrow) plus the interpreter itself.

    All four are hard dependencies (pyproject.toml), so this never needs to guess
    whether one is installed.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
    }


def platform_signature() -> str:
    """A short ``system-release-machine`` string identifying the execution platform."""
    return f"{platform.system()}-{platform.release()}-{platform.machine()}"


def content_hash(parameters: dict[str, Any]) -> str:
    """Stable SHA-256 hex digest of a JSON-serializable parameter mapping.

    Detects silent configuration drift between runs claimed to be reproducible: two
    runs with the same hash used bit-for-bit the same recorded parameter values, sorted
    so key order never changes the digest. This is a content check, not a security
    hash - it is not meant to resist deliberate tampering.
    """
    encoded = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
