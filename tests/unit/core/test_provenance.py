"""Git commit, library versions, platform, content-hash, timestamp capture."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from uowc.core import provenance


def test_utc_timestamp_is_iso8601_and_timezone_aware() -> None:
    stamp = provenance.utc_timestamp()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_code_version_reports_the_actual_repo_head() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".git").exists():
        pytest.skip("not running inside a git checkout")
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    version = provenance.code_version(repo_path=repo_root)
    assert version.split("-dirty")[0] == expected_head


def test_code_version_flags_a_dirty_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("v1")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)

    assert not provenance.code_version(repo_path=repo).endswith("-dirty")

    (repo / "f.txt").write_text("v2 - uncommitted")
    assert provenance.code_version(repo_path=repo).endswith("-dirty")


def test_code_version_falls_back_to_unknown_outside_a_repo(tmp_path: Path) -> None:
    assert provenance.code_version(repo_path=tmp_path) == "unknown"


def test_library_versions_reports_the_hard_dependencies() -> None:
    versions = provenance.library_versions()
    for name in ("python", "numpy", "scipy", "pandas", "pyarrow"):
        assert name in versions
        assert re.match(r"^\d+\.\d+", versions[name]), (
            f"{name} version looks malformed: {versions[name]!r}"
        )


def test_platform_signature_has_three_dash_separated_fields() -> None:
    assert len(provenance.platform_signature().split("-")) >= 3


def test_content_hash_is_order_independent() -> None:
    a = provenance.content_hash({"x": 1, "y": 2})
    b = provenance.content_hash({"y": 2, "x": 1})
    assert a == b


def test_content_hash_is_sensitive_to_value_changes() -> None:
    a = provenance.content_hash({"seed": 1})
    b = provenance.content_hash({"seed": 2})
    assert a != b


def test_content_hash_is_deterministic() -> None:
    params = {"seed": 42, "wavelength_nm": 520.0, "model": "haltrin"}
    assert provenance.content_hash(params) == provenance.content_hash(dict(params))


def test_content_hash_is_a_hex_sha256_digest() -> None:
    digest = provenance.content_hash({"a": 1})
    assert len(digest) == 64
    assert re.match(r"^[0-9a-f]{64}$", digest)


# --- scalar_fields -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FakePhase:
    asymmetry: float


@dataclass(frozen=True, slots=True)
class _FakeModel:
    wavelength_nm: float
    label: str
    enabled: bool
    phase: _FakePhase
    samples: object  # e.g. a realized random-field array - not a scalar coefficient


def test_scalar_fields_keeps_float_int_str_bool_verbatim() -> None:
    model = _FakeModel(
        wavelength_nm=520.0, label="x", enabled=True, phase=_FakePhase(0.5), samples=None
    )
    fields = provenance.scalar_fields(model)
    assert fields["wavelength_nm"] == 520.0
    assert fields["label"] == "x"
    assert fields["enabled"] is True


def test_scalar_fields_extracts_phase_function_asymmetry() -> None:
    model = _FakeModel(
        wavelength_nm=520.0, label="x", enabled=True, phase=_FakePhase(0.924), samples=None
    )
    fields = provenance.scalar_fields(model)
    assert fields["phase_asymmetry"] == pytest.approx(0.924)


def test_scalar_fields_skips_non_scalar_non_phase_fields() -> None:
    import numpy as np

    model = _FakeModel(
        wavelength_nm=520.0,
        label="x",
        enabled=True,
        phase=_FakePhase(0.0),
        samples=np.array([1.0, 2.0]),
    )
    fields = provenance.scalar_fields(model)
    assert "samples" not in fields


def test_scalar_fields_on_a_real_optical_model() -> None:
    from uowc.optics import HaltrinModel

    model = HaltrinModel(
        wavelength_nm=520.0,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.02,
    )
    fields = provenance.scalar_fields(model)
    assert fields["wavelength_nm"] == 520.0
    assert fields["pure_water_absorption_m_inv"] == 0.1
    assert fields["chlorophyll_absorption_exponent"] == 0.602
    assert "water_phase_asymmetry" in fields
    assert "particle_phase_asymmetry" in fields


def test_scalar_fields_output_is_json_serializable_for_content_hash() -> None:
    from uowc.optics import HaltrinModel

    model = HaltrinModel(
        wavelength_nm=520.0,
        pure_water_absorption_m_inv=0.1,
        chlorophyll_specific_absorption_m2_mg=0.04,
        pure_water_scattering_m_inv=0.02,
    )
    # Must not raise: scalar_fields() feeds content_hash() in real provenance use.
    provenance.content_hash(provenance.scalar_fields(model))
