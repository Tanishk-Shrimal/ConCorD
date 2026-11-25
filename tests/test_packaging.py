"""Packaging metadata regression tests for the production ConCorD install."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_runtime_dependencies_cover_direct_production_imports():
    project = _pyproject()["project"]
    requirements = tuple(project["dependencies"])
    names = {
        requirement.split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for requirement in requirements
    }

    assert {
        "numpy",
        "torch",
        "h5py",
        "opt-einsum",
        "pydantic",
        "pyyaml",
        "typer",
    } <= names


def test_optional_extras_separate_test_and_release_tooling():
    project = _pyproject()["project"]
    dev = tuple(project["optional-dependencies"]["dev"])
    release = tuple(project["optional-dependencies"]["release"])
    lowered_dev = tuple(requirement.lower() for requirement in dev)
    lowered_release = tuple(requirement.lower() for requirement in release)

    assert any(requirement.startswith("pytest") for requirement in lowered_dev)
    assert not any(requirement.startswith("build") for requirement in lowered_dev)
    assert not any(requirement.startswith("twine") for requirement in lowered_dev)
    assert any(requirement.startswith("build") for requirement in lowered_release)
    assert any(requirement.startswith("twine") for requirement in lowered_release)


def test_console_entry_point_targets_single_cli_main():
    project = _pyproject()["project"]
    assert project["scripts"]["concord"] == "concord_lqcd.cli:main"


def test_license_is_mit_and_shipped():
    project = _pyproject()["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n\nCopyright (c) 2026 ")


def test_release_readme_and_project_urls_are_declared():
    project = _pyproject()["project"]
    assert project["readme"] == "README.md"
    assert (ROOT / project["readme"]).is_file()

    urls = project["urls"]
    assert urls["Repository"] == "https://github.com/Tanishk-Shrimal/ConCorD"
    assert urls["Issues"] == "https://github.com/Tanishk-Shrimal/ConCorD/issues"


def test_package_version_matches_project_metadata_without_importing_package():
    project_version = _pyproject()["project"]["version"]
    init_text = (ROOT / "concord_lqcd" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == project_version == "1.0.0"

    plan_text = (ROOT / "concord_lqcd" / "job" / "plan.py").read_text(encoding="utf-8")
    fallback = re.search(
        r'except PackageNotFoundError:.*?\n\s*return "([^"]+)"', plan_text, re.DOTALL
    )
    assert fallback is not None
    assert fallback.group(1) == project_version


def test_cli_starts_in_a_fresh_python_process():
    """Catch package-initialization cycles hidden by pytest import ordering."""

    result = subprocess.run(
        [sys.executable, "-m", "concord_lqcd", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "hardware" in result.stdout
    assert "plan" in result.stdout
    assert "inspect" in result.stdout
    assert "run" in result.stdout
