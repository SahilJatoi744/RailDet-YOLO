"""
Smoke tests for the repository.

These do NOT require a GPU or any dataset — they just confirm that:
1. The vendored `ultralytics` package imports correctly.
2. A stock YOLO model can be built and can run a forward pass.
3. The dataset config YAMLs in `configs/` are well-formed.
4. The pipeline scripts are importable / expose a CLI.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ultralytics_imports():
    import ultralytics  # noqa: F401
    from ultralytics import YOLO  # noqa: F401


def test_stock_model_builds_and_runs_forward_pass():
    """A stock (non-proprietary) YOLO config should build and run on a dummy input."""
    import torch
    from ultralytics import YOLO

    model = YOLO("yolo11n.yaml")  # builds from the architecture config, no pretrained weights downloaded
    dummy = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out = model.model(dummy)
    assert out is not None


@pytest.mark.parametrize("config_name", ["fault_seg.yaml", "laser.yaml"])
def test_dataset_configs_are_well_formed(config_name):
    config_path = REPO_ROOT / "configs" / config_name
    assert config_path.exists(), f"Missing {config_path}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("path", "train", "val", "nc", "names"):
        assert key in cfg, f"'{key}' missing from {config_name}"
    assert cfg["nc"] == len(cfg["names"]), "nc must match len(names)"


def test_no_proprietary_architecture_names_leak_into_the_repo():
    """
    Guard-rail test: fails the build if any withheld architecture identifiers
    accidentally get committed back into the public tree.
    """
    banned_terms = ["raildet", "RWC2f", "RWBottleneck", "DSSA", "AdaptiveScaleFusion", "DySample"]
    skip_dirs = {".git", "__pycache__", ".pytest_cache"}
    hits = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".ipynb", ".cff", ".toml", ".txt"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for term in banned_terms:
            if term in text:
                hits.append((str(path.relative_to(REPO_ROOT)), term))
    assert not hits, f"Found withheld architecture references: {hits}"


def test_scripts_have_working_help():
    for script in ["scripts/train.py", "scripts/yolo_predict_and_benchmark.py"]:
        script_path = REPO_ROOT / script
        assert script_path.exists()
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"{script} --help failed:\n{result.stdout}\n{result.stderr}"
