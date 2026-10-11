"""Project configuration: where data, results and external tools live.

All machine-specific paths are read from a YAML file instead of being hardcoded.
The file is looked up in this order:

1. the path in the ``BRAINPRED_CONFIG`` environment variable;
2. ``configs/paths.yaml`` at the repository root.

Copy ``configs/paths.example.yaml`` to ``configs/paths.yaml`` and edit it.
"""
import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
ELASTIX_PARAMS_DIR = REPO_ROOT / "configs" / "elastix"


@dataclass
class Config:
    # Folder containing one sub-folder per visit (HUNT3/, HUNT4/), each with one folder per patient.
    data_dir: Path
    # Root folder where every method writes its predictions, statistics and figures.
    results_dir: Path
    # Folder with the MNI ICBM152 2009c template files.
    mni_template_dir: Path
    # Scratch folder used by FreeSurfer / FastSurfer (FastSurfer runs through docker).
    temp_dir: Path
    # Folder containing the FreeSurfer ``license.txt``.
    freesurfer_license_dir: Path
    # Folder where model checkpoints are stored.
    checkpoints_dir: Path
    # Folder for intermediate artefacts shared between methods
    # (longitudinal registrations, inter-patient similarity tables, 2D slices).
    work_dir: Path


def _to_path(value):
    return None if value is None else Path(os.path.expandvars(os.path.expanduser(str(value))))


@lru_cache(maxsize=1)
def get_config(path=None) -> Config:
    """Load and cache the project configuration."""
    path = Path(path or os.environ.get("BRAINPRED_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Copy configs/paths.example.yaml to configs/paths.yaml and edit it, "
            "or set BRAINPRED_CONFIG to point to your config."
        )
    raw = yaml.safe_load(path.read_text()) or {}

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")

    return Config(**{k: _to_path(v) for k, v in raw.items()})
