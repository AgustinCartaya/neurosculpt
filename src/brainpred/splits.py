"""Subject lists and the train / validation split used in every experiment.

The split is defined in ``configs/splits.yaml`` (git-ignored, falls back to
``configs/splits.example.yaml``):

* subjects are all patients that have both a HUNT3 and a HUNT4 scan, sorted by ID;
* the first ``n_train`` subjects are used for training / as the reference population;
* the remaining subjects, plus ``extra_validation_ids``, are used for validation;
* ``excluded_patients`` are never used as reference subjects.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

from brainpred import paths
from brainpred.config import REPO_ROOT

HUNTS = [3, 4]


@lru_cache(maxsize=1)
def _load_splits():
    path = os.environ.get("BRAINPRED_SPLITS")
    candidates = [Path(path)] if path else [REPO_ROOT / "configs" / "splits.yaml",
                                            REPO_ROOT / "configs" / "splits.example.yaml"]
    for candidate in candidates:
        if candidate.exists():
            raw = yaml.safe_load(candidate.read_text()) or {}
            return {
                "n_train": int(raw.get("n_train", 0)),
                "extra_validation_ids": [str(i) for i in raw.get("extra_validation_ids", [])],
                "excluded_patients": [str(i) for i in raw.get("excluded_patients", [])],
            }
    raise FileNotFoundError(f"No splits file found in {candidates}")


def get_matching_ids():
    """IDs of the patients that have a scan in both HUNT3 and HUNT4, sorted."""
    def patient_ids(hunt):
        hunt_folder = paths.get_hunt_folder(hunt)
        return {e for e in os.listdir(hunt_folder) if os.path.isdir(os.path.join(hunt_folder, e))}

    return sorted(patient_ids(3) & patient_ids(4))


def get_train_ids():
    return get_matching_ids()[:_load_splits()["n_train"]]


def get_validation_ids():
    splits = _load_splits()
    return splits["extra_validation_ids"] + get_matching_ids()[splits["n_train"]:]


def get_excluded_patients():
    return list(_load_splits()["excluded_patients"])
