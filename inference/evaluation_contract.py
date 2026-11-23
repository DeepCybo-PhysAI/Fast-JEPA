

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def audit_checkpoint(policy: str, checkpoint_root: str | Path) -> dict | None:

    if policy == "random":
        return None
    root = Path(checkpoint_root).expanduser().resolve()
    candidate = Path(policy).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else root / candidate
    if candidate.is_dir():
        weights = sorted(candidate.glob("*.pt"))
        if len(weights) != 1:
            raise ValueError(
                f"Checkpoint directory must contain exactly one .pt file: {candidate}"
            )
        checkpoint = weights[0]
    elif candidate.is_file():
        checkpoint = candidate
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")

    try:
        portable_path = checkpoint.relative_to(root).as_posix()
    except ValueError:
        portable_path = checkpoint.name
    record = {
        "policy_id": policy,
        "runtime_file": portable_path,
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "state_dict_load": "strict",
    }
    config_path = checkpoint.parent / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record["config_sha256"] = _sha256(config_path)
        record["legacy_ablation_config_detected"] = _contains_key(
            config, "actor_warmstart"
        )
        record["model_target"] = config.get("_target_")
    else:
        record["config_sha256"] = None
        record["legacy_ablation_config_detected"] = None
        record["model_target"] = None
    run_metadata_path = checkpoint.parent / "run_metadata.json"
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        record["run_metadata_sha256"] = _sha256(run_metadata_path)
        record["training_seed"] = run_metadata.get("seed")
        record["training_commit"] = run_metadata.get("git_commit")
        record["training_sdpa"] = run_metadata.get("sdpa")
        record["training_status"] = run_metadata.get("status")
    else:
        record["run_metadata_sha256"] = None
        record["training_seed"] = None
        record["training_commit"] = None
        record["training_sdpa"] = None
        record["training_status"] = None
    return record
