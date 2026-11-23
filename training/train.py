

from __future__ import annotations

import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import socket
import subprocess
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from sdpa_policy import (
    sdpa_kernel_context,
    sdpa_policy_metadata,
    validate_sdpa_backend,
)
from utils import (
    PreviousActionDataset,
    RunMetadataCallback,
    SaveCkptCallback,
    TrainingProgressCallback,
    get_column_normalizer,
    get_column_stats,
    get_img_preprocessor,
)




def construct_intents(
    embeddings: torch.Tensor, intent_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:

    current = embeddings[:, :-1]
    successor = embeddings[:, 1:]
    local = successor - current
    goal = embeddings[:, -1:].detach().expand_as(current)
    displacement = goal - current
    if intent_mode == "waypoint":
        remaining = torch.arange(
            current.size(1),
            0,
            -1,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).view(1, -1, 1)
        deployment = displacement / remaining
    elif intent_mode == "goal_displacement":
        deployment = displacement
    else:
        raise ValueError(f"Unknown intent mode: {intent_mode}")
    return local, deployment


def historical_construct_intents(
    embeddings: torch.Tensor,
    intent_mode: str,
    local_start: int = 0,
    goal_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:

    max_start = embeddings.size(1) - 2
    if local_start < 0 or local_start > max_start:
        raise ValueError(
            f"local_start must be in [0, T-2], got {local_start} for T={embeddings.size(1)}"
        )
    if goal_start < 0 or goal_start > max_start:
        raise ValueError(
            f"goal_start must be in [0, T-2], got {goal_start} for T={embeddings.size(1)}"
        )
    current = embeddings[:, local_start:-1]
    successor = embeddings[:, local_start + 1 :]
    local = successor - current
    deployment_current = embeddings[:, goal_start:-1]
    goal = embeddings[:, -1:].detach().expand_as(deployment_current)
    displacement = goal - deployment_current
    if intent_mode == "waypoint":
        remaining = torch.arange(
            deployment_current.size(1),
            0,
            -1,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).view(1, -1, 1)
        deployment = displacement / remaining
    elif intent_mode == "goal_displacement":
        deployment = displacement
    else:
        raise ValueError(f"Unknown intent mode: {intent_mode}")
    return local, deployment

def historical_paired_intent_action_loss(model, embeddings, actions, previous_actions, cfg):

    local_weight = float(cfg.loss.intent.local_weight)
    goal_weight = float(cfg.loss.intent.goal_weight)
    local_start = int(cfg.loss.intent.local_start)
    goal_start = int(cfg.loss.intent.goal_start)
    if local_weight <= 0 and goal_weight <= 0:
        zero = embeddings.new_zeros(())
        return {
            "action_loss": zero,
            "local_nll": zero,
            "goal_nll": zero,
            "local_mae": zero,
            "goal_mae": zero,
        }

    local_intent, goal_intent = historical_construct_intents(
        embeddings,
        model.intent_mode,
        local_start=local_start,
        goal_start=goal_start,
    )
    zero = embeddings.new_zeros(())
    local_stats = {"loss": zero, "mae": zero[None]}
    if local_weight > 0:
        local_action = actions[:, local_start : embeddings.size(1) - 1]
        local_previous = previous_actions[:, local_start : embeddings.size(1) - 1]
        local_stats = model.action_nll(
            z=embeddings[:, local_start:-1],
            intent=local_intent,
            previous_action=local_previous,
            target_action=local_action,
        )

    goal_stats = {"loss": zero, "mae": zero[None]}
    if goal_weight > 0:
        goal_action = actions[:, goal_start : embeddings.size(1) - 1]
        goal_previous = previous_actions[:, goal_start : embeddings.size(1) - 1]
        goal_stats = model.action_nll(
            z=embeddings[:, goal_start:-1],
            intent=goal_intent,
            previous_action=goal_previous,
            target_action=goal_action,
        )

    return {
        "action_loss": local_weight * local_stats["loss"]
        + goal_weight * goal_stats["loss"],
        "local_nll": local_stats["loss"],
        "goal_nll": goal_stats["loss"],
        "local_mae": local_stats["mae"].mean().detach(),
        "goal_mae": goal_stats["mae"].mean().detach(),
    }

def paired_intent_action_loss(model, embeddings, previous_actions, target_actions, cfg):
    result = historical_paired_intent_action_loss(model, embeddings, target_actions, previous_actions, cfg)
    loss = result.pop("action_loss")
    result = {key: value.detach() for key, value in result.items()}
    result["intent_loss"] = loss
    result["intent_mae"] = (float(cfg.loss.intent.local_weight) * result["local_mae"] + float(cfg.loss.intent.goal_weight) * result["goal_mae"]) / (float(cfg.loss.intent.local_weight) + float(cfg.loss.intent.goal_weight))
    return result



def fast_jepa_forward(self, batch, stage, cfg):
    if "previous_action" not in batch:
        raise KeyError("Boundary-aware previous_action is required")
    if not torch.isfinite(batch["action"]).all() or not torch.isfinite(
        batch["previous_action"]
    ).all():
        raise ValueError("Training action pairs contain non-finite values")
    output = self.model.encode(batch)
    embeddings = output["emb"]
    if self.model.world_model_type != 'fast':
        raise ValueError("Fast-JEPA training requires world_model_type='fast'")
    from fast_world_model import fast_prediction_loss
    pred_loss, wm_metrics = fast_prediction_loss(self.model, embeddings, batch["action"])
    sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
    intent = paired_intent_action_loss(
        self.model,
        embeddings,
        batch["previous_action"][:, :-1],
        batch["action"][:, :-1],
        cfg,
    )
    loss = (
        float(cfg.loss.forward_weight) * pred_loss
        + float(cfg.loss.sigreg.weight) * sigreg_loss
        + intent["intent_loss"]
    )

    metrics = {
        **wm_metrics,
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        **intent,
    }
    self.log_dict(
        {f"{stage}/{key}": value for key, value in metrics.items()},
        on_step=stage == "train",
        on_epoch=True,
        prog_bar=key_in_progress_bar(metrics),
        sync_dist=True,
    )
    return metrics


def key_in_progress_bar(metrics):

    return "loss" in metrics


def validate_training_config(cfg) -> None:

    if cfg.world_model_type != 'fast' or cfg.model.get('world_model_type') != 'fast':
        raise ValueError("Fast-JEPA training only accepts world_model_type='fast'")
    if cfg.model.get('_target_') != 'hybrid_jepa.JEPA':
        raise ValueError('Fast-JEPA training requires hybrid_jepa.JEPA')

    backend = validate_sdpa_backend(cfg.sdpa_backend)
    if backend != "math":
        raise ValueError(
            "Fast-JEPA training requires on Math SDPA; "
            f"received sdpa_backend={backend!r}"
        )
    if int(cfg.num_frames) < 3:
        raise ValueError("Actor training requires num_frames >= 3")
    if not 2 <= int(cfg.history_size) < int(cfg.num_frames):
        raise ValueError("history_size must be in [2, num_frames)")
    if not 0.0 < float(cfg.train_split) < 1.0:
        raise ValueError("train_split must be strictly between 0 and 1")
    if cfg.model.intent_mode not in {"waypoint", "goal_displacement"}:
        raise ValueError(f"Unknown intent mode: {cfg.model.intent_mode}")
    if int(cfg.loader.num_workers) == 0 and bool(cfg.loader.persistent_workers):
        raise ValueError("persistent_workers requires loader.num_workers > 0")
    if int(cfg.log_interval_steps) < 1:
        raise ValueError("log_interval_steps must be positive")
    weights = (
        float(cfg.loss.forward_weight),
        float(cfg.loss.sigreg.weight),
        float(cfg.loss.intent.local_weight),
        float(cfg.loss.intent.goal_weight),
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("Training loss weights must be non-negative")


def build_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    is_lance = str(dataset_name).endswith(".lance")
    if is_lance:


        dataset_cfg["keys_to_cache"] = []
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    action_mean, action_std = get_column_stats(dataset, "action")
    transforms = [
        get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)
    ]


    transforms.append(
        get_column_normalizer(
            dataset,
            "action",
            "action",
            stats=(action_mean, action_std),
            raw_nan_value=0.0,
        )
    )

    with open_dict(cfg):
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim
        if cfg.model.intent_actor is not None:
            cfg.model.intent_actor.action_dim = action_dim
    dataset.transform = spt.data.transforms.Compose(*transforms)
    if is_lance:
        getattr(dataset, "_cache", {}).pop("action", None)
        update_fetch_columns = getattr(dataset, "_update_fetch_columns", None)
        if update_fetch_columns is not None:
            update_fetch_columns()
    return PreviousActionDataset(dataset, action_mean, action_std)


def build_dataloaders(cfg, dataset):

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, validation_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader


def build_model(cfg):

    model = hydra.utils.instantiate(cfg.model)
    if cfg.init_weights_path:
        state = torch.load(cfg.init_weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    return model


def historical_lr_metadata(learning_rate):
    return {"type": "LinearWarmupCosineAnnealingLR", "peak_learning_rate": float(learning_rate), "warmup_start_lr": 0.0, "eta_min": 0.0, "warmup_steps": "max(1,int(0.01*estimated_stepping_batches))", "max_steps": "estimated_stepping_batches", "configured_interval": "epoch", "effective_interval": "optimizer_step (stable_pretraining manual optimization)"}



def build_training_module(cfg, model):

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    return spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(fast_jepa_forward, cfg=cfg),
        optim=optimizers,
    )


def current_git_commit() -> str:

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_run_artifacts(cfg, *, training_mode: str = "single_task"):

    checkpoint_root = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    )
    run_dir = checkpoint_root / cfg.output_model_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains artifacts: {run_dir}. "
            "Choose a new output_model_name; existing runs are never overwritten."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train_config.yaml")
    metadata = {
        "method": "Fast-JEPA",
        "world_model_type": str(cfg.world_model_type),
        "fast_horizon": int(cfg.fast_horizon),
        "fast_target_stop_gradient": bool(cfg.fast_target_stop_gradient),
        "training_mode": training_mode,
        "intent_mode": str(cfg.model.intent_mode),
        "seed": int(cfg.seed),
        "task": str(cfg.data.task_name),
        "dataset": str(cfg.data.dataset.name),
        "action_dim": int(cfg.model.intent_actor.action_dim),
        "num_frames": int(cfg.num_frames),
        "history_size": int(cfg.history_size),
        "max_epochs": int(cfg.trainer.max_epochs),
        "log_interval_steps": int(cfg.log_interval_steps),
        "metrics_jsonl": str(run_dir / "metrics" / "train.jsonl"),
        "optimizer": {
            "type": str(cfg.optimizer.type),
            "learning_rate": float(cfg.optimizer.lr),
            "weight_decay": float(cfg.optimizer.weight_decay),
            "fused": bool(cfg.optimizer.get("fused", False)),
            "scheduler": historical_lr_metadata(cfg.optimizer.lr),
        },
        "forward_supervision": "fixed_start_dense_prefix",
        "intent_supervision": "physical_0_6_goal_0_6",
        "physical_transition_count": int(cfg.num_frames) - 1,
        "goal_transition_count": int(cfg.num_frames) - 1,
        "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
        "git_commit": current_git_commit(),
        "hostname": socket.gethostname(),
        "numeric_policy": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "actor_batching": "separate_local_goal",
            "predictor_batching": "parallel_prefixes",
        },
        "sdpa": sdpa_policy_metadata(cfg.sdpa_backend),
        "status": "configured",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return run_dir, metadata


def build_trainer(cfg, run_dir, logger):

    return pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=1,
            ),
            RunMetadataCallback(run_dir / "run_metadata.json"),
            TrainingProgressCallback(
                run_dir / "metrics" / "train.jsonl",
                interval_steps=int(cfg.log_interval_steps),
                task=str(cfg.data.task_name),
            ),
        ],
        logger=logger,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        default_root_dir=run_dir,
    )


def configure_deterministic_math(seed: int) -> None:

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(seed, workers=True)

@hydra.main(
    version_base=None, config_path="./config/fast", config_name="pusht"
)
def run(cfg):
    validate_training_config(cfg)
    configure_deterministic_math(int(cfg.seed))
    dataset = build_dataset(cfg)
    train_loader, validation_loader = build_dataloaders(cfg, dataset)
    model = build_model(cfg)
    module = build_training_module(cfg, model)
    data_module = spt.data.DataModule(
        train=train_loader, val=validation_loader
    )
    run_dir, _ = prepare_run_artifacts(cfg)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = build_trainer(cfg, run_dir, logger)
    with sdpa_kernel_context(cfg.sdpa_backend):
        trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    run()
