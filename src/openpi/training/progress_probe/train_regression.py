"""Train Progress Regression Lite v3 on one configurable Split-A source subset."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional

from openpi.training.progress_probe.dataset import LiberoProgressData
from openpi.training.progress_probe.dataset import UniformVisionLanguageProgressSampler
from openpi.training.progress_probe.dataset import VisionLanguageRegressionBatch
from openpi.training.progress_probe.dataset import split_train_validation_episodes
from openpi.training.progress_probe.dataset import task_manifest
from openpi.training.progress_probe.model import ProgressRegressionVisionLanguageLite
from openpi.training.progress_probe.model import build_vision_language_regression_model
from openpi.training.progress_probe.regression_config import load_regression_config
from openpi.training.progress_probe.regression_config import resolved_regression_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_forward(
    model: ProgressRegressionVisionLanguageLite,
    batch: VisionLanguageRegressionBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch.images.to(device, non_blocking=True)
    predictions = model(images, batch.instructions)
    targets = batch.targets.to(device, non_blocking=True)
    return functional.smooth_l1_loss(predictions, targets), predictions, targets


@torch.no_grad()
def validate(
    model: ProgressRegressionVisionLanguageLite,
    sampler: UniformVisionLanguageProgressSampler,
    *,
    num_frames: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions, targets, losses = [], [], []
    count = 0
    while count < num_frames:
        current = min(batch_size, num_frames - count)
        loss, prediction, target = regression_forward(model, sampler.sample_batch(current), device)
        losses.append((float(loss), current))
        predictions.extend(prediction.float().cpu().tolist())
        targets.extend(target.float().cpu().tolist())
        count += current
    model.train()
    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    residual = np.square(target_array - prediction_array).sum()
    total = np.square(target_array - target_array.mean()).sum()
    return {
        "validation_loss": sum(loss * size for loss, size in losses) / count,
        "validation_mae": float(np.abs(target_array - prediction_array).mean()),
        "validation_r2": float(1 - residual / total) if total > 0 else 0.0,
    }


def write_curves(output_dir: pathlib.Path, rows: list[dict]) -> None:
    with (output_dir / "train_validation_curves.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "train_validation_curves.json").write_text(json.dumps(rows, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([row["optimizer_step"] for row in rows], [row["train_loss"] for row in rows], label="train")
    validation_rows = [row for row in rows if row["validation_loss"] is not None]
    if validation_rows:
        steps = [row["optimizer_step"] for row in validation_rows]
        axes[0].plot(steps, [row["validation_loss"] for row in validation_rows], marker="o", label="validation")
        axes[1].plot(steps, [row["validation_r2"] for row in validation_rows], marker="o")
    axes[0].set(xlabel="optimizer step", ylabel="SmoothL1 loss")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axes[1].set(xlabel="optimizer step", ylabel="sampled validation R²")
    figure.tight_layout()
    figure.savefig(output_dir / "train_validation_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_regression_config(args.config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved_regression_config(config), indent=2) + "\n")
    seed_everything(config.training.seed)
    device = torch.device(args.device)
    split = config.split()
    source_refs = list(config.source_refs())
    if config.training.max_source_tasks is not None:
        source_refs = source_refs[: config.training.max_source_tasks]
    if not source_refs:
        raise ValueError("No source tasks selected for regression training.")

    data = LiberoProgressData(
        config.dataset.repo_id,
        root=config.dataset.root,
        image_key=config.dataset.image_key,
    )
    source_episodes = data.task_episodes(split, tuple(source_refs))
    train_episodes, validation_episodes = split_train_validation_episodes(
        source_episodes,
        seed=config.training.seed,
        validation_demos_per_task=config.training.validation_demos_per_task,
        max_train_demos_per_task=config.training.max_train_demos_per_task,
    )
    manifest = {
        "split_id": split.split_id,
        "probe_source_subset": config.probe_source_subset,
        "source_suites": list(config.source_suites),
        "target_suites": list(config.target_suites),
        "successful_expert_demos_only": config.dataset.successful_expert_demos_only,
        "training_source_tasks": task_manifest(split, data, train_episodes),
        "within_task_validation": task_manifest(split, data, validation_episodes),
        "unseen_target_tasks": task_manifest(split, data, data.task_episodes(split, config.target_refs())),
    }
    (output_dir / "task_manifests.json").write_text(json.dumps(manifest, indent=2) + "\n")

    train_sampler = UniformVisionLanguageProgressSampler(data, split, train_episodes, seed=config.training.seed)
    validation_sampler = UniformVisionLanguageProgressSampler(
        data, split, validation_episodes, seed=config.training.seed + 1
    )
    model = build_vision_language_regression_model(config.model.clip_model).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    curves = []
    start_time = time.monotonic()
    for step in range(1, config.training.optimizer_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = regression_forward(model, train_sampler.sample_batch(config.training.batch_size), device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.training.max_grad_norm)
        optimizer.step()
        should_validate = (
            step == 1 or step % config.training.validate_every == 0 or step == config.training.optimizer_steps
        )
        should_log = step == 1 or step % config.training.log_every == 0 or step == config.training.optimizer_steps
        validation = (
            validate(
                model,
                validation_sampler,
                num_frames=config.training.validation_frames_per_task * len(validation_sampler.refs),
                batch_size=config.training.batch_size,
                device=device,
            )
            if should_validate
            else {"validation_loss": None, "validation_mae": None, "validation_r2": None}
        )
        if should_log or should_validate:
            row = {
                "optimizer_step": step,
                "train_loss": float(loss.detach()),
                **validation,
                "elapsed_seconds": time.monotonic() - start_time,
            }
            curves.append(row)
            print(json.dumps(row), flush=True)

    checkpoint = output_dir / f"progress_regression_lite_v{config.schema_version}.pt"
    torch.save(
        {
            "format_version": config.schema_version,
            "model": f"Progress Regression Lite v{config.schema_version}",
            "objective": "SmoothL1(normalized_demo_progress)",
            "clip_model": config.model.clip_model,
            "clip_weights_included": False,
            "use_proprio": False,
            "seed": config.training.seed,
            "optimizer_steps": config.training.optimizer_steps,
            "split_id": split.split_id,
            "probe_source_subset": config.probe_source_subset,
            "source_suites": list(config.source_suites),
            "trainable_parameter_count": model.trainable_parameter_count(),
            "trainable_state_dict": {key: value.detach().cpu() for key, value in model.trainable_state_dict().items()},
        },
        checkpoint,
    )
    write_curves(output_dir, curves)
    print(
        f"Saved {config.probe_source_subset} probe ({model.trainable_parameter_count():,} parameters) to {checkpoint}"
    )


if __name__ == "__main__":
    main()
