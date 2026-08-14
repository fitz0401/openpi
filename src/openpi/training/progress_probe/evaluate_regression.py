"""Evaluate one frozen source-trained v3 probe on unseen Split-A targets."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random

import numpy as np
import torch

from openpi.training.low_data.experiment import TaskRef
from openpi.training.progress_probe.dataset import LiberoProgressData
from openpi.training.progress_probe.dataset import minimum_separation
from openpi.training.progress_probe.dataset import task_manifest
from openpi.training.progress_probe.metrics import regression_metrics
from openpi.training.progress_probe.model import ProgressRegressionVisionLanguageLite
from openpi.training.progress_probe.model import build_vision_language_regression_model
from openpi.training.progress_probe.regression_config import load_regression_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-eval-demos", type=int)
    return parser.parse_args()


def wrong_language_map(
    instructions: dict[TaskRef, str], instruction_pool: dict[TaskRef, str], seed: int
) -> dict[TaskRef, str]:
    mapping = {}
    for ref, instruction in instructions.items():
        candidates = sorted((candidate, text) for candidate, text in instruction_pool.items() if text != instruction)
        if not candidates:
            raise ValueError("Wrong-language evaluation requires at least two distinct instructions.")
        mapping[ref] = random.Random(f"regression-wrong-language:{seed}:{ref.key}").choice(candidates)[1]
    return mapping


@torch.inference_mode()
def score_episode(
    model: ProgressRegressionVisionLanguageLite,
    data: LiberoProgressData,
    episode_id: int,
    instruction: str,
    wrong_instruction: str | None,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions, wrong_predictions = [], []
    length = data.episode_length(episode_id)
    for start in range(0, length, batch_size):
        indices = list(range(start, min(start + batch_size, length)))
        images = data.images(episode_id, indices).to(device, non_blocking=True)
        prediction = model(images, [instruction] * len(indices))
        predictions.extend(prediction.float().cpu().tolist())
        if wrong_instruction is not None:
            wrong_prediction = model(images, [wrong_instruction] * len(indices))
            wrong_predictions.extend(wrong_prediction.float().cpu().tolist())
    wrong = np.asarray(wrong_predictions) if wrong_instruction is not None else None
    return np.asarray(predictions), wrong


def aggregate(rows: list[dict], keys: tuple[str, ...], role: str) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, selected in groups.items():
        record = dict(zip(keys, key, strict=True))
        record.update(
            {
                "probe_source_subset": selected[0]["probe_source_subset"],
                "task_role": role,
                "num_target_demos": sum(int(row.get("num_target_demos", 1)) for row in selected),
                "checkpoint": selected[0]["checkpoint"],
                "seed": selected[0]["seed"],
            }
        )
        for field in ("r2", "mae", "spearman_rho", "pairwise_accuracy", "wrong_language_r2"):
            values = [float(row[field]) for row in selected if row[field] is not None]
            record[field] = float(np.mean(values)) if values else None
            record[f"{field}_std"] = float(np.std(values)) if values else None
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in keys))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_regression_config(args.config)
    checkpoint = args.checkpoint or config.output_dir / f"progress_regression_lite_v{config.schema_version}.pt"
    device = torch.device(args.device)
    data = LiberoProgressData(
        config.dataset.repo_id,
        root=config.dataset.root,
        image_key=config.dataset.image_key,
    )
    model = build_vision_language_regression_model(config.model.clip_model).to(device)
    payload = model.load_trainable_checkpoint(checkpoint, map_location=device)
    expected = {
        "format_version": config.schema_version,
        "clip_model": config.model.clip_model,
        "use_proprio": False,
        "split_id": config.split().split_id,
        "probe_source_subset": config.probe_source_subset,
    }
    mismatches = {
        key: {"checkpoint": payload.get(key), "config": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint/config mismatch: {mismatches}")
    model.requires_grad_(requires_grad=False)
    model.eval()

    split = config.split()
    target_refs = list(config.target_refs())
    if config.evaluation.max_target_tasks is not None:
        target_refs = target_refs[: config.evaluation.max_target_tasks]
    episodes_by_task = data.task_episodes(split, tuple(target_refs))
    max_demos = args.max_eval_demos if args.max_eval_demos is not None else config.evaluation.max_eval_demos
    instructions = {ref: data.task_instruction(split, ref) for ref in target_refs}
    pool_refs = tuple(dict.fromkeys((*split.source_task_refs(), *split.target_task_refs())))
    pool = {ref: data.task_instruction(split, ref) for ref in pool_refs}
    wrong = wrong_language_map(instructions, pool, config.training.seed)

    per_demo = []
    selected_episodes = {}
    for task_index, ref in enumerate(target_refs, start=1):
        episodes = episodes_by_task[ref][:max_demos] if max_demos is not None else episodes_by_task[ref]
        selected_episodes[ref] = episodes
        print(
            f"[{task_index}/{len(target_refs)}] {config.probe_source_subset} -> {ref.key}: {len(episodes)} demos",
            flush=True,
        )
        for episode_id in episodes:
            length = data.episode_length(episode_id)
            targets = np.arange(length, dtype=np.float64) / (length - 1)
            predictions, wrong_predictions = score_episode(
                model,
                data,
                episode_id,
                instructions[ref],
                wrong[ref] if config.evaluation.wrong_language else None,
                batch_size=config.evaluation.batch_size,
                device=device,
            )
            metrics = regression_metrics(
                targets,
                predictions,
                min_separation=minimum_separation(length, config.evaluation.min_separation_fraction),
            )
            wrong_r2 = None
            if wrong_predictions is not None:
                wrong_r2 = regression_metrics(
                    targets,
                    wrong_predictions,
                    min_separation=minimum_separation(length, config.evaluation.min_separation_fraction),
                )["r2"]
            per_demo.append(
                {
                    "probe_source_subset": config.probe_source_subset,
                    "target_suite": ref.suite,
                    "target_task_id": ref.task_id,
                    "episode_id": episode_id,
                    "num_frames": length,
                    **metrics,
                    "wrong_language_r2": wrong_r2,
                    "checkpoint": str(checkpoint.resolve()),
                    "seed": config.training.seed,
                }
            )

    per_task = aggregate(per_demo, ("target_suite", "target_task_id"), "target_task")
    per_suite = aggregate(per_task, ("target_suite",), "target_suite")
    output_dir = config.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, rows in (("per_demo", per_demo), ("per_task", per_task), ("per_suite", per_suite)):
        write_csv(output_dir / f"{stem}.csv", rows)
        (output_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {
        "checkpoint_metadata": {key: value for key, value in payload.items() if key != "trainable_state_dict"},
        "tasks": task_manifest(split, data, selected_episodes),
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote v{config.schema_version} regression evaluation to {output_dir}")


if __name__ == "__main__":
    main()
