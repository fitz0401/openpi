"""Continual benchmark evaluation: fill the success matrix + learning curves via LIBERO rollouts.

Runs in the LIBERO conda environment (the one that can import `libero` and `openpi_client`). It is
driven entirely by a ``manifest.json`` produced by ``scripts/continual_finetune.py`` -- it does NOT
import the main ``openpi`` package, so it has no JAX/training dependencies.

For each checkpoint it (re)launches a policy server subprocess in the openpi (uv) environment
(`uv run scripts/serve_policy.py policy:checkpoint ...`), waits for the port, runs rollouts, then
shuts the server down. Output:

  * ``success_matrix.csv`` : rows = train stage (0 = optional pretrained baseline), cols = task id.
  * ``learning_curves.csv``: (stage, task_id, step, success_rate) recovery curves.
  * ``eval_results.json``  : raw structured results.

Metrics (average accuracy / forgetting / BWT / FWT) are computed separately by
``scripts/continual_metrics.py`` (run in the uv env) from ``success_matrix.csv``.

Example:
    python scripts/continual_eval.py \
        --manifest checkpoints/pi05_libero_object_continual/slice_v0/budget10/seed0/manifest.json \
        --repo-root /dodrio/scratch/projects/starting_2026_047/openpi \
        --num-trials 20 --num-trials-lc 10
"""

from __future__ import annotations

import collections
import contextlib
import csv
import dataclasses
import json
import logging
import math
import os
import pathlib
import signal
import socket
import subprocess
import time
from typing import Iterator

import numpy as np
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Path to the manifest.json written by continual_finetune.py.
    manifest: str
    # Repo root used to launch the policy server (where scripts/serve_policy.py lives).
    repo_root: str = "."
    # Training config name to serve the finetuned checkpoints with.
    serve_config: str = "pi05_libero_object_continual"
    # LIBERO suite (task strings in the manifest are matched against this suite's task languages).
    task_suite_name: str = "libero_object"

    # Rollout counts.
    num_trials: int = 20  # trials per matrix cell (full row after each stage)
    num_trials_lc: int = 10  # trials per intermediate learning-curve checkpoint
    max_steps: int = 280  # libero_object longest demo ~254
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224

    # Optional pretrained baseline (stage 0) to enable forward-transfer. If both are set, this
    # checkpoint is served and evaluated on all tasks as matrix row 0.
    baseline_config: str | None = None
    baseline_dir: str | None = None

    # Whether to also evaluate intermediate checkpoints for learning curves.
    eval_learning_curves: bool = True

    # Server launch settings.
    port: int = 8000
    server_gpu: str = "0"  # CUDA_VISIBLE_DEVICES for the server subprocess
    server_timeout_s: float = 600.0

    seed: int = 7
    # Output directory (defaults to the manifest's directory).
    out_dir: str | None = None


def _port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


@contextlib.contextmanager
def policy_server(repo_root: str, config: str, ckpt_dir: str, port: int, gpu: str, timeout_s: float) -> Iterator[None]:
    """Launch serve_policy.py for one checkpoint; tear it down on exit."""
    # Wait for any previous server on this port to be gone.
    t0 = time.time()
    while _port_open("0.0.0.0", port):
        if time.time() - t0 > 60:
            raise RuntimeError(f"Port {port} still occupied; cannot start server.")
        time.sleep(1.0)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # ``--port`` belongs to serve_policy.py's top-level Args, so Tyro requires it before the
    # ``policy:checkpoint`` subcommand. Keep this as an argv list as paths are user-controlled.
    cmd = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "policy:checkpoint",
        "--policy.config",
        config,
        "--policy.dir",
        ckpt_dir,
    ]
    logging.info("Launching policy server: %s", ckpt_dir)
    proc = subprocess.Popen(cmd, cwd=repo_root, env=env, start_new_session=True)
    try:
        t0 = time.time()
        while not _port_open("0.0.0.0", port):
            if proc.poll() is not None:
                raise RuntimeError(f"Policy server exited early (code {proc.returncode}) for {ckpt_dir}")
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Policy server did not come up within {timeout_s}s for {ckpt_dir}")
            time.sleep(2.0)
        logging.info("Policy server is up (%.0fs).", time.time() - t0)
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            proc.wait(timeout=60)
        # Ensure the port is released before the next server starts.
        t0 = time.time()
        while _port_open("0.0.0.0", port) and time.time() - t0 < 60:
            time.sleep(1.0)


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eval_task(client, task_suite, task_id: int, args: Args, num_trials: int) -> float:
    """Run `num_trials` rollouts of one LIBERO task; return success rate."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    successes = 0
    for episode_idx in range(num_trials):
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
        t = 0
        done = False
        while t < args.max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])
                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    successes += 1
                    break
                t += 1
            except Exception as e:  # noqa: BLE001
                logging.error("Rollout exception (task %d, ep %d): %s", task_id, episode_idx, e)
                break
    env.close()
    sr = successes / max(num_trials, 1)
    logging.info("Task %d (%s): SR = %d/%d = %.3f", task_id, task_description, successes, num_trials, sr)
    return sr


def _build_task_id_map(task_suite, task_strings: list[str]) -> list[int]:
    """Map each manifest task string to its LIBERO benchmark task id (normalized match)."""

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    lang_to_id = {norm(task_suite.get_task(i).language): i for i in range(task_suite.n_tasks)}
    ids = []
    for s in task_strings:
        tid = lang_to_id.get(norm(s))
        if tid is None:
            raise ValueError(f"Manifest task {s!r} not found in suite. Known: {list(lang_to_id)}")
        ids.append(tid)
    return ids


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    out_dir = pathlib.Path(args.out_dir or pathlib.Path(args.manifest).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_strings = manifest["task_strings"]
    n_tasks = manifest["n_tasks"]
    stages = manifest["stages"]

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = _build_task_id_map(task_suite, task_strings)  # benchmark id per training-stage column
    logging.info("Task-string -> benchmark id map: %s", dict(zip(task_strings, task_ids)))

    def connect():
        return _websocket_client_policy.WebsocketClientPolicy("0.0.0.0", args.port)

    # matrix[stage][col] (stage 0 = baseline, cols 1..N over training order); learning_curves rows.
    matrix: dict[int, dict[int, float]] = collections.defaultdict(dict)
    lc_rows: list[dict] = []

    # Optional pretrained baseline row (stage 0) for forward transfer.
    if args.baseline_config and args.baseline_dir:
        with policy_server(args.repo_root, args.baseline_config, args.baseline_dir, args.port, args.server_gpu, args.server_timeout_s):
            client = connect()
            for col, tid in enumerate(task_ids, start=1):
                matrix[0][col] = eval_task(client, task_suite, tid, args, args.num_trials)

    # Per-stage evaluation.
    for stage in stages:
        k = stage["stage"]
        ckpt_dir = stage["checkpoint_dir"]
        final_step = stage["final_step"]
        col_self = k  # this stage's own task column

        # Learning-curve: intermediate checkpoints, current task only.
        if args.eval_learning_curves:
            for step in stage["learning_curve_steps"]:
                if step == final_step:
                    continue  # measured below with the full row
                step_dir = str(pathlib.Path(ckpt_dir) / str(step))
                if not pathlib.Path(step_dir).exists():
                    logging.warning("Missing learning-curve checkpoint %s; skipping.", step_dir)
                    continue
                with policy_server(args.repo_root, args.serve_config, step_dir, args.port, args.server_gpu, args.server_timeout_s):
                    client = connect()
                    sr = eval_task(client, task_suite, task_ids[k - 1], args, args.num_trials_lc)
                lc_rows.append({"stage": k, "task_id": task_ids[k - 1], "step": step, "success_rate": sr})

        # Full matrix row: final checkpoint evaluated on every task.
        final_dir = str(pathlib.Path(ckpt_dir) / str(final_step))
        with policy_server(args.repo_root, args.serve_config, final_dir, args.port, args.server_gpu, args.server_timeout_s):
            client = connect()
            for col, tid in enumerate(task_ids, start=1):
                sr = eval_task(client, task_suite, tid, args, args.num_trials)
                matrix[k][col] = sr
                if col == col_self:
                    lc_rows.append({"stage": k, "task_id": tid, "step": final_step, "success_rate": sr})

        _write_outputs(out_dir, matrix, lc_rows, n_tasks, task_ids, manifest)

    logging.info("Evaluation complete. Outputs in %s", out_dir)


def _write_outputs(out_dir, matrix, lc_rows, n_tasks, task_ids, manifest) -> None:
    # success_matrix.csv
    with (out_dir / "success_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage"] + [f"task{c}_id{task_ids[c - 1]}" for c in range(1, n_tasks + 1)])
        for stage in sorted(matrix):
            w.writerow([stage] + [matrix[stage].get(c, "") for c in range(1, n_tasks + 1)])

    # learning_curves.csv
    with (out_dir / "learning_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "task_id", "step", "success_rate"])
        for r in lc_rows:
            w.writerow([r["stage"], r["task_id"], r["step"], r["success_rate"]])

    # eval_results.json
    (out_dir / "eval_results.json").write_text(
        json.dumps(
            {
                "budget": manifest["budget"],
                "seed": manifest["seed"],
                "n_tasks": n_tasks,
                "task_strings": manifest["task_strings"],
                "task_ids": task_ids,
                "matrix": {str(k): v for k, v in matrix.items()},
                "learning_curves": lc_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
