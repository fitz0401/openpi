"""
RLBench evaluation script using a remote policy server (peract fork of RLBench).

Run the policy server first (in the openpi uv environment):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config pi05_rlbench_lora \
        --policy.dir checkpoints/pi05_rlbench_lora/<exp>/<step>

Then run this script (using the guide conda Python):
    /home/ze/miniconda3/envs/guide/bin/python examples/rlbench/main.py \
        --task_names push_button open_drawer \
        --num_trials_per_task 10
"""

import collections
import dataclasses
import importlib
import logging
import pathlib
from typing import Sequence

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

try:
    from rlbench.environment import Environment
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.observation_config import ObservationConfig
except ImportError as e:
    raise ImportError(
        f"RLBench (peract fork) import failed: {e}\n"
        "Install with:\n"
        "  git clone https://github.com/MohitShridhar/RLBench.git\n"
        "  cd RLBench && git checkout peract && pip install -e . && cd .."
    ) from e


class _JointVelocity(JointVelocity):
    """MoveArmThenGripper passes `ignore_collisions` but JointVelocity doesn't accept it."""

    def action(self, scene, action, ignore_collisions=True):
        super().action(scene, action)


# Standard 18-task peract suite
PERACT_TASKS: tuple[str, ...] = (
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_sorter",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "turn_tap",
    "place_wine_at_rack_location",
    "sweep_to_dustpan_of_size",
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Policy server
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000

    #################################################################################################################
    # RLBench evaluation
    #################################################################################################################
    task_names: Sequence[str] = ("push_button",)
    num_trials_per_task: int = 10
    max_steps_per_episode: int = 500
    # Policy is queried once every `replan_steps` env steps.
    replan_steps: int = 1
    resize_size: int = 224

    #################################################################################################################
    # Demo loading (optional)
    #################################################################################################################
    # Path to peract dataset root, e.g. /home/ze/data/peract.
    # Layout: <dataset_root>/test/<task_name>/variation0/episodes/episode<i>/
    dataset_root: str = ""
    use_demos: bool = False

    #################################################################################################################
    # Output
    #################################################################################################################
    video_out_path: str = "data/rlbench/videos"
    seed: int = 7


def _task_name_to_class(task_name: str):
    """Convert snake_case task name to its RLBench Task class."""
    class_name = "".join(w.capitalize() for w in task_name.split("_"))
    mod = importlib.import_module(f"rlbench.tasks.{task_name}")
    return getattr(mod, class_name)


def _get_obs_state(obs) -> np.ndarray:
    """8-dim state: 7 joint positions + 1 gripper open."""
    return np.concatenate([np.array(obs.joint_positions), [float(obs.gripper_open)]])


def _make_action(policy_action: np.ndarray) -> list:
    """
    Pass 8-dim policy output to RLBench.

    MoveArmThenGripper(JointVelocity, Discrete) expects:
        [joint_vel(7), gripper_binary(1)]  -- 8 dims total.
    """
    return np.asarray(policy_action).tolist()


def eval_rlbench(args: Args) -> None:
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to policy server at %s:%d", args.host, args.port)

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.wrist_camera.rgb = True
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    action_mode = MoveArmThenGripper(
        arm_action_mode=_JointVelocity(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        dataset_root=args.dataset_root,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    total_episodes = 0
    total_successes = 0
    results: dict[str, tuple[int, int]] = {}

    for task_name in args.task_names:
        logging.info("=" * 60)
        logging.info("Task: %s", task_name)
        logging.info("=" * 60)

        task_class = _task_name_to_class(task_name)
        task = env.get_task(task_class)

        demos = None
        if args.use_demos and args.dataset_root:
            try:
                demos = task.get_demos(args.num_trials_per_task, live_demos=False)
                logging.info("Loaded %d demos from %s", len(demos), args.dataset_root)
            except Exception as exc:
                logging.warning("Demo loading failed (%s). Using random reset.", exc)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if demos is not None and episode_idx < len(demos):
                descriptions, obs = task.reset_to_demo(demos[episode_idx])
            else:
                descriptions, obs = task.reset()

            task_description = descriptions[0] if descriptions else task_name.replace("_", " ")

            t = 0
            done = False
            replay_images: list[np.ndarray] = []
            action_plan: collections.deque = collections.deque()

            while t < args.max_steps_per_episode:
                try:
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            np.asarray(obs.front_rgb), args.resize_size, args.resize_size
                        )
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            np.asarray(obs.wrist_rgb), args.resize_size, args.resize_size
                        )
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": _get_obs_state(obs),
                            "prompt": task_description,
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan_steps={args.replan_steps} but policy returned "
                            f"{len(action_chunk)} actions"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    raw_action = action_plan.popleft()
                    obs, reward, done = task.step(_make_action(raw_action))

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as exc:
                    logging.error("Exception at step %d: %s", t, exc)
                    import traceback

                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            if replay_images:
                suffix = "success" if done else "failure"
                video_path = (
                    pathlib.Path(args.video_out_path)
                    / f"{task_name}_{episode_idx:03d}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    str(video_path),
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                logging.info("Saved %s", video_path)

            logging.info(
                "Episode %d/%d: %s  (task SR: %d/%d = %.1f%%)",
                episode_idx + 1,
                args.num_trials_per_task,
                "SUCCESS" if done else "failure",
                task_successes,
                task_episodes,
                100.0 * task_successes / task_episodes,
            )

        results[task_name] = (task_successes, task_episodes)
        logging.info(
            "Task %s final: %d/%d (%.1f%%)",
            task_name,
            task_successes,
            task_episodes,
            100.0 * task_successes / task_episodes,
        )

    env.shutdown()

    logging.info("\n" + "=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    for name, (succ, eps) in results.items():
        logging.info("  %-40s %d/%d (%.1f%%)", name, succ, eps, 100.0 * succ / eps)
    logging.info("-" * 60)
    logging.info(
        "  %-40s %d/%d (%.1f%%)",
        "Overall",
        total_successes,
        total_episodes,
        100.0 * total_successes / total_episodes if total_episodes else 0.0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_rlbench(tyro.cli(Args))
