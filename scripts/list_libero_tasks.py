"""List LIBERO task prompts and symbolic goals without importing the LIBERO runtime."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIBERO_ROOT = _REPO_ROOT / "third_party/libero/libero/libero"
_TASK_MAP_PATH = _LIBERO_ROOT / "benchmark/libero_suite_task_map.py"
_BENCHMARK_PATH = _LIBERO_ROOT / "benchmark/__init__.py"


def _literal_assignment(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal assignment {name!r} in {path}")


def _bddl_metadata(path: pathlib.Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    text = path.read_text()
    language_match = re.search(r"\(:language\s+([^\n\r]+)", text)
    language = language_match.group(1).strip().rstrip(")") if language_match else None
    goal_match = re.search(r"\(:goal\s+(.*?)\n\s*\)\s*\n\s*\)", text, flags=re.DOTALL)
    goal = " ".join(goal_match.group(1).split()) if goal_match else None
    return language, goal


def _parse_args(suites: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", *suites], default="all")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit JSON rather than a table.")
    return parser.parse_args()


def main() -> None:
    task_map = _literal_assignment(_TASK_MAP_PATH, "libero_task_map")
    task_orders = _literal_assignment(_BENCHMARK_PATH, "task_orders")
    suites = sorted(task_map)
    args = _parse_args(suites)
    if not 0 <= args.task_order_index < len(task_orders):
        raise ValueError(f"task_order_index must be in [0, {len(task_orders) - 1}]")

    records = []
    for suite in suites if args.suite == "all" else [args.suite]:
        names = task_map[suite]
        if suite != "libero_90":
            names = [names[index] for index in task_orders[args.task_order_index]]
        for task_id, name in enumerate(names):
            bddl_path = _LIBERO_ROOT / "bddl_files" / suite / f"{name}.bddl"
            bddl_language, symbolic_goal = _bddl_metadata(bddl_path)
            records.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_number": task_id + 1,
                    "prompt": name.replace("_", " "),
                    "bddl_language": bddl_language,
                    "symbolic_goal": symbolic_goal,
                    "bddl_path": str(bddl_path.relative_to(_REPO_ROOT)),
                }
            )

    if args.json:
        print(json.dumps(records, indent=2))
        return
    print("suite\ttask_id\ttask_number\tprompt\tbddl_language\tsymbolic_goal\tbddl_path")
    for record in records:
        print("\t".join("" if record[key] is None else str(record[key]) for key in record))


if __name__ == "__main__":
    main()
