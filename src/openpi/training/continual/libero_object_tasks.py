"""Canonical task ordering for the LIBERO-Object continual benchmark.

The 10 LIBERO-Object task instructions in their canonical benchmark order (task ids 0..9 of the
``libero_object`` suite). This ordering is the single source of truth shared by the training
orchestrator (which filters the LeRobot dataset by these strings) and the evaluation orchestrator
(which maps them back to LIBERO benchmark task ids). Keeping it dependency-free lets both the
openpi (uv) environment and the LIBERO (conda) environment import it.

The continual sequence uses the first ``N`` of these (default N=5).
"""

LIBERO_OBJECT_TASKS: tuple[str, ...] = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
)
