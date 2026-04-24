# from datasets import load_dataset
# ds = load_dataset("daixianjie/rlbench_joint_vel_action_lerobot_train")
# ds.save_to_disk("/dodrio/scratch/projects/starting_2026_047/dataset/peract_lerobot_train")

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="daixianjie/rlbench_joint_vel_action_lerobot_train",
    repo_type="dataset",
    local_dir="/dodrio/scratch/projects/starting_2026_047/dataset/peract_lerobot_train",
    token=None,
    resume_download=True
)