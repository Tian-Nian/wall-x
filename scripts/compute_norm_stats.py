#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

import pyarrow.parquet as pq
from tqdm import tqdm

from normalize import RunningStats
from wall_x.data.utils import KEY_MAPPINGS

DOF_ORDER_MAPPING = {
    "zju_data_20260209": [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ],
    "put_back_block-demo_clean_200": [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ],
}


def _to_serializable(stats):
    return {
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
        "q01": stats.q01.tolist(),
        "q99": stats.q99.tolist(),
    }


def _build_action_statistic_dof(repo_id: str, action_stats, state_stats) -> dict:
    dof_names = DOF_ORDER_MAPPING.get(repo_id)
    if dof_names is None:
        return {}

    robot_stats = {}
    action_q01 = action_stats.q01.tolist()
    action_q99 = action_stats.q99.tolist()
    state_q01 = state_stats.q01.tolist()
    state_q99 = state_stats.q99.tolist()

    # The official trainer loads one action-statistic dictionary for both action and
    # proprioception. For this dataset the joint ordering is identical in state/action,
    # so we use a merged q01-q99 envelope that safely covers both distributions.
    for idx, name in enumerate(dof_names):
        lower = min(action_q01[idx], state_q01[idx])
        upper = max(action_q99[idx], state_q99[idx])
        robot_stats[name] = {
            "min": [lower],
            "delta": [upper - lower],
        }

    return {repo_id: robot_stats}


def _iter_parquet_files(root: Path):
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"LeRobot data directory not found: {data_dir}")
    return sorted(data_dir.rglob("episode_*.parquet"))


def compute_norm_stats(repo_id: str, root: str | None, output_path: str) -> Path:
    action_key = KEY_MAPPINGS[repo_id]["action"]
    state_key = KEY_MAPPINGS[repo_id]["state"]

    running = {
        action_key: RunningStats(),
        state_key: RunningStats(),
    }

    if root is None:
        raise ValueError("`--root` is required for local LeRobot parquet statistics.")

    root_path = Path(root)
    parquet_files = _iter_parquet_files(root_path)

    for parquet_file in tqdm(parquet_files, desc=f"compute stats for {repo_id}"):
        table = pq.read_table(parquet_file, columns=[action_key, state_key])
        action_array = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
        state_array = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
        running[action_key].update(action_array)
        running[state_key].update(state_array)

    action_stats = running[action_key].get_statistics()
    state_stats = running[state_key].get_statistics()

    payload = {
        "repo_id": repo_id,
        "norm_stats": {
            action_key: _to_serializable(action_stats),
            state_key: _to_serializable(state_stats),
        },
    }
    payload.update(_build_action_statistic_dof(repo_id, action_stats, state_stats))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute WALL-X compatible norm_stats.json for a LeRobot dataset."
    )
    parser.add_argument(
        "--repo-id",
        default="lerobot/aloha_mobile_cabinet",
        help="LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Local LeRobot root. If omitted, LeRobot defaults are used.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output json path, e.g. /abs/path/norm_stats.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = compute_norm_stats(args.repo_id, args.root, args.output)
    print(f"Saved norm stats to {output_path}")


if __name__ == "__main__":
    main()
