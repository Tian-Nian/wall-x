import copy
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wall_x.model.action_head import Normalizer
from wall_x.model.model_utils import load_wallx_processors
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


DEFAULT_CONFIG_PATH = REPO_ROOT / "workspace" / "lerobot_example" / "config_qact.yml"
DEFAULT_CAMERA_KEY = ["face_view", "left_wrist_view", "right_wrist_view"]
OPENPI_TO_WALLX_ORDER = np.array(
    [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64
)


def _load_prepare_batch():
    utils_path = REPO_ROOT / "wall_x" / "serving" / "policy" / "utils.py"
    spec = importlib.util.spec_from_file_location("wall_x_policy_utils_local", utils_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Failed to load prepare_batch from {utils_path}")
    spec.loader.exec_module(module)
    return module.prepare_batch


prepare_batch = _load_prepare_batch()


def _resolve_train_config_path(train_config_name):
    if train_config_name:
        candidate = Path(train_config_name).expanduser()
        search_paths = [
            candidate,
            Path.cwd() / candidate,
            REPO_ROOT / candidate,
            REPO_ROOT / "workspace" / "lerobot_example" / candidate,
            REPO_ROOT / "workspace" / "lerobot_example" / f"{candidate.name}.yml",
        ]
        for path in search_paths:
            if path.exists():
                return path.resolve()
    return DEFAULT_CONFIG_PATH.resolve()


def _load_train_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_dataset_name(train_config, deploy_cfg):
    if deploy_cfg.get("dataset_name"):
        return deploy_cfg["dataset_name"]
    customized = train_config.get("customized_robot_config", {})
    if customized.get("name"):
        return customized["name"]
    return train_config["data"]["lerobot_config"]["repo_id"]


def _sum_dims(config_dict):
    return int(sum(config_dict.values()))


def _load_action_statistics(norm_stats_path):
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        raw_stats = json.load(f)

    filtered_stats = {}
    for robot_name, robot_stats in raw_stats.items():
        if not isinstance(robot_stats, dict):
            continue
        has_dof_stats = any(
            isinstance(value, dict) and "min" in value and "delta" in value
            for value in robot_stats.values()
        )
        if has_dof_stats:
            filtered_stats[robot_name] = robot_stats

    if not filtered_stats:
        raise ValueError(f"No valid robot statistics found in {norm_stats_path}")

    return filtered_stats


def _build_normalizers(train_config, model_path):
    action_ckpt = Path(model_path) / "normalizer_action.pth"
    propri_ckpt = Path(model_path) / "normalizer_propri.pth"
    stats = None

    if action_ckpt.exists():
        normalizer_action = Normalizer.from_ckpt(str(action_ckpt))
    else:
        stats = _load_action_statistics(train_config["norm_stats_path"])
        normalizer_action = Normalizer(
            stats,
            train_config["dof_config"],
            min_key=train_config.get("min_key", "min"),
            delta_key=train_config.get("delta_key", "delta"),
        )

    if propri_ckpt.exists():
        normalizer_propri = Normalizer.from_ckpt(str(propri_ckpt))
    else:
        if stats is None:
            stats = _load_action_statistics(train_config["norm_stats_path"])
        normalizer_propri = Normalizer(
            stats,
            train_config["agent_pos_config"],
            min_key=train_config.get("min_key", "min"),
            delta_key=train_config.get("delta_key", "delta"),
        )

    return normalizer_action, normalizer_propri


def _decode_image(image_blob):
    array = np.asarray(image_blob)
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        image = array
    else:
        jpeg_bytes = array.tobytes().rstrip(b"\0")
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from observation")

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")

    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.transpose(image, (2, 0, 1))


def _to_wallx_state(openpi_state):
    state = np.asarray(openpi_state, dtype=np.float32).reshape(-1)
    if state.shape[0] != OPENPI_TO_WALLX_ORDER.shape[0]:
        raise ValueError(f"Expected 14-dim state, got shape {state.shape}")
    return state[OPENPI_TO_WALLX_ORDER]


def _to_openpi_action(wallx_action):
    action = np.asarray(wallx_action, dtype=np.float32).reshape(-1)
    if action.shape[0] != OPENPI_TO_WALLX_ORDER.shape[0]:
        raise ValueError(f"Expected 14-dim action, got shape {action.shape}")
    return action[OPENPI_TO_WALLX_ORDER]


class WallXDualPolicy:
    def __init__(self, deploy_cfg):
        self.config_path = _resolve_train_config_path(deploy_cfg.get("train_config_name"))
        self.train_config = _load_train_config(self.config_path)
        self.model_path = deploy_cfg.get("model_path") or self.train_config["pretrained_wallx_path"]
        self.dataset_name = _infer_dataset_name(self.train_config, deploy_cfg)
        self.pred_horizon = int(self.train_config["data"].get("action_horizon", 32))
        self.action_dim = _sum_dims(self.train_config["dof_config"])
        self.agent_pos_dim = _sum_dims(self.train_config["agent_pos_config"])
        self.camera_key = deploy_cfg.get("camera_key", DEFAULT_CAMERA_KEY)
        self.predict_mode = deploy_cfg.get("predict_mode", "diffusion")
        self.device = deploy_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(deploy_cfg.get("max_length", self.train_config.get("max_length", 2048)))
        self.observation_window = None
        self.instruction = None

        print(f"Use config: {self.config_path}")
        self.normalizer_action, self.normalizer_propri = _build_normalizers(
            self.train_config, self.model_path
        )

        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            self.model_path,
            train_config=self.train_config,
            action_tokenizer_path=deploy_cfg.get("action_tokenizer_path"),
        )
        self.model.set_normalizer(
            copy.deepcopy(self.normalizer_action),
            copy.deepcopy(self.normalizer_propri),
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.device.startswith("cuda"):
            self.model.to_bfloat16_for_selected_params()

        self.processor = load_wallx_processors(self.train_config)["processor"]
        print("loading model success!")

    def set_language(self, instruction):
        self.instruction = instruction

    def update_obs(self, obs):
        state = np.concatenate(
            [
                np.asarray(obs[0]["left_arm"]["joint"], dtype=np.float32).reshape(-1),
                np.asarray(obs[0]["left_arm"]["gripper"], dtype=np.float32).reshape(-1),
                np.asarray(obs[0]["right_arm"]["joint"], dtype=np.float32).reshape(-1),
                np.asarray(obs[0]["right_arm"]["gripper"], dtype=np.float32).reshape(-1),
            ]
        )

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": _decode_image(obs[1]["cam_head"]["color"]),
                "cam_left_wrist": _decode_image(obs[1]["cam_left_wrist"]["color"]),
                "cam_right_wrist": _decode_image(obs[1]["cam_right_wrist"]["color"]),
            },
            "prompt": self.instruction,
        }
        return self.observation_window

    def infer(self, observation_window=None):
        if observation_window is None:
            observation_window = self.observation_window
        if observation_window is None:
            raise ValueError("No observation available. Call update_obs(obs) first.")

        prepared_obs = {
            "face_view": observation_window["images"]["cam_high"],
            "left_wrist_view": observation_window["images"]["cam_left_wrist"],
            "right_wrist_view": observation_window["images"]["cam_right_wrist"],
            "prompt": observation_window["prompt"],
            "state": _to_wallx_state(observation_window["state"]),
            "dataset_names": self.dataset_name,
        }

        input_batch = prepare_batch(
            prepared_obs,
            self.processor,
            self.normalizer_propri,
            self.camera_key,
            self.agent_pos_dim,
            self.action_dim,
            self.pred_horizon,
            self.action_dim,
            self.max_length,
            28,
            4 * 28 * 28,
            16384 * 28 * 28,
            self.predict_mode,
            self.device,
        )

        with torch.inference_mode():
            outputs = self.model(
                **input_batch,
                action_dim=self.action_dim,
                action_horizon=self.pred_horizon,
                mode="predict",
                predict_mode=self.predict_mode,
            )

        predicted_actions = outputs["predict_action"]
        if predicted_actions is None:
            predicted_actions = torch.zeros(
                (1, self.pred_horizon, self.action_dim),
                device=self.device,
                dtype=torch.float32,
            )

        predicted_actions = predicted_actions[0].detach().cpu().to(torch.float32).numpy()
        return np.stack([_to_openpi_action(action) for action in predicted_actions], axis=0)

    def get_action(self, obs=None):
        if obs is not None:
            self.update_obs(obs)

        actions = self.infer()
        ret_actions = []
        for action in actions:
            ret_actions.append(
                {
                    "arm": {
                        "left_arm": {
                            "joint": action[:6],
                            "gripper": action[6],
                        },
                        "right_arm": {
                            "joint": action[7:13],
                            "gripper": action[13],
                        },
                    }
                }
            )
        return ret_actions

    def reset(self):
        self.observation_window = None
        self.instruction = None
        print("successfully reset observation_window and instruction")
