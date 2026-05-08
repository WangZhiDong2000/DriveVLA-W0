"""
Pickle generation for navsim mini dataset.
Differences from the original pickle_generation_navsim_pre_1s.py:
- No nuplan dependency (nuplan pre_1s fallback is skipped; affected frames are dropped)
- Uses mini split paths
- Token list derived from all available logs (no external yaml filter)
- RunningStats inlined (utils/dataset/normalize_pi0.py not present in this repo)
"""
import os
import os.path as osp
import pickle
import sys
from tqdm import tqdm
import numpy as np
import yaml

from pyquaternion import Quaternion

sys.path.insert(0, osp.join(osp.dirname(__file__)))
from navsim_coor import StateSE2, convert_absolute_to_relative_se2_array, normalize_angle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.expanduser("~/Dataset/navsim/download")
PROJECT_ROOT = os.path.expanduser("~/Project/DriveVLA-W0")

split = "mini"
logs_path = osp.join(DATASET_ROOT, "mini_navsim_logs/mini")
vq_dir = osp.join(DATASET_ROOT, "processed_data/mini_vq_codes_256_144")
output_path = osp.join(DATASET_ROOT, "processed_data/meta")
output_file_name = f"navsim_emu_vla_256_144_{split}_pre_1s.pkl"
normalizer_save_path = osp.join(PROJECT_ROOT, f"configs/normalizer_navsim_{split}")

os.makedirs(output_path, exist_ok=True)
os.makedirs(normalizer_save_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW = 12
CENTER_IDX = 3

text_name_list = ["go left", "go straight", "go right", "unknown"]


# ---------------------------------------------------------------------------
# Minimal RunningStats (replaces utils.dataset.normalize_pi0)
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self, mean, std, q01, q99):
        self.mean = mean
        self.std = std
        self.q01 = q01
        self.q99 = q99


class RunningStats:
    def __init__(self):
        self._chunks = []

    def update(self, data):
        self._chunks.append(np.asarray(data))

    def get_statistics(self):
        data = np.concatenate(self._chunks, axis=0)
        return _Stats(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            q01=np.quantile(data, 0.01, axis=0),
            q99=np.quantile(data, 0.99, axis=0),
        )


def save_normalizer(path, stats_dict):
    for name, stats in stats_dict.items():
        np.save(osp.join(path, f"{name}_mean.npy"), stats.mean)
        np.save(osp.join(path, f"{name}_std.npy"), stats.std)
        np.save(osp.join(path, f"{name}_q01.npy"), stats.q01)
        np.save(osp.join(path, f"{name}_q99.npy"), stats.q99)
    print(f"Normalizer saved to {path}")


# ---------------------------------------------------------------------------
# Phase 1: Build scene_dict_all
# ---------------------------------------------------------------------------
scene_dict_all = {}

log_files = [f for f in os.listdir(logs_path) if f.endswith(".pkl")]
print(f"Found {len(log_files)} log files in {logs_path}")

for log_name in tqdm(log_files, desc="Processing logs"):
    log_path = osp.join(logs_path, log_name)
    scene = pickle.load(open(log_path, "rb"))
    num_frames = len(scene)

    # Compute all global SE2 poses
    global_ego_poses = []
    for fi in scene:
        t = fi["ego2global_translation"]
        q = Quaternion(*fi["ego2global_rotation"])
        yaw = q.yaw_pitch_roll[0]
        global_ego_poses.append([t[0], t[1], yaw])
    global_ego_poses = np.array(global_ego_poses, dtype=np.float64)

    # Compute all relative poses rel_all[i, j]: pose of j relative to i as origin
    rel_all = []
    for i in range(num_frames):
        origin = StateSE2(*global_ego_poses[i])
        rel = convert_absolute_to_relative_se2_array(origin, global_ego_poses)
        rel_all.append(rel)
    rel_all = np.stack(rel_all, axis=0)

    for i, fi in enumerate(scene):
        idxs = list(range(i - CENTER_IDX, i - CENTER_IDX + WINDOW))

        # action_list: relative motion between consecutive frames
        action_list = []
        for j in idxs:
            if 0 <= j < num_frames - 1:
                dx, dy, dtheta = rel_all[j, j + 1]
            else:
                dx = dy = dtheta = 0.0
            action_list.append([float(dx), float(dy), float(dtheta)])
        fi["relative_action_list"] = action_list

        # image_vq_list: paths to .npy VQ code files
        image_list = []
        for j in idxs:
            if 0 <= j < num_frames:
                cam_path = scene[j]["cams"]["CAM_F0"]["data_path"]
                log_n, _, fname = cam_path.split("/")
                vq_name = fname.replace("jpg", "npy")
                image_list.append(osp.join(vq_dir, log_n, vq_name))
            else:
                image_list.append(None)
        fi["image_vq_list"] = image_list

        # text_list
        text_list = []
        for j in idxs:
            if 0 <= j < num_frames:
                driving_text_idx = scene[j]["driving_command"].nonzero()[0].item()
                text_list.append(text_name_list[driving_text_idx])
            else:
                text_list.append(None)
        fi["text_list"] = text_list

        # pre_1s data: use frame i-2 as "previous 1 second" context
        if i < 2:
            fi["pre_1s_relative_action_list"] = fi["relative_action_list"]
            fi["pre_1s_text_list"] = fi["text_list"]
            fi["pre_1s_image_vq_list"] = fi["image_vq_list"]
        else:
            fi["pre_1s_relative_action_list"] = scene[i - 2]["relative_action_list"]
            fi["pre_1s_text_list"] = scene[i - 2]["text_list"]
            fi["pre_1s_image_vq_list"] = scene[i - 2]["image_vq_list"]

        token = fi.pop("token")
        scene_dict_all[token] = fi

print(f"Total frames indexed: {len(scene_dict_all)}")

# ---------------------------------------------------------------------------
# Phase 2: Build result_file using all available tokens
# ---------------------------------------------------------------------------
result_file = []
for token, info in tqdm(scene_dict_all.items(), desc="Building result_file"):
    result_file.append({
        "token": token,
        "text": info["text_list"],
        "image": info["image_vq_list"],
        "action": info["relative_action_list"],
        "pre_1s_text": info["pre_1s_text_list"],
        "pre_1s_image": info["pre_1s_image_vq_list"],
        "pre_1s_action": info["pre_1s_relative_action_list"],
    })

print(f"Total scenes before filtering: {len(result_file)}")

# Drop frames where pre_1s == image (first 2 frames of each log).
# The original script uses nuplan to fill these in; we skip nuplan.
result_file = [item for item in result_file if item["pre_1s_image"] != item["image"]]
print(f"Total scenes after dropping pre_1s==image frames: {len(result_file)}")

# ---------------------------------------------------------------------------
# Phase 3: Normalize actions
# ---------------------------------------------------------------------------
normalizer = RunningStats()
action_data = np.concatenate([np.array(scene["action"]) for scene in result_file])
normalizer.update(action_data)
norm_stats = normalizer.get_statistics()

print(f"Action mean:  {norm_stats.mean}")
print(f"Action std:   {norm_stats.std}")
print(f"Action q01:   {norm_stats.q01}")
print(f"Action q99:   {norm_stats.q99}")

save_normalizer(normalizer_save_path, {"libero": norm_stats})

for scene in result_file:
    for key in ("action", "pre_1s_action"):
        a = np.array(scene[key])
        normalized = 2 * (a - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
        scene[key] = np.clip(normalized, -1, 1)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_file = osp.join(output_path, output_file_name)
with open(output_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"\nSaved {len(result_file)} scenes to {output_file}")
