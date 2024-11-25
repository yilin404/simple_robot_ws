from pathlib import Path
import glob
import os
import re
import shutil
import json
import argparse
import warnings

import numpy as np
import torch
import cv2
from PIL import Image as PILImage

from safetensors.torch import save_file
from datasets import Dataset, Features, Image, Sequence, Value
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    check_repo_id,
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
    calculate_episode_data_index
)
from lerobot.common.datasets.video_utils import encode_video_frames, VideoFrame

import tqdm

from typing import Tuple, List, Dict, Any

JSON_FILE = "data.json"

def get_episodes(raw_dir: Path) -> List[Path]:
    episodes = glob.glob(os.path.join(raw_dir, '*'))

    return [path for path in episodes if os.path.isdir(path)]

def check_format(raw_dir) -> bool:
    episode_paths = get_episodes(raw_dir)
    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)
    assert len(episode_paths) > 0, "raw_dir is Empty..."
    
    for i, json_path in enumerate(episode_paths):
        json_path = os.path.join(json_path, JSON_FILE)

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            episode_data = json.load(jsonf)

            for sample_data in episode_data["data"]:
                assert "actions" in sample_data, f"actions must be a key, in episode_{i}/{JSON_FILE} [\"data\"]"
                assert len(sample_data["actions"]) > 0, f"the number of actions must be larger than zero, in in episode_{i}/{JSON_FILE} [\"data\"]"

                assert "states" in sample_data, f"states must be a key in episode_{i}/{JSON_FILE} [\"data\"]"
                assert len(sample_data["states"]) > 0, f"the number of states must be larger than zero, in in episode_{i}/{JSON_FILE} [\"data\"]"

                assert "colors" in sample_data, f"colors must be a key in episode_{i}/{JSON_FILE} [\"data\"]"
                assert len(sample_data["colors"]) > 0, f"the number of colors must be larger than zero, in in episode_{i}/{JSON_FILE} [\"data\"]"

def extract_qpos_data(episode_data: Dict, key: str, parts: List[str]) -> np.ndarray:
    result = []
    for sample_data in episode_data["data"]:
        data_array = np.array([], dtype=np.float32)
        for part in parts:
            if part in sample_data[key] and sample_data[key][part] is not None:
                qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                data_array = np.concatenate([data_array, qpos])
        result.append(data_array)

    return np.array(result) # [num_frames, num_data]

def get_actions_data(episode_data: Dict) -> np.ndarray:
    parts = ["arm"]
    return extract_qpos_data(episode_data, "actions", parts)

def get_states_data(episode_data: Dict) -> np.ndarray:
    parts = ["arm"]
    return extract_qpos_data(episode_data, "states", parts)

def get_cameras(episode_data: Dict, image_type: str) -> Tuple[str]:
    return episode_data["data"][0][image_type].keys()

def get_images_data(ep_path: Path, episode_data: Dict, image_type: str) -> Dict[str, List[np.ndarray]]:
    cameras = get_cameras(episode_data, image_type)

    images_dict = {}
    for cam in cameras:
        images_dict.setdefault(image_type + '_' + cam, [])

        for sample_data in episode_data["data"]:
            image_path = os.path.join(ep_path, sample_data[image_type].get(cam, ""))
            if not os.path.exists(image_path):
                print(f"Warning: Image path does not exist: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to read image at {image_path}")
                continue

            if image_type == "colors":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images_dict[image_type + '_' + cam].append(image)

    return images_dict

def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    episode_paths = get_episodes(raw_dir)
    print(f"Found {len(episode_paths)} episodes.")

    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)
    num_episodes = len(episode_paths)

    episode_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids, desc="Load raw data"):
        ep_path = episode_paths[ep_idx]

        json_path = os.path.join(ep_path, JSON_FILE)
        with open(json_path, 'r', encoding='utf-8') as jsonf:
            episode_data = json.load(jsonf)

            actions = torch.from_numpy(get_actions_data(episode_data)) # [num_frames, num_actions]
            states = torch.from_numpy(get_states_data(episode_data)) # [num_frames, num_states]

            # last step of demonstration is considered done
            done = torch.zeros(actions.size(0), dtype=torch.bool)
            done[-1] = True

            image_color_dicts = get_images_data(ep_path, episode_data, "colors")

            ep_dict = {}
            ep_dict["observation.state"] = states
            ep_dict["action"] = actions
            ep_dict["episode_index"] = torch.tensor([ep_idx] * actions.size(0))
            ep_dict["frame_index"] = torch.arange(0, actions.size(0), 1)
            ep_dict["timestamp"] = torch.arange(0, actions.size(0), 1) / fps
            ep_dict["next.done"] = done
            for cam, imgs_color_array in image_color_dicts.items():
                key = f"observation.images.{cam}"
                
                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    save_images_concurrently(imgs_color_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{key}_episode_{ep_idx:06d}.mp4"
                    video_path = os.path.join(videos_dir, fname)
                    encode_video_frames(tmp_imgs_dir, video_path, fps, vcodec='libx264')
                    # encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)
                    
                    # store the reference to the video frame
                    ep_dict[key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(actions.size(0))
                    ]
                else:
                    ep_dict[key] = [PILImage.fromarray(x) for x in imgs_color_array]

            episode_dicts.append(ep_dict)
    
    data_dict = concatenate_episodes(episode_dicts)

    return data_dict

def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 60

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()
    return hf_dataset, episode_data_index, info

def save_meta_data(
    info: dict[str, Any], stats: dict, episode_data_index: dict[str, list], meta_data_dir: Path
):
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # save episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)

def push_dataset_local(
    raw_dir: Path,
    repo_id: str,
    local_dir: Path,
    fps: int | None = None,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    episodes: list[int] | None = None,
    force_override: bool = False,
    resume: bool = False,
    encoding: dict | None = None,
):
    check_repo_id(repo_id)
    user_id, dataset_id = repo_id.split("/")

    # Robustify when `raw_dir` is str instead of Path
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise NotADirectoryError(
            f"{raw_dir} does not exists. Check your paths or run this command to download an existing raw dataset on the hub: "
            f"`python lerobot/common/datasets/push_dataset_to_hub/_download_raw.py --raw-dir your/raw/dir --repo-id your/repo/id_raw`"
        )

    # Robustify when `local_dir` is str instead of Path
    local_dir = Path(local_dir)

    # Send warning if local_dir isn't well formated
    if local_dir.parts[-2] != user_id or local_dir.parts[-1] != dataset_id:
        warnings.warn(
            f"`local_dir` ({local_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht'). Following this naming convention is advised, but not mandatory.",
            stacklevel=1,
        )

    # Check we don't override an existing `local_dir` by mistake
    if local_dir.exists():
        if force_override:
            shutil.rmtree(local_dir)
        elif not resume:
            raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override 1`.")

    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"
    
    fmt_kwgs = {
        "raw_dir": raw_dir,
        "videos_dir": videos_dir,
        "fps": fps,
        "video": video,
        "episodes": episodes,
        "encoding": encoding,
    }

    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(**fmt_kwgs)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    # mandatory for upload
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    return lerobot_dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `data/aloha_mobile_chair_raw` or `data/pusht_raw).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="Frame rate used to collect videos. If not provided, use the default one specified in the code.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=1,
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        help="When provided, only converts the provided episodes (e.g `--episodes 2 3 4`). Useful to test the code on 1 episode.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="When set to 1, removes provided output directory if it already exists. By default, raises a ValueError exception.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="When set to 1, resumes a previous run.",
    )

    args = parser.parse_args()
    push_dataset_local(**vars(args))

if __name__ == "__main__":
    main()