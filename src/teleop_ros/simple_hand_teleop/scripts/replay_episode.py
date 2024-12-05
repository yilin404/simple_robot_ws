import rospy

import cv2
import numpy as np

import time
from pathlib import Path
import glob
import os
import re
import json

from lib.driver.arm.arm_driver_wrapper import ArmDriverWrapper, ArmDriverWrapperCfg
from lib.realsense_wrapper import RealSenseWrapper, RealSenseWrapperCfg

from typing import List, Dict

JSON_FILE = "data.json"

def get_episodes(raw_dir: Path) -> List[Path]:
    episodes = glob.glob(os.path.join(raw_dir, '*'))

    return [path for path in episodes if os.path.isdir(path)]

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

def get_replay_episode_actions(raw_dir: str | Path,
                       ep_idx: int):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise NotADirectoryError(
            f"{raw_dir} does not exists. Check your paths"
        )

    episode_paths = get_episodes(raw_dir)
    print(f"Found {len(episode_paths)} episodes.")

    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)

    ep_path = episode_paths[ep_idx]
    json_path = os.path.join(ep_path, JSON_FILE)
    with open(json_path, 'r', encoding='utf-8') as jsonf:
        episode_data = json.load(jsonf)

        actions = get_actions_data(episode_data) # [num_frames, num_actions]

    return actions
    
def main():
    rospy.init_node("Replay_Episode_node", anonymous=True)
    rospy.loginfo("==> Start Replay Episode node...")

    driver_wrapper_cfg = ArmDriverWrapperCfg(joint_state_topic_name=rospy.get_param("~joint_state_topic_name"),
                                             curobo_config_file_path=rospy.get_param("~curobo_config_file_path"),
                                             arm_joint_position_control_topic_name=rospy.get_param("~arm_joint_position_control_topic_name"),
                                             gripper_joint_position_control_topic_name=rospy.get_param("~gripper_joint_position_control_topic_name"))
    driver_wrapper = ArmDriverWrapper(cfg=driver_wrapper_cfg)

    realsense_wrapper_cfg = RealSenseWrapperCfg(names=["camera_top", "camera_wrist", "camera_left"],
                                                sns=["238222073566", "238222071769", "238322071831"],
                                                color_shape=(640, 480), depth_shape=(640, 480),
                                                fps=30, timeout_ms=30)
    realsense_wrapper = RealSenseWrapper(cfg=realsense_wrapper_cfg)

    controlling = False
    replay_actions = get_replay_episode_actions(rospy.get_param("~episode_raw_dir"), 0)

    while not rospy.is_shutdown():
        # 获取realsense图像
        color_images, depth_images, point_clouds = realsense_wrapper.get_frames()

        if controlling:
            for action in replay_actions:
                driver_wrapper.command_arm_joint_position(action)
                time.sleep(1. / 30.)

        # 绘制并显示realsense图像, 捕获键盘输入
        cv2.imshow("realsense_image", cv2.hconcat(color_images))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            driver_wrapper.command_arm_ee_pose(arm_ee_position=np.array([0., 0.2, 0.3], dtype=np.float32),
                                               arm_ee_quaternion=np.array([0.707, 0.707, 0., 0.], dtype=np.float32))
            driver_wrapper.command_gripper_joint_position(0.03)
            time.sleep(3.)
                    
            controlling = False
            print("==> End controlling...")
        elif key == ord('c'):
            driver_wrapper.command_arm_ee_pose(arm_ee_position=np.array([0., 0.2, 0.3], dtype=np.float32),
                                               arm_ee_quaternion=np.array([0.707, 0.707, 0., 0.], dtype=np.float32))
            driver_wrapper.command_gripper_joint_position(0.03)
            time.sleep(3.)
                    
            controlling = True
            print("==> Start controlling...")
        else:
            pass

if __name__ == "__main__":
    main()