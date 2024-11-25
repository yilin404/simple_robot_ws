#!/home/yilin/miniconda3/envs/robotics/bin/python3

import rospy

import cv2
import numpy as np
import einops
import torch
from torch import nn

import os
import time
import logging
from contextlib import nullcontext

from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lib.lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy

from lib.arm_driver_wrapper import ArmDriverWrapper, ArmDriverWrapperCfg
from lib.realsense_wrapper import RealSenseWrapper, RealSenseWrapperCfg

from typing import Tuple
from omegaconf import DictConfig
from lerobot.common.policies.policy_protocol import Policy

def get_policy(pretrained_policy_path: str) -> Tuple[Policy, torch.device, DictConfig]:
    hydra_cfg = init_hydra_config(os.path.join(pretrained_policy_path, "config.yaml"))

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    logging.info("Making policy.")
    # Note: We need the dataset stats to pass to the policy's normalization modules.
    policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()

    return policy, device, hydra_cfg

def main():
    rospy.init_node("Eval_Lerobot_node", anonymous=True)
    rospy.loginfo("==> Start Eval Lerobot node...")

    # init_logging()
    
    driver_wrapper_cfg = ArmDriverWrapperCfg(joint_state_topic_name=rospy.get_param("~joint_state_topic_name"),
                                             curobo_config_file_path=rospy.get_param("~curobo_config_file_path"),
                                             arm_joint_position_control_topic_name=rospy.get_param("~arm_joint_position_control_topic_name"),
                                             gripper_joint_position_control_topic_name=rospy.get_param("~gripper_joint_position_control_topic_name"))
    driver_wrapper = ArmDriverWrapper(cfg=driver_wrapper_cfg)

    realsense_wrapper_cfg = RealSenseWrapperCfg(names=["camera_top", "camera_wrist"],
                                                sns=["238222073566", "238222071769"],
                                                color_shape=(640, 480), depth_shape=(640, 480),
                                                fps=60, timeout_ms=30)
    realsense_wrapper = RealSenseWrapper(cfg=realsense_wrapper_cfg)

    controlling = False

    # policy, policy_device, policy_hydra_cfg = get_policy(rospy.get_param("~pretrained_policy_path"))
    # policy.reset()

    while not rospy.is_shutdown():
        # 获取realsense图像
        realsense_wrapper.get_frames()

        # # 获取当前机械臂状态信息
        # qpos_curr, qvel_curr, torque_curr = driver_wrapper.get_arm_states()

        # if controlling:
        #     with torch.no_grad(), torch.autocast(device_type=policy_device.type) if policy_hydra_cfg.use_amp else nullcontext():
        #         # 计算机器人动作
        #         observation = {}

        #         imgs = {"observation.images.camera_top": torch.from_numpy(realsense_wrapper.color_images[0]).to(policy_device),
        #                 "observation.images.camera_top": torch.from_numpy(realsense_wrapper.color_images[1]).to(policy_device),}
        #         for imgkey, img in imgs.items():
        #             # convert to channel first of type float32 in range [0,1]
        #             img = einops.rearrange(img.unsqueeze(0), "b h w c -> b c h w").contiguous()
        #             img = img.type(torch.float32)
        #             img /= 255

        #             observation[imgkey] = img

        #         observation["observation.state"] = torch.from_numpy(qpos_curr).to(policy_device).float().unsqueeze()

        #         with torch.inference_mode():
        #             action = policy.select_action(observation)
        #         action = action.cpu().numpy()

        #     # 控制机器人运动
        #     for i in range(len(action)):
        #         driver_wrapper.command_arm_joint_position(action[i, :-1])
        #         driver_wrapper.command_gripper_joint_position(action[i, -1:])

        #         time.sleep(1. / 30.)

        # 绘制并显示realsense图像, 捕获键盘输入
        # cv2.imshow("realsense_image", cv2.hconcat(realsense_wrapper.color_images))
        cv2.imshow("realsense_image", realsense_wrapper.color_images[0])
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            driver_wrapper.command_arm_ee_pose(arm_ee_position=np.array([0., 0.2, 0.3], dtype=np.float32),
                                                arm_ee_quaternion=np.array([0.707, 0.707, 0., 0.], dtype=np.float32))
            time.sleep(5.)
            
            controlling = False
            print("==> End controlling...")
        elif key == ord('c'):
            driver_wrapper.command_arm_ee_pose(arm_ee_position=np.array([0., 0.2, 0.3], dtype=np.float32),
                                                arm_ee_quaternion=np.array([0.707, 0.707, 0., 0.], dtype=np.float32))
            time.sleep(5.)

            controlling = True
            print("==> Start controlling...")
        else:
            pass

if __name__ == "__main__":
    main()