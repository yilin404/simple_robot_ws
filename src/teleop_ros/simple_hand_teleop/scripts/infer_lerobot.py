#!/home/yilin/miniconda3/envs/robotics/bin/python3

import rospy

import cv2
import numpy as np
import einops
import torch

import time

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy

from lib.arm_driver_wrapper import ArmDriverWrapper, ArmDriverWrapperCfg
from lib.realsense_wrapper import RealSenseWrapper, RealSenseWrapperCfg

def main():
    rospy.init_node("Infer_Lerobot_node", anonymous=True)
    rospy.loginfo("==> Start Infer Lerobot node...")
    
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
    
    policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # policy = DiffusionPolicy.from_pretrained(rospy.get_param("~pretrained_policy_path"))
    policy = ACTPolicy.from_pretrained(rospy.get_param("~pretrained_policy_path"))
    policy.to(policy_device)
    policy.reset()

    while not rospy.is_shutdown():
        # 获取realsense图像
        color_images, depth_images, point_clouds = realsense_wrapper.get_frames()

        # 获取当前机械臂状态信息
        qpos_curr, qvel_curr, torque_curr = driver_wrapper.get_arm_states()

        if controlling:
            with torch.no_grad():
                # 计算机器人动作
                observation = {}

                imgs = {"observation.images.colors_camera_top": torch.from_numpy(color_images[0]).to(policy_device),
                        "observation.images.colors_camera_wrist": torch.from_numpy(color_images[1]).to(policy_device),
                        "observation.images.colors_camera_right": torch.from_numpy(color_images[2]).to(policy_device),}
                for imgkey, img in imgs.items():
                    # convert to channel first of type float32 in range [0,1]
                    img = einops.rearrange(img.unsqueeze(0), "b h w c -> b c h w").contiguous()
                    img = img.type(torch.float32)
                    img /= 255

                    observation[imgkey] = img

                observation["observation.state"] = torch.from_numpy(qpos_curr).to(policy_device).float().unsqueeze(0)

                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = action.cpu().numpy()

            # 控制机器人运动
            driver_wrapper.command_arm_joint_position(action[0, :-1])
            driver_wrapper.command_gripper_joint_position(action[0, -1])

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
            print("==> End controlling...")
        else:
            pass

if __name__ == "__main__":
    main()