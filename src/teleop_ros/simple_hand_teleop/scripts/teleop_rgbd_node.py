#!/home/yilin/miniconda3/envs/robotics/bin/python3

import rospy

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from lib.teleop_wrapper.rgbd.teleop_rgbd_wrapper import TeleOpRGBDWrapper, TeleOpRGBDWrapperCfg
from lib.arm_driver_wrapper import ArmDriverWrapperCfg
from lib.wrist_pose_tracker_wrapper import WristPoseTrackerWrapper, WristPoseTrackerWrapperCfg
from lib.realsense_wrapper import RealSenseWrapper, RealSenseWrapperCfg
from lib.episode_writer import EpisodeWriter

def main():
    rospy.init_node("TeleOp_RGBD_node", anonymous=True)
    rospy.loginfo("==> Start TeleOp RGBD node...")

    teleop_wrapper_cfg = TeleOpRGBDWrapperCfg(hand_type=rospy.get_param("~hand_type"),
                                              retarget_config_path=rospy.get_param("~retarget_config_path"))
    teleop_wrapper = TeleOpRGBDWrapper(cfg=teleop_wrapper_cfg)

    wrist_pose_tracker_wrapper_cfg = WristPoseTrackerWrapperCfg(driver_wrapper_cfg=ArmDriverWrapperCfg(joint_state_topic_name=rospy.get_param("~joint_state_topic_name"),
                                                                                                       curobo_config_file_path=rospy.get_param("~curobo_config_file_path"),
                                                                                                       arm_joint_position_control_topic_name=rospy.get_param("~arm_joint_position_control_topic_name"),
                                                                                                       gripper_joint_position_control_topic_name=rospy.get_param("~gripper_joint_position_control_topic_name")),
                                                                ee_position_initial=np.array([0., 0.2, 0.3], dtype=np.float32),
                                                                ee_quaternion_initial=np.array([0.707, 0.707, 0., 0.], dtype=np.float32),
                                                                delta_position_scale=1.5)
    wrist_pose_tracker_wrapper = WristPoseTrackerWrapper(cfg=wrist_pose_tracker_wrapper_cfg)
    wrist_pose_filter = False
    wrist_position_prev = None
    wrist_quaternion_prev = None
    
    realsense_wrapper_cfg = RealSenseWrapperCfg(names=["camera_top", "camera_wrist"],
                                                sns=["238222073566", "238222071769"],
                                                color_shape=(640, 480), depth_shape=(640, 480),
                                                fps=60, timeout_ms=30)
    realsense_wrapper = RealSenseWrapper(cfg=realsense_wrapper_cfg)

    controlling = False
    recording = False
    press_key_s_count = 0
    episode_writer = EpisodeWriter(rospy.get_param("~episode_save_path"))

    while not rospy.is_shutdown():
        # 获取手腕姿态(对应机械臂末端姿态)和机械臂手掌关节
        wrist_position, wrist_quaternion, hand_qpos = teleop_wrapper.get_teleop_data()
        # 获取手掌检测的图像, 并捕获键盘输入
        annotated_img = teleop_wrapper.get_annotated_img()
        if annotated_img is not None:
            cv2.imshow("annotated_img", annotated_img)
            cv2.waitKey(1)

        if wrist_position is not None and wrist_quaternion is not None and hand_qpos is not None: # 手腕姿态合理
            # 手势姿态滤波
            if wrist_pose_filter:
                if wrist_position_prev is not None and wrist_quaternion_prev is not None:
                    wrist_position = 0.5 * wrist_position + 0.5 * wrist_position_prev
                    wrist_quaternion = Slerp([0., 1.], R.concatenate([R.from_quat(wrist_quaternion_prev),
                                                                      R.from_quat(wrist_quaternion)]))(0.5).as_quat()
                
                wrist_position_prev = wrist_position
                wrist_quaternion_prev = wrist_quaternion

            # 获取realsense图像,
            realsense_wrapper.get_frames()

            # 获取当前机械臂状态信息
            qpos_curr, qvel_curr, torque_curr = wrist_pose_tracker_wrapper.get_arm_states()

            # 绘制并显示realsense图像, 捕获键盘输入
            cv2.imshow("relasense_image", cv2.hconcat(realsense_wrapper.color_images))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                wrist_pose_tracker_wrapper.reset()
                
                controlling = False
            elif key == ord('s'):
                press_key_s_count += 1
                press_key_s_count %= 2

                if press_key_s_count == 1:
                    print("==> Start recording...")
                    recording = True
                    episode_writer.create_episode()
                    print("==> Create episode ok.")
                else:
                    print("==> End recording...")
                    recording = False
                    episode_writer.save_episode()
                    print("==> Save episode ok.")
            elif key == ord('c'):
                controlling = True
                print("==> Start controlling...")
            else:
                pass
            
            if controlling:
                # 机械臂控制
                success, qpos_target = wrist_pose_tracker_wrapper(wrist_position, wrist_quaternion, hand_qpos)

                if recording and success:
                    episode_writer.add_item(colors={"camera_top": realsense_wrapper.color_images[0],
                                                    "camera_wrist": realsense_wrapper.color_images[1]},
                                            depths={"camera_top": realsense_wrapper.depth_images[0]},
                                            pointclouds={"camera_top": realsense_wrapper.point_clouds[0]},
                                            states={"arm": {"qpos": qpos_curr.tolist(), "qvel": qvel_curr.tolist(), "torque": torque_curr.tolist()}},
                                            actions={"arm": {"qpos": qpos_target.tolist()}})

if __name__ == "__main__":
    main()