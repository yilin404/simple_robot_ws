import numpy as np
from scipy.spatial.transform import Rotation as R

from .arm_driver_wrapper import ArmDriverWrapper, ArmDriverWrapperCfg

from dataclasses import dataclass
from typing import Tuple

@dataclass
class WristPoseTrackerWrapperCfg:
    driver_wrapper_cfg: ArmDriverWrapperCfg

    # 机械臂初始姿态设置
    ee_position_initial: np.ndarray
    ee_quaternion_initial: np.ndarray # xyzw format

    # # 控制量变换矩阵
    delta_position_scale: float

class WristPoseTrackerWrapper:
    def __init__(self, cfg: WristPoseTrackerWrapperCfg):
        super().__init__()

        print("==> WristPoseTrackerWrapper initial...")

        self.cfg = cfg

        self.wrist_position_initial = None
        self.wrist_quaternion_initial = None

        self.driver_wrapper = ArmDriverWrapper(cfg.driver_wrapper_cfg)

        print("==> WristPoseTrackerWrapper initial successfully...")
        
    def __call__(self, wrist_position: np.ndarray,
                 wrist_quaternion: np.ndarray,
                 hand_qpos: np.ndarray) -> Tuple[bool, np.ndarray]:
        if self.wrist_position_initial is None or self.wrist_quaternion_initial is None:
            self.wrist_position_initial = wrist_position
            self.wrist_quaternion_initial = wrist_quaternion # xyzw format

            return False, None
        else:
            wrist_pos = wrist_position
            wrist_rot = R.from_quat(wrist_quaternion)
            
            # 计算手腕位姿的增量
            # delta_wrist_pos = self.cfg.transform_matrix @ (wrist_pos - self.wrist_position_initial)
            delta_wrist_pos = R.from_quat(self.wrist_quaternion_initial).inv().apply(self.cfg.delta_position_scale * (wrist_pos - self.wrist_position_initial))
            delta_wrist_rot =  R.from_quat(self.wrist_quaternion_initial).inv() * wrist_rot

            # 计算末端位姿
            # ee_pos = self.cfg.ee_position_initial + delta_wrist_pos
            ee_pos = self.cfg.ee_position_initial + R.from_quat(self.cfg.ee_quaternion_initial).apply(delta_wrist_pos)
            ee_rot = R.from_quat(self.cfg.ee_quaternion_initial) * delta_wrist_rot

            ee_position = ee_pos
            ee_quaternion = ee_rot.as_quat() # xyzw format

            # 调用 末端位姿控制 服务
            success, arm_qpos_target = self.driver_wrapper.command_arm_ee_pose(ee_position, ee_quaternion)

            # 调用 夹爪位置控制 服务
            self.driver_wrapper.command_gripper_joint_position(hand_qpos[-1])

            if success:
                return True, np.concatenate([arm_qpos_target, np.array([hand_qpos[-1]], dtype=np.float32)])
            else:
                return False, None
    
    def get_arm_states(self):
        return self.driver_wrapper.get_arm_states()

    def reset(self):
        print("==> WristPoseTrackerWrapper reset...\n")

        ### 调用机械臂控制接口, 让机械臂回复初始位姿 ###
        self.driver_wrapper.command_arm_ee_pose(self.cfg.ee_position_initial,
                                                self.cfg.ee_quaternion_initial)

        # 调用 夹爪位置控制 服务
        self.driver_wrapper.command_gripper_joint_position(0.03)

        self.wrist_position_initial = None
        self.wrist_quaternion_initial = None

        print("==> WristPoseTrackerWrapper reset successfully...\n")