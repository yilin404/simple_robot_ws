import rospy
from sensor_msgs.msg import JointState
from a1arm_utils.msg import gripper_position_control

import numpy as np
import torch

import math
import time
from threading import Thread, Lock

from .cpin_robot_wrapper import CPinRobotWrapper

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

@dataclass
class ArmDriverWrapperCfg:
    # 机械臂状态
    joint_state_topic_name: str

    # 机械臂控制
    # curobo_config_file_path: str
    urdf_file_path: str
    arm_joint_position_control_topic_name: str
    gripper_joint_position_control_topic_name: str

class ArmDriverWrapper:
    JOINT_MAX_DELTA: float = 0.1
    def __init__(self, cfg: ArmDriverWrapperCfg) -> None:
        super().__init__()

        # 机械臂末端控制
        self.cpin_robot_wrapper = CPinRobotWrapper(urdf_filename=cfg.urdf_file_path,
                                                   locked_joints=["gripper1_axis", "gripper2_axis"],
                                                   ee_link="arm_seg6")

        # 机械臂夹爪位置控制
        self.GripperPositionControlPub = rospy.Publisher(cfg.gripper_joint_position_control_topic_name, gripper_position_control, queue_size=100)

        self.lock = Lock()
        # 机械臂关节状态
        self.is_curr_qpos_initialized = False
        self.curr_qpos = np.ndarray((7,), dtype=np.float64)
        self.is_curr_qvel_initialized = False
        self.curr_qvel = np.ndarray((7,), dtype=np.float64)
        self.is_curr_torque_initialized = False
        self.curr_torque = np.ndarray((7,), dtype=np.float64)
        self.joint_state_sub = rospy.Subscriber(cfg.joint_state_topic_name, JointState, self.joint_state_cb, queue_size=1000)
        # 机械臂关节位置控制
        self.is_arm_joint_position_target_initialized = False
        self.arm_joint_position_target = np.ndarray((6,), dtype=np.float64)
        self.arm_joint_position_target_pub = rospy.Publisher(cfg.arm_joint_position_control_topic_name, JointState, queue_size=100)
        self.arm_joint_position_target_pub_freq = 300
        self.arm_joint_position_target_pub_thread = Thread(target=self._pub_arm_joint_position_target)
        self.arm_joint_position_target_pub_thread.start()

    def __del__(self):
        self.arm_joint_position_target_pub_thread.join()

    # 单独线程, 不断发布机械臂关节位置期望
    def _pub_arm_joint_position_target(self) -> None:
        while not rospy.is_shutdown():
            if self.is_arm_joint_position_target_initialized and self.is_curr_qpos_initialized:
                with self.lock:
                    delta_q = np.clip(self.arm_joint_position_target - self.curr_qpos[:-1], a_min=-self.JOINT_MAX_DELTA, a_max=self.JOINT_MAX_DELTA)
                    q_target = self.curr_qpos[:-1] + delta_q

                joint_target = JointState()
                joint_target.header.stamp = rospy.Time.now()
                joint_target.name = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
                joint_target.position = q_target.tolist()

                self.arm_joint_position_target_pub.publish(joint_target)
        
            time.sleep(1. / self.arm_joint_position_target_pub_freq)
    
    def command_arm_joint_position(self, arm_joint_position: Sequence[float]) -> None:
        with self.lock:
            self.is_arm_joint_position_target_initialized = True
            self.arm_joint_position_target[:] = np.array(arm_joint_position)

    def command_arm_ee_pose(self, 
                            arm_ee_position: np.ndarray, # [3,]
                            arm_ee_quaternion: np.ndarray, # [4,] # xyzw format
                            use_initial_guess: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        if use_initial_guess:
            with self.lock:
                initial_guess = self.curr_qpos[:-1].copy()
            q_solution = self.cpin_robot_wrapper.inverseKinematics(target_translation=arm_ee_position,
                                                                   target_quaternion=arm_ee_quaternion,
                                                                   initial_guess=initial_guess)
        else:
            q_solution = self.cpin_robot_wrapper.inverseKinematics(target_translation=arm_ee_position,
                                                                   target_quaternion=arm_ee_quaternion)

        if len(q_solution) > 0:
            with self.lock:
                self.is_arm_joint_position_target_initialized = True
                self.arm_joint_position_target[:] = q_solution[:]
                
                # 对动作进行截断
                delta_qpos = np.clip(q_solution - self.curr_qpos[:-1], a_min=-self.JOINT_MAX_DELTA, a_max=self.JOINT_MAX_DELTA)
                action = self.curr_qpos[:-1] + delta_qpos

            return True, action
        else:
            return False, None

    def command_gripper_joint_position(self, gripper_joint_position: float) -> None:
        targetMsg = gripper_position_control()
        targetMsg.header.stamp = rospy.Time.now()
        targetMsg.gripper_stroke = gripper_joint_position * 1000 * 2.

        self.GripperPositionControlPub.publish(targetMsg)

    def joint_state_cb(self, msg: JointState):
        gripper_motor_position = msg.position[-1]
        gripper_position = (18.0277 * math.sin(gripper_motor_position)) / (math.sin(2.5536 - gripper_motor_position)) * 1.e-3

        with self.lock:
            self.curr_qpos[:] = np.array(list(msg.position[:-1]) + [gripper_position])
            self.curr_qvel[:] = np.array(list(msg.velocity[:-1]) + [0.])
            self.curr_torque[:] = np.array(list(msg.effort[:-1]) + [0.])

            self.is_curr_qpos_initialized = True
            self.is_curr_qvel_initialized = True
            self.is_curr_torque_initialized = True
    
    @property
    def initialized(self):
        with self.lock:
            return (self.is_curr_qpos_initialized and 
                    self.is_curr_qvel_initialized and 
                    self.is_curr_torque_initialized)

    def get_arm_states(self):
        if self.initialized:
            with self.lock:
                return self.curr_qpos.copy(), self.curr_qvel.copy(), self.curr_torque.copy()
        else:
            return None, None, None