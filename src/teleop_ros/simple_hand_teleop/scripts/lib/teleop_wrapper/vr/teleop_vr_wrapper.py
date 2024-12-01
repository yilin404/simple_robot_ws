import rospy

from multiprocessing import Lock, Process
from multiprocessing.managers import SyncManager, SharedMemoryManager
import time

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from .television import TeleVision
from ...dex_retargeting.retargeting_config import RetargetingConfig

from dataclasses import dataclass
from typing import List

COORDINATE_ROTATION_MATRIX_LEFT = np.array([[1., 0., 0.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]], dtype=np.float64)
COORDINATE_ROTATION_MATRIX_RIGHT = np.array([[0., 1., 0.],
                                             [0., 0., -1.],
                                             [-1., 0., 0.]], dtype=np.float64)


@dataclass
class TeleOpVRWrapperCfg:
    # 手势检测
    hand_type: str

    # Retarget
    retarget_config_path: str

    # TeleVision
    display_image_shape: List[int]

class TeleOpVRWrapper:
    def __init__(self, cfg: TeleOpVRWrapperCfg):
        super().__init__()

        print("==> TeleOpVRWrapper initial...")

        # 创建 Retargeting
        self.retargeting = RetargetingConfig.load_from_file(cfg.retarget_config_path).build()

        self.manager = SyncManager()
        self.shm_manager = SharedMemoryManager()
        self.manager.start()
        self.shm_manager.start()
        self.lock = Lock()

        self.is_wrist_position_initialized = self.manager.Value('b', False)
        self.wrist_position_shm = self.shm_manager.SharedMemory(size=3 * np.float64().nbytes)
        self.wrist_position = np.ndarray((3,), dtype=np.float64, buffer=self.wrist_position_shm.buf)
        self.is_wrist_quaternion_initialized = self.manager.Value('b', False)
        self.wrist_quaternion_shm = self.shm_manager.SharedMemory(size=4 * np.float64().nbytes)
        self.wrist_quaternion = np.ndarray((4,), dtype=np.float64, buffer=self.wrist_quaternion_shm.buf)
        self.is_hand_qpos_initialized = self.manager.Value('b', False)
        self.hand_qpos_shm = self.shm_manager.SharedMemory(size=1 * np.float64().nbytes)
        self.hand_qpos = np.ndarray((1,), dtype=np.float64, buffer=self.hand_qpos_shm.buf)

        self.display_image_shm = self.shm_manager.SharedMemory(size=math.prod(cfg.display_image_shape) * np.uint8().nbytes)
        self.display_image_array = np.ndarray(cfg.display_image_shape, dtype=np.uint8, buffer=self.display_image_shm.buf)

        self.television_process = Process(target=self._television_process, args=(cfg.hand_type,))
        self.television_process.start()

    def __del__(self) -> None:
        self.television_process.join()

    def _television_process(self, hand_type: str):
        tv = TeleVision(self.display_image_array)

        while not rospy.is_shutdown():
            if tv.initialized:
                # 获取手腕关节姿态, 以及手掌关键点位置
                if hand_type == "Left":
                    assert 1 == 2, "Left hand control has not been finished...."
                    wrist_pose_matrix = tv.left_hand_pose_matrix
                    wrist_position = wrist_pose_matrix[:3, 3]
                    wrist_quaternion = R.from_matrix(wrist_pose_matrix[:3, :3] @ COORDINATE_ROTATION_MATRIX_LEFT).as_quat() # xyzw format

                    # 手掌关键点位置转移到手掌局部坐标系下
                    joint_pos = tv.left_landmarks_position
                    joint_pos = joint_pos - wrist_position.reshape(1, 3)
                    joint_pos = joint_pos @ wrist_pose_matrix[:3, :3]
                elif hand_type == "Right":
                    wrist_pose_matrix = tv.right_hand_pose_matrix
                    wrist_position = wrist_pose_matrix[:3, 3]
                    wrist_quaternion = R.from_matrix(wrist_pose_matrix[:3, :3] @ COORDINATE_ROTATION_MATRIX_RIGHT).as_quat() # xyzw format

                    # 手掌关键点位置转移到手掌局部坐标系下
                    joint_pos = tv.right_landmarks_position
                    joint_pos = joint_pos - wrist_position.reshape(1, 3)
                    joint_pos = joint_pos @ wrist_pose_matrix[:3, :3]
                else:
                    raise NotImplementedError("hand_type must be \'Left\' or \'Right\'......")

                retargeting_type = self.retargeting.optimizer.retargeting_type
                indices = self.retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                
                qpos = self.retargeting.retarget(ref_value, fixed_qpos=np.zeros(6,))

                with self.lock:
                    self.wrist_position[:] = wrist_position[:]
                    self.wrist_quaternion[:] = wrist_quaternion[:]
                    self.hand_qpos[:] = qpos[-1:]

                    self.is_wrist_position_initialized.value = True
                    self.is_wrist_quaternion_initialized.value = True
                    self.is_hand_qpos_initialized.value = True

            time.sleep(0.01)

    def get_teleop_data(self):
        if self.initialized:
            with self.lock:
                return self.wrist_position.copy(), self.wrist_quaternion.copy(), self.hand_qpos.copy()
        else:
            return None, None, None

    @property
    def initialized(self):
        with self.lock:
            return (self.is_wrist_position_initialized.value and
                    self.is_wrist_quaternion_initialized.value and
                    self.is_hand_qpos_initialized.value)
    
    def set_display_image(self, display_image: np.ndarray):
        with self.lock:
            self.display_image_array[:] = display_image[:]
