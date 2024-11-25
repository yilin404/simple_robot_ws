import rospy

from multiprocessing import Process, Lock
from multiprocessing.managers import SyncManager, SharedMemoryManager
from queue import Empty
import time

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from .yd_depth_camera import YDDepthCamera
from .hand_detector import SingleHandDetector
from ...dex_retargeting.retargeting_config import RetargetingConfig

from dataclasses import dataclass
from typing import Tuple, Optional

COORDINATE_ROTATION_MATRIX = np.array([[-1., 0., 0.],
                                       [0., 0., -1.],
                                       [0., -1., 0.]], dtype=np.float64)

@dataclass
class TeleOpRGBDWrapperCfg:
    # 手势检测
    hand_type: str

    # Retarget
    retarget_config_path: str

class TeleOpRGBDWrapper:
    def __init__(self, cfg: TeleOpRGBDWrapperCfg) -> None:
        super().__init__()

        print("==> TeleOpRGBDWrapper initial...")

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
        
        self.shared_data = self.manager.Namespace()
        self.shared_data.annotated_img = None

        # 手势检测线程
        self.img_queue = self.manager.Queue(maxsize=1)
        self.pcd_queue = self.manager.Queue(maxsize=1)
        self.producer_process = Process(target=self._produce_process)
        self.consumer_process = Process(target=self._consume_process, args=(cfg.hand_type,))

        self.producer_process.start()
        self.consumer_process.start()

        print("==> TeleOpRGBDWrapper initial successfully...")
    
    def __del__(self) -> None:
        self.producer_process.join()
        self.consumer_process.join()

    def _produce_process(self) -> None:
        yd_cam = YDDepthCamera()

        while not rospy.is_shutdown():
            if not yd_cam.get_frames():
                time.sleep(1. / 100.)
            else:
                self.img_queue.put(cv2.cvtColor(yd_cam.color_image, cv2.COLOR_BGRA2BGR)) # [H, W, 3]
                self.pcd_queue.put(yd_cam.point_cloud) # [H, W, 3]

                time.sleep(1. / 30.)

    def _consume_process(self, hand_type: str,) -> None:
        detector = SingleHandDetector(hand_type, selfie=False)

        while not rospy.is_shutdown():
            try:
                img_bgr = self.img_queue.get(timeout=5)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                pcd = self.pcd_queue.get(timeout=5)
            except Empty:
                print(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
                return

            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(img_rgb)
            # the type of joint_pos is numpy.ndarray, shape is [21, 3] ---> 本质上是keypoint_3d
            # the type of keypoint_2d is landmark_pb2.NormalizedLandmarkList, shape is [21, 2]
            # the type of mediapipe_wrist_rot is numpy.ndarray, shape is [3, 3]
            
            if num_box > 0:
                # 计算 wrist 关节的姿态
                keypoint_2d_array = detector.parse_keypoint_2d(keypoint_2d, img_rgb.shape) # [21, 2], numpy.ndarray
                wrist_position = pcd[int(keypoint_2d_array[0, 1]), int(keypoint_2d_array[0, 0])] # [3,]
                wrist_quaternion = R.from_matrix(mediapipe_wrist_rot @ COORDINATE_ROTATION_MATRIX).as_quat() # [4,]

                if wrist_position[-1] < 0.1:
                    print("Incorrect depth camera data, the depth value should not be less than 0.5m. Please move your hand away from the camera.")
                else:
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
                        # 保存 wrist pose 和 hand qpos
                        self.wrist_position[:] = wrist_position[:]
                        self.wrist_quaternion[:] = wrist_quaternion[:]
                        self.hand_qpos[:] = qpos[-1:]

                        self.is_wrist_position_initialized.value = True
                        self.is_wrist_quaternion_initialized.value = True
                        self.is_hand_qpos_initialized.value = True
                    
            # 对输入图像进行标注
            self.shared_data.annotated_img = detector.draw_skeleton_on_image(img_bgr, keypoint_2d, style="default")

            time.sleep(1. / 25.)
    
    def get_teleop_data(self) -> Tuple[Optional[np.ndarray]]:
        with self.lock:
            if self.initialized:
                return self.wrist_position.copy(), self.wrist_quaternion.copy(), self.hand_qpos.copy()
            else:
                return None, None, None

    def get_annotated_img(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.shared_data.annotated_img.copy() if self.shared_data.annotated_img is not None else None
        
    @property
    def initialized(self):
        return (self.is_wrist_position_initialized.value and 
                self.is_wrist_quaternion_initialized.value and 
                self.is_hand_qpos_initialized.value)