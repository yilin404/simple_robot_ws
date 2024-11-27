import pyrealsense2 as rs

import numpy as np
import open3d as o3d
import time

import threading

from dataclasses import dataclass
from typing import List

@dataclass
class RealSenseWrapperCfg:
    names: List[str]
    sns: List[str]

    color_shape: List[int]
    depth_shape: List[int]

    fps: int

    timeout_ms: float

class RealSenseWrapper:
    def __init__(self, cfg: RealSenseWrapperCfg):
        super().__init__()

        print("==> RealSenseWrapper initial...")

        self.cfg = cfg

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        self.pipes = []
        self.pipe_cfgs = []
        self.profiles = []

        # 创建对齐对象，默认对齐到彩色图像
        self.align_to = rs.stream.color  # 默认将深度对齐到彩色图
        self.align = rs.align(self.align_to)  # 创建对齐对象

        for sn in self.cfg.sns:
            pipe = rs.pipeline()
            self.pipes.append(pipe)

            pipe_cfg = rs.config()
            pipe_cfg.enable_device(sn)
            pipe_cfg.enable_stream(rs.stream.color, self.cfg.color_shape[0], self.cfg.color_shape[1], rs.format.bgr8, self.cfg.fps)
            pipe_cfg.enable_stream(rs.stream.depth, self.cfg.depth_shape[0], self.cfg.depth_shape[1], rs.format.z16, self.cfg.fps)
            self.pipe_cfgs.append(pipe_cfg)

            profile = pipe.start(pipe_cfg)
            self.profiles.append(profile)
        
        print("==> Waiting for Frames...")
        for _ in range(3):
            for name, pipe in zip(self.cfg.names, self.pipes):
                t = time.time()
                try:
                    pipe.wait_for_frames()
                    print(f"{name} waited {time.time() - t}s")
                except:
                    print(f"{name} waited too long: {time.time() - t}s\n\n")
                    raise Exception

        # 初始化共享内存和锁
        self.lock = threading.Lock()

        # 使用 numpy 的共享内存方式
        self.color_images_array = np.zeros((len(self.cfg.names), cfg.color_shape[1], cfg.color_shape[0], 3), dtype=np.uint8)
        self.depth_images_array = np.zeros((len(self.cfg.names), cfg.depth_shape[1], cfg.depth_shape[0]), dtype=np.uint16)
        self.point_clouds_list = [None] * len(self.cfg.names)

        """启动多进程捕获图像"""
        self.threads = []
        for camera_index in range(len(self.cfg.names)):
            t = threading.Thread(target=self._get_frame_thread, args=(camera_index,))
            t.daemon = True
            self.threads.append(t)
            t.start()

            print(f"==> Start RealSense GetFrame Thread {camera_index}")

        print("==> RealSenseWrapper initial successfully...")

    def _get_frame_thread(self, camera_index: int):
        while True:
            pipe = self.pipes[camera_index]
            
            try:
                frameset = pipe.wait_for_frames(timeout_ms=self.cfg.timeout_ms)
                aligned_frameset = self.align.process(frameset)  # 对齐深度图和彩色图

                color_frame = aligned_frameset.get_color_frame()
                depth_frame = aligned_frameset.get_depth_frame()

                with self.lock:
                    self.color_images_array[camera_index] = np.array(color_frame.get_data())
                    self.depth_images_array[camera_index] = np.array(depth_frame.get_data())
                    self.point_clouds_list[camera_index] = self._depth_to_point_cloud(self.depth_images_array[camera_index], depth_frame.profile.as_video_stream_profile().intrinsics)
                
                time.sleep(1e-2)
            except:
                pass

            time.sleep(5e-3)

    def get_frames(self):
        with self.lock:
            return self.color_images_array.copy(), self.depth_images_array.copy(), self.point_clouds_list.copy()

    def _depth_to_point_cloud(self, depth_image, intrinsic):
        # Create Open3D Image from depth map
        o3d_depth = o3d.geometry.Image(depth_image)

        # Get intrinsic parameters
        fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

        # Create Open3D PinholeCameraIntrinsic object
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)

        # Create Open3D PointCloud object from depth image and intrinsic parameters
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

        return np.asarray(pcd.points)