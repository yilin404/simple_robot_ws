import numpy as np
import cv2

import ctypes

from .yd_depthcam_sdk import yd_people_sensor as yd

class YDDepthCamera(object):
    def __init__(self) -> None:
        error_code = yd.Sensor.get_count(yd.c_uint())
        if yd.ErrorCode.success.value != error_code:
            print("Failed to get sensor count with error code: %s" % error_code)
            exit(1)
        
        self.sensor = yd.Sensor()
        reslutions = [yd.ColorResolution.vga, yd.DepthResolution.vga]
        reslutions_size = [(320, 240), (640,480), (1280, 960)]
        error_code = self.sensor.initialize(reslutions[0], reslutions[1], yd.c_bool(True), yd.c_uint(0))
        if yd.ErrorCode.success.value != error_code:
            print("Failed to initialize sensor with error code: %s" % error_code)
            exit(1)

        error_code = self.sensor.set_depth_mapped_to_color(True)
        if not self.sensor.is_depth_mapped_to_color():
            print("Failed to set depth mapped to color")
            exit(1)

        error_code = self.sensor.set_near_mode(True)
        if not self.sensor.is_near_mode:
            print("Failed to set near mode")
            exit(1)

        error_code = self.sensor.start()
        if yd.ErrorCode.success.value != error_code:
            print("Failed to start sensor with error code: %s" % error_code)
            exit(1)
            
        
        self.color_frame = yd.ColorFrame()
        self.depth_frame = yd.DepthFrame()
        self.publish_data = yd.PublishData()
        
        # self.depth_range_x = [85, 590] #(85,35)  (590, 429)
        # self.depth_range_y = [35, 429]
        
        # self.color_image = None
        # self.depth_image = None
        
        self.color_size = reslutions_size[reslutions[0].value]
        color_length = self.color_size[0] * self.color_size[1] * 4
        self.color_array_type = yd.c_char * color_length
        
        self.depth_size = reslutions_size[reslutions[1].value]
        depth_length = self.depth_size[0] * self.depth_size[1]
        self.depth_array_type = yd.c_ushort * depth_length
    
    def get_frames(self) -> bool:
        error_code = self.sensor.get_color_frame(self.color_frame)
        if yd.ErrorCode.success.value != error_code:
            return False

        error_code = self.sensor.get_depth_frame(self.depth_frame)
        if yd.ErrorCode.success.value != error_code:
            return False

        error_code = self.sensor.get_publish_data(self.publish_data)
        if yd.ErrorCode.success.value != error_code:
            return False
        
        color_addr = yd.addressof(self.color_frame.pixels.contents)
        self.color_image = np.frombuffer(self.color_array_type.from_address(color_addr), dtype=np.uint8).reshape(self.color_frame.height, self.color_frame.width, 4)
        self.color_image = np.fliplr(self.color_image)  # 左右翻转
        
        # depth_addr = yd.addressof(self.depth_frame.pixels.contents)
        # self.depth_image = np.frombuffer(self.depth_array_type.from_address(depth_addr), dtype=np.uint16).reshape(self.depth_frame.height, self.depth_frame.width)
        # self.depth_image = np.fliplr(self.depth_image)  # 左右翻转

        self.get_point_cloud()

        return True

    def get_point_cloud(self) -> None:
        cloudframe = yd.PointCloudFrame()
        cloud_length = self.depth_frame.width * self.depth_frame.height * 3
        buffer = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_float) * cloud_length)
        cloudframe.point = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_float))
        
        self.sensor.convert_depth_frame_to_point_cloud(self.depth_frame.width, self.depth_frame.height, self.depth_frame.pixels.contents, cloudframe.point.contents)
        
        cloud_array_type = yd.c_float * cloud_length
        addr = ctypes.addressof(cloudframe.point.contents)
        point_cloud = np.frombuffer(cloud_array_type.from_address(addr), dtype=np.float32).reshape(self.depth_frame.height, self.depth_frame.width, 3) # [H, W, 3]
        self.point_cloud = np.fliplr(point_cloud).copy() # [H, W, 3]
        self.point_cloud[:, :, 0] = -self.point_cloud[:, :, 0]
        self.point_cloud[:, :, 1] = -self.point_cloud[:, :, 1]