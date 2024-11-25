from yd_people_sensor import *
import cv2
import math
import numpy as np


class Skeleton2D(Structure):
    _fields_ = [("user_id", c_uint),
                ("is_tracked", POINTER(Skeleton)),
                ("joints", Float2 * JointType.count.value)]


def draw_skeleton(image: np.ndarray, skeleton_2d: Skeleton2D):
    draw_joint(image, skeleton_2d, JointType.head)
    draw_joint(image, skeleton_2d, JointType.neck)
    draw_joint(image, skeleton_2d, JointType.torso)
    draw_joint(image, skeleton_2d, JointType.left_shoulder)
    draw_joint(image, skeleton_2d, JointType.right_shoulder)
    draw_joint(image, skeleton_2d, JointType.left_elbow)
    draw_joint(image, skeleton_2d, JointType.right_elbow)
    draw_joint(image, skeleton_2d, JointType.left_hand)
    draw_joint(image, skeleton_2d, JointType.right_hand)
    draw_joint(image, skeleton_2d, JointType.left_chest)
    draw_joint(image, skeleton_2d, JointType.right_chest)
    draw_joint(image, skeleton_2d, JointType.left_waist)
    draw_joint(image, skeleton_2d, JointType.right_waist)
    draw_joint(image, skeleton_2d, JointType.left_knee)
    draw_joint(image, skeleton_2d, JointType.right_knee)
    draw_joint(image, skeleton_2d, JointType.left_foot)
    draw_joint(image, skeleton_2d, JointType.right_foot)

    draw_bone(image, skeleton_2d, JointType.head, JointType.neck)
    draw_bone(image, skeleton_2d, JointType.neck, JointType.left_shoulder)
    draw_bone(image, skeleton_2d, JointType.neck, JointType.right_shoulder)
    draw_bone(image, skeleton_2d, JointType.left_shoulder, JointType.left_elbow)
    draw_bone(image, skeleton_2d, JointType.left_elbow, JointType.left_hand)
    draw_bone(image, skeleton_2d, JointType.right_shoulder, JointType.right_elbow)
    draw_bone(image, skeleton_2d, JointType.right_elbow, JointType.right_hand)
    draw_bone(image, skeleton_2d, JointType.left_shoulder, JointType.right_waist)
    draw_bone(image, skeleton_2d, JointType.right_shoulder, JointType.left_waist)
    draw_bone(image, skeleton_2d, JointType.left_waist, JointType.right_waist)
    draw_bone(image, skeleton_2d, JointType.left_waist, JointType.left_knee)
    draw_bone(image, skeleton_2d, JointType.left_knee, JointType.left_foot)
    draw_bone(image, skeleton_2d, JointType.right_waist, JointType.right_knee)
    draw_bone(image, skeleton_2d, JointType.right_knee, JointType.right_foot)


def draw_joint(image: np.ndarray, skeleton_2d: Skeleton2D, joint_type: JointType):
    if math.isnan(skeleton_2d.joints[joint_type.value].x) or math.isnan(skeleton_2d.joints[joint_type.value].y):
        return

    point_size = 6
    point_color = (0, 0, 255)  # BGR
    thickness = 8

    cv2.circle(image, (np.int32(skeleton_2d.joints[joint_type.value].x), np.int32(skeleton_2d.joints[joint_type.value].y)),
               point_size, point_color, thickness)


def draw_bone(image: np.ndarray, skeleton_2d: Skeleton2D, start: JointType, end: JointType):
    if (math.isnan(skeleton_2d.joints[start.value].x)
            or math.isnan(skeleton_2d.joints[start.value].y)
            or math.isnan(skeleton_2d.joints[end.value].x)
            or math.isnan(skeleton_2d.joints[end.value].y)):
        return

    point_color = (0, 255, 0)  # BGR
    thickness = 6
    line_type = 4

    cv2.line(image, (np.int32(skeleton_2d.joints[start.value].x), np.int32(skeleton_2d.joints[start.value].y)),
             (np.int32(skeleton_2d.joints[end.value].x), np.int32(skeleton_2d.joints[end.value].y)),
             point_color, thickness, line_type)


count = c_uint()
error_code = Sensor.get_count(count)
if ErrorCode.success.value != error_code:
    print("Failed to get sensor count with error code: %s" % error_code)
    exit(1)

sensor = Sensor()
error_code = sensor.initialize(ColorResolution.vga, DepthResolution.vga, c_bool(True), c_uint(0))
if ErrorCode.success.value != error_code:
    print("Failed to initialize sensor with error code: %s" % error_code)
    exit(1)

error_code = sensor.start()
if ErrorCode.success.value != error_code:
    print("Failed to start sensor with error code: %s" % error_code)
    exit(1)

color_frame = ColorFrame()
depth_frame = DepthFrame()
publish_data = PublishData()
skeleton_2d = Skeleton2D()
try:
    while True:
        error_code = sensor.get_color_frame(color_frame)
        if ErrorCode.success.value != error_code:
            continue

        error_code = sensor.get_depth_frame(depth_frame)
        if ErrorCode.success.value != error_code:
            continue

        error_code = sensor.get_publish_data(publish_data)
        if ErrorCode.success.value != error_code:
            continue

        # Color image
        color_length = color_frame.width * color_frame.height * 4
        color_array_type = c_char * color_length
        addr = addressof(color_frame.pixels.contents)
        color_image = np.frombuffer(color_array_type.from_address(addr), dtype=np.uint8).reshape(color_frame.height, color_frame.width, 4)
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        # Depth image
        depth_length = depth_frame.width * depth_frame.height
        depth_array_type = c_ushort * depth_length
        addr = addressof(depth_frame.pixels.contents)
        depth_image = np.frombuffer(depth_array_type.from_address(addr), dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # print(type(skeleton_2d.joints[1]))

        # Skeleton image
        for i in range(publish_data.skeletons.size):
            for j in range(JointType.count.value):
                # point_2d = Float2
                # sensor.depth_space_point_to_screen(publish_data.skeletons.data[i].joints[j].position, point_2d)
                sensor.depth_space_point_to_screen(publish_data.skeletons.data[i].joints[j].position, skeleton_2d.joints[j])
            draw_skeleton(depth_colormap, skeleton_2d)

        images = np.hstack((bgr_image, depth_colormap))

        cv2.namedWindow('YDViewer', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('YDViewer', images)

        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            break
finally:
    sensor.stop()
    sensor.uninitialize()
