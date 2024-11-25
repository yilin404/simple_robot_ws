from ctypes import *
from enum import Enum
import platform


class ErrorCode(Enum):
    """错误代码。

    函数返回值，标明函数执行结果。

    success: 函数执行成功 \n
    invalid_parameter: 无效参数 \n
    no_device: 未找到设备 \n
    no_permission: 未得到设备打开权限 \n
    device_unsupported: 不支持的图像设备 \n
    device_not_open: 设备未开启 \n
    stream_unsupported: 不支持的数据流类型 \n
    stream_disabled: 当前类型数据流未开启 \n
    timeout: 获取数据超时
    """
    success = 0
    invalid_parameter = 1
    no_device = 100
    no_permission = 101
    device_unsupported = 102
    device_not_open = 103
    stream_unsupported = 104
    stream_disabled = 105
    timeout = 106


class JointType(Enum):
    """骨骼节点类型。

    骨骼节点类型索引值，在骨骼节点数组中的不同索引的元素代表不同的节点。

    head: 头结点 \n
    neck: 脖子结点 \n
    left_shoulder: 左肩结点 \n
    right_shoulder: 右肩结点 \n
    left_elbow: 左肘结点 \n
    right_elbow: 右肘结点 \n
    left_hand: 左手结点 \n
    right_hand: 右手结点 \n
    left_chest: 左胸结点 \n
    right_chest: 右胸结点 \n
    left_waist: 左腰结点 \n
    right_waist: 右腰结点 \n
    left_knee: 左膝结点 \n
    right_knee: 右膝结点 \n
    left_foot: 左脚结点 \n
    right_foot: 右脚结点 \n
    torso: 躯干结点 \n
    count: 结点类型总数
    """
    head = 0
    neck = 1
    left_shoulder = 2
    right_shoulder = 3
    left_elbow = 4
    right_elbow = 5
    left_hand = 6
    right_hand = 7
    left_chest = 8
    right_chest = 9
    left_waist = 10
    right_waist = 11
    left_knee = 12
    right_knee = 13
    left_foot = 14
    right_foot = 15
    torso = 16
    count = 17


class ColorResolution(Enum):
    """彩色图像分辨率。

    qvga: 320x240 \n
    vga: 640x480 \n
    uvga: 1280x960
    """
    qvga = 0
    vga = 1
    uvga = 2


class DepthResolution(Enum):
    """深度图像分辨率。

    qvga: 320x240 \n
    vga: 640x480
    """
    qvga = 0
    vga = 1


class Float2(Structure):
    """包含 2 个浮点值的向量。

    屏幕向量。

    x: x 轴向量，向右为正。 \n
    y: y 轴向量，向上为正。
    """
    _fields_ = [("x", c_float),
                ("y", c_float)]


class Float3(Structure):
    """包含 3 个浮点值的向量。

    空间向量。

    x: x 轴向量，向右为正。 \n
    y: y 轴向量，向上为正。 \n
    z: z 轴向量，摄像头照射方向为正。
    """
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]


class Float4(Structure):
    """包含 4 个浮点值的向量。

    空间向量。

    x: x 轴向量，向右为正。 \n
    y: y 轴向量，向上为正。 \n
    z: z 轴向量，摄像头照射方向为正。 \n
    w: 辅助向量，用于空间转换。
    """
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float),
                ("w", c_float)]


class Joint(Structure):
    """骨骼节点。

    包含一个骨骼节点在应用场景中的三维空间信息。

    position: 前三个元素代表三维空间中的 x，y，z 坐标，最后一个元素 w 用来辅助做空间变换。 \n
    rotation: 代表一个四元数 {x，y，z，w}，表示当前节点相对于T-Pose在父节点空间坐标下的旋转。 \n
    orientation: 节点相对于T-Pose在世界空间坐标下的旋转表达，3个元素分别代表节点原始的 x, y 和 z 轴经旋转后的朝向。
    """
    _fields_ = [("position", Float4),
                ("rotation", Float4),
                ("orientation", Float3 * 3)]


class Skeleton(Structure):
    """骨骼数据。

    包含一个被识别用户的骨骼数据。

    user_id: 分配给用户的 ID （从 1 开始），用于区别场景中的多个用户。 \n
    is_tracked: 用户状态，表明在当前的一帧数据中用户是否被成功地追踪到。 \n
    position: 用户位置，表明用户身体中心在场景中的大致位置。 \n
    joints: 骨骼节点数据，每个元素代表用户身体上的某个关节在场景中的三维位置。
    """
    _fields_ = [("user_id", c_ushort),
                ("is_tracked", c_bool),
                ("position", Float4),
                ("joints", Joint * JointType.count.value)]


class Skeletons(Structure):
    """骨骼集合。

    包含所有被识别的骨骼信息。

    size: data 数组中的元素个数。 \n
    data: 每个数组元素是一个用户的骨骼信息。
    """
    _fields_ = [("size", c_uint),
                ("data", POINTER(Skeleton))]


class UserMask(Structure):
    """用户掩码图像。

    包含被识别用户在深度图像或彩色图像中的掩码信息。

    mask: 掩码图像像素集合。数组长度为 width * height ，每个像素值等于对应深度图中相应像素所属的用户 ID。
    如 x=100，y=100 的掩码像素值为 1，则对应深度图中 x=100，y=100 的深度像素属于 ID=1 的用户身体。 \n
    width: 掩码图像宽度，等于对应的深度图像宽度，以像素为单位。 \n
    height: 掩码图像高度，等于对应的深度图像高度，以像素为单位。
    """
    _fields_ = [("mask", POINTER(c_ushort)),
                ("width", c_uint),
                ("height", c_uint)]


class PublishData(Structure):
    """用户数据。

    包含由输入的深度数据图像计算生成的用户数据。

    skeletons: 用户骨骼数据集合。每个元素对应一位用户被识别的全身骨骼节点，数组的有效长度为 Skeleton.size 。 \n
    ground_model: 使用场景中被检测出来的地面的平面方程。4 个元素代表平面方程 Ax +By + Cz + D = 0 中的 A，B，C，D 四个系数。 \n
    user_mask: 用户掩码数据，此数据用来表示深度图像中的每个像素与用户之间的所属关系。 \n
    timestamp: 数据时间戳（以毫秒为单位），与对应输入的深度图像的时间戳相同，用来同步深度图像。
    """
    _fields_ = [("skeletons", Skeletons),
                ("ground_model", Float4),
                ("user_mask", UserMask),
                ("timestamp", c_longlong)]


class ColorFrame(Structure):
    """彩色图像帧。

    包含从硬件传感器采集的完整的一帧彩色图像。

    width: 图像宽度（以像素点为单位）。 \n
    height: 图像高度（以像素点为单位）。 \n
    frame_id: 图像帧序号（从 0 开始）。 \n
    timestamp: 图像采集时间戳（以毫秒为单位）。 \n
    pixels: 图像像素点集合。每个像素包含红绿蓝和 alpha 通道共四个字节，数组长度为 width * height 。
    """
    _fields_ = [("width", c_uint),
                ("height", c_uint),
                ("frame_id", c_long),
                ("timestamp", c_longlong),
                ("pixels", POINTER(c_uint))]


class DepthFrame(Structure):
    """深度图像帧。

    包含从硬件传感器采集的完整的一帧深度图像。

    width: 图像宽度（以像素点为单位）。 \n
    height: 图像高度（以像素点为单位）。 \n
    frame_id: 图像帧序号（从 0 开始）。 \n
    timestamp: 图像采集时间戳（以毫秒为单位）。 \n
    pixels: 图像像素点集合。每个像素代表拍摄的物体上的点到硬件传感器成像平面的距离，以毫米为单位。数组长度为 width * height 。
    """
    _fields_ = [("width", c_uint),
                ("height", c_uint),
                ("frame_id", c_long),
                ("timestamp", c_longlong),
                ("pixels", POINTER(c_ushort))]


class AudioFrame(Structure):
    """音频数据帧。

    包含从硬件传感器采集的完整的一帧音频。

    size: 音频数据数组长度。 \n
    duration: 音频数据持续时间。 \n
    frame_id: 音频帧序号（从 0 开始）。 \n
    timestamp: 音频采集时间戳（以毫秒为单位）。 \n
    data: 音频数据集合，数组长度为 size 。
    """
    _fields_ = [("size", c_uint),
                ("duration", c_uint),
                ("frame_id", c_long),
                ("timestamp", c_longlong),
                ("data", POINTER(c_ushort))]

class PointCloudFrame(Structure):
    _fields_ = [("point", POINTER(c_float))]


lib_name = ""
if platform.uname()[0] == "Windows":
    lib_name = ".\\YDPeopleSensor.dll"
elif platform.uname()[0] == "Linux":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = os.path.join(current_dir, "libYDPeopleSensor.so")
    print(lib_name)
    # lib_name = "./libYDPeopleSensor.so"

yd_people_sensor = cdll.LoadLibrary(lib_name)

get_people_sensor_count = yd_people_sensor.GetPeopleSensorCount
get_people_sensor_count.argtypes = [POINTER(c_uint)]
get_people_sensor_count.restype = c_int

create_people_sensor = yd_people_sensor.CreatePeopleSensor
create_people_sensor.argtypes = [POINTER(c_void_p)]
create_people_sensor.restype = c_bool

destroy_people_sensor = yd_people_sensor.DestroyPeopleSensor
destroy_people_sensor.argtypes = [c_void_p]

initialize_people_sensor = yd_people_sensor.InitializePeopleSensor
initialize_people_sensor.argtypes = [c_void_p, c_uint, c_uint, c_bool, c_uint]
initialize_people_sensor.restype = c_int

uninitialize_people_sensor = yd_people_sensor.UninitializePeopleSensor
uninitialize_people_sensor.argtypes = [c_void_p]

start_people_sensor = yd_people_sensor.StartPeopleSensor
start_people_sensor.argtypes = [c_void_p]
start_people_sensor.restype = c_int

stop_people_sensor = yd_people_sensor.StopPeopleSensor
stop_people_sensor.argtypes = [c_void_p]

is_people_sensor_running = yd_people_sensor.IsPeopleSensorRunning
is_people_sensor_running.argtypes = [c_void_p]
is_people_sensor_running.restype = c_bool

set_people_sensor_depth_mapped_to_color = yd_people_sensor.SetPeopleSensorDepthMappedToColor
set_people_sensor_depth_mapped_to_color.argtypes = [c_void_p, c_bool]

is_people_sensor_depth_mapped_to_color = yd_people_sensor.IsPeopleSensorDepthMappedToColor
is_people_sensor_depth_mapped_to_color.argtypes = [c_void_p]
is_people_sensor_depth_mapped_to_color.restype = c_bool

set_people_sensor_skeleton_mapped_to_color = yd_people_sensor.SetPeopleSensorSkeletonMappedToColor
set_people_sensor_skeleton_mapped_to_color.argtypes = [c_void_p, c_bool]

is_people_sensor_skeleton_mapped_to_color = yd_people_sensor.IsPeopleSensorSkeletonMappedToColor
is_people_sensor_skeleton_mapped_to_color.argtypes = [c_void_p]
is_people_sensor_skeleton_mapped_to_color.restype = c_bool

turn_on_people_sensor_infrared_emitter = yd_people_sensor.TurnOnPeopleSensorInfraredEmitter
turn_on_people_sensor_infrared_emitter.argtypes = [c_void_p]
turn_on_people_sensor_infrared_emitter.restype = c_int

turn_off_people_sensor_infrared_emitter = yd_people_sensor.TurnOffPeopleSensorInfraredEmitter
turn_off_people_sensor_infrared_emitter.argtypes = [c_void_p]
turn_off_people_sensor_infrared_emitter.restype = c_int

enable_people_sensor_color = yd_people_sensor.EnablePeopleSensorColor
enable_people_sensor_color.argtypes = [c_void_p]
enable_people_sensor_color.restype = c_int

disable_people_sensor_color = yd_people_sensor.DisablePeopleSensorColor
disable_people_sensor_color.argtypes = [c_void_p]
disable_people_sensor_color.restype = c_int

is_people_sensor_color_enabled = yd_people_sensor.IsPeopleSensorColorEnabled
is_people_sensor_color_enabled.argtypes = [c_void_p]
is_people_sensor_color_enabled.restype = c_bool

enable_people_sensor_audio = yd_people_sensor.EnablePeopleSensorAudio
enable_people_sensor_audio.argtypes = [c_void_p]
enable_people_sensor_audio.restype = c_int

disable_people_sensor_audio = yd_people_sensor.DisablePeopleSensorAudio
disable_people_sensor_audio.argtypes = [c_void_p]
disable_people_sensor_audio.restype = c_int

is_people_sensor_audio_enabled = yd_people_sensor.IsPeopleSensorAudioEnabled
is_people_sensor_audio_enabled.argtypes = [c_void_p]
is_people_sensor_audio_enabled.restype = c_bool

set_people_sensor_near_mode = yd_people_sensor.SetPeopleSensorNearMode
set_people_sensor_near_mode.argtypes = [c_void_p, c_bool]
set_people_sensor_near_mode.restype = c_int

is_people_sensor_near_mode = yd_people_sensor.IsPeopleSensorNearMode
is_people_sensor_near_mode.argtypes = [c_void_p]
is_people_sensor_near_mode.restype = c_bool

get_people_sensor_color_frame = yd_people_sensor.GetPeopleSensorColorFrame
get_people_sensor_color_frame.argtypes = [c_void_p, POINTER(ColorFrame)]
get_people_sensor_color_frame.restype = c_int

get_people_sensor_depth_frame = yd_people_sensor.GetPeopleSensorDepthFrame
get_people_sensor_depth_frame.argtypes = [c_void_p, POINTER(DepthFrame)]
get_people_sensor_depth_frame.restype = c_int

get_people_sensor_audio_frame = yd_people_sensor.GetPeopleSensorAudioFrame
get_people_sensor_audio_frame.argtypes = [c_void_p, POINTER(AudioFrame)]
get_people_sensor_audio_frame.restype = c_int

get_people_sensor_publish_data = yd_people_sensor.GetPeopleSensorPublishData
get_people_sensor_publish_data.argtypes = [c_void_p, POINTER(PublishData)]
get_people_sensor_publish_data.restype = c_int

get_people_sensor_color_frame_timeout = yd_people_sensor.GetPeopleSensorColorFrameTimeout
get_people_sensor_color_frame_timeout.argtypes = [c_void_p, POINTER(ColorFrame), c_uint]
get_people_sensor_color_frame_timeout.restype = c_int

get_people_sensor_depth_frame_timeout = yd_people_sensor.GetPeopleSensorDepthFrameTimeout
get_people_sensor_depth_frame_timeout.argtypes = [c_void_p, POINTER(DepthFrame), c_uint]
get_people_sensor_depth_frame_timeout.restype = c_int

get_people_sensor_audio_frame_timeout = yd_people_sensor.GetPeopleSensorAudioFrameTimeout
get_people_sensor_audio_frame_timeout.argtypes = [c_void_p, POINTER(AudioFrame), c_uint]
get_people_sensor_audio_frame_timeout.restype = c_int

get_people_sensor_publish_data_timeout = yd_people_sensor.GetPeopleSensorPublishDataTimeout
get_people_sensor_publish_data_timeout.argtypes = [c_void_p, POINTER(PublishData), c_uint]
get_people_sensor_publish_data_timeout.restype = c_int

people_sensor_depth_space_point_to_screen = yd_people_sensor.PeopleSensorDepthSpacePointToScreen
people_sensor_depth_space_point_to_screen.argtypes = [c_void_p, POINTER(Float4), POINTER(Float2)]
people_sensor_depth_space_point_to_screen.restype = c_int

people_sensor_color_space_point_to_screen = yd_people_sensor.PeopleSensorColorSpacePointToScreen
people_sensor_color_space_point_to_screen.argtypes = [c_void_p, POINTER(Float4), POINTER(Float2)]
people_sensor_color_space_point_to_screen.restype = c_int

people_sensor_convert_people_sensor_depth_frame_to_point_cloud= yd_people_sensor.ConvertPeopleSensorDepthFrameToPointCloud
people_sensor_convert_people_sensor_depth_frame_to_point_cloud.argtypes = [c_void_p, c_int,c_int,POINTER(c_ushort), POINTER(c_float)]
people_sensor_convert_people_sensor_depth_frame_to_point_cloud.restype = c_int

class Sensor(object):
    """传感器操作接口。"""

    def __init__(self):
        """构造函数。

        创建传感器实例。
        """
        self.sensor = c_void_p()
        create_people_sensor(byref(self.sensor))

    def __del__(self):
        """析构函数。

        销毁传感器实例。

        :rtype: None
        """
        destroy_people_sensor(self.sensor)

    @staticmethod
    def get_count(count):
        """获取可用的传感器数量。

        :param count: 可用的传感器数量。
        :type count: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_count(byref(count))

    def initialize(self, color_resolution, depth_resolution, generate_user_data, sensor_index):
        """初始化传感器实例。

        :param color_resolution: 彩色图像分辨率。
        :type color_resolution: ColorResolution
        :param depth_resolution: 深度图像分辨率。
        :type depth_resolution: DepthResolution
        :param generate_user_data: 是否要生成用户数据。
        :type generate_user_data: c_bool
        :param sensor_index: 要使用的传感器的索引值（从 0 开始）。
        :type sensor_index: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return initialize_people_sensor(self.sensor, color_resolution.value, depth_resolution.value, generate_user_data, sensor_index)

    def uninitialize(self):
        """终结传感器实例。

        :rtype: None
        """
        uninitialize_people_sensor(self.sensor)

    def start(self):
        """启动传感器实例。

        传感器启动后将不间断产生彩色、深度、音频和用户数据。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return start_people_sensor(self.sensor)

    def stop(self):
        """停止传感器实例。

        传感器停止后将结束彩色、深度、音频和用户数据流。

        :rtype: None
        """
        stop_people_sensor(self.sensor)

    def is_running(self):
        """检查传感器是否已启动。

        :return: 传感器已经启动返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_running(self.sensor)

    def set_depth_mapped_to_color(self, is_mapped):
        """设置深度图像是否匹配彩色图像。

        由于深度摄像头和彩色摄像头视角可能并不完全一致，以及两个摄像头安装位置不同等原因，
        空间中的点 P 在深度图像中的坐标 P1 和彩色图像中的坐标 P2 是不同的。
        通过将深度图像中的每一个像素映射到彩色图像视角，P1 映射后得到的新的深度图像中坐标为 P1'，P1' 和 P2 的坐标完全一致。
        一般该接口在 Start() 之前调用， GetDepthFrame() 所得到的每一帧图像都受其影响，
        GetPublishData() 所得到的每一帧数据也受其影响。

        :param is_mapped: 若为 True，深度图像与彩色图像视角匹配；否则不匹配。
        :type is_mapped: c_bool
        :rtype: None
        """
        set_people_sensor_depth_mapped_to_color(self.sensor, is_mapped)

    def is_depth_mapped_to_color(self):
        """检查深度图像是否与彩色图像匹配过。

        默认状态深度图像与彩色图像未匹配。

        :return: 深度图与彩色图像匹配过返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_depth_mapped_to_color(self.sensor)

    def set_skeleton_mapped_to_color(self, is_mapped):
        """设置骨骼是否匹配彩色图像。

        一般该接口在 Start() 之前调用， GetPublishData() 所得到的每一帧数据都受其影响。
        如果已经调用了 SetDepthMappedToColor() 并设置参数为 True， GetPublishData() 所得到的每一帧数据也都受其影响。
        GetDepthFrame() 不受该接口影响。

        :param is_mapped: 若为 True，骨骼与彩色图像视角匹配，否则不匹配。
        :type is_mapped: c_bool
        :rtype: None
        """
        set_people_sensor_skeleton_mapped_to_color(self.sensor, is_mapped)

    def is_skeleton_mapped_to_color(self):
        """检查骨骼是否与彩色图像匹配过。

        默认状态骨骼与彩色图像未匹配。

        :return: 骨骼与彩色图像匹配过返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_skeleton_mapped_to_color(self.sensor)

    def turn_on_infrared_emitter(self):
        """打开红外激光发射器。

        通常不需显式调用，在调用 Start() 函数后红外激光发射器已自动打开。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return turn_on_people_sensor_infrared_emitter(self.sensor)

    def turn_off_infrared_emitter(self):
        """闭红外激光发射器。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return turn_off_people_sensor_infrared_emitter(self.sensor)

    def enable_color(self):
        """打开彩色数据流。

        通常不需显式调用，默认情况下彩色数据流是打开的。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return enable_people_sensor_color(self.sensor)

    def disable_color(self):
        """关闭彩色数据流。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return disable_people_sensor_color(self.sensor)

    def is_color_enabled(self):
        """检查彩色数据流是否已打开。

        默认彩色数据流是打开状态。

        :return: 彩色数据流已打开返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_color_enabled(self.sensor)

    def enable_audio(self):
        """打开音频数据流。

        通常不需显式调用，默认情况下音频数据流是打开的。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return enable_people_sensor_audio(self.sensor)

    def disable_audio(self):
        """关闭音频数据流。

        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return disable_people_sensor_audio(self.sensor)

    def is_audio_enabled(self):
        """检查音频数据流是否已打开。

        默认音频数据流是打开状态。

        :return: 音频数据流已打开返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_audio_enabled(self.sensor)

    def set_near_mode(self, is_near_mode):
        """设置是否开启近距离模式。

        :param is_near_mode: True 表示打开近距离模式，否则 False。
        :type is_near_mode: c_bool
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return set_people_sensor_near_mode(self.sensor, is_near_mode)

    def is_near_mode(self):
        """检查当前是否处于近距离模式。

        :return: 近距离模式已打开返回 True，否则返回 False。
        :rtype: c_bool
        """
        return is_people_sensor_near_mode(self.sensor)

    def get_color_frame(self, frame):
        """获取一帧彩色图像。

        每个像素为4个通道，顺序为ARGB。

        :param frame: 彩色图像帧。
        :type frame: ColorFrame
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_color_frame(self.sensor, byref(frame))

    def get_depth_frame(self, frame):
        """获取一帧深度图像。

        默认得到原始深度图像，如果调用了 SetDepthMappedToColor() 并设置参数为 True，得到与彩色图像视角匹配后的深度图像。

        :param frame: 深度图像帧。
        :type frame: DepthFrame
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_depth_frame(self.sensor, byref(frame))

    def get_audio_frame(self, frame):
        """获取一帧音频数据。

        :param frame: 音频数据帧。
        :type frame: AudioFrame
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_audio_frame(self.sensor, byref(frame))

    def get_publish_data(self, data):
        """获取一帧用户数据。

        默认得到基于原始深度图像计算得到的 Skeletons 和 GroundModel，UserMask 是与彩色图像匹配后的人物剪影掩码。
        如果调用了 SetSkeletonMappedToColor() 函数并设置参数为 True，Skeletons 中每个骨骼点是转换到彩色图像空间后的三维坐标值。

        :param data: 用户数据。
        :type data: PublishData
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_publish_data(self.sensor, byref(data))

    def get_color_frame_timeout(self, frame, timeout):
        """在一定时限内获取一帧彩色图像。

        每个像素为4个通道，顺序为ARGB。

        :param frame: 彩色图像帧。
        :type frame: ColorFrame
        :param timeout: 等待超时时间。
        :type timeout: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_color_frame_timeout(self.sensor, byref(frame), timeout)

    def get_depth_frame_timeout(self, frame, timeout):
        """在一定时限内获取一帧深度图像。

        :param frame: 深度图像帧。
        :type frame: DepthFrame
        :param timeout: 等待超时时间。
        :type timeout: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_depth_frame_timeout(self.sensor, byref(frame), timeout)

    def get_audio_frame_timeout(self, frame, timeout):
        """在一定时限内获取一帧音频数据。

        :param frame: 音频数据帧。
        :type frame: AudioFrame
        :param timeout: 等待超时时间。
        :type timeout: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_audio_frame_timeout(self.sensor, byref(frame), timeout)

    def get_publish_data_timeout(self, data, timeout):
        """在一定时限内获取一帧用户数据。

        默认得到基于原始深度图像计算得到的 skeletons 和 ground_model，user_mask 是与彩色图像匹配后的人物剪影掩码。
        如果调用了 set_skeleton_mapped_to_color() 函数并设置参数为 True，skeletons 中每个骨骼点是转换到彩色图像空间后的三维坐标值。

        :param data: 用户数据。
        :type data: PublishData
        :param timeout: 等待超时时间。
        :type timeout: c_uint
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return get_people_sensor_publish_data_timeout(self.sensor, byref(data), timeout)

    def depth_space_point_to_screen(self, point_3d, point_2d):
        """获取深度相机空间中的骨骼点在深度图像二维平面上的投影。

        屏幕坐标原点(0, 0)位于屏幕左上方。

        :param point_3d: 骨骼点在三维空间中的坐标。
        :type point_3d: Float4
        :param point_2d: 骨骼点经投影计算后得到的二维坐标。
        :type point_2d: Float2
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return people_sensor_depth_space_point_to_screen(self.sensor, byref(point_3d), byref(point_2d))

    def color_space_point_to_screen(self, point_3d, point_2d):
        """获取彩色相机空间中的骨骼点在彩色图像二维平面上的投影。

        屏幕坐标原点(0, 0)位于屏幕左上方。

        :param point_3d: 骨骼点在三维空间中的坐标。
        :type point_3d: Float4
        :param point_2d: 骨骼点经投影计算后得到的二维坐标。
        :type point_2d: Float2
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return people_sensor_color_space_point_to_screen(self.sensor, byref(point_3d), byref(point_2d))
    

    def depth_frame_to_point_cloud(self, width, height, depth, cloud):
        
        
        cloud = c_float()
        return people_sensor_convert_people_sensor_depth_frame_to_point_cloud(self.sensor, width, height, byref(depth), byref(cloud))

    def depth_frame_to_point_cloud_simple(self, depth_frame:DepthFrame):
        width, height, depth = depth_frame.width, depth_frame.height, depth_frame.pixels.contents
        
        cloud = c_float()
        
        return people_sensor_convert_people_sensor_depth_frame_to_point_cloud(self.sensor, width, height, byref(depth), byref(cloud))

    def convert_depth_frame_to_point_cloud(self, width, height,depth,cloud):
        """从深度图像获取点云图像。

        :param width: 深度图像宽度。
        :type width: c_int
        :param height: 深度图像高度。
        :type height: c_int
        :param depth: 深度图像帧。
        :type depth: c_ushort
        :param cloud: 点云图像帧，空间由用户端分配及销毁，大小为 frame.Width * frame.Height * sizeof(float) * 3 字节。 每个点云图像像素由三个 float 值组成，分别代表点在真实空间中的 x, y, z 坐标，距离单位是米。
        :type cloud: c_float
        :return: 调用成功则返回 ErrorCode.success，否则参考 ErrorCode 以获得返回值具体含义。
        :rtype: c_int
        """
        return people_sensor_convert_people_sensor_depth_frame_to_point_cloud(self.sensor, width,height, byref(depth), byref(cloud))