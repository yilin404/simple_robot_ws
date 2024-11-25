sudo chmod 777 /dev/ttyACM0

source ~/3rdparty/A1_SDK-release-v1.2.0/install/setup.bash
roslaunch signal_arm single_arm_node.launch host_serial_port_path:=/dev/ttyACM0 & sleep 2;

source ~/3rdparty/A1_SDK-release-v1.2.0/install/setup.bash
roslaunch mobiman eeTrackerdemo.launch & sleep 2;

wait;