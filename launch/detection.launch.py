import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get the launch directory
    pkg_pose_detection = get_package_share_directory('capra_pose_tracking')
    pkg_rove_description = get_package_share_directory('rove_description')

    # Get the URDF file
    urdf_path = os.path.join(pkg_rove_description, 'urdf', 'rove.urdf.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    pose_detection = Node(
		package="capra_pose_tracking",
        executable="person_detection_node",
	)

    return LaunchDescription([
        pose_detection,
            ])
