"""Launch the existing Raptor model with simulated time and controllers."""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro


def setup(context):
    description = get_package_share_directory('raptor_description')
    robot = xacro.process_file(os.path.join(description, 'urdf', 'raptor.urdf.xacro'),
        mappings={'test_fixture': LaunchConfiguration('test_fixture').perform(context),
                  'detailed_visuals': LaunchConfiguration('detailed_visuals').perform(context),
                  'sensors': LaunchConfiguration('sensors').perform(context)}).toxml()
    spawn = Node(package='ros_gz_sim', executable='create', arguments=[
        '-world', 'raptor_world', '-topic', 'robot_description',
        '-name', 'raptor', '-z', '0.03'], output='screen')
    broadcaster = Node(package='controller_manager', executable='spawner',
                       arguments=['joint_state_broadcaster'], output='screen')
    controller = Node(package='controller_manager', executable='spawner',
                      arguments=['raptor_joint_controller'], output='screen')
    return [
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[broadcaster])),
        RegisterEventHandler(OnProcessExit(target_action=broadcaster, on_exit=[controller])),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', os.path.join(
            description, 'worlds', 'raptor_world.sdf')], output='screen',
            additional_env={'GZ_SIM_RESOURCE_PATH': os.path.dirname(description)+os.pathsep+
                            os.environ.get('GZ_SIM_RESOURCE_PATH','')}),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot, 'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             condition=IfCondition(LaunchConfiguration('sensors')),
             arguments=['/raptor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
                '/raptor/left_foot/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
                '/raptor/right_foot/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
                '/raptor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/raptor/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/raptor/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo']),
        spawn,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('test_fixture', default_value='false'),
        DeclareLaunchArgument('detailed_visuals', default_value='false'),
        DeclareLaunchArgument('sensors', default_value='false'),
        OpaqueFunction(function=setup),
    ])
