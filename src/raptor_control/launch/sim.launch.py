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
    hip = float(LaunchConfiguration('crouch_hip_pitch').perform(context))
    if not -.25 <= hip <= .05:
        raise ValueError('crouch_hip_pitch outside experimental bounds [-0.25, 0.05]')
    leg_design = LaunchConfiguration('leg_design').perform(context)
    if leg_design not in ('legacy', 'digitigrade'):
        raise ValueError('leg_design must be legacy or digitigrade')
    toe_scale = float(LaunchConfiguration('toe_stiffness_scale').perform(context))
    if not 1 <= toe_scale <= 5:
        raise ValueError('toe_stiffness_scale outside experimental bounds [1, 5]')
    robot = xacro.process_file(os.path.join(description, 'urdf', 'raptor.urdf.xacro'),
        mappings={'test_fixture': LaunchConfiguration('test_fixture').perform(context),
                  'detailed_visuals': LaunchConfiguration('detailed_visuals').perform(context),
                  'sensors': LaunchConfiguration('sensors').perform(context),
                  'passive_toes': LaunchConfiguration('passive_toes').perform(context),
                  'crouched_start': LaunchConfiguration('crouched_start').perform(context),
                  'toe_stiffness_scale': str(toe_scale),
                  'leg_design': leg_design,
                  'crouch_hip_pitch': str(hip)}).toxml()
    spawn = Node(package='ros_gz_sim', executable='create', arguments=[
        '-world', 'raptor_world', '-topic', 'robot_description',
        '-name', 'raptor', '-z', LaunchConfiguration('spawn_z').perform(context),
        '-P', LaunchConfiguration('spawn_pitch').perform(context)], output='screen')
    broadcaster = Node(package='controller_manager', executable='spawner',
                       arguments=['joint_state_broadcaster'], output='screen')
    controller = Node(package='controller_manager', executable='spawner',
                      arguments=['raptor_joint_controller'], output='screen')
    return [
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[broadcaster])),
        RegisterEventHandler(OnProcessExit(target_action=broadcaster, on_exit=[controller])),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', LaunchConfiguration('world_file').perform(context)], output='screen',
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
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             condition=IfCondition(LaunchConfiguration('passive_toes')),
             arguments=['/raptor/passive_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model'] + [
                 f'/raptor/{side}/toe_{digit}_{part}/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'
                 for side in ['left', 'right'] for digit in range(1, 4)
                 for part in ['proximal', 'distal']],
             remappings=[('/raptor/passive_joint_states', '/joint_states')]),
        spawn,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_file', default_value=os.path.join(
            get_package_share_directory('raptor_description'), 'worlds', 'raptor_world.sdf')),
        DeclareLaunchArgument('spawn_z', default_value='0.03'),
        DeclareLaunchArgument('spawn_pitch', default_value='0.0'),
        DeclareLaunchArgument('test_fixture', default_value='false'),
        DeclareLaunchArgument('detailed_visuals', default_value='false'),
        DeclareLaunchArgument('sensors', default_value='false'),
        DeclareLaunchArgument('passive_toes', default_value='false'),
        DeclareLaunchArgument('toe_stiffness_scale', default_value='1.0'),
        DeclareLaunchArgument('crouched_start', default_value='false'),
        DeclareLaunchArgument('leg_design', default_value='legacy'),
        DeclareLaunchArgument('crouch_hip_pitch', default_value='-0.15'),
        OpaqueFunction(function=setup),
    ])
