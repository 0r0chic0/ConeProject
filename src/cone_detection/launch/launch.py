from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('cone_detection'),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='cone_detection',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            parameters=[config]
        ),
        Node(
            package='cone_detection',
            executable='process_node',
            name='process_node',
            parameters=[config]
        ),
        Node(
            package='cone_detection',
            executable='camera_node',
            name='camera_node',
            parameters=[config]
        ),
        # Node(
        #     package='cone_detection',
        #     executable='waypoint_node',
        #     name='waypoints',
        #     parameters=[config]
        # )
    ])
