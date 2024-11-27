from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lab7_pkg'),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='lab7_pkg',
            executable='rrt_node',
            name='rrt_node',
            parameters=[config]
        ),
        Node(
            package='lab7_pkg',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            parameters=[config]
        ),
            Node(
            package='lab7_pkg',
            executable='waypoint_viz_node',
            name='waypoint_viz_node',
            parameters=[config]
        ),
    ])