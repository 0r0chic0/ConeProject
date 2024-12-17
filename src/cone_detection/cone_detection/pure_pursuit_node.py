#!/usr/bin/env python3
from typing import List
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car using waypoints from a topic.
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Waypoints topic and other topics
        waypoints_topic = '/waypoints'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        self.lookahead_distance = 0.5  # arbitrary parameter value
        self.waypoints = []  # Initialize an empty list for waypoints

        # Subscribe to the /waypoints topic (assuming waypoints are published as a PoseArray)
        self.waypoints_subscription = self.create_subscription(
            PoseArray,
            waypoints_topic,
            self.waypoints_callback,
            10
        )

        # Subscribe to the /ego_racecar/odom topic
        self.odomSubscription = self.create_subscription(
            Odometry, 
            odom_topic, 
            self.pose_callback, 
            10
        )

        # Publish to the drive topic
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # Parameters (if any)
        self.declare_parameters(
            namespace='',
            parameters=[      
            ]
        )

    def waypoints_callback(self, waypoints_msg: PoseArray):
        """Callback to update waypoints when new waypoints are received."""
        self.waypoints = [pose.position for pose in waypoints_msg.poses]

    def pose_callback(self, pose_msg: Odometry):
        """Callback to update car's pose and execute pure pursuit."""
        if len(self.waypoints) == 0:
            return

        # Extract position from the message
        current_pose = pose_msg.pose.pose
        
        # Extract the current position and heading from the pose message
        heading = self.get_yaw_from_pose(current_pose)
        
        # Transform waypoints to vehicle frame
        transformed_waypoints = self.transform_waypoints_to_vehicle_frame(current_pose, heading)
        
        # Find the lookahead waypoint
        lookahead_waypoint = self.find_lookahead_waypoint(transformed_waypoints, self.lookahead_distance)
        
        if lookahead_waypoint:
            # Calculate the steering angle to the lookahead point
            steering_angle = self.calculate_steering_angle(lookahead_waypoint)
            
            # Publish the drive command
            self.publish_drive_command(steering_angle)
        
    def get_yaw_from_pose(self, pose: Pose) -> float:
        """Extract yaw (heading) from a Pose message's quaternion orientation."""
        orientation = pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def transform_waypoints_to_vehicle_frame(self, current_pose: Pose, heading: float):
        """Transform all waypoints from the global frame to the vehicle's frame."""        
        transformed_waypoints = []
        for waypoint in self.waypoints:
            dx = waypoint.x - current_pose.position.x
            dy = waypoint.y - current_pose.position.y

            # Rotate to vehicle frame
            transformed_x = dx * np.cos(-heading) - dy * np.sin(-heading)
            transformed_y = dx * np.sin(-heading) + dy * np.cos(-heading)
            
            transformed_waypoints.append(Point(x=transformed_x, y=transformed_y))
        return transformed_waypoints

    def find_lookahead_waypoint(self, transformed_waypoints: List[Point], lookahead_distance: float) -> Point:
        """Find the waypoint at the lookahead distance in the vehicle's frame."""
        closest_waypoint = None
        min_distance_diff = float('inf')

        for wp in transformed_waypoints:
            dist_to_wp = np.sqrt(wp.x**2 + wp.y**2)
            
            # Ensure waypoint is ahead of the vehicle (positive x in vehicle frame)
            if wp.x > 0 and dist_to_wp >= lookahead_distance:
                # Minimize the difference between waypoint distance and lookahead distance
                distance_diff = abs(dist_to_wp - lookahead_distance)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    closest_waypoint = wp
        
        return closest_waypoint

    def calculate_steering_angle(self, lookahead_waypoint: Point):
        """Calculate the steering angle to the lookahead waypoint."""        
        L = self.lookahead_distance  # Lookahead distance
        y = lookahead_waypoint.y  # Lateral offset to the lookahead point
        if y == 0:
            return 0.0

        # Calculate the curvature (k)
        curvature = 2 * y / (L ** 2)

        # Calculate the steering angle (delta) with arctan
        steering_angle = np.arctan(curvature * L)

        # Limit the steering angle within allowable bounds for safety
        max_steering_angle = 0.4
        return np.clip(steering_angle, -max_steering_angle, max_steering_angle)

    def publish_drive_command(self, steering_angle):
        """Publish the drive command with the calculated steering angle."""        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
