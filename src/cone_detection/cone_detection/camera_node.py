# File: src/rrt/rrt_node.py

import numpy as np
import scipy.ndimage as nd
import math
from typing import List, Optional, Tuple

import rclpy
#import tf_transformations
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from sensor_msgs.msg import Range
from visualization_msgs.msg import Marker 
from nav_msgs.msg import Odometry, OccupancyGrid

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Camera Node Started")

        # Topics
        pose_topic = "ego_racecar/odom"
        leftwall_ranges_topic = "/left_wall_ranges"
        rightwall_ranges_topic = "/right_wall_ranges"
        occupancy_grid_topic = "/rrt_occupancy_grid"

        # Parameters
        self.map_width = 70
        self.map_height = 120
        self.grid_resolution = 0.025  # Grid cell size in meters
        self.lookahead_distance = 1.5
        self.car_length = 0.35  
        self.car_width = 0.20

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

        # Subscribers
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)

        # Publishers (#Keeping for now but we will have to change Rviz visuals)
        self.grid_pub = self.create_publisher(OccupancyGrid, occupancy_grid_topic, 10)
        self.goal_pub = self.create_publisher(Marker, "/curr_wallpoint", 10) 
        self.left_wall_publisher = self.create_publisher(PoseArray, leftwall_ranges_topic, 10) 
        self.right_wall_publisher = self.create_publisher(PoseArray, rightwall_ranges_topic, 10) 

        # Class attributes
        self.occupancy_grid = None
        self.start = None
        self.goal = None

        # Precompute kernel
        d = int(2 * (self.car_width / self.grid_resolution))
        self.kernel = np.fromfunction(lambda x, y: ((x+.5-d/2)**2 + (y+.5-d/2)**2 < (d/2)**2)*1, shape=(d, d), dtype=int)

        # Load wallpoints from file
        self.left_wall = []
        self.right_wall = []
        self.left_wall_path = "./src/cone_detection/config/left_wall.csv"  # Path to waypoints
        self.right_wall_path = "./src/cone_detection/config/right_wall.csv"  # Path to waypoints
        self.load_wallpoints()

        #not sure this works yet
        #self.publish_points(self.goals, self.goal_array_pub)


    #
    # Pose callback is used to update the car's position and find the closest walls in range
    # This will run on callback and always be updated whenever the car's position changes
    #
    def pose_callback(self, pose_msg: Odometry):

        # Return if occupancy grid doesnt exist yet
        if self.occupancy_grid is None:
            return
        
        current_pose = pose_msg.pose.pose
        self.x = current_pose.position.x
        self.y = current_pose.position.y
        self.yaw = self.get_yaw_from_pose(current_pose)

        # Calculate the offset in the car's frame
        half_grid_height = self.map_height * self.grid_resolution / 2
        offset_x = half_grid_height * math.sin(self.yaw)
        offset_y = -half_grid_height * math.cos(self.yaw)

        # Update the grid origin position and orientation in the global frame. This is for Rviz visualization
        self.occupancy_grid.info.origin.position.x = self.x + offset_x
        self.occupancy_grid.info.origin.position.y = self.y + offset_y

        self.occupancy_grid.info.origin.orientation.w = np.cos(self.yaw / 2)
        self.occupancy_grid.info.origin.orientation.z = np.sin(self.yaw / 2)
        self.occupancy_grid.info.origin.orientation.x = 0.0
        self.occupancy_grid.info.origin.orientation.y = 0.0

        # Publish update
        self.grid_pub.publish(self.occupancy_grid)

        # Setup the goal waypoints to be relative to car frame, then find the closest one ahead
        transformed_left_waypoints = self.transform_wallpoints_to_car_frame(current_pose, self.yaw, self.left_wall)
        transformed_right_waypoints = self.transform_wallpoints_to_car_frame(current_pose, self.yaw, self.right_wall)


        lookahead_waypoints_left = self.find_lookahead_waypoints(transformed_left_waypoints, self.lookahead_distance) 
        lookahead_waypoints_right = self.find_lookahead_waypoints(transformed_right_waypoints, self.lookahead_distance)

        if lookahead_waypoints_left:
            left_wallpoints_pub = []
            for wp in lookahead_waypoints_left:

                # Convert the lookahead waypoint back to the global frame for path planning
                lookahead_x = self.x + (
                    wp.x * math.cos(self.yaw) - wp.y * math.sin(self.yaw)
                )
                lookahead_y = self.y + (
                    wp.x * math.sin(self.yaw) + wp.y * math.cos(self.yaw)
                )
                distance = np.sqrt((lookahead_x - self.x)**2 + (lookahead_y - self.y)**2)
                fov = 0.5
                color = 0

                # Create Range message
                range_msg = Range()
                range_msg.header.stamp = self.get_clock().now().to_msg()
                range_msg.header.frame_id = "left_wall_ranges"
                range_msg.radiation_type = 0 #0 for left wall 1 for right
                range_msg.field_of_view = fov
                range_msg.min_range = 0.0
                range_msg.max_range = 0.0
                range_msg.range = distance

                left_wallpoints_pub.append(range_msg)
                #create Range message

                # Publish the goal pose for visualization
            self.publish_wall(left_wallpoints_pub, self.left_wall_publisher)
        

        if lookahead_waypoints_right:
            right_wallpoints_pub = []
            for wp in lookahead_waypoints_right:

                # Convert the lookahead waypoint back to the global frame for path planning
                lookahead_x = self.x + (
                    wp.x * math.cos(self.yaw) - wp.y * math.sin(self.yaw)
                )
                lookahead_y = self.y + (
                    wp.x * math.sin(self.yaw) + wp.y * math.cos(self.yaw)
                )
                distance = np.sqrt((lookahead_x - self.x)**2 + (lookahead_y - self.y)**2)
                fov = 0.5
                color = 0

                # Create Range message
                range_msg = Range()
                range_msg.header.stamp = self.get_clock().now().to_msg()
                range_msg.header.frame_id = "right_wall_ranges"
                range_msg.radiation_type = 1 #0 for left wall 1 for right
                range_msg.field_of_view = fov
                range_msg.min_range = 0.0
                range_msg.max_range = 0.0
                range_msg.range = distance

                right_wallpoints_pub.append(range_msg)
                #create Range message

                # Publish the goal pose for visualization
            self.publish_wall(right_wallpoints_pub, self.right_wall_publisher)

                
    def transform_wallpoints_to_car_frame(self, current_pose: Pose, heading: float, wallpoints):
        transformed_wallpoints = []
        for wallpoint in wallpoints:
            dx = wallpoint.x - current_pose.position.x
            dy = wallpoint.y - current_pose.position.y
            transformed_x = dx * np.cos(-heading) - dy * np.sin(-heading)
            transformed_y = dx * np.sin(-heading) + dy * np.cos(-heading)
            transformed_wallpoints.append(Point(x=transformed_x, y=transformed_y))
        return transformed_wallpoints

    def find_lookahead_waypoints(self, transformed_waypoints: List[Point], lookahead_distance: float) -> Optional[Point]:
        valid_waypoints = []

        for wp in transformed_waypoints:
            # Ensure waypoint is ahead of the vehicle (positive x in vehicle frame)
            dist_to_wp = np.sqrt((wp.x)**2 + (wp.y)**2)
            if wp.x <= 0 or dist_to_wp < lookahead_distance:
                continue

            # Ensure the waypoint is not in a wall
            wp_grid_x = int(wp.x / self.grid_resolution)
            wp_grid_y = int(wp.y / self.grid_resolution) + self.map_height // 2
            if self.check_around_point(wp_grid_x, wp_grid_y, 3):
                continue

            # Add waypoint to the valid list
            valid_waypoints.append(wp)

        return valid_waypoints

    def check_around_point(self, x: int, y: int, rad: int) -> bool:
        for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    nx, ny = x + dx, y + dy
                    
                    # Check boundaries
                    point_idx = int(ny * self.map_width + nx)
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height and self.occupancy_grid.data[point_idx] > 0:
                        return True
        return False

    def get_yaw_from_pose(self, pose: Pose) -> float:
        orientation = pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def load_wallpoints(self):
        try:
            with open(self.left_wall_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    self.left_wall.append(Point(x=x, y=y, z=0.0))
                    
            with open(self.right_wall_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    self.right_wall.append(Point(x=x, y=y, z=0.0))
                    
        except FileNotFoundError:
            self.get_logger().warn("Waypoints file not found.")

    def publish_wall(self, wallpoints_pub: List[Range], publisher):
        """
        Publish a list of Range messages to the specified publisher.

        Args:
            wallpoints_pub (List[Range]): List of Range messages to publish.
            publisher: The ROS2 publisher to send the messages.
        """
        for range_msg in wallpoints_pub:
            publisher.publish(range_msg)
        self.get_logger().info(f"Published {len(wallpoints_pub)} range messages to {publisher.topic_name}.")


    # Publishes the goal poses as a PoseArray message to a new topic for visualization in RViz
    def publish_points(self, points: List[Point], publisher):
        point_array = PoseArray()
       
        for i in range(len(points)):
            if i < len(points) - 1:
                p1, p2 = points[i], points[i+1]
            else:
                p1, p2 = points[i], points[i]
            pose = Pose()
            pose.position.x = p1.x
            pose.position.y = p1.y
            pose.position.z = 0.0

            # Calculate yaw (heading)
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)

            # Calculate quaternion
            #quaternion = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
            pose.orientation = Quaternion(
                w=quaternion[0],
                x=quaternion[1],
                y=quaternion[2],
                z=quaternion[3],
            )
            
            point_array.poses.append(pose)
       
        publisher.publish(point_array)

       
def main(args=None):
    rclpy.init(args=args)
    print("Camera Node Initialized")
    camera_node = CameraNode()
    rclpy.spin(camera_node)

    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":#Keeping for now but we will have to change Rviz visuals
    main()