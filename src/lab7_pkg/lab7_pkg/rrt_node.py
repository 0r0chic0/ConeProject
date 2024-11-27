# File: src/rrt/rrt_node.py

import numpy as np
import scipy.ndimage as nd
import math
from typing import List, Optional, Tuple

import rclpy
import tf_transformations
from .rrt_algo import RRTAlgorithm
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker 
from nav_msgs.msg import Odometry, OccupancyGrid


class TreeNode:
    def __init__(self, point: Tuple[float, float], parent=None):
        self.x = point[0]
        self.y = point[1]
        self.children = []
        self.parent = parent

# RRT Node
# This node is used to implement the RRT path planning algorithm

# CALLBACKS:
#   - pose_callback: Used to update the car's position and find the closest goal, then run the path planning algorithm
#   - scan_callback: Used for dynamic obstacle detection and addition to the occupancy grid

# IMPORTANT METHODS:
# execute: Run the path planning algorithm (currently BFS) to find the shortest path
# transform_waypoints_to_vehicle_frame: Transform the goal waypoints to the vehicle frame
# find_lookahead_waypoint: Find the closest goal waypoint ahead of the vehicle
# reconstruct_path: Reconstruct the path from the start to the goal using the came_from dictionary

class RRT(Node):
    def __init__(self):
        super().__init__("rrt_node")
        self.get_logger().info("RRT Node Started")

        # Topics
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"
        waypoints_topic = "/waypoints"
        goalpoints_topic = "/goalpoints"
        occupancy_grid_topic = "/rrt_occupancy_grid"
        rrt_tree_topic = "/rrt_tree"

        # Parameters
        self.map_width = 120
        self.map_height = 120
        self.grid_resolution = 0.025  # Grid cell size in meters
        self.step_size = 0.2  # Step size between waypoints
        self.goal_threshold = 0.2  # Goal hit radius in meters
        self.max_iterations = 500  
        self.lookahead_distance = 1.5
        self.car_length = 0.35  
        self.car_width = 0.20
        self.rrt_goal_bias = 0.15#%

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

        # Subscribers
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseArray, waypoints_topic, 10) #For pure pursuit
        self.grid_pub = self.create_publisher(OccupancyGrid, occupancy_grid_topic, 10) #For Rviz Visuals
        self.goal_pub = self.create_publisher(Marker, "/curr_goalpoint", 10) #For Rviz Visuals
        self.goal_array_pub = self.create_publisher(PoseArray, goalpoints_topic, 10) #For Rviz Visuals
        self.rrt_tree_pub = self.create_publisher(Marker, rrt_tree_topic, 10) #For Rviz Visuals

        # Initialize RRT Algorithm
        self.rrt_algo = RRTAlgorithm(
            self.map_width, 
            self.map_height, 
            self.step_size, 
            self.goal_threshold, 
            self.max_iterations,
            self.rrt_goal_bias,
            self.rrt_tree_pub,
            self.grid_pub,
            self.get_clock(),
            self.get_logger()
        )

        # Class attributes
        self.occupancy_grid = None
        self.start = None
        self.goal = None

        # Precompute kernel
        d = int(2 * (self.car_width / self.grid_resolution))
        self.kernel = np.fromfunction(lambda x, y: ((x+.5-d/2)**2 + (y+.5-d/2)**2 < (d/2)**2)*1, shape=(d, d), dtype=int)

        # Load waypoints from file
        self.goals = []
        self.goal_path = "./src/lab7_pkg/config/waypoints.csv"  # Path to waypoints
        self.load_goals()
        self.publish_points(self.goals, self.goal_array_pub)

    #
    # Scan callback is used for dynamic obstacle detection and addition to the occupancy grid
    # This will run on callback and always be updated whenever lidar scans come in
    # 
    def scan_callback(self, scan_msg: LaserScan):
        """
        LaserScan callback to update the occupancy grid and expand obstacles.
        """

        # Initialize occupancy grid if it doesn't exist
        if self.occupancy_grid is None:
            self.occupancy_grid = OccupancyGrid()
            self.occupancy_grid.info.resolution = self.grid_resolution
            self.occupancy_grid.info.width = self.map_width
            self.occupancy_grid.info.height = self.map_height
            self.occupancy_grid.info.origin.position.x = -self.map_width * self.grid_resolution / 2
            self.occupancy_grid.info.origin.position.y = -self.map_height * self.grid_resolution / 2

        # Reset occupancy grid to free space (0)
        grid_data = np.zeros((self.map_height, self.map_width), dtype=int)

        # Define robot's position in grid coordinates
        car_x = 0
        car_y = self.map_height // 2

        # Convert scan ranges and angles to NumPy arrays for vectorized operations
        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment

        # Filter out invalid ranges
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Calculate obstacle positions in grid coordinates
        obstacle_x = np.round(valid_ranges * np.cos(valid_angles) / self.grid_resolution).astype(int) + car_x
        obstacle_y = np.round(valid_ranges * np.sin(valid_angles) / self.grid_resolution).astype(int) + car_y

        # Mark in-bounds obstacles
        in_bounds = (
            (obstacle_x >= 0) & (obstacle_x < self.map_width) &
            (obstacle_y >= 0) & (obstacle_y < self.map_height)
        )
        obstacle_x = obstacle_x[in_bounds]
        obstacle_y = obstacle_y[in_bounds]
        grid_data[obstacle_y, obstacle_x] = 1

        # Apply bubbling with a circular kernel, ala minecraft circle generation       
        max_filtered = nd.maximum_filter(grid_data, footprint=self.kernel) > 0
        expanded_grid = (max_filtered * 100).astype(int)
        self.occupancy_grid.data = expanded_grid.ravel().tolist()

        # Publish the updated occupancy grid
        self.occupancy_grid.header.frame_id = "map"
        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.grid_pub.publish(self.occupancy_grid)

    #
    # Pose callback is used to update the car's position and find the closest goal
    # This will run on callback and always be updated whenever the car's pose changes
    #
    def pose_callback(self, pose_msg: Odometry):
        """
        Pose callback to set the car's current position and find the closest goal.
        """

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
        transformed_waypoints = self.transform_goalpoints_to_car_frame(current_pose, self.yaw)
        lookahead_waypoint = self.find_lookahead_waypoint(transformed_waypoints, self.lookahead_distance)

        if lookahead_waypoint:
            # Convert the lookahead waypoint back to the global frame for path planning
            global_lookahead_x = self.x + (
                lookahead_waypoint.x * math.cos(self.yaw) - lookahead_waypoint.y * math.sin(self.yaw)
            )
            global_lookahead_y = self.y + (
                lookahead_waypoint.x * math.sin(self.yaw) + lookahead_waypoint.y * math.cos(self.yaw)
            )

            # Publish the goal pose for visualization
            self.publish_goal(global_lookahead_x, global_lookahead_y)

            # Set the start and goal for path planning
            self.start = (self.x + self.car_length / 2, self.y)
            self.goal = (global_lookahead_x, global_lookahead_y)

            # Execute the path planning algorithm (RRT)
            path = self.rrt_algo.execute(self.start, self.goal, self.occupancy_grid)
            # self.get_logger().info(f"Poses: {path}")
            if (path):
                self.publish_points(path, self.waypoint_pub)

    def transform_goalpoints_to_car_frame(self, current_pose: Pose, heading: float):
        transformed_goalpoints = []
        for goalpoint in self.goals:
            dx = goalpoint.x - current_pose.position.x
            dy = goalpoint.y - current_pose.position.y
            transformed_x = dx * np.cos(-heading) - dy * np.sin(-heading)
            transformed_y = dx * np.sin(-heading) + dy * np.cos(-heading)
            transformed_goalpoints.append(Point(x=transformed_x, y=transformed_y))
        return transformed_goalpoints

    def find_lookahead_waypoint(self, transformed_waypoints: List[Point], lookahead_distance: float) -> Optional[Point]:
        closest_waypoint = None
        min_distance_diff = float('inf')

        for wp in transformed_waypoints:
            # Ensure waypoint is ahead of the vehicle (positive x in vehicle frame)
            dist_to_wp = np.sqrt((wp.x)**2 + (wp.y)**2)
            if wp.x <= 0 or dist_to_wp < lookahead_distance:
                continue

            # Ensure we are not in a wall
            wp_grid_x = int(wp.x / self.grid_resolution)
            wp_grid_y = int(wp.y / self.grid_resolution) + self.map_height // 2
            if (self.check_around_point(wp_grid_x, wp_grid_y, 3)):
                continue
                
            # Minimize the difference between waypoint distance and lookahead distance
            distance_diff = abs(dist_to_wp - lookahead_distance)
            if distance_diff < min_distance_diff:
                min_distance_diff = distance_diff
                closest_waypoint = wp
        
        return closest_waypoint
    
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

    def load_goals(self):
        try:
            with open(self.goal_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    self.goals.append(Point(x=x, y=y, z=0.0))
        except FileNotFoundError:
            self.get_logger().warn("Waypoints file not found.")

    # Publishes the next goal pose for visualization in RViz
    def publish_goal(self, x, y):
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame id for RViz, adjust as necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_pose_marker"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the marker size (scale) and color
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.goal_pub.publish(marker)

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
            quaternion = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
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
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()