from dataclasses import dataclass, field
import os
import time
import numpy as np
from typing import List, Tuple

from .rviz_utils import Rviz, RED, BLUE, GREEN
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray

LEFT = 1.0
RIGHT = 0.0

@dataclass
class Cone:
    position: Point
    count: int = 1
    first_seen: float = field(default_factory=lambda: time.time())
    last_seen: float = field(default_factory=lambda: time.time())
    used: bool = True

class ProcessNode(Node):
    def __init__(self):
        super().__init__("process_node")
        self.get_logger().info("Process Node Started")

        # Topics
        wall_ranges_topic = "/wall_ranges"
        leftwall_markers_topic = "/left_wall_markers"
        rightwall_markers_topic = "/right_wall_markers"
        pose_topic = "/ego_racecar/odom"
        waypoints_topic = "/waypoints"
        lidarscan_topic = "/scan"
        waypoint_viz_topic = "/waypoints_viz"

        # Parameters
        self.last_seen_prune = (10, 0.1) # prune if not seen atleast [0] readings before [1] seconds since last seen
        self.first_seen_prune = (30, 5) # prune if not seen atleast [0] readings before [1] seconds since first seen
        self.car_length = 0.35 # in meters
        self.car_width = 0.20 # in meters
        self.cone_diameter = 0.25 # in meters
        self.merge_point_threshold = 2 * self.cone_diameter # in meters
        self.frame_id = "map"
        self.log = True

        # Subscribers
        self.create_subscription(Float64MultiArray, wall_ranges_topic, self.wall_callback, 10)
        self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseArray, waypoints_topic, 10)

        # Persistant state
        self.left_wall: List[Cone] = []
        self.right_wall: List[Cone] = []

        # Lidar data
        self.lidar_array = None
        self.lidar_data = None

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

        # Visualization publishers for RViz
        if self.log:
            self.waypoint_viz_pub = self.create_publisher(MarkerArray, waypoint_viz_topic, 10)
            self.last_waypoint_len = 0
            self.leftwall_markers_pub = self.create_publisher(MarkerArray, leftwall_markers_topic, 10)
            self.last_leftwall_markers_len = 0
            self.rightwall_markers_pub = self.create_publisher(MarkerArray, rightwall_markers_topic, 10)
            self.last_rightwall_markers_len = 0

            self.rviz = Rviz(self.get_clock(), self.frame_id)

    def pose_callback(self, pose_msg: Odometry):
        # get current position
        current_pose = pose_msg.pose.pose
        self.x = current_pose.position.x
        self.y = current_pose.position.y
        self.yaw = self.get_yaw_from_pose(current_pose)

    def lidar_callback(self, scan_msg: LaserScan):
        self.lidar_array = np.array(scan_msg.ranges, dtype=float)
        self.lidar_data = scan_msg

    def wall_callback(self, wall_msg: Float64MultiArray):
        if self.x is None or self.lidar_array is None:
            return
        
        new_left_points, new_right_points = self.get_points_from_ranges(wall_msg.data)

        # Old data is better than no data
        if len(new_left_points) == 0 or len(new_right_points) == 0:
            return
        
        left_updated, self.left_wall = self.merge_points(self.left_wall, new_left_points, [])
        right_updated, self.right_wall = self.merge_points(self.right_wall, new_right_points, self.left_wall)

        # No need to recalculate if no changes
        if not left_updated and not right_updated:
            return

        if self.log:
            self.last_leftwall_markers_len = self.rviz.publish_points(
                [cone.position for cone in self.left_wall], 
                self.last_leftwall_markers_len, 
                "leftwall_markers", 
                RED, 
                self.leftwall_markers_pub
            )
            self.last_rightwall_markers_len = self.rviz.publish_points(
                [cone.position for cone in self.right_wall], 
                self.last_rightwall_markers_len, 
                "rightwall_markers", 
                BLUE, 
                self.rightwall_markers_pub
            )

        self.recalculate_midpoints()

    def get_points_from_ranges(self, range_data: List[float]) ->  Tuple[List[Point], List[Point]]:
        angle_min = self.lidar_data.angle_min
        total_points = len(self.lidar_array)
        angle_increment = self.lidar_data.angle_increment

        left_points = []
        right_points = []
        ranges = [(range_data[i], range_data[i + 1], range_data[i + 2]) for i in range(0, len(range_data), 3)]
        for (theta_start, theta_end, color) in ranges:
            # find angle
            i_start = int((theta_start - angle_min) / angle_increment)
            i_end = int((theta_end - angle_min) / angle_increment)
            i_start = np.clip(i_start, 0, total_points - 1)
            i_end = np.clip(i_end, 0, total_points - 1)

            # slice
            segment = self.lidar_array[i_start:i_end+1]

            # exclude nonfinite or negative values
            valid_mask = np.isfinite(segment) & (segment > 0)
            valid_values = segment[valid_mask]
            if valid_values.size == 0:
                continue

            # find average distance, rejecting outliers
            avg_dist = np.mean(valid_values[abs(valid_values - np.mean(valid_values)) < 1 * np.std(valid_values)]) + self.cone_diameter / 2
            theta_mid = (theta_start + theta_end) / 2.0

            if not np.isfinite(avg_dist) or avg_dist < 0 or avg_dist > 6:
                continue

            # get global coords and store
            global_angle = self.yaw + theta_mid
            x_g = self.x + avg_dist * np.cos(global_angle)
            y_g = self.y + avg_dist * np.sin(global_angle)
            p = Point(x=x_g, y=y_g, z=0.0)

            if color == LEFT:
                left_points.append(p)
            else:
                right_points.append(p)

        return left_points, right_points
    
    def merge_points(self, existing_cones: List[Cone], new_points: List[Point], overriding_cones: List[Cone]) -> Tuple[bool, List[Point]]:
        updated = False
        for new_pt in new_points:
            overridden = False
            for cone in overriding_cones:
                distance = self.euclidean_distance(new_pt, cone.position)
                if distance < self.merge_point_threshold:
                    overridden = True
                    break
            if overridden:
                continue

            matched = False
            for cone in existing_cones:
                distance = self.euclidean_distance(new_pt, cone.position)
                if distance < self.merge_point_threshold:
                    # Update the cone's position
                    cone.position.x = (cone.position.x * cone.count + new_pt.x) / (cone.count + 1)
                    cone.position.y = (cone.position.y * cone.count + new_pt.y) / (cone.count + 1)
                    cone.count += 1
                    cone.last_seen = time.time()
                    matched = True
                    break
            if not matched:
                # Add as a new cone
                updated = True
                existing_cones.append(Cone(position=new_pt))
        
        # Prune cones that don't meet the detection count or are outdated
        current_time = time.time()
        pruned_cones = []
        for cone in existing_cones:
            if (cone.count < self.last_seen_prune[0] and current_time - cone.last_seen > self.last_seen_prune[1]) or \
                (cone.count < self.first_seen_prune[0] and current_time - cone.first_seen > self.first_seen_prune[1]):
                if cone.used:
                    updated = True
                continue

            pruned_cones.append(cone)
        return updated, pruned_cones

    def recalculate_midpoints(self):
        midpoint_poses = PoseArray()

        for left_cone in self.left_wall:
            left_cone.used = False

        # For each point in the right wall, find the two closest points in the left wall
        for right_cone in self.right_wall:
            closest1 = None
            closest2 = None
            closest_dist1 = np.inf
            closest_dist2 = np.inf
            for left_cone in self.left_wall:
                dx = left_cone.position.x - right_cone.position.x
                dy = left_cone.position.y - right_cone.position.y
                distance = np.sqrt(dx**2 + dy**2)

                # Update closest1 and closest2 based on the current distance
                if distance < closest_dist1:
                    closest_dist2 = closest_dist1
                    closest2 = closest1
                    closest_dist1 = distance
                    closest1 = left_cone
                elif distance < closest_dist2:
                    closest_dist2 = distance
                    closest2 = left_cone

            # Add midpoint for the first closest cones if they exist
            if closest1 is not None:
                closest1.used = True
                mx1 = (right_cone.position.x + closest1.position.x) / 2.0
                my1 = (right_cone.position.y + closest1.position.y) / 2.0

                pose1 = Pose()
                pose1.position.x = mx1
                pose1.position.y = my1
                pose1.position.z = 0.0
                pose1.orientation.w = 1.0
                midpoint_poses.poses.append(pose1)

            if closest2 is not None:
                closest1.used = True
                mx2 = (right_cone.position.x + closest2.position.x) / 2.0
                my2 = (right_cone.position.y + closest2.position.y) / 2.0

                pose2 = Pose()
                pose2.position.x = mx2
                pose2.position.y = my2
                pose2.position.z = 0.0
                pose2.orientation.w = 1.0
                midpoint_poses.poses.append(pose2)

        # Publish the collected midpoints to the waypoint publisher
        self.waypoint_pub.publish(midpoint_poses)

        # If logging is enabled, publish the waypoints to RViz for visualization
        if self.log:
            waypoint_positions = [pose.position for pose in midpoint_poses.poses]
            self.last_waypoint_len = self.rviz.publish_points(
                waypoint_positions,
                self.last_waypoint_len,
                "waypoint_markers",
                GREEN,
                self.waypoint_viz_pub
            )
    
    def dist_from_self(self, point: Point) -> float:
        return self.euclidean_distance(Point(x=self.x, y=self.y), point)

    @staticmethod
    def euclidean_distance(p1: Point, p2: Point) -> float:
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    @staticmethod
    def get_yaw_from_pose(pose: Pose) -> float:
        orientation = pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)
       
def main(args=None):
    rclpy.init(args=args)
    process_node = ProcessNode()
    rclpy.spin(process_node)
    process_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()