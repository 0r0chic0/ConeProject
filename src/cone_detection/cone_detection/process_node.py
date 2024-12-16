import numpy as np
import math
from typing import List

import rclpy
from rclpy.node import Node, Publisher
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
GREEN = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
BLUE = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)

class ProcessNode(Node):
    def __init__(self):
        super().__init__("process_node")
        self.get_logger().info("Process Node Started")

        # Topics
        leftwall_ranges_topic = "/left_wall_ranges"
        rightwall_ranges_topic = "/right_wall_ranges"
        leftwall_markers_topic = "/left_wall_markers"
        rightwall_markers_topic = "/right_wall_markers"
        pose_topic = "/ego_racecar/odom"
        waypoints_topic = "/waypoints"
        lidarscan_topic = "/scan"
        waypoint_viz_topic = "/waypoints_viz"

        # Parameters
        self.car_length = 0.35 # in meters
        self.car_width = 0.20 # in meters
        self.cone_diameter = 0.25 # in meters
        self.merge_point_threshold = 0.25 # in meters
        self.frame_id = "map"

        # Subscribers
        self.create_subscription(Float64MultiArray, leftwall_ranges_topic, self.leftwall_callback, 10) 
        self.create_subscription(Float64MultiArray, rightwall_ranges_topic, self.rightwall_callback, 10) 
        self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseArray, waypoints_topic, 10)
        self.waypoint_viz_pub = self.create_publisher(MarkerArray, waypoint_viz_topic, 10)
        self.last_waypoint_len = 0
        self.leftwall_markers_pub = self.create_publisher(MarkerArray, leftwall_markers_topic, 10)
        self.last_leftwall_markers_len = 0
        self.rightwall_markers_pub = self.create_publisher(MarkerArray, rightwall_markers_topic, 10)
        self.last_rightwall_markers_len = 0

        # Persistant state
        self.left_wall: List[Point] = []
        self.right_wall: List[Point] = []

        # Lidar data
        self.lidar_array = None
        self.lidar_data = None

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

    def pose_callback(self, pose_msg: Odometry):
        # get current position
        current_pose = pose_msg.pose.pose
        self.x = current_pose.position.x
        self.y = current_pose.position.y
        self.yaw = self.get_yaw_from_pose(current_pose)

    def lidar_callback(self, scan_msg: LaserScan):
        self.lidar_array = np.array(scan_msg.ranges, dtype=float)
        self.lidar_data = scan_msg

    def leftwall_callback(self, wall_msg: Float64MultiArray):
        new_points = self.get_points_from_ranges(wall_msg.data)
        self.left_wall = self.merge_points(self.left_wall, new_points, self.merge_point_threshold)
        self.last_leftwall_markers_len = self.publish_waypoint_viz(self.left_wall, self.last_leftwall_markers_len, "leftwall_markers", RED, self.leftwall_markers_pub)


        self.recalculate_midpoints()

    def rightwall_callback(self, wall_msg: Float64MultiArray):
        new_points = self.get_points_from_ranges(wall_msg.data)
        self.right_wall = self.merge_points(self.right_wall, new_points, self.merge_point_threshold)
        self.last_rightwall_markers_len = self.publish_waypoint_viz(self.right_wall, self.last_rightwall_markers_len, "rightwall_markers", BLUE, self.rightwall_markers_pub)

        self.recalculate_midpoints()

    def recalculate_midpoints(self):
        midpoint_poses = PoseArray()

        # For each point in the left wall and each point in the right wall, find the midpoint
        for lp in self.left_wall:
            if self.dist_from_self(lp) > 7:
                continue

            for rp in self.right_wall:
                if self.dist_from_self(rp) > 7:
                    continue

                mx = (lp.x + rp.x) / 2.0
                my = (lp.y + rp.y) / 2.0
                

                # Add point at midpoint
                pose = Pose()
                pose.position.x = mx
                pose.position.y = my
                pose.position.z = 0.0
                pose.orientation.w = 1.0
                midpoint_poses.poses.append(pose)

        # Publish to pure pursuit
        self.waypoint_pub.publish(midpoint_poses)
        self.last_waypoint_len = self.publish_waypoint_viz([pose.position for pose in midpoint_poses.poses], self.last_waypoint_len, "waypoint_markers", GREEN, self.waypoint_viz_pub)

    def merge_points(self, existing_points: List[Point], new_points: List[Point], threshold: float) -> List[Point]:
        # for new in new_points:
        #     merged = False
        #     for _, existing in enumerate(existing_points):
        #         dx = existing.x - new.x
        #         dy = existing.y - new.y
        #         dist = math.sqrt(dx*dx + dy*dy)
        #         if dist < threshold:
        #             # merge points
        #             existing.x = (existing.x + new.x) / 2.0
        #             existing.y = (existing.y + new.y) / 2.0
        #             merged = True
        #             break

        #     if not merged:
        #         # if no match, add in
        #         existing_points.append(new)
        
        return new_points

    def get_points_from_ranges(self, range_data: List[float]) ->  List[Point]:
        if self.x is None or self.lidar_array is None:
            return []

        # constants
        angle_min = self.lidar_data.angle_min
        total_points = len(self.lidar_array)
        angle_increment = self.lidar_data.angle_increment

        points = []
        ranges = [(range_data[i], range_data[i + 1]) for i in range(0, len(range_data), 2)]
        for (theta_start, theta_end) in ranges:
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
            # self.get_logger().info(f"{np.mean(valid_values)}, {np.std(valid_values)} {segment} {avg_dist}")

            # get global coords and store
            global_angle = self.yaw + theta_mid
            x_g = self.x + avg_dist * math.cos(global_angle)
            y_g = self.y + avg_dist * math.sin(global_angle)
            p = Point(x=x_g, y=y_g, z=0.0)
            points.append(p)

        return points
    
    def dist_from_self(self, point: Point) -> float:
        dx = self.x - point.x
        dy = self.y - point.y
        return np.sqrt(dx**2 + dy**2)

    def get_yaw_from_pose(self, pose: Pose) -> float:
        orientation = pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def publish_waypoint_viz(self, waypoints: List[Point], last_len: int, namespace: str, color: ColorRGBA, publisher: Publisher) -> int:
        waypoint_markers = MarkerArray()

        for i, point in enumerate(waypoints):
            marker = self.point_to_marker(point, i, namespace, color)

            waypoint_markers.markers.append(marker)

        # Delete old
        curr_waypoints_len = len(waypoint_markers.markers)
        if (curr_waypoints_len < last_len):
            for i in range(curr_waypoints_len, last_len):
                marker = self.remove_marker(i, namespace)

                waypoint_markers.markers.append(marker)
        # Publish the markers
        publisher.publish(waypoint_markers)
        return curr_waypoints_len
    
    def point_to_marker(self, point: Point, idx: int, namespace: str, color: ColorRGBA) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.w = 1.0

        # Set the marker size and color
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = color

        return marker
    
    def remove_marker(self, idx: int, namespace: str) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.DELETE

        return marker

       
def main(args=None):
    rclpy.init(args=args)
    process_node = ProcessNode()
    rclpy.spin(process_node)
    process_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()