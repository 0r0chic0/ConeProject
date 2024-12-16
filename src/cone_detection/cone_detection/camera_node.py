import numpy as np
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node, Publisher
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, ColorRGBA

RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
GREEN = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
BLUE = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Camera Node Started")

        # Topics
        pose_topic = "ego_racecar/odom"
        leftwall_ranges_topic = "/left_wall_ranges"
        rightwall_ranges_topic = "/right_wall_ranges"
        left_marker_point_topic = "/left_wall_marker_points"
        right_marker_point_topic = "/right_wall_marker_points"
        left_marker_range_topic = "/left_wall_marker_ranges"
        right_marker_range_topic = "/right_wall_marker_ranges"

        # Parameters
        self.fov = 70 # in degrees
        self.car_length = 0.35 # in meters
        self.car_width = 0.20 # in meters
        self.cone_diameter = 0.25 # in meters
        self.frame_id = "map"

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

        # Subscribers
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)

        # Publishers (#Keeping for now but we will have to change Rviz visuals)
        self.left_wall_publisher = self.create_publisher(Float64MultiArray, leftwall_ranges_topic, 10) 
        self.right_wall_publisher = self.create_publisher(Float64MultiArray, rightwall_ranges_topic, 10) 

        # Load wallpoints from file
        self.left_wall = []
        self.right_wall = []
        self.left_wall_path = "./src/cone_detection/config/left_wall.csv"  # Path to waypoints
        self.right_wall_path = "./src/cone_detection/config/right_wall.csv"  # Path to waypoints
        self.load_wallpoints()

        # Visualization publishers for RViz
        self.left_marker_point_pub = self.create_publisher(MarkerArray, left_marker_point_topic, 10)
        self.right_marker_point_pub = self.create_publisher(MarkerArray, right_marker_point_topic, 10)
        self.left_marker_range_pub = self.create_publisher(MarkerArray, left_marker_range_topic, 10)
        self.last_left_wallranges_len = 0
        self.right_marker_range_pub = self.create_publisher(MarkerArray, right_marker_range_topic, 10)
        self.last_right_wallranges_len = 0
        self.border_pub = self.create_publisher(MarkerArray, '/border', 10)

    def pose_callback(self, pose_msg: Odometry):
        self.publish_wallpoints(self.left_wall, "left_wall", RED, self.left_marker_point_pub)
        self.publish_wallpoints(self.right_wall, "right_wall", BLUE, self.right_marker_point_pub)

        # get current position
        current_pose = pose_msg.pose.pose
        self.x = current_pose.position.x
        self.y = current_pose.position.y
        self.yaw = self.get_yaw_from_pose(current_pose)

        # Compute cones for both sides with distances
        left_cones = self.cones_in_fov(self.left_wall) 
        right_cones = self.cones_in_fov(self.right_wall)

        # Tag them to know which side they belong to
        cones = [("L", *c) for c in left_cones] + [("R", *c) for c in right_cones]

        # Remove overlaps between left and right sets
        indices_to_remove = set()
        for i in range(len(cones)):
            if i in indices_to_remove:
                continue
            for j in range(i+1, len(cones)):
                if j in indices_to_remove:
                    continue

                _, t1_start, t1_end, d1 = cones[i]
                _, t2_start, t2_end, d2 = cones[j]

                # Check for overlap
                if not (t1_end < t2_start or t2_end < t1_start):
                    if d1 > d2:
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)

        filtered_cones = []
        for idx, (side, tstart, tend, _) in enumerate(cones):
            if idx not in indices_to_remove:
                filtered_cones.append((side, tstart, tend))

        # Separate back into left and right
        final_left_cones = [(tstart, tend) for (side, tstart, tend) in filtered_cones if side == "L"]
        final_right_cones = [(tstart, tend) for (side, tstart, tend) in filtered_cones if side == "R"]

        leftWallArray = Float64MultiArray()
        leftWallArray.data = []
        for theta_start, theta_end in final_left_cones:
            leftWallArray.data.extend([theta_start, theta_end])

        # Publish the left wall
        self.left_wall_publisher.publish(leftWallArray)
        self.last_left_wallranges_len = self.publish_wallranges(leftWallArray.data, self.last_left_wallranges_len, "left_ranges", RED, self.left_marker_range_pub)
        
        rightWallArray = Float64MultiArray()
        rightWallArray.data = []
        for theta_start, theta_end in final_right_cones:
            rightWallArray.data.extend([theta_start, theta_end])

        # Publish the right wall
        self.right_wall_publisher.publish(rightWallArray)
        self.last_right_wallranges_len = self.publish_wallranges(rightWallArray.data, self.last_right_wallranges_len, "right_ranges", BLUE, self.right_marker_range_pub)

        # Publish border
        self.publish_wallranges([0.610865, -0.610865], 2, "border", GREEN, self.border_pub)

    def cones_in_fov(self, waypoints: List[Point]) -> List[Tuple]:
        fov_radians = math.radians(self.fov)
        half_fov = fov_radians / 2.0

        cones_in_view_ranges = []
        for point in waypoints:
            # Compute vector from robot to point
            dx = point.x - self.x
            dy = point.y - self.y
            if dx == 0 and dy == 0:
                continue

            # find angle from point
            point_angle = math.atan2(dy, dx)
            angle_diff = point_angle - self.yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            # Check angle difference is in fov
            if -half_fov <= angle_diff <= half_fov:
                distance = math.sqrt(dx**2 + dy**2)
                if distance == 0:
                    continue

                # Half-angle that the cone subtends
                half_angle = math.atan2((self.cone_diameter / 2.0), distance)

                theta_start = angle_diff - half_angle
                theta_end = angle_diff + half_angle
                theta_start = (theta_start + math.pi) % (2 * math.pi) - math.pi
                theta_end = (theta_end + math.pi) % (2 * math.pi) - math.pi

                cones_in_view_ranges.append((theta_start, theta_end, distance))

        return cones_in_view_ranges

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

    def publish_wallranges(self, rangeArray: List[float], last_len: int, namespace: str, color: ColorRGBA, publisher: Publisher) -> int:
        wallranges_markers = MarkerArray()

        for i, angle in enumerate(rangeArray):
            marker = self.angle_to_arrow(self.yaw + angle, i, namespace, color)

            wallranges_markers.markers.append(marker)

        # Delete old
        curr_wallranges_len = len(wallranges_markers.markers)
        if (curr_wallranges_len < last_len):
            for i in range(curr_wallranges_len, last_len):
                marker = self.remove_marker(i, namespace)

                wallranges_markers.markers.append(marker)

        # Publish the markers
        publisher.publish(wallranges_markers)
        return curr_wallranges_len

    def publish_wallpoints(self, points: List[Point], namespace: str, color: ColorRGBA, publisher: Publisher):
        wallpoints_markers = MarkerArray()

        for i, point in enumerate(points):
            marker = self.point_to_marker(point, i, namespace, color)

            wallpoints_markers.markers.append(marker)

        # Publish the markers
        publisher.publish(wallpoints_markers)

    def angle_to_arrow(self, angle: float, idx: int, namespace: str, color: ColorRGBA) -> Marker:
        # Create and publish a marker for this direction angle
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # tail at current position
        marker.points.append(Point(x=self.x, y=self.y, z=0.0))

        # Set arrow length
        arrow_length = 10.0
        end_x = self.x + arrow_length * math.cos(angle)
        end_y = self.y + arrow_length * math.sin(angle)
        marker.points.append(Point(x=end_x, y=end_y, z=0.0))

        # Set the marker size and color
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = color

        return marker

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
        marker.type = Marker.ARROW
        marker.action = Marker.DELETE

        return marker

       
def main(args=None):
    rclpy.init(args=args)
    print("Camera Node Initialized")
    camera_node = CameraNode()
    rclpy.spin(camera_node)

    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()