import numpy as np
from typing import List, Tuple

from .rviz_utils import Rviz, RED, BLUE, GREEN
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

LEFT = 1.0
RIGHT = 0.0

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Camera Node Started")

        # Topics
        pose_topic = "ego_racecar/odom"
        wall_ranges_topic = "/wall_ranges"
        left_marker_point_topic = "/left_wall_marker_points"
        right_marker_point_topic = "/right_wall_marker_points"
        left_marker_range_topic = "/left_wall_marker_ranges"
        right_marker_range_topic = "/right_wall_marker_ranges"

        # Parameters
        self.car_length = 0.35 # in meters
        self.car_width = 0.20 # in meters
        self.fov = 70 # in degrees
        self.fov_radians = np.radians(self.fov)
        self.half_fov = self.fov_radians / 2.0
        self.cone_diameter = 0.30 # in meters
        self.frame_id = "map"
        self.log = False

        # Current location
        self.x = None
        self.y = None
        self.yaw = None

        # Subscribers
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)

        # Publishers (#Keeping for now but we will have to change Rviz visuals)
        self.wall_publisher = self.create_publisher(Float64MultiArray, wall_ranges_topic, 10) 

        # Load wallpoints from file
        self.left_wall = []
        self.right_wall = []
        self.left_wall_path = "./src/cone_detection/config/left_wall.csv"  # Path to waypoints
        self.right_wall_path = "./src/cone_detection/config/right_wall.csv"  # Path to waypoints
        self.load_wallpoints()

        # Visualization publishers for RViz
        if self.log:
            self.left_marker_point_pub = self.create_publisher(MarkerArray, left_marker_point_topic, 10)
            self.right_marker_point_pub = self.create_publisher(MarkerArray, right_marker_point_topic, 10)
            self.left_marker_range_pub = self.create_publisher(MarkerArray, left_marker_range_topic, 10)
            self.last_left_wallranges_len = 0
            self.right_marker_range_pub = self.create_publisher(MarkerArray, right_marker_range_topic, 10)
            self.last_right_wallranges_len = 0
            self.border_pub = self.create_publisher(MarkerArray, '/border', 10)

            self.rviz = Rviz(self.get_clock(), self.frame_id)

    def pose_callback(self, pose_msg: Odometry):
        if self.log:
            self.rviz.publish_points(self.left_wall, len(self.left_wall), "left_wall", RED, self.left_marker_point_pub)
            self.rviz.publish_points(self.right_wall, len(self.right_wall), "right_wall", BLUE, self.right_marker_point_pub)

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

        # Remove overlaps between cones
        sorted_cones = sorted(cones, key=lambda x: x[3], reverse=True)
        for i in range(len(sorted_cones)):
            for j in range(i+1, len(sorted_cones)):
                further = sorted_cones[i]
                closer = sorted_cones[j]

                # no overlap
                if closer[1] > further[2] or closer[2] < further[1]:
                    continue

                # find valid segments
                valid_segments = [(0,0)]
                if further[1] > closer[1]:
                    valid_segments.append((further[1], closer[1]))
                if closer[2] < further[2]:
                    valid_segments.append((closer[2], further[2]))

                best_segment_idx = np.argmax([s[1] - s[0] for s in valid_segments])
                best_segment = valid_segments[best_segment_idx]

                sorted_cones[i] = (
                    further[0],
                    best_segment[0],
                    best_segment[1],
                    further[3]
                )    

        filtered_cones = []
        for _, (side, tstart, tend, distance) in enumerate(sorted_cones):
            fourth_angle = np.arctan2((self.cone_diameter / 4.0), distance)
            # Must be able to see 3/4 of it
            if tstart + 3*fourth_angle < tend:
                filtered_cones.append((side, tstart + fourth_angle, tend - fourth_angle))

        # Separate back into left and right
        final_left_cones = [(tstart, tend) for (side, tstart, tend) in filtered_cones if side == "L"]
        final_right_cones = [(tstart, tend) for (side, tstart, tend) in filtered_cones if side == "R"]

        wallArray = Float64MultiArray()
        wallArray.data = []
        for theta_start, theta_end in final_left_cones:
            wallArray.data.extend([theta_start, theta_end, LEFT])

        for theta_start, theta_end in final_right_cones:
            wallArray.data.extend([theta_start, theta_end, RIGHT])

        # Publish the right wall
        self.wall_publisher.publish(wallArray)

        if self.log:
            self.last_left_wallranges_len = self.rviz.publish_angles(
                (self.x, self.y, self.yaw), 
                [r for cone in final_left_cones for r in cone], 
                self.last_left_wallranges_len, 
                "left_ranges", 
                RED, 
                self.left_marker_range_pub
            )
            self.last_right_wallranges_len = self.rviz.publish_angles(
                (self.x, self.y, self.yaw), 
                [r for cone in final_right_cones for r in cone], 
                self.last_right_wallranges_len, 
                "right_ranges", 
                BLUE, 
                self.right_marker_range_pub
            )
            self.rviz.publish_angles(
                (self.x, self.y, self.yaw), 
                [0.610865, -0.610865], # [35deg, -35deg]
                2, 
                "border", 
                GREEN, 
                self.border_pub
            )

    def cones_in_fov(self, waypoints: List[Point]) -> List[Tuple]:
        cones_in_view_ranges = []
        for point in waypoints:
            # Compute vector from us to point
            dx = point.x - self.x
            dy = point.y - self.y
            if dx == 0 and dy == 0:
                continue

            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - self.yaw
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            distance = np.sqrt(dx**2 + dy**2)
            if distance == 0:
                continue

            # Calculate angular size of cone
            half_angle = np.arctan2((self.cone_diameter / 2.0), distance)
            theta_start = angle_diff - half_angle
            theta_end = angle_diff + half_angle

            # normalize angles 
            theta_start = (theta_start + np.pi) % (2 * np.pi) - np.pi
            theta_end = (theta_end + np.pi) % (2 * np.pi) - np.pi

            if theta_end >= -self.half_fov and theta_start <= self.half_fov:
                cones_in_view_ranges.append((theta_start, theta_end, distance))

        return cones_in_view_ranges

    def get_yaw_from_pose(self, pose: Pose) -> float:
        q = pose.orientation
        qy2 = q.y * q.y
        qz2 = q.z * q.z
        qwqz = q.w * q.z
        qxqy = q.x * q.y

        siny_cosp = 2.0 * (qwqz + qxqy)
        cosy_cosp = 1.0 - 2.0 * (qy2 + qz2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def load_wallpoints(self):
        try:
            with open(self.left_wall_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    self.left_wall.append(Point(x=x, y=y))
                    
            with open(self.right_wall_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    self.right_wall.append(Point(x=x, y=y))
                    
        except FileNotFoundError:
            self.get_logger().warn("Waypoints file not found.")
       
def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()