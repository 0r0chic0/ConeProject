#!/usr/bin/env python3
from ctypes import Array
import os
import rclpy
from rclpy.qos import DurabilityPolicy
from rclpy.node import Node, QoSProfile

from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray

class WaypointNode(Node):
    """ 
    Log and store waypoints for future use
    """
    def __init__(self):
        super().__init__('waypoint_node')
        # Topics
        new_point_topic = '/clicked_point'

        # Waypoint subscription and service
        self.new_point_sub = self.create_subscription(PointStamped, new_point_topic, self.new_point_callback, 10)

        # Visualization publisher for RViz
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        self.waypoint_visual_pub = self.create_publisher(MarkerArray, '/waypoint_markers', qos_profile)

        # Store waypoints
        self.waypoints = []
        self.waypoint_markers = MarkerArray()
        self.marker_namespace = "waypoints"
        self.waypoint_path = os.path.realpath(os.path.join('src', 'cone_detection', 'config', 'right_wall.csv'))

        # Try read waypoints from csv
        try:
            with open(self.waypoint_path, "r") as file:
                for line in file.readlines():
                    x, y = [float(val) for val in line.split(',')]
                    waypoint = Point()
                    waypoint = Point(x=x, y=y, z=0.0015)
                    self.add_waypoint(waypoint)
                    self.get_logger().info(f"Added point at x: {x}, y: {y}")
            self.waypoint_visual_pub.publish(self.waypoint_markers)
        except FileNotFoundError:
            pass # Ignore file not found

    def new_point_callback(self, point_msg: PointStamped):
        self.add_waypoint(point_msg.point)

        # Publish the markers
        self.waypoint_visual_pub.publish(self.waypoint_markers)
        self.get_logger().info(f"Added point at x: {point_msg.point.x}, y: {point_msg.point.y}")

    def add_waypoint(self, point: Point):
        # Append the new point to the list of waypoints
        self.waypoints.append(point)

        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame id for RViz, adjust as necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.marker_namespace
        marker.id = len(self.waypoint_markers.markers)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.w = 1.0

        # Set the marker size (scale) and color
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Save the marker
        self.waypoint_markers.markers.append(marker)

    def clear_waypoints(self):
        # Create a new MarkerArray message to delete all existing markers
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns = self.marker_namespace
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.waypoint_visual_pub.publish(marker_array_msg)
        
    def save_waypoints(self):
        self.get_logger().info('Saving waypoints!')

        # Write to csv
        with open(self.waypoint_path, "w") as file:
            for waypoint in self.waypoints:
                file.write(f'{waypoint.x},{waypoint.y}\n')


def main(args=None):
    try:
        rclpy.init(args=args)
        print("Waypoints Initialized")
        waypoint_node = WaypointNode()
        rclpy.spin(waypoint_node)

        waypoint_node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        waypoint_node.save_waypoints()
        waypoint_node.clear_waypoints()


if __name__ == '__main__':
    main()