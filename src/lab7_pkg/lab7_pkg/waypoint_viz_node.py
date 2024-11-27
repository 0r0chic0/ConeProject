#!/usr/bin/env python3
import rclpy
from rclpy.qos import DurabilityPolicy
from rclpy.node import Node, QoSProfile

from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

class WaypointVizNode(Node):
    """ 
    Log and store waypoints for future use
    """
    def __init__(self):
        super().__init__('waypoint_viz_node')
        # Topics
        waypoint_topic = '/waypoints'
        goalpoint_topic = '/goalpoints'
        waypoint_markers_topic = '/waypoint_markers'
        goalpoint_markers_topic = '/goalpoint_markers'

        # Waypoint subscription (is actually PointArray)
        self.waypoint_sub = self.create_subscription(PoseArray, waypoint_topic, self.waypoints_callback, 10)

        # Goalpoint subscription (is actually PointArray)
        self.goalpoint_sub = self.create_subscription(PoseArray, goalpoint_topic, self.goalpoints_callback, 10)

        # Visualization publishers for RViz
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        self.waypoint_visual_pub = self.create_publisher(Marker, waypoint_markers_topic, qos_profile)
        self.goalpoint_visual_pub = self.create_publisher(MarkerArray, goalpoint_markers_topic, qos_profile)

        # Store waypoints
        self.waypoints_namespace = "waypoints"
        self.goalpoints_namespace = "goalpoints"
        self.last_goalpoints_len = 0

    
    def waypoints_callback(self, points_msg: PoseArray):
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame id for RViz, adjust as necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.waypoints_namespace
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        for _, pose in enumerate(points_msg.poses):
            marker.points.append(pose.position)

        # Set the marker size (scale) and color
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Publish the markers
        self.waypoint_visual_pub.publish(marker)

    def goalpoints_callback(self, points_msg: PoseArray):
        goalpoint_markers = MarkerArray()

        for i, point in enumerate(points_msg.poses):
            marker = self.point_to_marker(point.position, i)

            goalpoint_markers.markers.append(marker)

        # Delete old
        curr_goalpoints_len = len(goalpoint_markers.markers)
        if (curr_goalpoints_len < self.last_goalpoints_len):
            for i in range(curr_goalpoints_len, self.last_goalpoints_len):
                marker = self.remove_marker(i, False)

                goalpoint_markers.markers.append(marker)
        self.last_goalpoints_len = curr_goalpoints_len

        # Publish the markers
        self.goalpoint_visual_pub.publish(goalpoint_markers)

    def point_to_marker(self, point: Point, idx: int) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame id for RViz, adjust as necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.goalpoints_namespace
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.w = 1.0

        # Set the marker size (scale) and color
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker
    
    def remove_marker(self, idx: int, waypoint: bool) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = "base_link"  # Set the frame id for RViz, adjust as necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.marker_namespace
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.DELETE

        return marker

def main(args=None):
    rclpy.init(args=args)
    print("Waypoints Initialized")
    waypoint_viz_node = WaypointVizNode()
    rclpy.spin(waypoint_viz_node)

    waypoint_viz_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
