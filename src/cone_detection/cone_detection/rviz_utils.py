from typing import List, Tuple
from rclpy.node import Publisher, Clock
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np

RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
GREEN = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
BLUE = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)

class Rviz:
    def __init__(self, clock: Clock, frame_id: str):
        self.clock = clock
        self.frame_id = frame_id
    
    def publish_points(self, waypoints: List[Point], last_len: int, namespace: str, color: ColorRGBA, publisher: Publisher) -> int:
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
    
    def publish_angles(self, origin: Tuple, rangeArray: List[float], last_len: int, namespace: str, color: ColorRGBA, publisher: Publisher) -> int:
        wallranges_markers = MarkerArray()

        for i, angle in enumerate(rangeArray):
            marker = self.angle_to_arrow(origin, angle, i, namespace, color)

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

    def point_to_marker(self, point: Point, idx: int, namespace: str, color: ColorRGBA) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.clock.now().to_msg()
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
    
    def angle_to_arrow(self, origin: Tuple, angle: float, idx: int, namespace: str, color: ColorRGBA) -> Marker:
        # Create and publish a marker for this direction angle
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.clock.now().to_msg()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        x, y, yaw = origin

        # tail at current position
        marker.points.append(Point(x=x, y=y, z=0.0))

        # Set arrow length
        arrow_length = 10.0
        end_x = x + arrow_length * np.cos(angle + yaw)
        end_y = y + arrow_length * np.sin(angle + yaw)
        marker.points.append(Point(x=end_x, y=end_y, z=0.0))

        # Set the marker size and color
        marker.scale.x = 0.025
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = color

        return marker

    def remove_marker(self, idx: int, namespace: str) -> Marker:
        # Create and publish a marker for this point
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.clock.now().to_msg()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.DELETE

        return marker