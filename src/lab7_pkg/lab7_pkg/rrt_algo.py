from logging import Logger
from rclpy.clock import Clock
import numpy as np
from typing import List, Tuple, Optional
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from scipy.spatial import cKDTree

class TreeNode:
    def __init__(self, point: Tuple[float, float], parent=None):
        self.x = point[0]
        self.y = point[1]
        self.point = point
        self.parent = parent

class RRTAlgorithm:
    def __init__(
        self,
        map_width: int,
        map_height: int,
        step_size: float,
        goal_threshold: float,
        max_iterations: int,
        goal_bias: float,
        rrt_tree_pub,
        grid_pub,
        clock,
        logger
    ):
        self.map_width: int = map_width
        self.map_height: int = map_height
        self.step_size: float = step_size
        self.goal_threshold: float = goal_threshold
        self.max_iterations: int = max_iterations
        self.goal_bias: float = goal_bias
        self.grid: OccupancyGrid = None  # To be set externally
        self.tree_pub = rrt_tree_pub
        self.grid_pub = grid_pub
        self.clock: Clock = clock
        self.logger: Logger = logger
        self.log = True

        # For KD-Tree
        self.nodes = []
        self.positions = np.empty((0, 2))
        self.kdtree = None
        self.kdtree_rebuild_interval = 25
        self.kdtree_counter = 0

        if self.log:
            # For marker publishing frequency
            self.publish_interval = 50
            self.publish_counter = 0

            # Initialize tree marker
            self.tree_marker = Marker()
            self.tree_marker.header.frame_id = "map"
            self.tree_marker.header.stamp = self.clock.now().to_msg()
            self.tree_marker.ns = "tree_marker"
            self.tree_marker.id = 1
            self.tree_marker.type = Marker.LINE_LIST
            self.tree_marker.action = Marker.ADD

            # Set the marker size scale and color
            self.tree_marker.scale.x = 0.01
            self.tree_marker.scale.y = 0.01
            self.tree_marker.scale.z = 0.01
            self.tree_marker.color.r = 0.0
            self.tree_marker.color.g = 0.0
            self.tree_marker.color.b = 1.0
            self.tree_marker.color.a = 1.0

    def execute(self, start: Tuple[float, float], goal: Tuple[float, float], occupancy_grid) -> Optional[List[Point]]:
        self.grid = occupancy_grid
        self.root = TreeNode(start)
        self.goal_node = TreeNode(goal)
        self.nodes = [self.root]
        self.kdtree = None

        # Coordinate transformations
        origin = self.grid.info.origin
        self.origin_x = origin.position.x
        self.origin_y = origin.position.y
        self.resolution = self.grid.info.resolution
        q = np.array([origin.orientation.w, origin.orientation.x, origin.orientation.y, origin.orientation.z])
        self.yaw = self.quaternion_to_yaw(q)
        self.cos_yaw = np.cos(-self.yaw)  # Negative for inverse rotation
        self.sin_yaw = np.sin(-self.yaw)

        # Set sample width and height in grid coordinates to include goal
        goal_grid = self.tf_to_grid(goal)
        goal_radius = (self.goal_bias / self.resolution)
        self.sample_width = (min(0, goal_grid[0] - goal_radius), max(self.map_width, goal_grid[0] + goal_radius))
        self.sample_height = (min(0, goal_grid[1] - goal_radius), max(self.map_height, goal_grid[1] + goal_radius))

        # Reset marker
        if self.log:
            self.tree_marker.points = []

        for _ in range(self.max_iterations):
            # Rebuild KD-Tree every kdtree_rebuild_interval iterations
            if self.kdtree_counter % self.kdtree_rebuild_interval == 0 or self.kdtree is None:
                self.positions = np.array([[node.x, node.y] for node in self.nodes])
                self.kdtree = cKDTree(self.positions)
            self.kdtree_counter += 1

            sampled_point = self.sample()
            nearest_node = self.nearest(sampled_point)
            new_node = self.steer(nearest_node, sampled_point)

            if self.check_collision(nearest_node, new_node):
                self.nodes.append(new_node)

                # Update marker points
                if self.log:
                    self.tree_marker.points.append(Point(x=new_node.x, y=new_node.y, z=0.0))
                    self.tree_marker.points.append(Point(x=new_node.parent.x, y=new_node.parent.y, z=0.0))

                    # Publish markers less frequently
                    self.publish_counter += 1
                    if self.publish_counter % self.publish_interval == 0:
                        self.tree_pub.publish(self.tree_marker)

                if self.is_goal(new_node):
                    return self.reconstruct_path(new_node)

        return None  # No path found

    def sample(self) -> Tuple[float, float]:
        # Bias the sample towards the goal
        if np.random.rand() < self.goal_bias:
            return self.goal_node.point
        else:
            # Sample uniformly
            sampled_x = np.random.uniform(self.sample_width[0], self.sample_width[1])
            sampled_y = np.random.uniform(self.sample_height[0], self.sample_height[1])

            # Transform to global frame
            global_sampled_x, global_sampled_y = self.tf_to_global((sampled_x, sampled_y))

            return (global_sampled_x, global_sampled_y)

    def nearest(self, sampled_point: Tuple[float, float]) -> TreeNode:
        _, idx = self.kdtree.query(np.array(sampled_point))
        nearest_node = self.nodes[idx]
        return nearest_node

    def steer(self, nearest_node: TreeNode, sampled_point: Tuple[float, float]) -> TreeNode:
        direction = np.array(sampled_point) - np.array(nearest_node.point)
        distance = np.linalg.norm(direction)
        if distance == 0:
            return nearest_node
        direction_unit = direction / distance
        step_distance = min(self.step_size, distance)
        new_point = (nearest_node.x + direction_unit[0] * step_distance,
                     nearest_node.y + direction_unit[1] * step_distance)
        return TreeNode(new_point, parent=nearest_node)

    def check_collision(self, nearest_node: TreeNode, new_node: TreeNode) -> bool:
        pt0 = self.tf_to_grid(nearest_node.point)
        pt1 = self.tf_to_grid(new_node.point)
        line_points = self.bresenham(pt0, pt1)
        for x, y in line_points:
            idx = y * self.map_width + x
            if 0 <= x < self.map_width and 0 <= y < self.map_height and self.grid.data[idx] > 0:
                return False
        return True

    def is_goal(self, node: TreeNode) -> bool:
        return self.distance(node.point, self.goal_node.point) < self.goal_threshold

    def reconstruct_path(self, goal_node: TreeNode) -> List[Point]:
        path = []
        current_node = goal_node
        while current_node is not None:
            pose = Point(x=current_node.x, y=current_node.y, z=0.0)
            path.append(pose)
            current_node = current_node.parent
        return path[::-1]  # Reverse the path

    def tf_to_grid(self, global_point: Tuple[float, float]):
        x = global_point[0] - self.origin_x
        y = global_point[1] - self.origin_y
        x_rot = x * self.cos_yaw - y * self.sin_yaw
        y_rot = x * self.sin_yaw + y * self.cos_yaw
        grid_x = x_rot / self.resolution
        grid_y = y_rot / self.resolution
        return (grid_x, grid_y)

    def tf_to_global(self, grid_point: Tuple[float, float]) -> Tuple[float, float]:
        x_rot = grid_point[0] * self.resolution
        y_rot = grid_point[1] * self.resolution
        x = x_rot * np.cos(self.yaw) - y_rot * np.sin(self.yaw) + self.origin_x
        y = x_rot * np.sin(self.yaw) + y_rot * np.cos(self.yaw) + self.origin_y
        return (x, y)

    def quaternion_to_yaw(self, q):
        # q = [w, x, y, z]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    @staticmethod
    def bresenham(p0, p1):
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points

    @staticmethod
    def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return np.hypot(point1[0] - point2[0], point1[1] - point2[1])
