import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__("cone_detection_node")
        self.get_logger().info("Cone Detection Node Started")

        # Publishers
        self.color_publisher = self.create_publisher(String, "/detected_cone_colors", 10)

        # Subscribers
        self.create_subscription(Image, "/camera/video", self.image_callback, 10)

        # OpenCV bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()

    def image_callback(self, msg: Image):
        # Convert ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Detect cones by their colors
        detected_colors = self.detect_cones(frame)

        # Publish detected cone colors
        color_msg = String()
        color_msg.data = ",".join(detected_colors)
        self.color_publisher.publish(color_msg)

        # Display the video stream with detections (for debugging purposes)
        cv2.imshow("Cone Detection", frame)
        cv2.waitKey(1)

    def detect_cones(self, frame):
        detected_colors = []

        # Define color ranges in HSV format for red and blue
        color_ranges = {
            "RED": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
            "BLUE": [(100, 150, 0), (140, 255, 255)]
        }

        # Convert the image to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color, ranges in color_ranges.items():
            mask = None

            # Create masks for the defined color ranges
            for i in range(0, len(ranges), 2):
                lower_bound = np.array(ranges[i], dtype=np.uint8)
                upper_bound = np.array(ranges[i + 1], dtype=np.uint8)
                
                if mask is None:
                    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                else:
                    mask |= cv2.inRange(hsv_frame, lower_bound, upper_bound)

            # Check if any pixels in the mask indicate the presence of the color
            if np.any(mask):
                detected_colors.append(color)

            # Draw bounding boxes around detected cones 
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return detected_colors

def main(args=None):
    rclpy.init(args=args)
    cone_detection_node = ConeDetectionNode()
    rclpy.spin(cone_detection_node)
    cone_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
