import torch
import cv2
import numpy as np
from pathlib import Path
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

# Redirect PosixPath to WindowsPath
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class ConeDetectionYOLOv5:
    def __init__(self, weights='best.pt', device='cpu', img_size=640, conf_thres=0.70, iou_thres=0.30):
        # Initialize YOLOv5 model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.stride = self.model.stride
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.names  # Class names

        # OpenCV video capture (using default webcam)
        self.cap = cv2.VideoCapture(0)  # 0 refers to the default webcam
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.")
            exit()

    def preprocess_image(self, frame):
        # Resize and normalize image
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)  # Contiguous array
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # Normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_cones(self, frame):
        # Preprocess frame
        img = self.preprocess_image(frame)

        # Inference
        pred = self.model(img, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        detected_cones = []
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Ensure ROI coordinates are within bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    # Extract region of interest (ROI)
                    roi = frame[y1:y2, x1:x2]

                    # Ensure ROI is not empty
                    if roi.size == 0:
                        continue

                    # Convert ROI to HSV color space
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Define color ranges for red and blue in HSV
                    red_lower1 = np.array([0, 120, 70])
                    red_upper1 = np.array([10, 255, 255])
                    red_lower2 = np.array([170, 120, 70])
                    red_upper2 = np.array([180, 255, 255])
                    blue_lower = np.array([100, 150, 0])
                    blue_upper = np.array([140, 255, 255])

                    # Create masks for red and blue
                    red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
                    red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
                    red_mask = red_mask1 | red_mask2
                    blue_mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)

                    # Determine the color based on the mask with the largest non-zero area
                    red_area = cv2.countNonZero(red_mask)
                    blue_area = cv2.countNonZero(blue_mask)

                    if red_area > blue_area:
                        color = "Red Cone"
                    elif blue_area > red_area:
                        color = "Blue Cone"
                    else:
                        color = "Unknown"

                    # Adjust bounding box for tight fit
                    x1, y1, x2, y2 = map(int, [x1 + 2, y1 + 2, x2 - 2, y2 - 2])

                    # Draw bounding box and label
                    label = f'{color} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    detected_cones.append((color, conf, (x1, y1, x2, y2)))

        return detected_cones

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Detect cones
            detected_cones = self.detect_cones(frame)

            # Print detected cones
            for cone in detected_cones:
                print(f"Detected: {cone[0]} with confidence {cone[1]:.2f}")

            # Display the video stream with detections
            cv2.imshow("Cone Detection", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

        # Restore PosixPath
        pathlib.PosixPath = temp

if __name__ == "__main__":
    weights_path = r'src/cone_detection/cone_detection/best.pt'
    cone_detection = ConeDetectionYOLOv5(weights=weights_path, device='cpu')
    cone_detection.run()
