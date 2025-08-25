from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.model = YOLO('yolov8n.pt')
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10
        )
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model(cv_image)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])  # YOLO 클래스 ID
                    label = self.model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)  # 기본 바운딩 박스 색상 (녹색)

                    # person이면 색상 분석
                    if label == "person":
                        person_region = cv_image[y1:y2, x1:x2]
                        if person_region.size > 0:
                            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)

                            # 파란색 마스크
                            blue_lower = (100, 150, 50)
                            blue_upper = (140, 255, 255)
                            mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

                            # 빨간색 마스크 (두 구간)
                            red_lower1 = (0, 150, 50)
                            red_upper1 = (10, 255, 255)
                            red_lower2 = (170, 150, 50)
                            red_upper2 = (180, 255, 255)
                            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
                            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
                            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                            # 픽셀 비율 계산
                            blue_ratio = cv2.countNonZero(mask_blue) / mask_blue.size
                            red_ratio = cv2.countNonZero(mask_red) / mask_red.size

                            # 팀 판별
                            if blue_ratio > 0.1:
                                team = "Blue Team"
                                color = (255, 0, 0)
                            elif red_ratio > 0.1:
                                team = "Red Team"
                                color = (0, 0, 255)
                            else:
                                team = "Unknown"
                                color = (255, 255, 255)

                            cv2.putText(cv_image, team, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # YOLO 바운딩 박스 그리기
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cv_image, label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Detection with Team Color", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image : {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
