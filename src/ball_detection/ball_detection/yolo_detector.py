from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.model = YOLO('yolov8n.pt')
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            result = self.model(cv_image)

            annotated_frame = result[0].plot()
            cv2.imshow("YoloV8 Detection (ROS2)", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)

if __name__ == "__main__":
    main()