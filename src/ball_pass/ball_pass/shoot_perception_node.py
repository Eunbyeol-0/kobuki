from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo # CameraInfo 받아오기
from cv_bridge import CvBridge
import cv2
import numpy as np
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D, Pose2D
from geometry_msgs.msg import PointStamped 
import message_filters  # 메세지 동기화


class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.model = YOLO('best.pt')
        self.bridge = CvBridge()
        self.pub_dets = self.create_publisher(Detection2DArray, 'object_detection_2d', 10)
        self.midpoint_pub = self.create_publisher(PointStamped, 'cone', 10)

        # 카메라 내부 파라미터 (fx fy cx cy ) 저장 변수
        self.camera_intrinsics = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Depth image, rgb image 동기화
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,  # 0.1초 이내의 타임스탬프 차이를 가진 메시지들을 동기화
        )
        self.ts.registerCallback(self.image_callback)

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info("카메라 파라미터 수신")

    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_intrinsics is None:
            self.get_logger().warn("카메라 파라미터를 기다리는 중...")
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            results = self.model(cv_image)

            det_array = Detection2DArray()
            det_array.header = rgb_msg.header


            cone_positions_data = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])  # YOLO 클래스 ID
                    label = self.model.names[cls_id]

                    if label != "cone":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = float((x1 + x2) / 2.0)
                    score = float(box.conf)
                    color = (255, 0, 255)
                    
                    # 2D 탐지 메시지 생성
                    det_array.detections.append(
                        self.create_detection_msg(rgb_msg.header, label, score, center_x, float(y2))
                    )
                    
                    obj_x, obj_y, obj_z = self.calculate_cone_position(depth_image, center_x, float(y2))
                        
                    # (시각화는 유지해도 되고 지워도 됨)
                    
                    if obj_x is not None:
                        cone_positions_data.append({'3d': np.array([obj_x, obj_y, obj_z]), '2d': (center_x, float(y2))})
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                        distance = (obj_x**2 + obj_y**2 + obj_z**2)**0.5
                        
                        coord_text = f"cone (x:{obj_x:.2f} y:{obj_y:.2f} z:{obj_z:.2f})"
                        dist_text = f"dist: {distance:.2f}m"
                        cv2.putText(cv_image, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(cv_image, dist_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if len(cone_positions_data) == 2:
                midpoint_3d = (cone_positions_data[0]['3d'] + cone_positions_data[1]['3d']) / 2.0
                
                # 중점의 2D 픽셀 위치는 시각화를 위해 두 cone의 2D 픽셀 좌표의 중점을 사용
                midpoint_2d_x = int((cone_positions_data[0]['2d'][0] + cone_positions_data[1]['2d'][0]) / 2)
                midpoint_2d_y = int((cone_positions_data[0]['2d'][1] + cone_positions_data[1]['2d'][1]) / 2)

                # 중점 시각화 (초록색 원과 텍스트)
                cv2.circle(cv_image, (midpoint_2d_x, midpoint_2d_y), 8, (0, 255, 0), -1)
                midpoint_text = f"Midpoint (x:{midpoint_3d[0]:.2f} y:{midpoint_3d[1]:.2f} z:{midpoint_3d[2]:.2f})"
                cv2.putText(cv_image, midpoint_text, (midpoint_2d_x + 10, midpoint_2d_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 3D 중점 메시지 발행
                self.publish_midpoint(rgb_msg.header, midpoint_3d)

            if det_array.detections:
                self.pub_dets.publish(det_array)

            cv2.imshow("YOLOv8 Detection with Team & Center", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Error processing image : {e}")
            self.get_logger().error(traceback.format_exc())

    def create_detection_msg(self, header, class_id, score, cx, cy):
        # person과 ball 모두 사용할 수 있음
        det = Detection2D()
        det.header = header
        ohp = ObjectHypothesisWithPose()
        ohp.hypothesis.class_id = class_id
        ohp.hypothesis.score = score
        det.results.append(ohp)
        bbox = BoundingBox2D()
        bbox.center.position.x = cx
        bbox.center.position.y = cy
        det.bbox = bbox
        return det
        
    def calculate_cone_position(self, depth_image, cx, cy):
        cx, cy = int(cx), int(cy)
        
        try:
            distance_mm = depth_image[cy, cx]
            if distance_mm == 0: 
                return None, None, None

            distance_m = distance_mm / 1000.0
            
            fx = self.camera_intrinsics.k[0]
            fy = self.camera_intrinsics.k[4]
            cam_cx = self.camera_intrinsics.k[2]
            cam_cy = self.camera_intrinsics.k[5]

            obj_x = (cx - cam_cx) * distance_m / fx
            obj_y = (cy - cam_cy) * distance_m / fy
            obj_z = distance_m
             
            return obj_x, obj_y, obj_z

        except IndexError:
            return None, None, None
        
    def publish_midpoint(self, header, midpoint):
        point_msg = PointStamped()
        point_msg.header = header
        point_msg.header.frame_id = 'camera_depth_optical_frame'
        
        point_msg.point.x = midpoint[0]
        point_msg.point.y = midpoint[1]
        point_msg.point.z = midpoint[2]
        self.midpoint_pub.publish(point_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
