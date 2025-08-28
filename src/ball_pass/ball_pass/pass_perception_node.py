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
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.pub_dets = self.create_publisher(Detection2DArray, 'object_detection_2d', 10)
        self.ball_pub = self.create_publisher(PointStamped, 'ball', 10)
        self.human_pub = self.create_publisher(PointStamped, 'human', 10)

        
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

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0])  # YOLO 클래스 ID
                    label = self.model.names[cls_id]

                    if label not in ["person", "sports ball"]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = float((x1 + x2) / 2.0)
                    center_y = float((y1 + y2) / 2.0)
                    
                    team = label # 기본값은 객체 레이블
                    color = (0, 255, 0) # 기본 초록색
                    score = float(box.conf)
                    
                    if label == "person":
                        # 팀 판별 (HSV)
                        team = "Unknown"
                        person_region = cv_image[y1:y2, x1:x2]
                        if person_region.size > 0:
                            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)

                            blue_lower = (100, 150, 50)
                            blue_upper = (140, 255, 255)
                            mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

                            red_lower1 = (0, 150, 50)
                            red_upper1 = (10, 255, 255)
                            red_lower2 = (170, 150, 50)
                            red_upper2 = (180, 255, 255)
                            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
                            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
                            mask_red  = cv2.bitwise_or(mask_red1, mask_red2)

                            blue_ratio = cv2.countNonZero(mask_blue) / mask_blue.size
                            red_ratio  = cv2.countNonZero(mask_red)  / mask_red.size

                            if blue_ratio > 0.1 and blue_ratio >= red_ratio:
                                team = "Blue"
                                color = (255, 0, 0)
                            elif red_ratio > 0.1 and red_ratio > blue_ratio:
                                team = "Red"
                                color = (0, 0, 255)
                            else:
                                team = "Unknown"
                                color = (255, 255, 255)

                    elif label == "spdrtsball":
                        color = (0, 255, 255) # 공은 노란색으로 표시
                    
                    # 2D 탐지 메시지 생성
                    det_array.detections.append(
                        self.create_detection_msg(rgb_msg.header, team, score, center_x, center_y)
                    )
                    if label == "person":
                    # 3D 위치 계산 및 발행
                        obj_x, obj_y, obj_z = self.publish_human_position(depth_image, rgb_msg.header, center_x, center_y)
                    elif label == "sportsball":
                        obj_x, obj_y, obj_z = self.publish_ball_position(depth_image, rgb_msg.header, center_x, center_y)
                        
                    # (시각화는 유지해도 되고 지워도 됨)
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                    if obj_x is not None:

                        distance = (obj_x**2 + obj_y**2 + obj_z**2)**0.5
                        
                        # 2. 표시할 텍스트 생성
                        coord_text = f"{team} (x:{obj_x:.2f} y:{obj_y:.2f} z:{obj_z:.2f})"
                        dist_text = f"distance: {distance:.2f}m"
                        
                        cv2.putText(cv_image, coord_text, (x1, y2 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(cv_image, dist_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        cv2.putText(cv_image, label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
        
    def publish_ball_position(self, depth_image, header, cx, cy):
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
            
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.header.frame_id = 'camera_depth_optical_frame' # 좌표계 명시
            
            point_msg.point.x = obj_x
            point_msg.point.y = obj_y
            point_msg.point.z = obj_z
            self.ball_pub.publish(point_msg)
             
            return obj_x, obj_y, obj_z

        except IndexError:
            return None, None, None
        
    def publish_human_position(self, depth_image, header, cx, cy):
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
            
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.header.frame_id = 'camera_depth_optical_frame' # 좌표계 명시
            
            point_msg.point.x = obj_x
            point_msg.point.y = obj_y
            point_msg.point.z = obj_z
            self.human_pub.publish(point_msg)
             
            return obj_x, obj_y, obj_z

        except IndexError:
            return None, None, None
        
def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
