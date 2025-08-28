#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D, Pose2D
from geometry_msgs.msg import PointStamped 
import message_filters
from typing import Optional


class BestNode(LifecycleNode):
    def __init__(self):
        # 노드 이름을 'best_node'로 변경했습니다.
        super().__init__('best_node')
        self.model = None
        self.bridge = None
        self.pub_dets = None
        self.midpoint_pub = None
        
        self.camera_intrinsics = None
        self.camera_info_sub = None
        self.depth_sub = None
        self.image_sub = None
        self.ts = None
        self._last_warn_time = 0

    def on_configure(self, state: LifecycleState):
        """
        'unconfigured'에서 'inactive'로 전환될 때 호출됩니다.
        노드가 사용할 모든 리소스(YOLO 모델, 퍼블리셔, 서브스크라이버)를 초기화합니다.
        """
        self.get_logger().info("BestNode: 구성(on_configure) 상태로 전환 중...")
        
        # ROS 리소스 초기화
        self.model = YOLO('best.pt')
        self.bridge = CvBridge()
        self.pub_dets = self.create_publisher(Detection2DArray, 'object_detection_2d', 10)
        self.midpoint_pub = self.create_publisher(PointStamped, 'cone', 10)

        # 카메라 정보 구독
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # 이미지 동기화 구독
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.image_callback)
        
        self.get_logger().info("BestNode: 리소스 초기화 완료.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """
        'inactive'에서 'active'로 전환될 때 호출됩니다.
        노드가 항상 켜져 있어야 하므로, 이 상태에서 핵심 로직을 활성화합니다.
        """
        self.get_logger().info("BestNode: 활성화(on_activate) 상태입니다. 퍼셉션 로직을 시작합니다.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """
        'active'에서 'inactive'로 전환될 때 호출됩니다.
        퍼셉션 로직을 중지하고 리소스를 일시 정지시킵니다.
        """
        self.get_logger().info("BestNode: 비활성화(on_deactivate) 상태입니다. 퍼셉션 로직을 중지합니다.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """
        'inactive'에서 'unconfigured'로 전환될 때 호출됩니다.
        `on_configure`에서 생성한 모든 리소스를 해제합니다.
        """
        self.get_logger().info("BestNode: 정리(on_cleanup) 상태로 전환 중...")
        self.destroy_subscription(self.camera_info_sub)
        self.ts.unregister()
        self.destroy_subscription(self.depth_sub)
        self.destroy_subscription(self.image_sub)
        self.destroy_publisher(self.pub_dets)
        self.destroy_publisher(self.midpoint_pub)
        
        self.model = None
        self.bridge = None
        self.pub_dets = None
        self.midpoint_pub = None
        self.camera_info_sub = None
        self.depth_sub = None
        self.image_sub = None
        self.ts = None
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState):
        self.get_logger().info("BestNode: 종료(on_shutdown) 중...")
        return TransitionCallbackReturn.SUCCESS

    # 기존 콜백 함수와 로직은 그대로 유지합니다.
    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info("카메라 파라미터 수신")

    def image_callback(self, rgb_msg, depth_msg):
        # 노드가 'active' 상태일 때만 이 함수가 실행되도록 합니다.
        if self.get_current_state().label != 'active':
            return
            
        if self.camera_intrinsics is None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_warn_time > 1.0:
                self.get_logger().warn("카메라 파라미터를 기다리는 중...")
                self. _last_warn_time = current_time
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
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]

                    if label != "cone":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = float((x1 + x2) / 2.0)
                    score = float(box.conf)
                    color = (255, 0, 255)
                    
                    det_array.detections.append(
                        self.create_detection_msg(rgb_msg.header, label, score, center_x, float(y2))
                    )
                    
                    obj_x, obj_y, obj_z = self.calculate_cone_position(depth_image, center_x, float(y2))
                        
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
                midpoint_2d_x = int((cone_positions_data[0]['2d'][0] + cone_positions_data[1]['2d'][0]) / 2)
                midpoint_2d_y = int((cone_positions_data[0]['2d'][1] + cone_positions_data[1]['2d'][1]) / 2)

                cv2.circle(cv_image, (midpoint_2d_x, midpoint_2d_y), 8, (0, 255, 0), -1)
                midpoint_text = f"Midpoint (x:{midpoint_3d[0]:.2f} y:{midpoint_3d[1]:.2f} z:{midpoint_3d[2]:.2f})"
                cv2.putText(cv_image, midpoint_text, (midpoint_2d_x + 10, midpoint_2d_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                self.publish_midpoint(rgb_msg.header, midpoint_3d)

            if det_array.detections:
                self.pub_dets.publish(det_array)

            cv2.imshow("YOLOv8 Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Error processing image : {e}")
            self.get_logger().error(traceback.format_exc())

    def create_detection_msg(self, header, class_id, score, cx, cy):
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
    best_node = BestNode()
    rclpy.spin(best_node)

if __name__ == "__main__":
    main()