from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class YOLOBallFollower(Node):
    def __init__(self):
        super().__init__('yolo_ball_follower')

        # ---- Parameters (기존 유지 + 연속형 전진용 추가) ----
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('target_label', 'sports ball')   # COCO의 공 클래스
        self.declare_parameter('ang_kp', 0.004)                 # 회전 게인 (px → rad/s)
        self.declare_parameter('center_tol', 0.12)              # 중앙 판정 폭(정규화). 연속형에서도 정렬 게인에 사용
        self.declare_parameter('forward_speed', 0.25)           # 전진 기본 스케일(최대치 근사)
        self.declare_parameter('stop_box_frac', 0.45)           # 너무 가까우면 정지 (박스너비/이미지너비)
        self.declare_parameter('min_forward', 0.06)             # 최소 전진속도(정지 떨림 방지)
        self.declare_parameter('ema_alpha', 0.35)               # err 스무딩(0~1, 클수록 민감)

        self.image_topic   = self.get_parameter('image_topic').get_parameter_value().string_value
        self.target_label  = self.get_parameter('target_label').get_parameter_value().string_value
        self.ang_kp        = self.get_parameter('ang_kp').get_parameter_value().double_value
        self.center_tol    = self.get_parameter('center_tol').get_parameter_value().double_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.stop_box_frac = self.get_parameter('stop_box_frac').get_parameter_value().double_value
        self.min_forward   = self.get_parameter('min_forward').get_parameter_value().double_value
        self.ema_alpha     = self.get_parameter('ema_alpha').get_parameter_value().double_value

        # YOLO 모델
        self.model = YOLO('yolov8n.pt')

        # ROS I/O
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # 스무딩 상태
        self._ema_err = 0.0

        self.get_logger().info(f'Following target: "{self.target_label}" on {self.image_topic} (continuous forward mode)')

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]
            cx_img = w / 2.0

            # YOLO 추론
            results = self.model(frame)
            best = None  # 가장 큰 공(가까울 가능성 ↑)을 선택

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    if label != self.target_label:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    area = bw * bh
                    if best is None or area > best['area']:
                        best = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'w': bw, 'h': bh, 'area': area}

            twist = Twist()

            if best is not None:
                # 중심 오차 계산
                target_cx = (best['x1'] + best['x2']) / 2.0
                err_px   = target_cx - cx_img
                err_norm = err_px / w                          # -0.5 ~ 0.5 근사
                box_frac = best['w'] / w                       # 가까울수록 큼

                # --- 회전: P 제어 (부호가 반대로 느껴지면 아래 - 제거) ---
                ang_cmd = - self.ang_kp * err_px

                # --- 전진: 연속형 제어 ---
                # 정렬 게인: 중앙에 가까울수록 1, 벗어날수록 0
                if self.center_tol <= 1e-6:
                    align_gain = 0.0
                else:
                    align_gain = 1.0 - min(1.0, abs(err_norm) / self.center_tol)
                align_gain = max(0.0, align_gain)

                # 거리 게인: 멀면 1, 너무 가까우면 0
                dist_gain = 1.0 - min(1.0, box_frac / max(self.stop_box_frac, 1e-6))

                # 오차 스무딩(EMA)으로 align_gain 안정화
                self._ema_err = self.ema_alpha * err_norm + (1.0 - self.ema_alpha) * self._ema_err
                # 스무딩된 오차로 align_gain 다시 계산 (선택)
                if self.center_tol > 1e-6:
                    align_gain = 1.0 - min(1.0, abs(self._ema_err) / self.center_tol)
                    align_gain = max(0.0, align_gain)

                # 최종 전진속도 (최대 forward_speed에 대해 게인 적용)
                lin_cmd = self.forward_speed * align_gain * dist_gain

                # 최소 전진속도 확보(완전 0으로 떨리는 것 방지) — 단, 너무 가까우면 정지
                if box_frac >= self.stop_box_frac:
                    lin_cmd = 0.0
                else:
                    if lin_cmd > 0.0:
                        lin_cmd = max(self.min_forward, lin_cmd)

                twist.linear.x  = float(lin_cmd)
                twist.angular.z = float(ang_cmd)

                # 디버그 오버레이
                color = (0, 255, 0)
                cv2.rectangle(frame, (best['x1'], best['y1']), (best['x2'], best['y2']), color, 2)
                cv2.line(frame, (int(cx_img), 0), (int(cx_img), h), (255, 255, 255), 1)  # 화면 중앙선
                cv2.putText(
                    frame,
                    f'err={self._ema_err:.3f} box={box_frac:.3f} lin={lin_cmd:.2f} ang={ang_cmd:.2f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            else:
                # 공 미탐지: 천천히 회전 탐색
                twist.angular.z = 0.3
                twist.linear.x = 0.0
                cv2.putText(frame, 'SEARCHING...', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.cmd_pub.publish(twist)

            # 시각화 창
            cv2.imshow('Follow Ball (continuous forward)', frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in image_cb: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLOBallFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
