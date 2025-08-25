from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


def smoothstep(t: float) -> float:
    """0~1 구간에서 3t^2 - 2t^3 (부드러운 S-curve)"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class YOLOBallFollower(Node):
    def __init__(self):
        super().__init__('yolo_ball_follower')

        # ---- Parameters ----
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('target_label', 'sports ball')
        # 회전 P 제어
        self.declare_parameter('ang_kp', 0.004)         # [rad/s per px]
        self.declare_parameter('max_ang', 1.2)          # 회전속도 상한
        self.declare_parameter('ang_deadband', 0.02)    # 너무 작은 각속도는 0으로(떨림 방지)
        self.declare_parameter('ang_slew', 2.0)         # 각속도 가감속 [rad/s^2]
        self.declare_parameter('ang_ema', 0.3)          # 각속도 EMA 필터(0~1)

        # 전진 P 제어(연속형) + 부드러운 정지
        self.declare_parameter('center_tol', 0.12)      # 정렬 허용 폭(정규화)
        self.declare_parameter('forward_speed', 0.25)   # 전진 최고 스케일
        # 정지 히스테리시스: stop_enter >= stop_exit
        self.declare_parameter('stop_enter', 0.50)      # 박스가 이 이상이면 "가까움" 상태 진입
        self.declare_parameter('stop_exit', 0.42)       # 이 미만으로 줄어들면 "가까움" 해제
        self.declare_parameter('min_forward', 0.06)     # 최소 전진 속도(스틱션 극복)
        # 전진 속도 가감속/필터
        self.declare_parameter('lin_slew_up', 0.6)      # 가속 한계 [m/s^2]
        self.declare_parameter('lin_slew_down', 0.8)    # 감속 한계 [m/s^2]
        self.declare_parameter('lin_ema', 0.3)          # 전진속도 EMA 필터(0~1)

        # ---- Get params ----
        p = lambda k: self.get_parameter(k).get_parameter_value()
        self.image_topic   = p('image_topic').string_value
        self.target_label  = p('target_label').string_value

        self.ang_kp        = p('ang_kp').double_value
        self.max_ang       = p('max_ang').double_value
        self.ang_deadband  = p('ang_deadband').double_value
        self.ang_slew      = p('ang_slew').double_value
        self.ang_ema_alpha = p('ang_ema').double_value

        self.center_tol    = p('center_tol').double_value
        self.forward_speed = p('forward_speed').double_value
        self.stop_enter    = p('stop_enter').double_value
        self.stop_exit     = p('stop_exit').double_value
        self.min_forward   = p('min_forward').double_value

        self.lin_slew_up   = p('lin_slew_up').double_value
        self.lin_slew_down = p('lin_slew_down').double_value
        self.lin_ema_alpha = p('lin_ema').double_value

        # ---- YOLO / ROS I-O ----
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # ---- States ----
        self._prev_t = None
        self._lin_cmd_prev = 0.0
        self._ang_cmd_prev = 0.0
        self._lin_cmd_filt = 0.0
        self._ang_cmd_filt = 0.0
        self._ema_err = 0.0
        self._too_close = False  # 히스테리시스 상태

        self.get_logger().info(
            f'Following "{self.target_label}" on {self.image_topic} (smooth P with hysteresis/slew/filter)'
        )

    # 공용: 슬루레이트 제한
    def _slew(self, prev: float, target: float, max_rate: float, dt: float) -> float:
        if dt <= 0.0:
            return target
        max_delta = max_rate * dt
        delta = target - prev
        if delta > max_delta:
            return prev + max_delta
        if delta < -max_delta:
            return prev - max_delta
        return target

    def image_cb(self, msg: Image):
        twist = Twist()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]
            cx_img = w / 2.0

            # 시간
            now = self.get_clock().now().nanoseconds / 1e9
            if self._prev_t is None:
                self._prev_t = now
            dt = max(1e-3, now - self._prev_t)

            # YOLO 추론 (기본 설정 사용)
            results = self.model(frame)
            best = None
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
                                'w': bw, 'h': bh, 'area': area,
                                'conf': float(box.conf[0]) if box.conf is not None else 0.0}

            if best is not None:
                # 오차/거리 지표
                target_cx = (best['x1'] + best['x2']) / 2.0
                err_px   = target_cx - cx_img
                err_norm = err_px / w
                box_frac = best['w'] / w

                # --- (1) 정지 히스테리시스 ---
                if self._too_close:
                    if box_frac < self.stop_exit:
                        self._too_close = False
                else:
                    if box_frac >= self.stop_enter:
                        self._too_close = True

                # --- (2) 회전: P + 슬루 + EMA + 데드밴드 ---
                # ---- 기존 계산: err_px, err_norm, box_frac 까지 끝난 상태라고 가정 ----
                # 회전: 기존 ang_target/ang_slew/EMA/데드밴드 로직 그대로 쓰면 OK
                ang_target = - self.ang_kp * err_px
                ang_target = max(-self.max_ang, min(self.max_ang, ang_target))

                # 선택: 간단히 쓸 경우 바로 적용
                ang_cmd = ang_target

                # ===== 1) 정렬 히스테리시스 게이트 =====
                # 파라미터(원하면 __init__에서 declare_parameter로 뺄 수 있음)
                align_enter = 0.08   # 정규화 오차가 이보다 작아지면 "정렬됨"으로 인정
                align_exit  = 0.12   # 이보다 커지면 "정렬 해제"
                if not hasattr(self, "_aligned"):
                    self._aligned = False

                abs_err = abs(err_norm)
                if self._aligned:
                    if abs_err > align_exit:
                        self._aligned = False
                else:
                    if abs_err < align_enter:
                        self._aligned = True

                # ===== 2) 헤딩오차 기반 강한 감속 =====
                # 오차가 0이면 1, align_exit이면 0에 가깝게; 지수로 더 강하게 깎음
                err_limit = max(1e-6, align_exit)    # 스케일 기준
                err_pow   = 1.5                      # ↑값일수록 오차에 더 민감하게 감속
                align_gain = max(0.0, 1.0 - (abs_err / err_limit) ** err_pow)

                # ===== 3) 각속도 기반 감속/차단 =====
                # 회전이 큰 동안엔 전진을 더 억제
                ang_block   = 0.6      # |ang_cmd|가 이보다 크면 전진 거의 안 함
                ang_softcap = 1.0      # 소프트 감속 상한
                ang_gain = 1.0
                if abs(ang_cmd) >= ang_block:
                    ang_gain = 0.05    # 거의 정지(스틱션 극복용 소량만 남기려면 0.03~0.08)
                else:
                    # 0~ang_block 구간에서 선형으로 1→0.3 감속 (원하면 smoothstep으로 바꿔도 됨)
                    ang_gain = 0.3 + 0.7 * (1.0 - min(1.0, abs(ang_cmd) / ang_softcap))

                # ===== 거리 게인(너무 가까우면 정지) =====
                if self.stop_enter > self.stop_exit + 1e-6:
                    t = (self.stop_enter - box_frac) / (self.stop_enter - self.stop_exit)
                else:
                    t = 0.0
                # 부드러운 감속(S-curve)
                def smoothstep(t): 
                    t = max(0.0, min(1.0, t)); return t*t*(3 - 2*t)
                dist_gain = smoothstep(max(0.0, min(1.0, t)))
                too_close = getattr(self, "_too_close", False)
                if too_close:
                    if box_frac < self.stop_exit: self._too_close = False
                else:
                    if box_frac >= self.stop_enter: self._too_close = True
                too_close = self._too_close

                # ===== 최종 전진 속도 =====
                if too_close:
                    lin_cmd = 0.0
                else:
                    # (a) 정렬 게이트를 통과한 상태가 아니면 더 강하게 감속
                    gate_gain = 1.0 if self._aligned else 0.2   # 정렬 전엔 20%만 허용
                    base = self.forward_speed
                    lin_cmd = base * align_gain * ang_gain * dist_gain * gate_gain
                    # 최소 전진 속도는 과감히 낮춤(가다서다 방지), 정렬되기 전엔 적용 안 함
                    if self._aligned and lin_cmd > 0.0:
                        lin_cmd = max(self.min_forward, lin_cmd)

                # 적용
                twist.angular.z = float(ang_cmd)
                twist.linear.x  = float(lin_cmd)
                
                

                # 디버그 오버레이
                cv2.rectangle(frame, (best['x1'], best['y1']), (best['x2'], best['y2']), (0, 255, 0), 2)
                cv2.line(frame, (int(cx_img), 0), (int(cx_img), h), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"conf={best['conf']:.2f} err={err_norm:.3f} box={box_frac:.3f} "
                    f"lin={self._lin_cmd_filt:.2f} ang={self._ang_cmd_filt:.2f} close={self._too_close}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
            else:
                # 미탐지: 천천히 회전 탐색
                twist.angular.z = 0.3
                twist.linear.x  = 0.0
                cv2.putText(frame, "SEARCHING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Publish
            self.cmd_pub.publish(twist)

            # 상태 업데이트
            self._prev_t = now
            self._lin_cmd_prev = twist.linear.x
            self._ang_cmd_prev = twist.angular.z

            # 시각화
            cv2.imshow('Follow Ball (smooth P control)', frame)
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
