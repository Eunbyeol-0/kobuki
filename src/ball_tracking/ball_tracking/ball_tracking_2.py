from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import torch

# 0~1 구간 S-curve: 3t^2 - 2t^3 (부드러운 감속/가속 프로파일)
def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

class YOLOBallFollower(Node):
    def __init__(self):
        super().__init__('yolo_ball_follower')

        # ---- Parameters ----
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('cmd_topic', '/commands/velocity')   # 기본 퍼블리시 토픽
        self.declare_parameter('target_label', 'sports ball')

        # 정렬 히스테리시스 & 정렬 전 전진 허용 게이트 (완화용)
        self.declare_parameter('align_enter', 0.10)   # "정렬됨" 인정폭 (기본값 완화)
        self.declare_parameter('align_exit',  0.15)   # "정렬 해제" 폭 (기본값 완화)
        self.declare_parameter('unaligned_gate', 0.6) # 정렬 전 전진 스케일(0.2 → 0.6로 완화)

        # 회전(헤딩) P제어 관련
        self.declare_parameter('ang_kp', 0.004)      # 픽셀 오차→각속도 비례계수
        self.declare_parameter('max_ang', 1.2)       # 회전 속도 상한(rad/s)
        self.declare_parameter('ang_deadband', 0.02) # 너무 작은 회전은 0 (떨림 방지)
        self.declare_parameter('ang_slew', 2.0)      # 회전 가/감속 상한(rad/s^2)
        self.declare_parameter('ang_ema', 0.3)       # 회전 명령 EMA 계수(0~1)

        # 전진 스케일 + 부드러운 정지(거리 기반 감속/히스테리시스)
        # self.declare_parameter('center_tol', 0.12)  # 더 이상 사용하지 않음
        self.declare_parameter('forward_speed', 0.25)
        self.declare_parameter('stop_enter', 0.50)
        self.declare_parameter('stop_exit', 0.42)
        self.declare_parameter('min_forward', 0.06)
        self.declare_parameter('lin_slew_up', 0.6)
        self.declare_parameter('lin_slew_down', 0.8)
        self.declare_parameter('lin_ema', 0.3)

        # YOLO 추론 관련
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('yolo_conf_main', 0.25)
        self.declare_parameter('yolo_conf_scout', 0.15)
        self.declare_parameter('yolo_iou', 0.5)
        self.declare_parameter('yolo_imgsz_main', 960)
        self.declare_parameter('yolo_imgsz_scout', 1280)
        self.declare_parameter('yolo_use_tta', False)

        # ---- Get parameters ----
        p = lambda k: self.get_parameter(k).get_parameter_value()
        self.image_topic   = p('image_topic').string_value
        self.cmd_topic     = p('cmd_topic').string_value
        self.target_label  = p('target_label').string_value

        self.ang_kp        = p('ang_kp').double_value
        self.max_ang       = p('max_ang').double_value
        self.ang_deadband  = p('ang_deadband').double_value
        self.ang_slew      = p('ang_slew').double_value
        self.ang_ema_alpha = p('ang_ema').double_value

        # center_tol 제거(선언 안 했으므로 읽지 않음)
        self.forward_speed = p('forward_speed').double_value
        self.stop_enter    = p('stop_enter').double_value
        self.stop_exit     = p('stop_exit').double_value
        self.min_forward   = p('min_forward').double_value
        self.lin_slew_up   = p('lin_slew_up').double_value
        self.lin_slew_down = p('lin_slew_down').double_value
        self.lin_ema_alpha = p('lin_ema').double_value

        self.yolo_model_path = p('yolo_model').string_value
        self.yolo_conf_main  = p('yolo_conf_main').double_value
        self.yolo_conf_scout = p('yolo_conf_scout').double_value
        self.yolo_iou        = p('yolo_iou').double_value
        self.yolo_imgsz_main = int(p('yolo_imgsz_main').integer_value)
        self.yolo_imgsz_scout= int(p('yolo_imgsz_scout').integer_value)
        self.yolo_use_tta    = p('yolo_use_tta').bool_value

        self.align_enter     = p('align_enter').double_value
        self.align_exit      = p('align_exit').double_value
        self.unaligned_gate  = p('unaligned_gate').double_value

        # ---- YOLO / ROS I-O ----
        self.model = YOLO(self.yolo_model_path)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # ---- States ----
        self._prev_t = None
        self._lin_cmd_prev = 0.0
        self._ang_cmd_prev = 0.0
        self._lin_cmd_filt = 0.0
        self._ang_cmd_filt = 0.0
        self._too_close = False

        # 클래스 ID 매핑
        try:
            names = self.model.names
            if isinstance(names, dict):
                inv = {name: idx for idx, name in names.items()}
            else:
                inv = {name: idx for idx, name in enumerate(names)}
            self._ball_class_id = inv.get(self.target_label, 32)
        except Exception:
            self._ball_class_id = 32

        self.get_logger().info(
            f'Following "{self.target_label}" on {self.image_topic} -> {self.cmd_topic} '
            f'(imgsz_main={self.yolo_imgsz_main}, imgsz_scout={self.yolo_imgsz_scout}, '
            f'conf_main={self.yolo_conf_main}, conf_scout={self.yolo_conf_scout}, '
            f'align_enter={self.align_enter}, align_exit={self.align_exit}, '
            f'unaligned_gate={self.unaligned_gate})'
        )

    # 슬루레이트 제한
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

    # YOLO 추론 실행
    def _run_yolo(self, frame, conf, imgsz, use_tta, classes):
        device = 0 if torch.cuda.is_available() else None
        return self.model.predict(
            source=frame, imgsz=imgsz, conf=conf, iou=self.yolo_iou,
            classes=classes, verbose=False, augment=use_tta, max_det=50, device=device
        )

    # 공 후보 중 최상 선택
    def _pick_best(self, results, w):
        best = None
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != self._ball_class_id:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = max(1, x2 - x1)
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                area_norm = min(1.0, bw / max(1.0, w))     # 폭 기준 근사 정규화
                score = conf * (0.6 + 0.4 * math.sqrt(area_norm))
                cand = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'w': bw, 'conf': conf, 'score': score}
                if best is None or cand['score'] > best['score']:
                    best = cand
        return best

    def image_cb(self, msg: Image):
        twist = Twist()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]
            cx_img = w / 2.0

            now = self.get_clock().now().nanoseconds / 1e9
            if self._prev_t is None:
                self._prev_t = now
            dt = max(1e-3, now - self._prev_t)

            # 1차 추론
            results_main = self._run_yolo(
                frame=frame, conf=self.yolo_conf_main, imgsz=self.yolo_imgsz_main,
                use_tta=self.yolo_use_tta, classes=[self._ball_class_id]
            )
            best = self._pick_best(results_main, w)

            # 미탐 시 스카우트
            scout_used = False
            if best is None:
                scout_used = True
                results_scout = self._run_yolo(
                    frame=frame, conf=self.yolo_conf_scout, imgsz=self.yolo_imgsz_scout,
                    use_tta=self.yolo_use_tta, classes=[self._ball_class_id]
                )
                best = self._pick_best(results_scout, w)

            if best is not None:
                # 지표 계산
                target_cx = (best['x1'] + best['x2']) / 2.0
                err_px   = target_cx - cx_img
                err_norm = err_px / w
                box_frac = best['w'] / w

                # 근접 히스테리시스
                if self._too_close:
                    if box_frac < self.stop_exit:
                        self._too_close = False
                else:
                    if box_frac >= self.stop_enter:
                        self._too_close = True

                # 회전 P + 포화
                ang_cmd = - self.ang_kp * err_px
                ang_cmd = max(-self.max_ang, min(self.max_ang, ang_cmd))

                # 정렬 히스테리시스(파라미터 사용!)
                if not hasattr(self, "_aligned"):
                    self._aligned = False
                abs_err = abs(err_norm)
                if self._aligned:
                    if abs_err > self.align_exit:
                        self._aligned = False
                else:
                    if abs_err < self.align_enter:
                        self._aligned = True

                # 헤딩오차 감속
                err_limit = max(1e-6, self.align_exit)
                err_pow   = 1.5
                align_gain = max(0.0, 1.0 - (abs_err / err_limit) ** err_pow)

                # 각속도 기반 감속/차단
                ang_block   = 0.6
                ang_softcap = 1.0
                if abs(ang_cmd) >= ang_block:
                    ang_gain = 0.05
                else:
                    ang_gain = 0.3 + 0.7 * (1.0 - min(1.0, abs(ang_cmd) / ang_softcap))

                # 거리 게인: 가까우면 감속
                if self.stop_enter > self.stop_exit + 1e-6:
                    t = (self.stop_enter - box_frac) / (self.stop_enter - self.stop_exit)
                else:
                    t = 0.0
                dist_gain = smoothstep(max(0.0, min(1.0, t)))

                # 전진 속도 합성 (정렬 전엔 unaligned_gate 적용)
                if self._too_close:
                    lin_cmd = 0.0
                else:
                    gate_gain = 1.0 if self._aligned else self.unaligned_gate
                    base = self.forward_speed
                    lin_cmd = base * align_gain * ang_gain * dist_gain * gate_gain
                    if self._aligned and lin_cmd > 0.0:
                        lin_cmd = max(self.min_forward, lin_cmd)

                # 퍼블리시 전: 데드밴드 → 슬루 → EMA
                if abs(ang_cmd) < self.ang_deadband:
                    ang_cmd = 0.0
                ang_cmd_slewed = self._slew(self._ang_cmd_prev, ang_cmd, self.ang_slew, dt)
                lin_rate = self.lin_slew_up if (lin_cmd > self._lin_cmd_prev) else self.lin_slew_down
                lin_cmd_slewed = self._slew(self._lin_cmd_prev, lin_cmd, lin_rate, dt)

                a_ang = max(0.0, min(1.0, self.ang_ema_alpha))
                a_lin = max(0.0, min(1.0, self.lin_ema_alpha))
                self._ang_cmd_filt = (1 - a_ang) * self._ang_cmd_filt + a_ang * ang_cmd_slewed
                self._lin_cmd_filt = (1 - a_lin) * self._lin_cmd_filt + a_lin * lin_cmd_slewed

                twist.angular.z = float(self._ang_cmd_filt)
                twist.linear.x  = float(self._lin_cmd_filt)

                # 디버그 오버레이
                color = (0, 165, 255) if scout_used else (0, 255, 0)
                cv2.rectangle(frame, (best['x1'], best['y1']), (best['x2'], best['y2']), color, 2)
                cv2.line(frame, (int(cx_img), 0), (int(cx_img), h), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"conf={best['conf']:.2f} score={best['score']:.3f} err={err_norm:.3f} "
                    f"box={box_frac:.3f} lin={self._lin_cmd_filt:.2f} ang={self._ang_cmd_filt:.2f} "
                    f"close={self._too_close} scout={scout_used}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                )
            else:
                # 탐지 실패 시: 느린 좌회전 서치(필터 적용)
                raw_ang = 0.3
                raw_lin = 0.0

                if abs(raw_ang) < self.ang_deadband:
                    raw_ang = 0.0
                raw_ang = self._slew(self._ang_cmd_prev, raw_ang, self.ang_slew, dt)
                raw_lin = self._slew(self._lin_cmd_prev, raw_lin, self.lin_slew_down, dt)

                a_ang = max(0.0, min(1.0, self.ang_ema_alpha))
                a_lin = max(0.0, min(1.0, self.lin_ema_alpha))
                self._ang_cmd_filt = (1 - a_ang) * self._ang_cmd_filt + a_ang * raw_ang
                self._lin_cmd_filt = (1 - a_lin) * self._lin_cmd_filt + a_lin * raw_lin

                twist.angular.z = float(self._ang_cmd_filt)
                twist.linear.x  = float(self._lin_cmd_filt)

                cv2.putText(frame, "SEARCHING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Publish & state update
            self.cmd_pub.publish(twist)
            self._prev_t = now
            self._lin_cmd_prev = float(twist.linear.x)
            self._ang_cmd_prev = float(twist.angular.z)

            cv2.imshow('Follow Ball (with filtering)', frame)
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
