import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def cam_to_base_link(x_cam, z_cam, tx=0.0, ty=0.0, yaw_off_deg=0.0):
    u, v = z_cam, -x_cam
    th = math.radians(yaw_off_deg)
    ct, st = math.cos(th), math.sin(th)
    x_b = ct*u - st*v + tx
    y_b = st*u + ct*v + ty
    return x_b, y_b

def cam_quat_to_base_yaw(qx, qy, qz, qw, yaw_off_deg=0.0):
    """
    camera optical frame의 쿼터니언을 base_link의 yaw(Z축)로 변환.
    yaw_base = - pitch_cam  (축 대응: +Y_cam == -Z_base)
    yaw_off_deg: 카메라가 base_z에 대해 추가로 돌아붙어 있으면 보정값(도)
    """
    # pitch_cam (Y축) 추출: ZYX 기준
    sinp = 2.0 * (qw*qy - qz*qx)
    # 수치 안전
    sinp = max(-1.0, min(1.0, sinp))
    pitch_cam = math.asin(sinp)            # 라디안

    yaw_base = -pitch_cam                   # 핵심!
    yaw_base += math.radians(yaw_off_deg)   # 필요시 오프셋
    # -pi..pi
    while yaw_base > math.pi:
        yaw_base -= 2.0*math.pi
    while yaw_base < -math.pi:
        yaw_base += 2.0*math.pi
    return yaw_base


class CameraPointServo(Node):
    def __init__(self):
        super().__init__("camera_point_servo")

        self.camera_point_sub = self.create_subscription(Pose, '/goal', self.point_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        self.timer = self.create_timer(0.02, self.control_step)

        self.k_lin = 0.5
        self.k_ang = 1.5
        self.v_max = 0.4
        self.w_max = 1.5

        self.angle_deadband = 0.15
        self.angle_slowdown = 0.7
        self.pos_tol = 0.3
        self.yaw_tol = 0.03
        self.timeout_sec = 0.5
        self.reverse_ok = False

        self.last_point = None
        self.last_yaw = None
        self.last_stamp = time.time()

        # ▶ 1초 전진 모드 종료 시각 (None이면 평상시)
        self.boost_end_time = None

    def point_cb(self, msg: Pose):
            self.last_point = (msg.position.x, msg.position.y, msg.position.z)
            qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
            self.last_yaw = cam_quat_to_base_yaw(qx, qy, qz, qw)

            self.last_stamp = time.time()

    def control_step(self):
        now = time.time()
        twist = Twist()

        # ▶ 부스트 모드면 1초 동안 linear.x=0.5로 계속 퍼블리시
        if self.boost_end_time is not None:
            if now < self.boost_end_time:
                twist.linear.x = 0.5
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return
            else:
                # 1초 종료 → 정지 후 부스트 종료
                self.cmd_pub.publish(Twist())
                self.boost_end_time = None
                self.get_logger().info("정지 완료(부스트 종료)")
                return

        # 타임아웃: 회전 유지(원래 로직)
        if self.last_point is None or (now - self.last_stamp) > self.timeout_sec:
            twist.linear.x = 0.0
            twist.angular.z = 0.25
            self.cmd_pub.publish(twist)
            return

        x, _, z = self.last_point
        x_b, y_b = cam_to_base_link(x, z)
        self.get_logger().info(f'x: {x_b} , y: {y_b}')

        dist = math.hypot(x_b, y_b)
        yaw_error = self.last_yaw if self.last_yaw is not None else 0.0
        yaw_error = normalize_angle(yaw_error)

        if dist > self.pos_tol:
            yaw_for_position = math.atan2(y_b, x_b)
            self.get_logger().info(f'yaw_for_position : {yaw_for_position}')

            w = clamp(self.k_ang * yaw_for_position, -self.w_max, self.w_max)

            if abs(yaw_for_position) > self.angle_deadband:
                v = 0.0
            else:
                v_raw = self.k_lin * x_b
                scale = max(0.0, 1.0 - abs(yaw_for_position) / self.angle_slowdown)
                v = clamp(v_raw * scale, -self.v_max, self.v_max)
                if not self.reverse_ok and v < 0.0:
                    v = 0.0

            twist.linear.x = v
            twist.angular.z = w
            self.cmd_pub.publish(twist)
            return
        else:
            # ▶ 도착: 1초 전진 모드 시작
            self.get_logger().info("도착: 1초간 앞으로 이동 시작")
            self.boost_end_time = now + 2.5
            # 즉시 한 번 퍼블리시해서 반응 빠르게
            twist.linear.x = 0.5
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return


def main():
    rclpy.init()
    node = CameraPointServo()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()


