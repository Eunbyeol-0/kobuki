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

        self.k_lin = 0.5 # m/s per meter of z 
        self.k_ang = 1.5 # rad/s per rad of yaw_error
        self.v_max = 0.4 # m/s
        self.w_max = 1.5 # rad/s

        self.angle_deadband = 0.15 # rad : 이보다 크면 전진 0으로 (회전 우선)
        self.angle_slowdown = 0.7 # rad : 각도 크면 전진 속도 스케일 다운
        self.pos_tol = 0.05 # m : sqrt(x^2 + z^2) < pos_tol이면 정지
        self.yaw_tol = 0.03 # rad이 이정도 내에 들어오면 정지
        self.timeout_sec = 0.5 # 최근 포인트 수신 없으면 정지
        self.reverse_ok = False # z < 0 일 때 후진 허용 여부

        self.last_point = None
        self.last_yaw = None

        self.last_stamp = time.time()

    def point_cb(self, msg: Pose):
        self.last_point = (msg.position.x, msg.position.y, msg.position.z)
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.last_yaw = cam_quat_to_base_yaw(qx, qy, qz, qw)

        self.last_stamp = time.time()

    def control_step(self):
        now = time.time()
        twist = Twist()

        # last_point가 없거나 타임아웃 시 정지 -> last_yaw는 필요없나?
        if self.last_point is None or (now - self.last_stamp) > self.timeout_sec:
            twist.linear.x = 0.0
            twist.angular.z = 0.25   # 원하는 회전 속도로 조정 (rad/s)
            self.cmd_pub.publish(twist)
            return

        x, _, z = self.last_point # 카메라 좌표계 기준 x, z값 

        x_b, y_b = cam_to_base_link(x, z) # 카메라 좌표계 -> base_link 좌표계
        self.get_logger().info(f'x: {x_b} , y: {y_b}')

        dist = math.hypot(x_b, y_b)
        yaw_error = self.last_yaw if self.last_yaw is not None else 0.0 # 쿼터니언이 어떤 기준으로 publish 되는지 확인 필요
        yaw_error = normalize_angle(yaw_error)

        if dist > self.pos_tol:
            # 목표 지점을 위한 yaw 계산
            yaw_for_position = math.atan2(y_b, x_b)
            self.get_logger().info(f'yaw_for_position : {yaw_for_position}')
            
            #회전 속도 계산
            w = clamp(self.k_ang * yaw_for_position, -self.w_max, self.w_max)

            # 목표 지점으로 위한 yaw가 크면 우선 회전 먼저?
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

        elif abs(yaw_error) > self.yaw_tol:
            # 회전 진행
            twist.linear.x = 0.0
            twist.angular.z = clamp(self.k_ang * yaw_error, -self.w_max, self.w_max)
            self.cmd_pub.publish(twist)
            return

        else:
            self.cmd_pub.publish(twist) # 정지
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


