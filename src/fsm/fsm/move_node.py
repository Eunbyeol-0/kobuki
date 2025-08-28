#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool

# 기존 헬퍼 함수들은 그대로 유지합니다.
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
    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch_cam = math.asin(sinp)
    yaw_base = -pitch_cam
    yaw_base += math.radians(yaw_off_deg)
    while yaw_base > math.pi:
        yaw_base -= 2.0*math.pi
    while yaw_base < -math.pi:
        yaw_base += 2.0*math.pi
    return yaw_base


class MoveNode(LifecycleNode):
    def __init__(self):
        # 노드 이름을 'drive_node' 또는 'move_node'로 변경하여 FSM 테이블에 맞춥니다.
        super().__init__("drive_node")

        # 인스턴스 변수를 초기화합니다.
        self.camera_point_sub = None
        self.cmd_pub = None
        self.timer = None

        self.k_lin = 0.5
        self.k_ang = 1.5
        self.v_max = 0.4
        self.w_max = 1.5

        self.angle_deadband = 0.15
        self.angle_slowdown = 0.7
        self.pos_tol = 0.05
        self.yaw_tol = 0.03
        self.timeout_sec = 0.5
        self.reverse_ok = False

        self.last_point = None
        self.last_yaw = None
        self.last_stamp = time.time()

    def on_configure(self, state: LifecycleState):
        """
        'unconfigured'에서 'inactive'로 전환될 때 호출됩니다.
        퍼블리셔, 서브스크라이버, 타이머 등 리소스를 생성합니다.
        """
        self.get_logger().info("DriveNode: 구성(on_configure) 상태로 전환 중...")
        
        self.camera_point_sub = self.create_subscription(Pose, '/goal', self.point_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        self.game_pub = self.create_publisher(Bool, 'game_event', 10)
        
        self.timer = self.create_timer(0.02, self.control_step)
        
        self.get_logger().info("DriveNode: 리소스 초기화 완료.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """
        'inactive'에서 'active'로 전환될 때 호출됩니다.
        노드의 주요 기능을 시작합니다 (제어 로직 활성화).
        """
        self.get_logger().info("DriveNode: 활성화(on_activate) 상태입니다. 구동 제어를 시작합니다.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """
        'active'에서 'inactive'로 전환될 때 호출됩니다.
        구동 동작을 중지하고, 필요 시 로봇을 정지시킵니다.
        """
        self.get_logger().info("DriveNode: 비활성화(on_deactivate) 상태입니다. 구동을 중지합니다.")
        # 비활성화 시 로봇을 정지시키기 위해 Twist 메시지를 발행합니다.
        if self.cmd_pub:
            self.cmd_pub.publish(Twist())
        
        # 만약 control_step()이 계속 실행된다면 타이머를 취소해야 하지만,
        # 라이프사이클 노드 구조에서는 on_activate/deactivate에 맞춰
        # 제어 루프를 시작/정지하는 것이 더 일반적입니다.
        # 이 예제에서는 타이머가 계속 실행되지만, on_deactivate에서
        # 발행하는 정지 명령이 우선시되므로 안전합니다.
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """
        'inactive'에서 'unconfigured'로 전환될 때 호출됩니다.
        `on_configure`에서 생성한 리소스를 해제합니다.
        """
        self.get_logger().info("DriveNode: 정리(on_cleanup) 상태로 전환 중...")
        self.destroy_subscription(self.camera_point_sub)
        self.destroy_publisher(self.cmd_pub)
        self.destroy_timer(self.timer)
        self.camera_point_sub = None
        self.cmd_pub = None
        self.timer = None
        return TransitionCallbackReturn.SUCCESS

    # 기존 콜백 함수와 제어 로직은 그대로 유지합니다.
    def point_cb(self, msg: Pose):
        self.last_point = (msg.position.x, msg.position.y, msg.position.z)
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.last_yaw = cam_quat_to_base_yaw(qx, qy, qz, qw)
        self.last_stamp = time.time()

    def control_step(self):
        # 노드가 'active' 상태가 아닐 때는 제어 로직을 실행하지 않습니다.
        if self.get_current_state().label != 'active':
            return
            
        now = time.time()
        twist = Twist()

        # ... (기존 제어 로직 코드)
        if self.last_point is None or (now - self.last_stamp) > self.timeout_sec:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
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

        elif abs(yaw_error) > self.yaw_tol:
            twist.linear.x = 0.0
            twist.angular.z = clamp(self.k_ang * yaw_error, -self.w_max, self.w_max)
            self.cmd_pub.publish(twist)
            return
        
        else:
            self.cmd_pub.publish(twist)
            done = Bool()
            done.msg = True
            game_event.publish(done)
            return

# 기존 main 함수는 라이프사이클 노드에 맞춰 수정하거나, 런치 파일로 대체해야 합니다.
def main(args=None):
    rclpy.init(args=args)
    
    # 런치 파일에서 노드를 실행할 때 이 부분이 자동으로 처리됩니다.
    drive_node = MoveNode()
    rclpy.spin(drive_node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()