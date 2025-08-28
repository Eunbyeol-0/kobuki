#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.duration import Duration
import rclpy.time as rclpy_time

from geometry_msgs.msg import PointStamped, Pose, Quaternion
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def quat_about_y(theta: float) -> Quaternion:
    q = Quaternion()
    half = 0.5 * theta
    q.y = math.sin(half)
    q.w = math.cos(half)
    return q

class ShootGoalNode(LifecycleNode):
    def __init__(self):
        # 노드 이름을 'shoot_goal_node'로 변경하여 FSM 테이블에 맞춥니다.
        super().__init__("shoot_goal_node")

        # 인스턴스 변수들을 초기화합니다.
        self.ball_topic = None
        self.teammate_topic = None
        self.goal_topic = None
        self.target_frame = None
        self.tf_timeout = None
        self.d_kick = None
        self.min_L = None
        self.min_L_for_kick = None
        self.max_msg_age = None
        self.tf_buffer = None
        self.tf_listener = None
        self.ball_msg: Optional[PointStamped] = None
        self.teammate_msg: Optional[PointStamped] = None
        self.ball_sub = None
        self.teammate_sub = None
        self.goal_pub = None
        self.timer = None
        self._last_tf_warn_sec = -1.0

    def on_configure(self, state: LifecycleState):
        """
        'unconfigured'에서 'inactive'로 전환될 때 호출됩니다.
        파라미터 선언, TF 리스너, 퍼블리셔, 서브스크라이버 등 리소스를 생성합니다.
        """
        self.get_logger().info("ShootGoalNode: 구성(on_configure) 상태로 전환 중...")

        self.declare_parameter("ball_topic", "/ball")
        self.declare_parameter("teammate_topic", "/cone")
        self.declare_parameter("goal_topic", "/goal")
        self.declare_parameter("target_frame", "camera_depth_optical_frame")
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("d_kick", 0.50)
        self.declare_parameter("min_L", 1e-3)
        self.declare_parameter("min_L_for_kick", 0.10)
        self.declare_parameter("max_msg_age_sec", 0.5)

        self.ball_topic = self.get_parameter("ball_topic").get_parameter_value().string_value
        self.teammate_topic = self.get_parameter("teammate_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.tf_timeout = Duration(seconds=self.get_parameter("tf_timeout_sec").get_parameter_value().double_value)

        self.d_kick = self.get_parameter("d_kick").get_parameter_value().double_value
        self.min_L = self.get_parameter("min_L").get_parameter_value().double_value
        self.min_L_for_kick = self.get_parameter("min_L_for_kick").get_parameter_value().double_value
        self.max_msg_age = self.get_parameter("max_msg_age_sec").get_parameter_value().double_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self.ball_sub = self.create_subscription(PointStamped, self.ball_topic, self._ball_cb, qos)
        self.teammate_sub = self.create_subscription(PointStamped, self.teammate_topic, self._teammate_cb, qos)
        self.goal_pub = self.create_publisher(Pose, self.goal_topic, 10)
        
        self.get_logger().info("ShootGoalNode: 리소스 초기화 완료.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """
        'inactive'에서 'active'로 전환될 때 호출됩니다.
        노드의 핵심 기능을 시작합니다 (타이머를 활성화하여 주기적인 _step 함수 실행).
        """
        self.get_logger().info("ShootGoalNode: 활성화(on_activate) 상태입니다. 슛 목표 계산을 시작합니다.")
        self.timer = self.create_timer(0.05, self._step)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """
        'active'에서 'inactive'로 전환될 때 호출됩니다.
        핵심 작업을 중지합니다 (타이머를 취소하여 _step 함수 실행 중지).
        """
        self.get_logger().info("ShootGoalNode: 비활성화(on_deactivate) 상태입니다. 슛 목표 계산을 중지합니다.")
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """
        'inactive'에서 'unconfigured'로 전환될 때 호출됩니다.
        `on_configure`에서 생성한 리소스를 해제합니다.
        """
        self.get_logger().info("ShootGoalNode: 정리(on_cleanup) 상태로 전환 중...")
        self.destroy_subscription(self.ball_sub)
        self.destroy_subscription(self.teammate_sub)
        self.destroy_publisher(self.goal_pub)
        
        self.ball_sub = None
        self.teammate_sub = None
        self.goal_pub = None
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState):
        self.get_logger().info("ShootGoalNode: 종료(on_shutdown) 중...")
        return TransitionCallbackReturn.SUCCESS

    # 기존 콜백 함수와 로직은 그대로 유지합니다.
    def _ball_cb(self, msg: PointStamped):
        self.ball_msg = msg

    def _teammate_cb(self, msg: PointStamped):
        self.teammate_msg = msg

    def _now(self) -> rclpy_time.Time:
        return self.get_clock().now()

    def _age_ok(self, msg: PointStamped) -> bool:
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            return True
        age = (self._now() - rclpy_time.Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        return age <= self.max_msg_age

    def _lookup_transform_at(self, src_frame: str, stamp_msg) -> Optional[object]:
        try:
            if stamp_msg.sec == 0 and stamp_msg.nanosec == 0:
                tr = self.tf_buffer.lookup_transform(self.target_frame, src_frame, rclpy_time.Time())
            else:
                tr = self.tf_buffer.lookup_transform(
                    self.target_frame, src_frame, rclpy_time.Time.from_msg(stamp_msg), self.tf_timeout
                )
            return tr
        except TransformException as e:
            now_s = self._now().nanoseconds * 1e-9
            if now_s - self._last_tf_warn_sec > 1.0:
                self.get_logger().warn(f"TF transform failed {src_frame}->{self.target_frame}: {e}")
                self._last_tf_warn_sec = now_s
            return None

    def _to_target_frame(self, point_msg: PointStamped) -> Optional[PointStamped]:
        src_frame = point_msg.header.frame_id or "map"
        if src_frame == self.target_frame:
            return point_msg
        tr = self._lookup_transform_at(src_frame, point_msg.header.stamp)
        if tr is None:
            return None
        out: PointStamped = do_transform_point(point_msg, tr)
        out.header.stamp = point_msg.header.stamp
        out.header.frame_id = self.target_frame
        return out

    def _extract_xz(self, ps: PointStamped):
        x = ps.point.x
        z = ps.point.z
        return x, z

    def _publish_goal_pose_xz(self, xg: float, zg: float, quat_yaw_in_xz: Quaternion):
        goal = Pose()
        goal.position.x = xg
        goal.position.y = 0.0
        goal.position.z = zg
        goal.orientation = quat_yaw_in_xz
        self.goal_pub.publish(goal)

    def _step(self):
        # 노드가 'active' 상태일 때만 이 함수가 실행되도록 타이머를 제어합니다.
        if self.ball_msg is None or self.teammate_msg is None:
            return
        if not (self._age_ok(self.ball_msg) and self._age_ok(self.teammate_msg)):
            return

        ball_b = self._to_target_frame(self.ball_msg)
        mate_b = self._to_target_frame(self.teammate_msg)
        if ball_b is None or mate_b is None:
            return

        bx, bz = self._extract_xz(ball_b)
        px, pz = self._extract_xz(mate_b)

        dx = px - bx
        dz = pz - bz
        L = math.hypot(dx, dz)
        if not math.isfinite(L) or L < self.min_L:
            return

        ux = dx / L
        uz = dz / L

        d_offset = self.d_kick if L >= self.min_L_for_kick else 0.0
        xg = bx - d_offset * ux
        zg = bz - d_offset * uz

        theta = math.atan2(ux, uz)
        quat = quat_about_y(theta)

        self._publish_goal_pose_xz(xg, zg, quat)

# 기존 main 함수는 라이프사이클 노드에 맞춰 수정하거나 런치 파일로 대체해야 합니다.
def main(args=None):
    rclpy.init(args=args)
    shoot_goal_node = ShootGoalNode()
    rclpy.spin(shoot_goal_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()