#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
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


class ShootGoalPublisher(Node):  
    def __init__(self):
        super().__init__("shoot_goal_publisher")  

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

        self.ball_msg: Optional[PointStamped] = None
        self.teammate_msg: Optional[PointStamped] = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # PointStamped로 변경
        self.create_subscription(PointStamped, self.ball_topic, self._ball_cb, qos)
        self.create_subscription(PointStamped, self.teammate_topic, self._teammate_cb, qos)

        self.goal_pub = self.create_publisher(Pose, self.goal_topic, 10)

        self.timer = self.create_timer(0.05, self._step)

        self.get_logger().info(
            f"ShootGoalPublisher started. ball={self.ball_topic}, teammate={self.teammate_topic}, "
            f"goal(Pose)={self.goal_topic}, target_frame={self.target_frame}"
        )

        self._last_tf_warn_sec = -1.0

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

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = ShootGoalPublisher()  # 클래스명 변경
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
