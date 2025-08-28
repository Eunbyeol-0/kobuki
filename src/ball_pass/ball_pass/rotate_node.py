#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class Rotate180(Node):
    def __init__(self):
        super().__init__('rotate_180')
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)

        # 회전 파라미터
        self.angular_speed = -1.0  # rad/s, 음수면 오른쪽(시계 방향)
        self.angle = math.pi       # 여전히 180도
        correction_factor = 1.5    # 50% 더 돌기
        self.duration = abs(self.angle / self.angular_speed) * correction_factor

        # 시작 시간 기록
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz

    def timer_callback(self):
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        twist = Twist()

        if current_time - self.start_time < self.duration:
            twist.angular.z = self.angular_speed
            self.publisher_.publish(twist)
        else:
            # 회전 완료 → 정지 후 노드 종료
            twist.angular.z = 0.0
            self.publisher_.publish(twist)
            self.get_logger().info('180-degree rotation complete')
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Rotate180()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
