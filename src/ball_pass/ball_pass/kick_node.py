#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SingleKick(Node):
    def __init__(self):
        super().__init__('single_kick')
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)

        # Twist 메시지 생성 후 퍼블리시
        twist = Twist()
        twist.linear.x = 2.0   # 전진 속도 (m/s)
        twist.angular.z = 0.0  # 회전 속도 (rad/s)

        self.publisher_.publish(twist)
        self.get_logger().info('Kick: 전진 명령 전송 완료')

        # 노드 바로 종료
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SingleKick()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
