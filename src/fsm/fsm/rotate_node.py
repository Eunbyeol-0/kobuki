#!/usr/bin/env python3
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from geometry_msgs.msg import Twist
import math
import time
from std_msgs import Bool

class RotateNode(LifecycleNode):
    def __init__(self):
        # 노드 이름을 FSM 테이블에 맞게 'rotate_node'로 변경했습니다.
        super().__init__('rotate_node')
        self.publisher_ = None
        self.timer = None
        self.start_time = None
        
        # 회전 파라미터는 그대로 유지
        self.angular_speed = -1.0
        self.angle = math.pi
        correction_factor = 1.5
        self.duration = abs(self.angle / self.angular_speed) * correction_factor

    def on_configure(self, state: LifecycleState):
        """
        'unconfigured'에서 'inactive' 상태로 전환될 때 호출됩니다.
        이 단계에서 퍼블리셔를 초기화합니다.
        """
        self.get_logger().info("RotateNode: 구성(on_configure) 상태로 전환 중...")
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)
        self.game_pub = self.create_publisher(Bool, 'game_event', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """
        'inactive'에서 'active' 상태로 전환될 때 호출됩니다.
        이 단계에서 로봇의 '회전' 동작을 시작하는 핵심 로직을 실행합니다.
        """
        self.get_logger().info("RotateNode: 활성화(on_activate) 상태입니다. 회전 동작을 시작합니다.")
        
        # 회전 시작 시간 기록
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # 회전 동작을 위한 타이머 시작
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """
        'active'에서 'inactive' 상태로 전환될 때 호출됩니다.
        활성화된 작업을 중지하고 필요한 정리를 수행합니다.
        """
        self.get_logger().info("RotateNode: 비활성화(on_deactivate) 상태입니다. 회전 동작을 중지합니다.")
        # 회전 중인 로봇을 즉시 정지시킵니다.
        if self.publisher_ is not None:
            stop_twist = Twist()
            stop_twist.angular.z = 0.0
            self.publisher_.publish(stop_twist)
        
        # 타이머를 취소하여 더 이상 회전 명령을 보내지 않게 합니다.
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
            
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """
        'inactive'에서 'unconfigured' 상태로 전환될 때 호출됩니다.
        `on_configure`에서 생성한 리소스를 해제합니다.
        """
        self.get_logger().info("RotateNode: 정리(on_cleanup) 상태로 전환 중...")
        self.destroy_publisher(self.publisher_)
        self.publisher_ = None
        self.start_time = None
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState):
        self.get_logger().info("RotateNode: 종료(on_shutdown) 중...")
        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        """
        주기적으로 호출되어 회전 명령을 보냅니다.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        twist = Twist()

        if (current_time - self.start_time) < self.duration:
            # 회전이 끝나지 않았으면 계속 회전 명령을 보냅니다.
            twist.angular.z = self.angular_speed
            self.publisher_.publish(twist)
        else:
            # 회전 완료 -> FSM 상태 제어 노드에 알려야 합니다.
            # 이 노드 자체는 비활성화 상태로 전환될 준비를 합니다.
            self.get_logger().info('RotateNode: 180도 회전 완료')
            # FSM State Manager가 이 노드를 비활성화하도록 요청합니다.
            # 이 부분은 FSMStateManager의 로직에 따라 구현해야 합니다.
            # (예: 특정 토픽에 메시지 발행)
            self.timer.cancel() # 타이머를 취소하여 재발을 막습니다.
            done = Bool()
            done.msg = True
            game_pub.publish(done)


def main(args=None):
    rclpy.init(args=args)
    rotate_node = RotateNode()
    rclpy.spin(rotate_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()