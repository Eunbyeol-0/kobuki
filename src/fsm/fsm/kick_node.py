#!/usr/bin/env python3
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class KickNode(LifecycleNode):
    def __init__(self):
        # 노드 이름은 FSM 테이블에 맞게 'kick_node'로 변경했습니다.
        super().__init__('kick_node')
        self.publisher_ = None
        self.kick_command_sent = False

    def on_configure(self, state: LifecycleState):
        """
        노드가 'unconfigured'에서 'inactive' 상태로 전환될 때 호출됩니다.
        이 단계에서 퍼블리셔, 서브스크라이버 등을 초기화합니다.
        """
        self.get_logger().info("KickNode: 구성(on_configure) 상태로 전환 중...")
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)
        self.game_pub = self.create_publisher(Bool, '/demo_event', 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """
        노드가 'inactive'에서 'active' 상태로 전환될 때 호출됩니다.
        이 단계에서 로봇의 '킥' 동작을 수행하는 핵심 로직을 실행합니다.
        """
        self.get_logger().info("KickNode: 활성화(on_activate) 상태입니다. 킥 명령을 전송합니다.")
        
        # Twist 메시지 생성
        twist = Twist()
        twist.linear.x = 2.0   # 전진 속도 (m/s)
        twist.angular.z = 0.0  # 회전 속도 (rad/s)

        # 킥 명령 전송
        self.publisher_.publish(twist)
        self.get_logger().info('KickNode: 전진 명령 전송 완료')
        
        done = Bool()
        done.data = True
        self.game_pub.publish(done)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """
        노드가 'active'에서 'inactive' 상태로 전환될 때 호출됩니다.
        활성화된 작업을 중지하고 필요한 정리를 수행합니다.
        """
        self.get_logger().info("KickNode: 비활성화(on_deactivate) 상태입니다. 타이머를 중지합니다.")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """
        노드가 'inactive'에서 'unconfigured' 상태로 전환될 때 호출됩니다.
        `on_configure`에서 생성한 리소스를 해제합니다.
        """
        self.get_logger().info("KickNode: 정리(on_cleanup) 상태로 전환 중...")
        self.destroy_publisher(self.publisher_)
        self.publisher_ = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState):
        """
        노드가 종료될 때 호출됩니다.
        """
        self.get_logger().info("KickNode: 종료(on_shutdown) 중...")
        return TransitionCallbackReturn.SUCCESS
    



def main(args=None):
    rclpy.init(args=args)
    
    # 런치 파일에서 노드를 실행할 때 이 부분이 자동으로 처리됩니다.
    # 따라서 이 main 함수는 단독 실행 테스트용으로만 사용하거나 런치 파일에 맞게 수정해야 합니다.
    kick_node = KickNode()
    rclpy.spin(kick_node)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()