import rclpy
from rclpy.node import Node
from rclpy.lifecycle import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
import sys

# FSM 테이블을 딕셔너리 형태로 표현합니다.
# 이 데이터는 외부 YAML 파일에서 불러오는 것이 더 유연합니다.
FSM_TABLE = {
    'S1_ReadyPass':    {'shoot_goal_node': 'off', 'pass_goal_node': 'on', 'drive_node': 'on', 'tracking_node': 'off', 'kick_node': 'off', 'rotate_node': 'off'},
    'S2_Pass':         {'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'off', 'kick_node': 'on', 'rotate_node': 'off'},
    'S3_RotateForGoal':{'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'off', 'kick_node': 'off', 'rotate_node': 'on'},
    'S4_GoToGoalpost': {'shoot_goal_node': 'on', 'pass_goal_node': 'off', 'drive_node': 'on', 'tracking_node': 'off', 'kick_node': 'off', 'rotate_node': 'off'},
    'S5_Kick':         {'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'off', 'kick_node': 'on', 'rotate_node': 'off'},
    'S6_RotateForTrack':{'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'off', 'kick_node': 'off', 'rotate_node': 'on'},
    'S7_BallTracking': {'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'on', 'kick_node': 'off', 'rotate_node': 'off'},
    'S8_Finish':       {'shoot_goal_node': 'off', 'pass_goal_node': 'off', 'drive_node': 'off', 'tracking_node': 'off', 'kick_node': 'off', 'rotate_node': 'off'}
}

class FSMStateManager(Node):
    def __init__(self):
        super().__init__('fsm_state_manager')
        self.node_clients = {}

        # FSM 테이블에 있는 모든 노드에 대한 서비스 클라이언트를 생성합니다.
        for node_name in list(FSM_TABLE.values())[0].keys():
            self.node_clients[node_name] = self.create_client(ChangeState, f'/{node_name}/change_state')
            # 서비스가 사용 가능해질 때까지 기다립니다.
            while not self.node_clients[node_name].wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{node_name}/change_state 서비스 대기 중...')

        self.current_fsm_state = 'S1_ReadyPass'
        self.apply_fsm_state(self.current_fsm_state)

    def apply_fsm_state(self, new_state: str):
        """
        새로운 FSM 상태에 따라 각 노드의 라이프사이클 상태를 변경합니다.
        """
        self.get_logger().info(f"FSM 상태를 '{self.current_fsm_state}'에서 '{new_state}'로 변경합니다.")
        
        # 이전 상태와 새 상태를 비교하여 변경이 필요한 노드만 제어합니다.
        previous_state_config = FSM_TABLE[self.current_fsm_state]
        new_state_config = FSM_TABLE[new_state]

        for node_name, target_status in new_state_config.items():
            if previous_state_config[node_name] != target_status:
                transition_id = Transition.TRANSITION_DEACTIVATE if target_status == 'off' else Transition.TRANSITION_ACTIVATE
                self.send_transition_request(node_name, transition_id)

        self.current_fsm_state = new_state
        self.get_logger().info(f"FSM 상태 변경 완료. 현재 상태: {self.current_fsm_state}")

    def send_transition_request(self, node_name: str, transition_id: int):
        """
        특정 노드에 라이프사이클 상태 변경 요청을 보냅니다.
        """
        req = ChangeState.Request()
        req.transition.id = transition_id
        
        future = self.node_clients[node_name].call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            self.get_logger().info(f"'{node_name}' 노드의 상태 변경 요청 성공.")
        else:
            self.get_logger().error(f"'{node_name}' 노드의 상태 변경 요청 실패.")

    # 외부 트리거에 따라 상태를 변경하는 메소드 (예: 메시지 수신, 특정 조건 충족)
    def trigger_next_state(self, next_state: str):
        if next_state in FSM_TABLE:
            self.apply_fsm_state(next_state)
        else:
            self.get_logger().error(f"알 수 없는 FSM 상태: {next_state}")
