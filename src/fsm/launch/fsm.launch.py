from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 개별 라이프사이클 노드들을 실행합니다.
        Node(package='your_package_name', executable='shoot_goal_node', name='shoot_goal_node', output='screen'),
        Node(package='your_package_name', executable='pass_goal_node', name='pass_goal_node', output='screen'),
        Node(package='your_package_name', executable='drive_node', name='drive_node', output='screen'),
        Node(package='your_package_name', executable='tracking_node', name='tracking_node', output='screen'),
        Node(package='your_package_name', executable='kick_node', name='kick_node', output='screen'),
        Node(package='your_package_name', executable='rotate_node', name='rotate_node', output='screen'),
        
        # 상태를 제어하는 메인 FSM 노드를 실행합니다.
        Node(package='your_package_name', executable='fsm_state_manager', name='fsm_state_manager', output='screen'),
    ])