import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eunbyeol/kobuki_ws/src/ball_tracking/install/ball_tracking'
