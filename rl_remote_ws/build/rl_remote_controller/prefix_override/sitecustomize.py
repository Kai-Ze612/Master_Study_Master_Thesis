import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Master_Study_Master_Thesis/rl_remote_ws/install/rl_remote_controller'
