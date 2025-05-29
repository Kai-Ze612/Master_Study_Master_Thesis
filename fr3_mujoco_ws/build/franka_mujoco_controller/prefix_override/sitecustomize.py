import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/kai/Kai_Backup/Study/Master_Thesis/My_Master_Thesis/PD_with_object_franka_mujoco_ws/install/franka_mujoco_controller'
