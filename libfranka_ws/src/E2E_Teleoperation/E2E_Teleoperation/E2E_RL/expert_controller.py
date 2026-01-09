import numpy as np
import E2E_Teleoperation.config.robot_config as cfg

class ImprovedExpertAction:
    """
    Smoothed Expert Controller for Data Generation.
    Features:
    1. Reduced PD Gains: Prevents high-frequency jitter.
    2. Rate Limiting: Prevents sudden torque spikes.
    3. State Reset: Clears history on new episodes.
    """
    
    def __init__(self, follower_robot):
        self.follower = follower_robot
        self.prev_torque = np.zeros(cfg.ROBOT.N_JOINTS)
        
        # --- TUNED GAINS (Lower for Smoothness) ---
        self.kp = np.array([50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 20.0]) 
        self.kd = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])    
        
        # Torque Rate Limit (Nm per step)
        self.max_torque_rate = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    
    def compute(self, leader_state):
        if leader_state is None:
            return np.zeros(cfg.ROBOT.N_JOINTS)
        
        # Unpack Leader State
        if len(leader_state) == 3:
            target_q, target_qd, target_qdd_noisy = leader_state
        else:
            target_q, target_qd = leader_state
            target_qdd_noisy = np.zeros_like(target_q)
            
        # >>> CRITICAL FIX: IGNORE NOISY ACCELERATION <<<
        # We assume the target is moving at constant velocity for the physics calc.
        # This removes the vibration caused by IK finite differencing.
        target_qdd = np.zeros_like(target_qdd_noisy) 
        
        # Get Follower State
        f_q, f_qd = self.follower.get_joint_state()
        
        # 1. Compute Errors
        q_err = target_q - f_q
        qd_err = target_qd - f_qd
        
        # 2. PD Control Law
        # We use the Position and Velocity targets, but ignore the Acceleration target
        qdd_des = target_qdd + (self.kp * q_err) + (self.kd * qd_err)
        
        # 3. Inverse Dynamics (Compute required Torque)
        raw_torque = self.follower.compute_inverse_dynamics(f_q, f_qd, qdd_des)
        
        # ... (Rate limiting code remains the same) ...
        torque_diff = raw_torque - self.prev_torque
        torque_diff = np.clip(torque_diff, -self.max_torque_rate, self.max_torque_rate)
        smooth_torque = self.prev_torque + torque_diff
        smooth_torque = np.clip(smooth_torque, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        
        self.prev_torque = smooth_torque.copy()
        return smooth_torque
    
    def reset(self):
        self.prev_torque = np.zeros(cfg.ROBOT.N_JOINTS)