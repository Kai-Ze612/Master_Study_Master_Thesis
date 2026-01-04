"""
Deployment of the trained RL Agent (JointActor: LSTM + SAC).

Pipeline:
1. Subscribe to 'local_robot/joint_states' (Ground Truth Leader) -> Buffer History.
2. Subscribe to 'remote_robot/joint_states' (Real-time Remote) -> Buffer History.
3. Calculate Delay (simulated network latency).
4. Construct Observation Vector (matching TeleoperationEnv._get_obs()).
5. Pass Observation to JointActor -> Get Action (Torque) and Prediction.
6. Publish 'agent/tau_rl' and 'agent/predict_target'.
"""

import numpy as np
import torch
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from pathlib import Path

from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from E2E_Teleoperation.E2E_Teleoperation.E2E_RL.e2e_network import JointActor, ContinuousLSTMEncoder
import E2E_Teleoperation.config.robot_config as cfg

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        
        # --- 1. Setup & Config ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.control_freq = cfg.CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        
        # Experiment Config
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        exp_config_val = self.get_parameter('experiment_config').value
        self.delay_config = ExperimentConfig(exp_config_val)
        
        # --- 2. Load Model ---
<<<<<<< HEAD
        # [FIX] Correct initialization - JointActor takes encoder as positional argument
        self.encoder = ContinuousLSTMEncoder().to(self.device)
        self.actor = JointActor(self.encoder).to(self.device)
=======
        # We must initialize the architecture exactly as in training
        # [Correction 1] Correct Class Name
        self.shared_encoder = ContinuousLSTMEncoder().to(self.device)
        self.actor = JointActor(
            shared_encoder=self.shared_encoder, 
            action_dim=cfg.N_JOINTS
        ).to(self.device)
>>>>>>> 8eed1054bf496d83b0505b42e253c39917e304bd
        
        model_path = cfg.DEFAULT_RL_MODEL_PATH
        self._load_checkpoint(model_path)
        
        # --- 3. Buffers & State ---
        # Delay Simulator
        self.delay_sim = DelaySimulator(self.control_freq, self.delay_config, seed=42)
        
        # Buffers
        self.leader_hist_q = deque(maxlen=cfg.BUFFER_SIZE)
        self.leader_hist_qd = deque(maxlen=cfg.BUFFER_SIZE)
        self.remote_hist_q = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        self.remote_hist_qd = deque(maxlen=cfg.RNN_SEQUENCE_LENGTH)
        
        # Current State
        self.curr_remote_q = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        self.curr_remote_qd = np.zeros(cfg.N_JOINTS, dtype=np.float32)
        
<<<<<<< HEAD
        # Hidden State for LSTM (initialized to None, will be set on first forward pass)
        self.hidden_state = None
=======
        # [Correction 2] Initialize Hidden State for LSTM
        self.hidden_state = None  # Will be initialized on first forward pass
>>>>>>> 8eed1054bf496d83b0505b42e253c39917e304bd
        
        # State Flags
        self.is_leader_ready = False
        self.is_remote_ready = False
        self.joint_names = [f'panda_joint{i+1}' for i in range(cfg.N_JOINTS)]

        # --- 4. ROS Interfaces ---
        self.tau_pub = self.create_publisher(Float64MultiArray, 'agent/tau_rl', 10)
        self.pred_pub = self.create_publisher(JointState, 'agent/predict_target', 10)

        self.sub_leader = self.create_subscription(
            JointState, 'local_robot/joint_states', self.leader_callback, 10
        )
        self.sub_remote = self.create_subscription(
            JointState, 'remote_robot/joint_states', self.remote_callback, 10
        )
        
        self.timer = self.create_timer(self.dt, self.control_step)
        
        self.get_logger().info(f"Agent Node Initialized | Device: {self.device}")
        self.get_logger().info(f"Delay Config: {self.delay_config.name}")

    def _load_checkpoint(self, path):
        path = Path(path)
        if not path.exists():
            self.get_logger().error(f"Model checkpoint not found: {path}")
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'actor' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor'])
            elif 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            else:
                # Assume the checkpoint IS the state dict
                self.actor.load_state_dict(checkpoint)
                
            self.actor.eval()
            self.get_logger().info(f"Loaded model from {path}")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load model: {e}")

    def leader_callback(self, msg: JointState):
        """Buffer Ground Truth Leader State."""
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            q = np.array([msg.position[name_map[n]] for n in self.joint_names], dtype=np.float32)
            qd = np.array([msg.velocity[name_map[n]] for n in self.joint_names], dtype=np.float32)
            
            self.leader_hist_q.append(q)
            self.leader_hist_qd.append(qd)
            
            if not self.is_leader_ready and len(self.leader_hist_q) > cfg.RNN_SEQUENCE_LENGTH + 50:
                self.is_leader_ready = True
                self.get_logger().info("Leader Buffer Ready.")
        except Exception:
            pass

    def remote_callback(self, msg: JointState):
        """Buffer Remote State."""
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            q = np.array([msg.position[name_map[n]] for n in self.joint_names], dtype=np.float32)
            qd = np.array([msg.velocity[name_map[n]] for n in self.joint_names], dtype=np.float32)
            
            self.curr_remote_q = q
            self.curr_remote_qd = qd
            
            self.remote_hist_q.append(q)
            self.remote_hist_qd.append(qd)
            
            if not self.is_remote_ready and len(self.remote_hist_q) >= cfg.RNN_SEQUENCE_LENGTH:
                self.is_remote_ready = True
                self.get_logger().info("Remote Buffer Ready.")
        except Exception:
            pass

    def _get_observation(self):
<<<<<<< HEAD
        """Constructs the observation vector (must match training_env._get_obs())."""
=======
        """Constructs the observation vector."""
>>>>>>> 8eed1054bf496d83b0505b42e253c39917e304bd
        # 1. Normalize Current Remote State
        curr_q_norm = (self.curr_remote_q - cfg.Q_MEAN) / cfg.Q_STD
        curr_qd_norm = (self.curr_remote_qd - cfg.QD_MEAN) / cfg.QD_STD
        state_norm = np.concatenate([curr_q_norm, curr_qd_norm])

        # 2. Normalize Remote History
        hist_seq = []
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            if i < len(self.remote_hist_q):
                q = (self.remote_hist_q[i] - cfg.Q_MEAN) / cfg.Q_STD
                qd = (self.remote_hist_qd[i] - cfg.QD_MEAN) / cfg.QD_STD
                hist_seq.extend(np.concatenate([q, qd]))
            else:
                hist_seq.extend(np.zeros(cfg.ROBOT_STATE_DIM))

        # 3. Process Delayed Target History
        history_len = len(self.leader_hist_q)
        delay_steps = self.delay_sim.get_state_delay_steps(history_len)
        norm_delay = delay_steps / cfg.DELAY_INPUT_NORM_FACTOR
        
        target_seq = []
        end_idx = history_len - 1 - delay_steps
        start_idx = end_idx - cfg.RNN_SEQUENCE_LENGTH + 1
        start_idx = max(0, start_idx)
        
        for i in range(cfg.RNN_SEQUENCE_LENGTH):
            curr_idx = start_idx + i
            if curr_idx < history_len and curr_idx >= 0:
                q = self.leader_hist_q[curr_idx]
                qd = self.leader_hist_qd[curr_idx]
            else:
                q = self.leader_hist_q[0]
                qd = self.leader_hist_qd[0]
            
            q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
            qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
            target_seq.extend(np.concatenate([q_norm, qd_norm, [norm_delay]]))

        obs_np = np.concatenate([state_norm, hist_seq, target_seq], dtype=np.float32)
        return torch.tensor(obs_np, device=self.device).unsqueeze(0)

    def control_step(self):
        if not self.is_leader_ready or not self.is_remote_ready:
            return

        with torch.no_grad():
            # 1. Get Observation
            obs_tensor = self._get_observation()
            
<<<<<<< HEAD
            # 2. Run JointActor
            # [FIX] sample() returns 4 values: action, log_prob, pred, next_hidden
            action, log_prob, pred_state, next_hidden = self.actor.sample(
                obs_tensor, 
                hidden=self.hidden_state,
                has_new_obs=True
            )
            
            # Update hidden state for next step
            self.hidden_state = next_hidden
            
            # 3. Process Outputs
            tau_rl = action.cpu().numpy().flatten()
            
            # Denormalize prediction (pred_state is normalized)
            pred_state_np = pred_state.cpu().numpy().flatten()
=======
            # 2. Run JointActor (Policy + Encoder)
            # [Correction 2] Handle 5 return values
            # [Correction 3] Pass and update hidden_state
            action_scaled, _, _, pred_state_norm, self.hidden_state = self.actor.sample(
                obs_tensor, 
                hidden=self.hidden_state, # Pass previous hidden state
                deterministic=True
            )
            
            # 3. Process Outputs
            tau_rl = action_scaled.cpu().numpy().flatten()
            
            pred_state_np = pred_state_norm.cpu().numpy().flatten()
>>>>>>> 8eed1054bf496d83b0505b42e253c39917e304bd
            pred_q_norm = pred_state_np[:7]
            pred_qd_norm = pred_state_np[7:] if len(pred_state_np) > 7 else np.zeros(7)
            
            pred_q = pred_q_norm * cfg.Q_STD + cfg.Q_MEAN
            pred_qd = pred_qd_norm * cfg.QD_STD + cfg.QD_MEAN

            # 4. Publish
            self._publish_tau(tau_rl)
            self._publish_prediction(pred_q, pred_qd)

    def _publish_tau(self, tau):
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.tau_pub.publish(msg)

    def _publish_prediction(self, q, qd):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.name = self.joint_names
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.pred_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()