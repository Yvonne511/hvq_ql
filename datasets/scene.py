import numpy as np
import torch
import env.ogbench.ogbench as ogbench
from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct, idct
import numpy as np

class OgbenchDataset:
    def __init__(self, dataset, use_terminals=True):
        self.trajectories = []
        observations = dataset['observations']
        actions = dataset['actions']
        
        ends = dataset['terminals']
        
        terminal_indices = np.where(ends == 1.0)[0]

        # Robot arm and gripper (19 dims):
        # - joint_pos (6,)           # UR5e arm joint positions
        # - joint_vel (6,)           # UR5e arm joint velocities
        # - effector_pos (3,)        # Cartesian position of the gripper (centered and scaled)
        # - effector_yaw (2,)        # [cos(yaw), sin(yaw)]
        # - gripper_opening (1,)     # Gripper opening amount (scaled)
        # - gripper_contact (1,)     # Normalized contact force on right pad
        #
        # Cube (9 dims):
        # - block_pos (3,)           # Cube position (centered and scaled)
        # - block_quat (4,)          # Cube orientation as quaternion
        # - block_yaw (2,)           # [cos(yaw), sin(yaw)] of cube orientation
        #
        # Buttons (8 dims total for 2 buttons):
        # - button_states (4,)       # One-hot states for 2 buttons (2 per button)
        # - button_pos (2,)          # Joint positions for 2 buttons
        # - button_vel (2,)          # Joint velocities for 2 buttons
        #
        # Drawer and window (4 dims):
        # - drawer_pos (1,)          # Drawer slide position (scaled)
        # - drawer_vel (1,)          # Drawer slide velocity
        # - window_pos (1,)          # Window slide position (scaled)
        # - window_vel (1,)          # Window slide velocity

        start_idx = 0
        for traj_id, end_idx in enumerate(terminal_indices):
            traj_obs = observations[start_idx:end_idx + 1]
            traj_act = actions[start_idx:end_idx + 1]
            obs_tensor = torch.tensor(traj_obs, dtype=torch.float32).unsqueeze(0)
            act_tensor = torch.tensor(traj_act, dtype=torch.float32).unsqueeze(0)

            K = 250
            coeffs = dct(traj_act, axis=0, norm='ortho')      # (T, D)
            coeffs[K:, :] = 0                                      # zero out high-freq bins
            smoothed_obs = idct(coeffs, axis=0, norm='ortho')      # (T, D)
            obs_vqvae_tensor = torch.tensor(smoothed_obs, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
            self.trajectories.append({
                "obs_vqvae": act_tensor,
                "obs": obs_tensor,
                "acts": act_tensor
            })
            start_idx = end_idx + 1

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def load_dataset_and_env(cfg):
    dataset_name = "scene-play-v0"
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    train_data = OgbenchDataset(train_dataset, use_terminals=True)
    val_data = OgbenchDataset(val_dataset, use_terminals=True)

    return train_data, val_data, env

if __name__ == "__main__":
    dataset_name = "scene-play-v0"
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)
    print(train_dataset['observations'].shape)