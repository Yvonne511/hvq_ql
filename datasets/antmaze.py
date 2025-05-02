import numpy as np
import torch
import env.ogbench.ogbench as ogbench
from scipy.spatial.transform import Rotation as R
from scipy.fftpack import dct, idct
import numpy as np

def energy_based_k(traj_act: np.ndarray, alpha: float = 0.90):
    """
    Choose the smallest K so that the first K DCT bins
    explain >= alpha fraction of total energy.
    """
    # traj_act: (T, D) np array of joint angles / actions
    coeffs = dct(traj_act, axis=0, norm='ortho')     # (T, D)
    power = coeffs**2                                  # elementwise power
    total_energy = power.sum()
    # cumulative energy along time axis
    cum_energy = np.cumsum(power.sum(axis=1))         # (T,)
    # find the first index where cum_energy/total_energy >= alpha
    K = int(np.searchsorted(cum_energy / total_energy, alpha)) + 1
    return K

class OgbenchDataset:
    def __init__(self, dataset, use_terminals=True):
        self.trajectories = []
        observations = dataset['observations']
        actions = dataset['actions']
        
        ends = dataset['terminals']
        
        terminal_indices = np.where(ends == 1.0)[0]

        start_idx = 0
        for traj_id, end_idx in enumerate(terminal_indices):
            traj_obs = observations[start_idx:end_idx + 1]
            traj_act = actions[start_idx:end_idx + 1]
            obs_tensor = torch.tensor(traj_obs, dtype=torch.float32).unsqueeze(0)
            # qpos (position) 15
                # body position (x, y, z)
                # body orientation (x, y, z, w)
                # joint angle 8DoF
            # qvel (velocity) 14
                # body linear and angular velocity 6DOF
                # joint velocity 8DoF
            xy = torch.tensor(traj_obs[:, 0:2], dtype=torch.float32).unsqueeze(0)
            act_tensor = torch.tensor(traj_act, dtype=torch.float32).unsqueeze(0)

            # vqvae feature
            # 1. Change all Angles to sin(θ), cos(θ)
            orientations = traj_obs[:, 3:7]
            r = R.from_quat(orientations)
            yaw = r.as_euler('zyx')[:, 0]
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            sin_cos_yaw = np.stack([sin_yaw, cos_yaw], axis=-1)
            modified_obs = np.concatenate([
                # traj_obs[:, 0:3],          # (x, y, z)
                sin_cos_yaw,               # 2D heading
                traj_obs[:, 7:]            # joint angles + velocities
            ], axis=-1)
            obs_vqvae_tensor = torch.tensor(modified_obs, dtype=torch.float32).unsqueeze(0)
            # 2. Stack Consecutive States
            # modified_obs = traj_obs
            # window_size = 7
            # pad = window_size // 2  # to center window on each timestep
            # padded = np.pad(modified_obs, ((pad, pad), (0, 0)), mode='edge')  # pad on both sides

            # stacked_obs = np.lib.stride_tricks.sliding_window_view(padded, (window_size, modified_obs.shape[1]))[:, 0, :, :]
            # stacked_obs = stacked_obs.reshape(stacked_obs.shape[0], -1)  # shape: (T, window_size*D)

            # obs_vqvae_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)  # (1, T, window_size*D)
            # 3. DCT
            # K = energy_based_k(traj_act, alpha=0.55)
            # print(f"K = {K}")
            # K = 230
            # coeffs = dct(traj_act, axis=0, norm='ortho')      # (T, D)
            # coeffs[K:, :] = 0                                      # zero out high-freq bins
            # smoothed_obs = idct(coeffs, axis=0, norm='ortho')      # (T, D)
            # obs_vqvae_tensor = torch.tensor(smoothed_obs, dtype=torch.float32).unsqueeze(0)  # (1, T, D)

            self.trajectories.append({
                "obs_vqvae": act_tensor,
                "obs": obs_tensor,
                "acts": act_tensor,
                "xy": xy,
            })
            start_idx = end_idx + 1

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def load_dataset_and_env(cfg):
    dataset_name = "antmaze-medium-navigate-v0"
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    train_data = OgbenchDataset(train_dataset)
    val_data = OgbenchDataset(val_dataset)

    return train_data, val_data, env


