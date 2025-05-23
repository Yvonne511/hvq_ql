import numpy as np
import torch
import d4rl
import gym
from EpisodeMonitor import EpisodeMonitor

class FrankaKitchenDataset:
    def __init__(self, dataset):
        self.trajectories = []

        observations = dataset['observations']
        actions = dataset['actions']

        ends = dataset['terminals']

        terminal_indices = np.where(ends == 1.0)[0]

        start_idx = 0
        for traj_id, end_idx in enumerate(terminal_indices):
            traj_obs = observations[start_idx:end_idx + 1]
            traj_act = actions[start_idx:end_idx + 1]
            obs_tensor = torch.tensor(traj_obs, dtype=torch.float32).unsqueeze(0)  # (1, T, D_obs)
            act_tensor = torch.tensor(traj_act, dtype=torch.float32).unsqueeze(0)  # (1, T, D_act)
            self.trajectories.append({
                "obs": obs_tensor,
                "acts": act_tensor
            })
            start_idx = end_idx + 1

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]['obs']

def load_dataset_and_env(cfg):
    dataset_name = "kitchen-mixed-v0"
    # env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    # train_data = FrankaKitchenDataset(train_dataset)
    # val_data = FrankaKitchenDataset(val_dataset)

    # return train_data, val_data, env

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

if __name__ == "__main__":
    dataset_name = "kitchen-mixed-v0"
    env = make_env(dataset_name)
    dataset = d4rl.qlearning_dataset(env)