import numpy as np
import torch
import env.ogbench.ogbench as ogbench

class OgbenchDataset:
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
            obs_tensor = torch.tensor(traj_obs, dtype=torch.float32).unsqueeze(0)
            act_tensor = torch.tensor(traj_act, dtype=torch.float32).unsqueeze(0)
            self.trajectories.append({
                # "obs_vqvae": obs_tensor,
                "obs_vqvae": act_tensor,
                "obs": obs_tensor,
                "acts": act_tensor,
                "xy": obs_tensor,
            })
            start_idx = end_idx + 1

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def load_dataset_and_env(cfg):
    dataset_name = "pointmaze-medium-navigate-v0"
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    train_data = OgbenchDataset(train_dataset)
    val_data = OgbenchDataset(val_dataset)

    return train_data, val_data, env
