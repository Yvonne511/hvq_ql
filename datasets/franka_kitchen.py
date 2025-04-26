import numpy as np
import torch
import env.ogbench.ogbench as ogbench  # Replace this if you're using another loader like d4rl.kitchen

class FrankaKitchenDataset:
    def __init__(self, dataset):
        self.trajectories = []

        observations = dataset['observations']
        actions = dataset['actions']

        # Use terminals if available, otherwise fall back to timeouts
        if 'terminals' in dataset:
            ends = dataset['terminals']
        elif 'timeouts' in dataset:
            ends = dataset['timeouts']
        else:
            raise ValueError("No 'terminals' or 'timeouts' found in dataset")

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
    dataset_name = cfg.dataset_name  # should be something like 'kitchen-mixed-v0'
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    train_data = FrankaKitchenDataset(train_dataset)
    val_data = FrankaKitchenDataset(val_dataset)

    return train_data, val_data, env
