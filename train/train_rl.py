import sys
import os
from datetime import datetime
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/rl_agents/impls")))

import numpy as np
from tqdm import tqdm, trange
import wandb
import random
import hydra
from omegaconf import OmegaConf
import torch
import logging
from datasets import antmaze, pointmaze, scene
from models.HVQ.hvq.models.hvq_model import SingleVQModel, DoubleVQModel
from models.simpleVQ import VQSegmentationModel

## rl imports
import jax
from collections import defaultdict
from models.rl_agents.impls.agents import agents
from ml_collections import config_flags
from models.rl_agents.impls.utils.datasets import Dataset, GCDataset, HGCDataset
from models.rl_agents.impls.utils.env_utils import make_env_and_datasets
from models.rl_agents.impls.utils.evaluation import evaluate
from models.rl_agents.impls.utils.flax_utils import restore_agent, save_agent
from models.rl_agents.impls.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

from models.rl_agents.impls.agents.hiql import get_config
config = get_config()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_vq_model(hvq_cfg, cfg, device):
    checkpoint_path = os.path.join(
        cfg.model_path, "checkpoints", f"vq_model_{hvq_cfg.env}_{hvq_cfg.model_type}_epoch_{cfg.model_epoch}.pth") if "model_path" in cfg and cfg.model_path else os.path.join(
        os.getcwd(), "checkpoints", f"vq_model_{hvq_cfg.env}_{hvq_cfg.model_type}_epoch_{cfg.model_epoch}.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    logger.info(f"📦 Loading VQ model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct the model
    # if hvq_cfg.model_type == "single":
    #     model = SingleVQModel(
    #         num_stages=hvq_cfg.num_stages,
    #         num_layers=hvq_cfg.num_layers,
    #         num_f_maps=hvq_cfg.f_maps,
    #         dim=hvq_cfg.vqt_input_dim,
    #         num_classes=hvq_cfg.num_classes,
    #         latent_dim=hvq_cfg.f_maps,
    #         cfg=hvq_cfg,
    #     ).to(device)
    # else:
    #     model = DoubleVQModel(
    #         num_stages=hvq_cfg.num_stages,
    #         num_layers=hvq_cfg.num_layers,
    #         num_f_maps=hvq_cfg.f_maps,
    #         dim=hvq_cfg.vqt_input_dim,
    #         num_classes=hvq_cfg.num_classes,
    #         latent_dim=hvq_cfg.f_maps,
    #         ema_dead_code=hvq_cfg.ema_dead_code,
    #         cfg=hvq_cfg,
    #     ).to(device)
    model = VQSegmentationModel(
        input_dim=hvq_cfg.vqt_input_dim,
        hidden_dim=256,
        latent_dim=128,
        codebook_size=hvq_cfg.num_classes,
        num_quantizers=4,
        cfg=hvq_cfg,
        device=device,
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    logger.info("✅ Model loaded successfully and set to eval mode.")
    return model

def train_rl(cfg, vq_model, train_dset, val_dset, env, device):
    # vq_model.get_labels() to generate skill segments
    def label_dataset(dset, desc):
        all_obs = []
        all_actions = []
        all_subgoals = []
        all_valids = []
        all_terminals = []

        for idx in tqdm(range(len(dset)), desc=desc):
            traj = dset[idx]
            obs = traj["obs"]
            obs_vqvae = traj["obs_vqvae"]
            acts = traj["acts"]

            feats = obs_vqvae.permute(0, 2, 1).to(device)
            mask = torch.ones_like(feats).to(device)
            labels = vq_model.get_labels(feats, mask).squeeze().detach().cpu().numpy()

            subgoals = np.zeros_like(labels)
            subgoals[1:][labels[1:] != labels[:-1]] = 1  # mark subgoals where label changes

            obs_np = obs.squeeze(0).cpu().numpy()
            acts_np = acts.squeeze(0).cpu().numpy()

            T = obs_np.shape[0]
            terminals = np.zeros((T,))
            terminals[-1] = 1.0  # last step of each trajectory

            all_obs.append(obs_np)
            all_actions.append(acts_np)
            all_subgoals.append(subgoals)
            all_terminals.append(terminals)

        dataset = {
            "observations": np.concatenate(all_obs, axis=0),
            "actions": np.concatenate(all_actions, axis=0),
            "subgoals": np.concatenate(all_subgoals, axis=0),
            "terminals": np.concatenate(all_terminals, axis=0),
        }
        return dataset
    
    train_dataset = label_dataset(train_dset, desc="Labeling Train w. VQ")
    val_dataset = label_dataset(val_dset, desc="Labeling Val w. VQ")
    # train a hierarchical RL agent using those segments

    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{cfg.env}"
    # exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_pointmaze_baseline"
    # exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_antmaze_baseline"
    # exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_scene_baseline"
    setup_wandb(project='hvq_ql', group=cfg.run_group, name=exp_name, cfg_dict=OmegaConf.to_container(cfg, resolve=True))

    # train_dataset, val_dataset = None, None
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    # -- Training --
    train_logger = CsvLogger(os.path.join(cfg.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(cfg.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % cfg.eval_interval == 0:
            if cfg.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = cfg.eval_tasks if cfg.eval_tasks is not None else len(task_infos)
            for task_id in trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=cfg.eval_episodes,
                    num_video_episodes=cfg.video_episodes,
                    video_frame_skip=cfg.video_frame_skip,
                    eval_temperature=cfg.eval_temperature,
                    eval_gaussian=cfg.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

    train_logger.close()
    eval_logger.close()
    

@hydra.main(config_path="../config", config_name="rl")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)

    save_dir = os.getcwd()  # Hydra changes working directory automatically
    OmegaConf.save(cfg, os.path.join(save_dir, "rl.yaml"))
    cfg.save_dir = save_dir

    DATASET_LOADERS = {
        "pointmaze": pointmaze.load_dataset_and_env,
        "antmaze": antmaze.load_dataset_and_env,
        "scene": scene.load_dataset_and_env,
        # "franka_kitchen": franka_kitchen.load_dataset_and_env,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, env = DATASET_LOADERS[cfg.env](cfg)
    cfg.vqt_input_dim = train_dataset[0]["obs_vqvae"].shape[-1]

    hvq_cfg = OmegaConf.load(os.path.join(cfg.model_path, "hvq.yaml"))
    vq_model = load_vq_model(hvq_cfg, cfg, device)
    train_rl(cfg, vq_model, train_dataset, val_dataset, env, device)

if __name__ == "__main__":
    main()
