import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/rl_agents/impls")))

import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import torch
import logging
from datasets import antmaze, pointmaze, scene, franka_kitchen
from models.HVQ.hvq.models.hvq_model import SingleVQModel, DoubleVQModel

## rl imports
import jax
from absl import app, flags
from models.rl_agents.impls.agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_vq_model(cfg, device):
    checkpoint_path = os.path.join(
        cfg.model_path, "checkpoints", f"vq_model_{cfg.env}_{cfg.model_type}_epoch_latest.pth") if "model_path" in cfg and cfg.model_path else os.path.join(
        os.getcwd(), "checkpoints", f"vq_model_{cfg.env}_{cfg.model_type}_epoch_latest.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    logger.info(f"📦 Loading VQ model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct the model
    if cfg.model_type == "single":
        model = SingleVQModel(
            num_stages=cfg.num_stages,
            num_layers=cfg.num_layers,
            num_f_maps=cfg.f_maps,
            dim=cfg.vqt_input_dim,
            num_classes=cfg.num_classes,
            latent_dim=cfg.f_maps
        ).to(device)
    else:
        model = DoubleVQModel(
            num_stages=cfg.num_stages,
            num_layers=cfg.num_layers,
            num_f_maps=cfg.f_maps,
            dim=cfg.vqt_input_dim,
            num_classes=cfg.num_classes,
            latent_dim=cfg.f_maps,
            ema_dead_code=cfg.ema_dead_code
        ).to(device)

    model.load_state_dict(checkpoint['model'])
    logger.info("✅ Model loaded successfully and set to eval mode.")
    return model

def train_rl(cfg, vq_model, dataset, env, device):
    # vq_model.get_labels() to generate skill segments
    for idx in tqdm(range(len(dataset)), desc=f"Labeling w. VQ"):
        obs = dataset[idx]["obs"]
        acts = dataset[idx]["acts"]
        
        feats = obs.permute(0, 2, 1).to(device)
        mask = torch.ones_like(feats).to(device)
        labels = vq_model.get_labels(feats, mask).squeeze().detach().cpu().numpy()
        subgoals = np.zeros_like(labels)
        subgoals[1:][labels[1:] != labels[:-1]] = 1

    # train a hierarchical RL agent using those segments
    

@hydra.main(config_path="../config", config_name="rl")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)

    save_dir = os.getcwd()  # Hydra changes working directory automatically
    OmegaConf.save(cfg, os.path.join(save_dir, "rl.yaml"))

    DATASET_LOADERS = {
        "pointmaze": pointmaze.load_dataset_and_env,
        "antmaze": antmaze.load_dataset_and_env,
        "scene": scene.load_dataset_and_env,
        "franka_kitchen": franka_kitchen.load_dataset_and_env,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, env = DATASET_LOADERS[cfg.env](cfg)
    cfg.vqt_input_dim = train_dataset[0]["obs"].shape[-1]

    hvq_cfg = OmegaConf.load(os.path.join(cfg.model_path, "hvq.yaml"))
    vq_model = load_vq_model(cfg, device)
    train_rl(cfg, vq_model, train_dataset, env, device)

if __name__ == "__main__":
    main()
