import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

import hydra
import wandb
# import submitit_patch
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
import numpy as np
from tqdm import tqdm
# from datasets import antmaze, pointmaze, scene, franka_kitchen
from datasets import antmaze, pointmaze, scene
from models.HVQ.hvq.models.hvq_model import SingleVQModel, DoubleVQModel
from models.simpleVQ import VQSegmentationModel
from utils.visual_seg import visual_2d, visual_3d
import torch.nn.functional as F
import torch
import random
import warnings
import logging
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_vq(cfg, dataset, val_data, env, device):

    num_classes = cfg.num_classes
    
    # Initialize the HVQ model with different levels of hierarchies
    model = VQSegmentationModel(
        input_dim=cfg.vqt_input_dim,
        hidden_dim=256,
        latent_dim=128,
        codebook_size=cfg.num_classes,
        num_quantizers=4,
        chunk_size=cfg.chunk_size,
        cfg=cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.vqt_lr, weight_decay=1e-5) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    epochs = cfg.vqt_epochs
        
    # Reconstruction loss
    last_model = None
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    for e in range(epochs):  
        pbar = tqdm(dataloader, desc=f"Training VQ Epoch {e}")
        for batch in pbar:
            loss = 0
            model.train()

            # Load features
            # obs = dataset[idx]["obs"]
            obs = batch["obs_vqvae"].squeeze(1)
            acts = batch["acts"] #([1, 1000, 29])
            feats = obs.permute(0, 2, 1).to(device) # feats dim 1*vqt_input_dim*T breakfast one video eg. torch.Size([1, 2048, 917])

            mask = torch.ones_like(feats).to(device)
            _, loss, loss_components = model(feats, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # We take the last model as best model
            last_model = model.state_dict()    
        
        logger.info(f'VQ - epoch {e}: total {loss.item():.4f} = {loss_components["recon_loss"].item():.4f} + {loss_components["vq_loss"].item():.4f} + {loss_components["smooth_loss"].item():.4f}')

        # Load the last model and store embeddings for clustering
        if (e) % cfg.save_every_epoch_num == 0 or (e) == epochs: # save model, epoch, optimizer

            save_root = os.path.join(os.getcwd(), "checkpoints")
            os.makedirs(save_root, exist_ok=True)
            model_name = f"vq_model_{cfg.env}_{cfg.model_type}_epoch_{e}.pth"
            final_model_name = f"vq_model_{cfg.env}_{cfg.model_type}_epoch_latest.pth"
            model_path = os.path.join(save_root, model_name)
            final_model_path = os.path.join(save_root, final_model_name)
            torch.save({
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)
            torch.save({
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, final_model_path)
            logger.info(f"✅ Saved model checkpoint to {model_path}")

            if cfg.env == "scene":
                visual_3d(env, model, val_data, e, cfg)
            else:
                visual_2d(env, model, val_data, e, cfg)
        
@hydra.main(config_path="../config", config_name="hvq")
def main(cfg: OmegaConf):

    set_seed(cfg.seed)

    DATASET_LOADERS = {
        "pointmaze": pointmaze.load_dataset_and_env,
        "antmaze": antmaze.load_dataset_and_env,
        "scene": scene.load_dataset_and_env,
        # "franka_kitchen": franka_kitchen.load_dataset_and_env,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, env = DATASET_LOADERS[cfg.env](cfg)
    # cfg.vqt_input_dim = train_dataset[0]["obs"].shape[-1]
    cfg.vqt_input_dim = train_dataset[0]["obs_vqvae"].shape[-1]
    train_vq(cfg, train_dataset, val_dataset, env, device)

    save_dir = os.getcwd()
    OmegaConf.save(cfg, os.path.join(save_dir, "hvq.yaml"))

if __name__ == "__main__":
    main()