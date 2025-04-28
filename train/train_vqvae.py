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
from datasets import antmaze, pointmaze, scene, franka_kitchen
from models.HVQ.hvq.models.hvq_model import SingleVQModel, DoubleVQModel
from utils.visual_seg import visual_2d, visual_3d
import torch.nn.functional as F
import torch
import random
import warnings
import logging

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
    if cfg.model_type == "single":
        model = SingleVQModel(num_stages=cfg.num_stages, num_layers=cfg.num_layers, num_f_maps=cfg.f_maps, dim=cfg.vqt_input_dim, num_classes=num_classes, latent_dim=cfg.f_maps).to(device)
    elif cfg.model_type == "double":
        model = DoubleVQModel(num_stages=cfg.num_stages, num_layers=cfg.num_layers, num_f_maps=cfg.f_maps, dim=cfg.vqt_input_dim, num_classes=num_classes, latent_dim=cfg.f_maps, ema_dead_code=cfg.ema_dead_code).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.vqt_lr, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    epochs = cfg.vqt_epochs
        
    # Reconstruction loss
    loss_rec_fn = torch.nn.MSELoss(reduce='sum')
    last_model = None

    for e in range(epochs):  
        for idx in tqdm(range(len(dataset)), desc=f"Training VQ Epoch {e}"):
            loss = 0
            model.train()

            # Load features
            obs = dataset[idx]["obs"]
            acts = dataset[idx]["acts"] #([1, 1000, 29])
            feats = obs.permute(0, 2, 1).to(device) # feats dim 1*vqt_input_dim*T breakfast one video eg. torch.Size([1, 2048, 917])

            # Our batch size is 1, there is no padding and no need to mask
            # Change it if needed
            # feats = F.normalize(feats)
            mask = torch.ones_like(feats).to(device)
            # Reconstructed features, predicted labels, commitment loss, distances, encoder output
            reconstructed_feats, _, commit_loss, _, _ = model(feats, mask, return_enc=True)

            reconstruction_loss = 0 # Compute reconstruction loss for each stage of MS-TCN
            for i in range(reconstructed_feats.shape[0]):
                reconstruction_loss += loss_rec_fn(reconstructed_feats[i].unsqueeze(0), feats)

            loss += cfg.vqt_commit_weight * commit_loss
            loss += cfg.vqt_rec_weight * reconstruction_loss  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            # We take the last model as best model
            last_model = model.state_dict()    
        
        logger.info(f'VQ - epoch {e}: total {loss.item():.4f}, commit {commit_loss.item():.4f}, rec {reconstruction_loss.item():.4f}')  

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
        
            visual_2d(env, model, val_data, e, cfg)
        
@hydra.main(config_path="../config", config_name="hvq")
def main(cfg: OmegaConf):

    save_dir = os.getcwd()  # Hydra changes working directory automatically
    OmegaConf.save(cfg, os.path.join(save_dir, "hvq.yaml"))

    set_seed(cfg.seed)

    DATASET_LOADERS = {
        "pointmaze": pointmaze.load_dataset_and_env,
        "antmaze": antmaze.load_dataset_and_env,
        "scene": scene.load_dataset_and_env,
        "franka_kitchen": franka_kitchen.load_dataset_and_env,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, env = DATASET_LOADERS[cfg.env](cfg)
    cfg.vqt_input_dim = train_dataset[0]["obs"].shape[-1]
    train_vq(cfg, train_dataset, val_dataset, env, device)

if __name__ == "__main__":
    main()