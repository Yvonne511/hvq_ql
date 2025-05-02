import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.vector_quantize_pytorch import ResidualVQ
import math

def get_padding(kernel_size, dilation=1):
    return dilation * (kernel_size - 1) // 2

def get_pooling_padding(kernel_size):
    return (kernel_size - 1) // 2

class VQSegmentationModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        codebook_size: int = 4,
        num_quantizers: int = 2,
        cfg=None,
        device = "cuda",
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.latent_dim = latent_dim
        self.predict_steps = 20
        self.hist_steps = 10
        # 1D dilated CNN encoder: receptive field ~ (11 + 5*dilation)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                    out_channels=hidden_dim,
                    kernel_size=7,
                    padding=get_padding(7)),
            nn.ReLU(),

            nn.Conv1d(in_channels=hidden_dim,
                    out_channels=latent_dim,
                    kernel_size=9,
                    padding=get_padding(9)),
        )

        self.future_predictor = nn.Sequential(
            nn.Linear(2 * latent_dim * self.hist_steps, latent_dim * self.predict_steps),
            nn.ReLU(),
            nn.Linear(latent_dim * self.predict_steps, latent_dim * self.predict_steps)
        )
        self.code_embeddings = nn.Embedding(codebook_size, latent_dim)

        # two-stage ResidualVQ: level-1 = coarse trend, level-2 = local detail
        self.vq = ResidualVQ(
            dim               = latent_dim,
            codebook_size     = codebook_size,
            num_quantizers    = num_quantizers,
            shared_codebook   = False,
            rotation_trick    = True,
        )

    def forward(self, traj_xy: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            traj_xy: (B, 2, T) — your raw actions
            masks:   (B, T)   — optional validity mask (not used here)
        Returns:
            codes:    (B, T, num_quantizers)  — discrete segment IDs
            vq_loss:  scalar                  — sum of all VQ losses
        """
        B, C, T = traj_xy.shape
        H, P, D = self.hist_steps, self.predict_steps, self.latent_dim
        
        # traj_xy is (B, 2, T)
        x = traj_xy.to(self.device)
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 1)
        quantized, codes, vq_loss = self.vq(z_e)
        vq_loss = vq_loss.mean()

        # q_z = quantized.permute(0, 2, 1)
        # x_recon = self.decoder(q_z)
        c0 = codes[:, :, 0]   
        code_emb = self.code_embeddings(c0)
        z_q = torch.cat([quantized, code_emb], dim=-1)
        N = T - H - P + 1                                  # Number of valid windows
        z_hist = torch.stack(                             # (B, N, H, 2D)
            [z_q[:, i:N+i, :] for i in range(H)],
            dim=2
        )
        z_hist_flat = z_hist.reshape(B * N, H * 2 * D)     # (B*N, H*2D)

        z_target = torch.stack(                           # (B, N, P, D)
            [quantized[:, i+H:i+H+N, :] for i in range(P)],
            dim=2
        )

        z_pred_flat = self.future_predictor(z_hist_flat)  # (B*N, P*D)
        z_pred = z_pred_flat.view(B, N, P, D)              # (B, N, P, D)
        pred_sim = F.cosine_similarity(z_target, z_pred, dim=-1)
        code_start = c0[:, H:H+N]     # (B, N)
        code_next  = c0[:, H+1:H+N+1] # (B, N), shift by 1

        same_code = (code_start == code_next).float()             # (B, N)
        same_code = same_code.unsqueeze(-1).expand(-1, -1, P)     # (B, N, P)

        group_loss = F.mse_loss(pred_sim, same_code)

        c0 = codes[:, :, 0]                      # (B, T)
        transitions = (c0[:, 1:] != c0[:, :-1]).float()
        smooth_loss = transitions.mean()

        codes = F.interpolate(
            codes.permute(0, 2, 1).float(),  # (B, num_quantizers, T_pooled)
            size=T,
            mode="nearest"
        ).permute(0, 2, 1).long()

        # total loss
        loss = (
            group_loss                  * self.cfg.vqt_rec_weight
            + vq_loss                   * self.cfg.vqt_commit_weight
            + smooth_loss               * self.cfg.vqt_smoothness_weight
        )

        loss_components = {
            "recon_loss": group_loss,
            "vq_loss": vq_loss,
            "smooth_loss": smooth_loss,
        }

        return codes, loss, loss_components

    @torch.no_grad()
    def get_labels(self, traj_xy: torch.Tensor, masks: torch.Tensor):
        """
        Just return the code indices for each timestep.

        Returns:
            codes: (B, T, num_quantizers)
        """
        self.eval()
        codes, _, _= self.forward(traj_xy, masks)

        labels = codes[..., 0].clone()

        left  = labels[:, :-2]            # (B, T-2)
        mid   = labels[:, 1:-1]
        right = labels[:, 2:]

        isolate = (left == right) & (mid != left)

        mid[isolate] = left[isolate]

        smoothed = torch.zeros_like(labels)
        smoothed[:, 0]    = labels[:, 0]
        smoothed[:, 1:-1] = mid
        smoothed[:, -1]   = labels[:, -1]

        return smoothed

    



