import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.vector_quantize_pytorch import ResidualVQ

def get_padding(kernel_size, dilation=1):
    return dilation * (kernel_size - 1) // 2

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
        # 1D dilated CNN encoder: receptive field ~ (11 + 5*dilation)
        self.encoder = nn.Sequential(
            # input: (B, C=2, T) → (B, hidden_dim, T)
            nn.Conv1d(in_channels=input_dim,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      padding=get_padding(3)),
            nn.ReLU(),
            # → (B, latent_dim, T)
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=latent_dim,
                      kernel_size=5,
                      padding=get_padding(5))
        )

        self.decoder = nn.Sequential(
            # quantized comes in as (B, latent_dim, T)
            nn.Conv1d(in_channels=latent_dim,
                      out_channels=hidden_dim,
                      kernel_size=5,
                      padding=get_padding(5)),
            nn.ReLU(),
            # → (B, input_dim=2, T)
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=input_dim,
                      kernel_size=3,
                      padding=get_padding(3))
        )

        # two-stage ResidualVQ: level-1 = coarse trend, level-2 = local detail
        self.vq = ResidualVQ(
            dim               = latent_dim,
            codebook_size     = codebook_size,
            num_quantizers    = num_quantizers,
            shared_codebook   = False,
        )

        self.register_buffer("init_recon", torch.tensor(0.0))
        self.register_buffer("init_vq",    torch.tensor(0.0))
        self.register_buffer("init_smooth",torch.tensor(0.0))

    def forward(self, traj_xy: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            traj_xy: (B, 2, T) — your raw actions
            masks:   (B, T)   — optional validity mask (not used here)
        Returns:
            codes:    (B, T, num_quantizers)  — discrete segment IDs
            vq_loss:  scalar                  — sum of all VQ losses
        """
        # B, C, T = traj_xy.shape
        # x = traj_xy

        # x = x.permute(0, 2, 1)

        # z = self.encoder(x)

        # # quantize: returns (quantized, codes, loss)
        # quantized, codes, vq_loss = self.vq(z)

        # x_recon = self.decoder(quantized)
        # recon_loss = F.mse_loss(x_recon, x).mean()
        
        # c0 = codes[:, :, 0]                      # (B, T)
        # transitions = (c0[:, 1:] != c0[:, :-1]).float()
        # smooth_loss = transitions.mean()

        # loss = recon_loss           * self.cfg.vqt_rec_weight \
        #      + vq_loss.mean()       * self.cfg.vqt_commit_weight \
        #      + smooth_loss          * self.cfg.vqt_smoothness_weight
        
        # traj_xy is (B, 2, T)
        x = traj_xy.to(self.device)
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 1)
        quantized, codes, vq_loss = self.vq(z_e)
        vq_loss = vq_loss.mean()

        q_z = quantized.permute(0, 2, 1)
        x_recon = self.decoder(q_z)
        recon_loss = F.mse_loss(x_recon, x)

        c0 = codes[:, :, 0]                      # (B, T)
        transitions = (c0[:, 1:] != c0[:, :-1]).float()
        smooth_loss = transitions.mean()

        # total loss
        loss = (
            recon_loss                  * self.cfg.vqt_rec_weight
            + vq_loss                   * self.cfg.vqt_commit_weight
            + smooth_loss               * self.cfg.vqt_smoothness_weight
        )

        loss_components = {
            "recon_loss": recon_loss,
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



