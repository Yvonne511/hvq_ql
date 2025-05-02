import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.vector_quantize_pytorch import ResidualVQ
import math

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
        self.input_dim = input_dim
        # 1D dilated CNN encoder: receptive field ~ (11 + 5*dilation)
        self.encoder = nn.Sequential(
            # input: (B, C=2, T) → (B, hidden_dim, T)
            nn.Conv1d(in_channels=input_dim,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      padding=get_padding(3)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=11, stride=1, padding=5),
            # → (B, latent_dim, T)
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=hidden_dim,
                      kernel_size=5,
                      padding=get_padding(5)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=latent_dim,
                      kernel_size=7,
                      padding=get_padding(7)),
        )

        # self.decoder = nn.Sequential(
        #     # quantized comes in as (B, latent_dim, T)
        #     nn.Conv1d(in_channels=latent_dim+input_dim,
        #               out_channels=hidden_dim,
        #               kernel_size=5,
        #               padding=get_padding(5)),
        #     nn.ReLU(),
        #     # → (B, input_dim=2, T)
        #     nn.Conv1d(in_channels=hidden_dim,
        #               out_channels=input_dim,
        #               kernel_size=3,
        #               padding=get_padding(3))
        # )

        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim + num_quantizers, hidden_dim, kernel_size=7, padding=get_padding(7)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=get_padding(5)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2*input_dim, kernel_size=3, padding=get_padding(3)),
        )

        # two-stage ResidualVQ: level-1 = coarse trend, level-2 = local detail
        self.vq = ResidualVQ(
            dim               = latent_dim,
            codebook_size     = codebook_size,
            num_quantizers    = num_quantizers,
            shared_codebook   = False,
            rotation_trick    = True,
        )

        # self.register_buffer("recon_ema",  torch.tensor(0.0))
        # self.register_buffer("vq_ema",     torch.tensor(0.0))
        # self.register_buffer("smooth_ema", torch.tensor(0.0))
        # self.register_buffer("ema_decay",  torch.tensor(0.99))
        # self.register_buffer("eps",        torch.tensor(1e-8))

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

        # q_z = quantized.permute(0, 2, 1)
        # x_recon = self.decoder(q_z)
        # recon_loss = F.mse_loss(x_recon, x)
        codes_z = codes.permute(0, 2, 1)
        W = 50
        B, D, T = codes_z.shape
        recon_loss = 0.0
        count = 0

        # permute quantized back to (B, D, T) so decoder conv1d can apply on time-axis

        for t in range(0, T - 2*W + 1, W):
            # input q: steps [t+W : t+2*W]
            # input x: steps [t : t+W]
            # pred x: steps [t+W : t+2*W]
            # chunk_q = q_z[:, :, t+W : t+2*W]          # (B, Latent D, W)
            chunk_q = codes_z[:, :, t : t+W]          # (B, num_quantizers, W)
            past_chunk = x[:, :, t : t+W]             # (B, Input D, W)
            d_in = torch.cat([past_chunk, chunk_q], dim=1)  # (B, num_quantizers + Input D, W)
            dec_out = self.decoder(d_in)           # (B, 2, W)
            μ = dec_out[:, :self.input_dim, :]        # (B,2,W)
            log2 = dec_out[:, self.input_dim:, :]    # (B,2,W)
            theta = torch.exp(0.5 * log2)
            true_chunk = x[:, :, t+W : t+2*W]         # (B, Input D, W)
            recon_loss += 0.5 * (
                  ((true_chunk - μ) / theta).pow(2)
                + 2*log2
                + math.log(2*math.pi)
            ).mean()
            count += 1
        recon_loss = recon_loss / max(1, count)

        c0 = codes[:, :, 0]                      # (B, T)
        transitions = (c0[:, 1:] != c0[:, :-1]).float()
        smooth_loss = transitions.mean()

        # if self.recon_ema == 0:
        #     self.recon_ema   = recon_loss.detach()
        #     self.vq_ema      = vq_loss.detach()
        #     self.smooth_ema  = smooth_loss.detach()

        # self.recon_ema   = self.ema_decay * self.recon_ema   + (1 - self.ema_decay) * recon_loss.detach()
        # self.vq_ema      = self.ema_decay * self.vq_ema      + (1 - self.ema_decay) * vq_loss.detach()
        # self.smooth_ema  = self.ema_decay * self.smooth_ema  + (1 - self.ema_decay) * smooth_loss.detach()

        # recon_loss  = recon_loss  / (self.recon_ema   + self.eps)
        # vq_loss     = vq_loss      / (self.vq_ema      + self.eps)
        # smooth_loss = smooth_loss  / (self.smooth_ema  + self.eps)

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



