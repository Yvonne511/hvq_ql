
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
# import cv2
from PIL import Image
from matplotlib import cm
import torch
import os
import logging

logger = logging.getLogger(__name__)

# def create_timeline(skill_ids, width, height=20):
#     timeline = np.zeros((height, width, 3), dtype=np.uint8)
#     cmap = cm.get_cmap("tab20")

#     for i, sid in enumerate(skill_ids):
#         rgb = np.array(cmap(sid % 20)[:3]) * 255
#         timeline[:, i, :] = rgb.astype(np.uint8)

#     return timeline

# def combine_frame_with_timeline(frame, timeline, t):
#     h, _ = timeline.shape[:2]
#     timeline_overlay = timeline.copy()
#     cv2.line(timeline_overlay, (t, 0), (t, h - 1), color=(128, 128, 128), thickness=2)
#     combined = np.vstack([frame, timeline_overlay])
#     return combined

# Dummy 2D env visualization
def visual_2d(env, model, dset, epoch, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "segmentation_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get background frame and maze bounds
    env.reset()
    bg_frame = env.render().copy()
    h, w = bg_frame.shape[:2]
    unit = env.unwrapped._maze_unit
    offset_x, offset_y = env.unwrapped._offset_x, env.unwrapped._offset_y
    maze_h, maze_w = env.unwrapped.maze_map.shape

    xmin = -offset_x + unit * 0.5 # fix latter for all envs
    xmax = maze_w * unit - offset_x - unit * 1.5
    ymin = -offset_y + unit * 0.5
    ymax = maze_h * unit - offset_y - unit * 1.5
    extent = [xmin, xmax, ymin, ymax]
    # extent = [-2, 22, -2, 22]

    # Prepare color map
    cmap = plt.get_cmap('tab10', cfg.num_classes)

    model.eval()
    with torch.no_grad():
        # --- 1) 3x3 grid of individual trajectories ---
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            ax.imshow(bg_frame, extent=extent, origin='upper')
            if idx < len(dset):
                data = dset[idx]
                obs = data["obs"]
                feats = obs.permute(0, 2, 1).to(device)
                mask = torch.ones_like(feats).to(device)
                labels = model.get_labels(feats, mask)\
                            .squeeze().detach().cpu().numpy()
                obs_np = data["xy"].squeeze(0).detach().cpu().numpy()
                # Plot segments by class
                for cls in range(cfg.num_classes):
                    cls_mask = labels == cls
                    ax.plot(obs_np[cls_mask, 0], obs_np[cls_mask, 1],
                            '.', ms=2, label=f"cls {cls}", color=cmap(cls))
            ax.set_axis_off()
        fig.tight_layout()
        grid_path = os.path.join(save_dir, f"trajectories_grid_epoch_{epoch}.png")
        fig.savefig(grid_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # --- 2) Overlay of all trajectories ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(bg_frame, extent=extent, origin='upper')
        for idx in range(len(dset)):
            data = dset[idx]
            obs = data["obs"]
            feats = obs.permute(0, 2, 1).to(device)
            mask = torch.ones_like(feats).to(device)
            labels = model.get_labels(feats, mask).squeeze().detach().cpu().numpy()
            obs_np = data["xy"].squeeze(0).detach().cpu().numpy()
            for cls in range(cfg.num_classes):
                cls_mask = labels == cls
                ax.plot(obs_np[cls_mask, 0], obs_np[cls_mask, 1],
                        '.', ms=2, alpha=0.3, color=cmap(cls))
        ax.set_axis_off()
        overlay_path = os.path.join(save_dir, f"trajectories_overlay_epoch_{epoch}.png")
        fig.savefig(overlay_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        logger.info(f"Saved grid of trajectories to {grid_path}")
        logger.info(f"Saved overlay of trajectories to {overlay_path}")

def visual_3d(env, model, dset, epoch, cfg):
    # for obs_seq, _ in dset:
    #     obs_seq = obs_seq.squeeze(0).numpy()  # (T, D)
    #     skill_ids = model.predict_cluster(obs_seq)  # fake API
    #     frames = []

    #     timeline = create_timeline(skill_ids, width=len(skill_ids), height=20)

    #     for t in range(len(obs_seq)):
    #         # Fake 3D frame - we’ll draw a point in 3D space and render a projection
    #         fig = plt.figure(figsize=(3, 3))
    #         ax = fig.add_subplot(111, projection='3d')
    #         traj = obs_seq[:t+1]
    #         ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=cm.tab20(skill_ids[t] % 20))
    #         ax.set_xlim(-10, 10)
    #         ax.set_ylim(-10, 10)
    #         ax.set_zlim(-10, 10)
    #         ax.axis("off")
    #         fig.canvas.draw()
            
    #         # Convert matplotlib figure to numpy frame
    #         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         plt.close(fig)

    #         frame = combine_frame_with_timeline(img, timeline, t)
    #         frames.append(frame)

    #     # Save to video
    #     imageio.mimsave("visual_3d_output.mp4", frames, fps=10)
    #     break  # Only one trajectory
    pass