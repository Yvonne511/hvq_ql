
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
from tqdm import trange, tqdm
from env.ogbench.ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from env.ogbench.ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from env.ogbench.ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from env.ogbench.ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
from env.ogbench.ogbench.manipspace import lie
from moviepy.editor import ImageSequenceClip
import cv2

logger = logging.getLogger(__name__)

def get_label_colors(num_classes, scale=255):
    """Returns a dict mapping class index → RGB tuple of ints (for OpenCV & Matplotlib)"""
    cmap = plt.get_cmap("tab10")
    colors_rgb = {}
    colors_float = {}

    for i in range(num_classes):
        rgb_float = np.array(cmap(i)[:3])  # (R, G, B) in [0, 1]
        rgb_int = tuple((rgb_float * scale).astype(int))  # (R, G, B) in [0, 255]
        colors_rgb[i] = rgb_int
        colors_float[i] = rgb_float

    return colors_rgb, colors_float

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
                obs = data["obs_vqvae"]
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
            obs = data["obs_vqvae"]
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
    
    if cfg.env == "antmaze":
        # visulize each one of the labelled action
        task_id = 5
        for label_id in range(cfg.num_classes):
            frames = []
            data = dset[0]  # or a random one
            obs = data["obs_vqvae"]
            feats = obs.permute(0, 2, 1).to(device)
            mask = torch.ones_like(feats).to(device)
            labels = model.get_labels(feats, mask)\
                            .squeeze().detach().cpu().numpy()
            ob, info = env.reset(options=dict(task_id=task_id))
            num_steps = 100
            selected_indices = np.where(labels == label_id)[0]
            selected_actions = obs.squeeze()[selected_indices]
            for action in selected_actions:
                action = np.clip(action, -1, 1)
                ob, _, _, _, info = env.step(action)
                frame = env.render()
                frames.append(frame)
            video_path = os.path.join(save_dir, f"label_{label_id}_action_{epoch}.mp4")
            clip = ImageSequenceClip(frames, fps=60)
            clip.write_videofile(video_path, codec="libx264")
            logger.info(f"Saved video for label {label_id} to {video_path}")


def save_video_moviepy(frames, path, fps=10):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path, codec='libx264')

def add_labels_to_frames(frames, labels, label_colors, font_scale=0.6, font_thickness=2):
    color_map = plt.get_cmap('tab10')
    labeled_frames = []

    for i, frame in enumerate(frames):
        label = int(labels[i])
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Label color and text
        color = tuple(int(c) for c in label_colors[label])
        text = f"Label: {label}"

        # Draw filled rectangle for better visibility
        cv2.rectangle(frame_bgr, (0, 0), (130, 30), color, thickness=-1)

        # Put label text
        cv2.putText(frame_bgr, text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Convert back to RGB
        labeled_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        labeled_frames.append(labeled_frame)

    return labeled_frames

def save_label_timeline(labels, save_path, label_colors, title="Label Timeline"):
    """
    Save a label timeline image where each timestep is a colored segment.
    
    Args:
        labels (np.ndarray): array of shape (T,), one label per timestep.
        save_path (str): path to save the timeline image.
        title (str): title for the plot.
    """
    T = len(labels)
    img = np.zeros((20, T, 3), dtype=np.uint8)

    for t, label in enumerate(labels):
        rgb = label_colors[int(label)]
        if isinstance(rgb[0], float):  # if using float RGB (e.g., from cmap)
            rgb = (np.array(rgb) * 255).astype(np.uint8)
        img[:, t, :] = rgb

    plt.figure(figsize=(10, 1))
    plt.imshow(img)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visual_3d(env, model, dset, epoch, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "segmentation_plots"
    os.makedirs(save_dir, exist_ok=True)
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = len(task_infos)
    should_render = True
    for task_id in trange(1, num_tasks + 1):
        if task_id == 1:
            continue
        if task_id == 2:
            continue
        if task_id == 3:
            continue
        task_name = task_infos[task_id - 1]['task_name']

        ob, info = env.reset(options=dict(task_id=task_id))
        agents = {
            'cube': CubePlanOracle(env=env, noise=0, noise_smoothing=0),
            'button': ButtonPlanOracle(env=env, noise=0, noise_smoothing=0),
            'drawer': DrawerPlanOracle(env=env, noise=0, noise_smoothing=0),
            'window': WindowPlanOracle(env=env, noise=0, noise_smoothing=0),
        }
        agent = agents['button']
        has_yet_reset = True

        step = 0
        obs = []
        actions = []
        frames = []
        phase = 0
        for _ in range(400):
            if has_yet_reset:
                if task_id == 4:
                    if phase == 0:
                        env.unwrapped._target_button = 0
                        info['privileged/target_button'] = env.unwrapped._target_button
                        info['privileged/target_button_state'] = env.unwrapped._target_button_states[env.unwrapped._target_button]
                        info['privileged/target_button_top_pos'] = env.unwrapped._data.site_xpos[
                            env.unwrapped._button_site_ids[env.unwrapped._target_button]
                        ].copy()
                    elif phase == 1:
                        info['privileged/target_drawer_pos'] = -0.16
                        info['privileged/target_drawer_handle_pos'] = env.unwrapped._data.site_xpos[env.unwrapped._drawer_target_site_id].copy() + [0, 0.16, 0]
                    elif phase == 2:
                        target_mocap_id = env.unwrapped._cube_target_mocap_ids[env.unwrapped._target_block]
                        info['privileged/target_block'] = env.unwrapped._target_block
                        info['privileged/target_block_pos'] = env.unwrapped._drawer_center.copy()
                        info['privileged/target_block_yaw'] = np.array(
                            [lie.SO3(wxyz=env.unwrapped._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
                        )
                    elif phase == 3:
                        info['privileged/target_drawer_pos'] = 0.0
                        info['privileged/target_drawer_handle_pos'] = env.unwrapped._data.site_xpos[env.unwrapped._drawer_target_site_id].copy()
                elif task_id == 5:
                    if phase == 0:
                        env.unwrapped._target_button = 0
                        info['privileged/target_button'] = 0
                        info['privileged/target_button_state'] = 1
                        info['privileged/target_button_top_pos'] = env.unwrapped._data.site_xpos[
                            env.unwrapped._button_site_ids[env.unwrapped._target_button]
                        ].copy()
                    elif phase == 1:
                        info['privileged/target_drawer_pos'] = -0.16
                        info['privileged/target_drawer_handle_pos'] = env.unwrapped._data.site_xpos[env.unwrapped._drawer_target_site_id].copy() + [0, 0.16, 0]
                    elif phase == 2:
                        env.unwrapped._target_button = 1
                        info['privileged/target_button'] = 1
                        info['privileged/target_button_state'] = 1
                        info['privileged/target_button_top_pos'] = env.unwrapped._data.site_xpos[
                            env.unwrapped._button_site_ids[env.unwrapped._target_button]
                        ].copy()
                    elif phase == 3:
                        target_mocap_id = env.unwrapped._cube_target_mocap_ids[env.unwrapped._target_block]
                        info['privileged/target_block'] = env.unwrapped._target_block
                        info['privileged/target_block_pos'] = env.unwrapped._drawer_center.copy()
                        info['privileged/target_block_yaw'] = np.array(
                            [lie.SO3(wxyz=env.unwrapped._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
                        )
                    elif phase == 4:
                        info['privileged/target_drawer_pos'] = 0.0
                        info['privileged/target_drawer_handle_pos'] = env.unwrapped._data.site_xpos[env.unwrapped._drawer_target_site_id].copy()
                    elif phase == 5:
                        info['privileged/target_window_pos'] = 0.2
                        info['privileged/target_window_handle_pos'] = env.unwrapped._data.site_xpos[env.unwrapped._window_target_site_id].copy()
                    elif phase == 6:
                        env.unwrapped._target_button = 0
                        info['privileged/target_button'] = 0
                        info['privileged/target_button_state'] = 0
                        info['privileged/target_button_top_pos'] = env.unwrapped._data.site_xpos[
                            env.unwrapped._button_site_ids[env.unwrapped._target_button]
                        ].copy()
                    elif phase == 7:
                        env.unwrapped._target_button = 1
                        info['privileged/target_button'] = 1
                        info['privileged/target_button_state'] = 0
                        info['privileged/target_button_top_pos'] = env.unwrapped._data.site_xpos[
                            env.unwrapped._button_site_ids[env.unwrapped._target_button]
                        ].copy()
                agent.reset(ob, info)
                has_yet_reset = False

            action = agent.select_action(ob, info)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            ob, _, _, _, info = env.step(action)
            obs.append(ob)
            actions.append(action)
            step += 1

            if agent.done:
                # print('done', step)
                if task_id == 4:
                    phase += 1
                    if phase == 1:
                        agent = agents['drawer']
                    elif phase == 2:
                        agent = agents['cube']
                    elif phase == 3:
                        agent = agents['drawer']
                    else:
                        break
                elif task_id == 5:
                    phase += 1
                    if phase == 1:
                        agent = agents['drawer']
                    elif phase == 2:
                        agent = agents['button']
                    elif phase == 3:
                        agent = agents['cube']
                    elif phase == 4:
                        agent = agents['drawer']
                    elif phase == 5:
                        agent = agents['window']
                    elif phase == 6:
                        agent = agents['button']
                    elif phase == 7:
                        agent = agents['button']
                    else:
                        break
                has_yet_reset=True

            frames.append(env.render())

        if task_id == 4:
            target_ep_len = 225
        elif task_id == 5:
            target_ep_len = 375
        while len(frames) < target_ep_len:
            action = np.zeros(5)
            action[0] = -1
            action[1] = -1
            action[2] = 1
            env.step(action)
            frames.append(env.render())
            
        obs = np.array(obs)
        actions = np.array(actions)
        T_orig = actions.shape[0]
        action_dim = actions.shape[1]
        T_target = 1000

        # Pad actions to 1000 using empty action
        if T_orig < T_target:
            repeat_factor = (T_target + T_orig - 1) // T_orig  # ceiling division
            actions_repeated = np.tile(actions, (repeat_factor, 1))
            actions_padded = actions_repeated[:T_target]
        else:
            actions_padded = actions[:T_target]
        model.eval()
        with torch.no_grad():
            act_tensor = torch.tensor(actions_padded, dtype=torch.float32).unsqueeze(0).to(device)
            feats = act_tensor.permute(0, 2, 1).to(device)
            mask = torch.ones_like(feats).to(device)
            labels = model.get_labels(feats, mask)\
                        .squeeze().detach().cpu().numpy()
        frames = frames[:len(actions)]
        num_classes = cfg.num_classes
        labels = labels[:len(actions)]
        label_colors_rgb, label_colors_float = get_label_colors(num_classes)
        frames_with_labels = add_labels_to_frames(frames, labels, label_colors_rgb)
        label_timeline_path = os.path.join(save_dir, f"label_timeline_{task_name}_{epoch}.png")
        save_label_timeline(labels, label_timeline_path, label_colors_float, title=f"Label Timeline for {task_name}")
        save_video_moviepy(frames_with_labels, f'{save_dir}/{task_name}_{epoch}.mp4')

