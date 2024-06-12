import random
import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path

import datasets as ds
import einops
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from rich import print
from simple_parsing import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import utils.inference
import utils.option
import utils.render
import utils.training
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.efficient_unet import EfficientUNet
from models.refinenet import LiDARGenRefineNet
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles


def generate_random_point_cloud(num_points=1000, resolution=(64, 1024)):
    """
    生成随机点云数据，并将其投影到深度图和反射图上

    Args:
        num_points (int): 点的数量
        resolution (tuple): 图像分辨率 (H, W)

    Returns:
        dict: 包含随机点云数据的字典
    """
    H, W = resolution
    points = np.random.rand(num_points, 4).astype(np.float32)  # 随机生成 x, y, z, intensity
    xyz = points[:, :3]
    intensity = points[:, 3]

    # 投影到图像平面
    depth = xyz[:, 2]  # 使用 z 作为深度
    reflectance = intensity

    # 随机生成 grid 的索引
    grid_h = np.random.randint(0, H, size=(num_points,))
    grid_w = np.random.randint(0, W, size=(num_points,))

    depth_img = np.zeros((H, W), dtype=np.float32)
    reflectance_img = np.zeros((H, W), dtype=np.float32)

    depth_img[grid_h, grid_w] = depth
    reflectance_img[grid_h, grid_w] = reflectance

    data = {
        'sample_id': list(range(num_points)),
        'xyz': xyz,  # x, y, z
        'depth': depth_img,  # 结果维度为 (H, W)
        'reflectance': reflectance_img  # 结果维度为 (H, W)
    }
    return data

class PointCloudDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sample_id'])

    def __getitem__(self, idx):
        return {
            'sample_id': self.data['sample_id'][idx],
            'xyz': torch.tensor(self.data['xyz'][idx], dtype=torch.float32),
            'depth': torch.tensor(self.data['depth'], dtype=torch.float32).unsqueeze(0),
            'reflectance': torch.tensor(self.data['reflectance'], dtype=torch.float32).unsqueeze(0)
        }
def train(cfg: utils.option.Config):
    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.training.output_dir) / cfg.data.dataset / cfg.data.projection

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
    ]

    if cfg.model.architecture == "efficient_unet":
        model = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            ring=True,
        )
    elif cfg.model.architecture == "refinenet":
        model = LiDARGenRefineNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            channel_multiplier=cfg.model.channel_multiplier,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    if "spherical" in cfg.data.projection:
        model.coords = get_hdl64e_linear_ray_angles(*cfg.data.resolution)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if accelerator.is_main_process:
        print(f"number of parameters: {utils.inference.count_parameters(model):,}")

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            num_training_steps=cfg.diffusion.num_training_steps,
        )
    elif cfg.diffusion.timestep_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")
    ddpm.train()
    ddpm.to(device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=cfg.training.lr_warmup_steps
                              * cfg.training.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    )
    lidar_utils.to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    random_data = generate_random_point_cloud()
    dataset = PointCloudDataset(random_data)

    if accelerator.is_main_process:
        print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=8,  # 调整batch_size
        shuffle=True,
        num_workers=1,  # 调整num_workers为1
        drop_last=True,
        pin_memory=True,
    )




    lr_scheduler = utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps
                         * cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.num_steps
                           * cfg.training.gradient_accumulation_steps,
    )

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utility
    # =================================================================================

    def preprocess(batch):
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x


    def split_channels(image: torch.Tensor):
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        image = lidar_utils.denormalize(image)
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f"{tag}/depth"] = utils.render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = utils.render.colorize(
                metric / lidar_utils.max_depth
            )
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -utils.render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            bev = utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
                t=torch.tensor([0, 0, 1.0]).to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f"{tag}/reflectance"] = utils.render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================

    progress_bar = tqdm(
        range(cfg.training.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    while global_step < cfg.training.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_0 = preprocess(batch)



            with accelerator.accumulate(ddpm):
                loss = ddpm(x_0=x_0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                if global_step == 1:
                    log_images(x_0, "image", global_step)

                if global_step % cfg.training.steps_save_image == 0:
                    ddpm_ema.ema_model.eval()
                    sample = ddpm_ema.ema_model.sample(
                        batch_size=cfg.training.batch_size_eval,
                        num_steps=cfg.diffusion.num_sampling_steps,
                        rng=torch.Generator(device=device).manual_seed(0),
                    )
                    log_images(sample, "sample", global_step)

                if global_step % cfg.training.steps_save_model == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.training.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(utils.option.Config, dest="cfg")
    train(parser.parse_args().cfg)
