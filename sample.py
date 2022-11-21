import copy
import math
from diffusion.modules import UNet, EMA
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from diffusion.diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from data.coco import CocoStuffDataset
from data.utils import save_images, save_graphs
from torch.utils.data import DataLoader
from main import load_data

def sample_images():
        device = 'cuda'
        val_set = 'val'
        batch_size = 64
        num_workers = 4
        embedding_dim = 128
        sample_size = 50
        min_objects_per_image = 2
        max_objects_per_image = 4
        val_dataloader, dataset = load_data(val_set, min_objects_per_image, max_objects_per_image, device, batch_size, num_workers, shuffle=False)
        model = UNet(time_dim=embedding_dim, device=device).to(device)
        model.load_state_dict(torch.load("saved_models/model_base200_epoch_80.pt"))
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load("saved_models/ema_model_base200_epoch_80.pt"))
        diffusion = Diffusion(device=device)
        graphs, sample_layouts = dataset.sample_graphs(sample_size)
        sampled_images = diffusion.sample(model, n=sample_size, layout=sample_layouts)
        ema_sampled_images = diffusion.sample(ema_model, n=sample_size, layout=sample_layouts)
        save_graphs(graphs, f"sampled_graphs/val_graphs_320.json")
        save_images(sampled_images, f"sampled_images/val_model_320_img.jpg")
        save_images(ema_sampled_images, f"sampled_images/val_ema_model_320.jpg")


if __name__ == '__main__':
    sample_images()