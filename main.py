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

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def load_data(data_type, min_objects_per_image, max_object_per_image, device, batch_size, num_workers, shuffle):
    image_dir = f'coco_dataset/images/{data_type}2017'
    instances_json = f'coco_dataset/annotations/instances_{data_type}2017.json'
    stuff_json = f'coco_dataset/annotations/stuff_{data_type}2017.json'
    dataset = CocoStuffDataset(image_dir, instances_json, stuff_json, device, min_objects_per_image=min_objects_per_image, max_objects_per_image=max_object_per_image)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, dataset


def train(device, run_name):
    train_set, val_set = "train", "val"
    # Preferált batch_size = 64
    batch_size = 64
    num_workers = 4
    learning_rate = 1e-3
    embedding_dim = 128
    num_epochs = 80
    sample_size = 8
    min_objects_per_image = 2
    max_objects_per_image = 4

    #val_dataloader, dataset = load_data(val_set, min_objects_per_image, max_objects_per_image, device, batch_size, num_workers, shuffle=False)
    train_dataloader, dataset = load_data(train_set, min_objects_per_image, max_objects_per_image, device, batch_size, num_workers, shuffle=False)
    total_samples = len(dataset)

    model = UNet(time_dim=embedding_dim, device=device).to(device)
    model.load_state_dict(torch.load("saved_models/model_base200_epoch_80.pt"))
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load("saved_models/ema_model_base200_epoch_80.pt"))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(torch.load("saved_models/optimizer_base200_epoch_80.pt"))
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    logger = SummaryWriter()


    # training loop
    n_iterations = math.ceil(total_samples / batch_size)
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (imgs, layout_emb) in enumerate(pbar):
            imgs = imgs.to(device)
            layout_emb = torch.reshape(layout_emb, (-1, embedding_dim)).to(device)
            t = diffusion.sample_timesteps(imgs.size(0)).to(device)
            x_t, noise = diffusion.noise_images(imgs, t)

            if np.random.random() < 0.1:
                layout_emb = None

            predicted_noise = model(x_t, t, layout_emb)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * n_iterations + i)

                
                
                
        if epoch % 10 == 9:
            graphs, sample_layouts = dataset.sample_graphs(sample_size)
            sampled_images = diffusion.sample(model, n=sample_size, layout=sample_layouts)
            ema_sampled_images = diffusion.sample(ema_model, n=sample_size, layout=sample_layouts)
            save_graphs(graphs, f"graphs/{run_name}_{epoch}_graphs.json")
            save_images(sampled_images, f"images/{run_name}_{epoch}.jpg")
            save_images(ema_sampled_images, f"images/{run_name}_{epoch}_ema.jpg")
        
        if epoch % 40 == 39:
            torch.save(model.state_dict(), f"saved_models/model_{run_name}_epoch_{epoch}.pt")
            torch.save(ema_model.state_dict(), f"saved_models/ema_model_{run_name}_epoch_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"saved_models/optimizer_{run_name}_epoch_{epoch}.pt")


def sample_images():
        device = 'cuda'
        train_set, val_set = 'train', 'val'
        batch_size = 64
        num_workers = 4
        learning_rate = 1e-3
        embedding_dim = 128
        num_epochs = 80
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
        save_graphs(graphs, f"test_graphs/val_graphs_320.json")
        save_images(sampled_images, f"test_images/val_model_320_img.jpg")
        save_images(ema_sampled_images, f"test_images/val_ema_model_320.jpg")


if __name__ == '__main__':
    train('cuda', 'base320')
