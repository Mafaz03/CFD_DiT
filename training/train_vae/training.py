import numpy as np
import os
import sys

from torch.utils.data import Dataset, DataLoader
from Data import dataset_cfd

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from VAE import VAE

from pathlib import Path
import json

ROOT = Path(os.getcwd())#.resolve().parent.parent

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

kl_weight  = config["VAE"]["kl_weight"]
num_epochs = config["Training_VAE"]["epochs"]
batch_size = config["Training_VAE"]["batch_size"]
lr         = config["Training_VAE"]["learning_rate"]
acc_steps  = config["Training_VAE"]["accumulation_step"]

dataset    = dataset_cfd.dataset_csv(folder = f"{ROOT}/Data/Problems/{config['Data']['name']}", meta = "Lid_Driven")
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True,)# num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = VAE(
    device = device, 
    freeze = False, 
    scaling_factor = config["VAE"]["scaling_factor"], 
    path = f"{ROOT}/pretrained/sd-vae-ft-mse"
    ).to(device)
# use path = `stabilityai/sd-vae-ft-mse` for pulling weights from the internet


optimizer = torch.optim.Adam(
    vae.parameters(),
    lr = lr
)

vae.train()

optimizer.zero_grad()

for epoch in range(num_epochs):
    total_loss = 0.0
    for step, (images, numbers, mask) in enumerate(dataloader):
        images = images.to(device)
        
        reconstructed, mu, logvar = vae(images)

        sq = (reconstructed - images).pow(2)
        mask_ = mask.float().unsqueeze(1).to(device)
        sq = sq * mask_
        recon_loss = sq.sum() / mask_.sum()

        kl_loss = 0.5 * torch.sum(logvar.exp() + mu.pow(2) -1 - logvar)
        kl_loss = kl_loss / mu.shape[0]     # average over batch size

        loss = recon_loss + kl_weight * kl_loss

        (loss / acc_steps).backward()
        if (step + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    print(
        f"[VAE] Epoch {epoch + 1} finished, "
        f"Average Loss: {total_loss / (step + 1):.6f}"
    )

    if epoch % config["saves"]["VAE_Save_every"] == 0:
                torch.save(vae.state_dict(), config["saves"]["VAE_Path"])