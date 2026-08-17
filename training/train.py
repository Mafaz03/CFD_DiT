import torch
import matplotlib.pyplot as plt

from VAE import VAE
from training import train

from torch.utils.data import Dataset, DataLoader

import numpy as np
from Data import dataset_cfd

import json

from DiT_model import *
from DDPM import *


ROOT = Path(__file__).resolve().parent.parent

########################################
########## Loading modules #############
########################################

print("Loading modules.....")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device found: {device}")


scheduler = LinearNoiseScheduler(num_timesteps  = config["Scheduler"]["num_timesteps"],
                                     beta_start = config["Scheduler"]["beta_start"],
                                     beta_end   = config["Scheduler"]["beta_end"])


vae = VAE(device = device, freeze = True, scaling_factor = config["VAE"]["scaling_factor"], path = f"{ROOT}/pretrained/sd-vae-ft-mse").to(device)

dit = DiT(d_model           = config["DiT"]["d_model"],
          g_channels        = config["DiT"]["g_channels"],
          grid_size         = config["DiT"]["grid_size"],
          patch_size        = config["DiT"]["patch_size"],
          timestep_emb_dim  = config["DiT"]["timestep_emb_dim"],
          number_emb_dim    = config["DiT"]["number_emb_dim"],
          num_layers        = config["DiT"]["num_layers"],
          num_heads         = config["DiT"]["num_heads"])


dit = dit.to(device)
vae = vae.to(device)

########################################
########### Training DiT  ##############
########################################

print("Training DiT ......")

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

dataset    = dataset_cfd.dataset_csv(folder = f"{ROOT}/Data/Problems/{config['Data']['name']}")
dataloader = DataLoader(dataset, batch_size = config["Training"]["batch_size"], shuffle=True, num_workers=2)

dit_losses = train(0, config["Training"]["epochs"], dataloader, dit, vae, scheduler, device, acc_steps = config["Training"]["accumulation_step"])