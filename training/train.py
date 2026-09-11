import torch
import matplotlib.pyplot as plt

from VAE import VAE
from training import train_dit, train_pinn

from torch.utils.data import Dataset, DataLoader

import numpy as np
from Data import dataset_cfd

import json

from DiT_model import *
from DiT_model import FourierDiT
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
vae.load_state_dict(torch.load(f"{ROOT}/models/VAE.pth", map_location = device))

physics_informed = config["Training"].get("physics_informed", True)

if physics_informed:
    dit = FourierDiT(
            d_model               = config["DiT"]["d_model"],
            g_channels            = config["DiT"]["g_channels"],
            grid_size             = config["DiT"]["grid_size"],
            patch_size            = config["DiT"]["patch_size"],
            timestep_emb_dim      = config["DiT"]["timestep_emb_dim"],
            number_emb_dim        = config["DiT"]["number_emb_dim"],
            num_layers            = config["DiT"]["num_layers"],
            num_heads             = config["DiT"]["num_heads"],
            
            out_channels          = 3,    # (u, v, p)
            num_frequencies       = config["DiT"]["num_frequencies"], 
            sigmas                = config["DiT"]["sigmas"],
        ).to(device)
    train_fn = train_pinn
else:
    dit = DiT(d_model         = config["DiT"]["d_model"],
            g_channels        = config["DiT"]["g_channels"],
            grid_size         = config["DiT"]["grid_size"],
            patch_size        = config["DiT"]["patch_size"],
            timestep_emb_dim  = config["DiT"]["timestep_emb_dim"],
            number_emb_dim    = config["DiT"]["number_emb_dim"],
            num_layers        = config["DiT"]["num_layers"],
            num_heads         = config["DiT"]["num_heads"])
    train_fn = train_dit


dit = dit.to(device)
vae = vae.to(device)

########################################
########### Training DiT  ##############
########################################

print("Training DiT ......")

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

dataset    = dataset_cfd.dataset_csv(folder = f"{ROOT}/Data/Problems/{config['Data']['name']}", meta = config['Data']['meta'], grid_size = 256)
dataloader = DataLoader(dataset, batch_size = config["Training"]["batch_size"], shuffle=True, num_workers=2)

if config['saves']['load_model_for_traning']:
    dit.load_state_dict(torch.load(f"{ROOT}/{config['saves']['DiT_Path']}", map_location = device))
    
dit_losses = train_fn(start_epoch = 0, 
                      epochs      = config["Training"]["epochs"], 
                      dataloader  = dataloader, 
                      dit         = dit, 
                      vae         = vae, 
                      scheduler   = scheduler, 
                      device      = device, 
                      acc_steps   = config["Training"]["accumulation_step"], 
                      meta        = config['Data']['meta'])
