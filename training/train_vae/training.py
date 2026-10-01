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

dataset    = dataset_cfd.dataset_csv(folder = f"{ROOT}/Data/Problems/{config['Data_VAE']['name']}", meta = config['Data_VAE']['meta'])
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True,)# num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = VAE(
    device = device, 
    freeze = False, 
    scaling_factor = config["VAE"]["scaling_factor"], 
    path = f"{ROOT}/pretrained/sd-vae-ft-mse"
    ).to(device)
# use path = `stabilityai/sd-vae-ft-mse` for pulling weights from the internet

if config["saves"]["load_model_for_traning_vae"]:
    vae.load_state_dict(torch.load(f"{ROOT}/{config['saves']['VAE_Path']}", map_location = device))

optimizer = torch.optim.Adam(
    vae.parameters(),
    lr = lr
)

save_path = Path(ROOT) / config["saves"]["VAE_Path"]
save_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Relative (scale-free) masked reconstruction loss
# Every sample/channel counts by relative error, so low-Re fields
# are not drowned out by high-Re ones.
# ---------------------------------------------------------------
def vae_recon_loss(rec, tgt, mask, eps=1e-3):
    m = mask.float().unsqueeze(1)                              # [B, 1, H, W]
    n = m.sum((2, 3)).clamp(min=1)                             # [B, 1]
    mean = (tgt * m).sum((2, 3)) / n                           # [B, C]
    var = (((tgt - mean[..., None, None]) ** 2) * m).sum((2, 3)) / n # [B, C]
    std = (var + eps).sqrt()                                   # [B, C]

    diff = rec - tgt
    mse = ((diff ** 2) * m).sum((2, 3)) / n
    l1  = (diff.abs() * m).sum((2, 3)) / n

    # gradient term: keeps sharp structure that MSE blurs
    mx = m[..., 1:] * m[..., :-1]
    my = m[:, :, 1:, :] * m[:, :, :-1, :]
    gx = ((rec[..., 1:] - rec[..., :-1]) - (tgt[..., 1:] - tgt[..., :-1])).abs()
    gy = ((rec[:, :, 1:] - rec[:, :, :-1]) - (tgt[:, :, 1:] - tgt[:, :, :-1])).abs()
    C = tgt.shape[1]
    gl = ((gx * mx / std[..., None, None]).sum() / (mx.sum() * C)
        + (gy * my / std[..., None, None]).sum() / (my.sum() * C))

    rel_mse = (mse / (std ** 2)).mean()
    rel_l1  = (l1 / std).mean()
    return rel_mse + 0.5 * rel_l1 + 0.5 * gl


# ---------------------------------------------------------------
# Relative RMSE per Re bin (the metric that actually shows if low Re works)
# ---------------------------------------------------------------
@torch.no_grad()
def eval_by_re(vae, dataloader, device, n_bins=6):
    vae.eval()
    rows = []
    for images, numbers, mask, _, _ in dataloader:
        images = images.to(device)
        mu, _ = vae.encode(images)
        rec = vae.decode(mu)
        m = mask.float().unsqueeze(1).to(device)
        n = m.sum((2, 3)).clamp(min=1)
        mean = (images * m).sum((2, 3)) / n
        var = (((images - mean[..., None, None]) ** 2) * m).sum((2, 3)) / n
        err = (((rec - images) ** 2) * m).sum((2, 3)) / n
        rel = (err / var.clamp(min=1e-8)).sqrt().cpu()          # [B,3]
        for re, r in zip(numbers.view(-1).tolist(), rel.tolist()):
            rows.append([re] + r)
    rows = np.array(rows); rows = rows[np.argsort(rows[:, 0])]
    for c in np.array_split(rows, n_bins):
        print(f"[VAE eval] Re~{c[:,0].mean():+.2f}  rel-RMSE u={c[:,1].mean():.3f} v={c[:,2].mean():.3f} p={c[:,3].mean():.3f}")
    vae.train()


vae.train()

optimizer.zero_grad()

for epoch in range(num_epochs):
    total_loss = 0.0
    n_optim_steps = 0
    for step, (images, numbers, mask, _, _) in enumerate(dataloader):
        images = images.to(device)
        
        reconstructed, mu, logvar = vae(images)

        recon_loss = vae_recon_loss(reconstructed, images, mask.to(device))

        kl_loss = 0.5 * torch.sum(logvar.exp() + mu.pow(2) -1 - logvar)
        kl_loss = kl_loss / mu.shape[0]     # average over batch size

        loss = recon_loss + kl_weight * kl_loss

        (loss / acc_steps).backward()
        if (step + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            n_optim_steps += 1

        total_loss += loss.item()

    # flush whatever's left so gradients never leak across epoch boundaries
    if (step + 1) % acc_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        n_optim_steps += 1


    print(f"[VAE] Epoch {epoch+1} finished, Average Loss: {total_loss/(step+1):.6f}, "
          f"optim steps: {n_optim_steps}")

    is_last = (epoch + 1) == num_epochs
    if (epoch + 1) % config["saves"]["VAE_Save_every"] == 0 or is_last:
        torch.save(vae.state_dict(), save_path)
        print(f"[VAE] Saved checkpoint to {save_path} (epoch {epoch+1})")
        eval_by_re(vae, dataloader, device)