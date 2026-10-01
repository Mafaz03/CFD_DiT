import torch
from torch.optim import AdamW
import torch.nn.functional as F

import json
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingLR


ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)


def train_dit(start_epoch, epochs, dataloader, dit, vae, scheduler, device, acc_steps, **kwargs):
    optimizer = AdamW(dit.parameters(), lr=config["Training"]["learning_rate"], weight_decay=0)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=epochs)
    # loss_fn   = torch.nn.MSELoss()

    for param in vae.parameters():
        param.requires_grad = False

    losses = []
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        step_count = 0
        for images, numbers, mask, _, _ in dataloader:
            step_count += 1
            images  = images.to(device)
            numbers = numbers.float().to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

            # Sample random noise
            noise = torch.randn_like(z).to(device)

            # Sample timestep
            t = torch.randint(0, 1000,(z.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(z, noise, t)


            mask_px = mask.to(device)                                   # (B, 256, 256), conditioning input
            pred = dit(noisy_im, t, numbers, mask_px)

            sq = (pred - noise).pow(2)
            mask_lat = F.interpolate(mask_px.float().unsqueeze(1), size=sq.shape[-2:], mode="nearest")
            sq = sq * mask_lat
            loss = sq.sum() / (mask_lat.sum() * sq.shape[1])

            
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if step_count % 50 == 0:
                avg_so_far = epoch_loss / step_count
                print(f"[DiT] Epoch {epoch+1}/{epochs}  "
                      f"Step {step_count}/{len(dataloader)}  "
                      f"loss={avg_so_far:.6f}", flush=True)

        scheduler_lr.step()
        avg = epoch_loss / len(dataloader)
        losses.append(avg)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")

        is_last = (epoch + 1) == epochs
        if (epoch + 1) % config["saves"]["DiT_Save_every"] == 0 or is_last:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses