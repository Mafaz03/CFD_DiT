import torch
from torch.optim import AdamW
import torch.nn.functional as F

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)


def train(start_epoch, epochs, dataloader, dit, vae, scheduler, device, acc_steps):
    optimizer = AdamW(dit.parameters(), lr=config["Training"]["learning_rate"], weight_decay=0)
    # loss_fn   = torch.nn.MSELoss()

    for param in vae.parameters():
        param.requires_grad = False

    losses = []
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        step_count = 0
        for images, numbers, mask in dataloader:
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

            pred = dit(noisy_im, t, numbers)
            
            # loss = loss_fn(pred, noise)
            sq = (pred - noise).pow(2) # [B, 4, 32, 32]

            mask = mask.float().unsqueeze(1).to(device)                     # [B, 1, 256, 256]
            mask = F.interpolate(mask, size=sq.shape[-2:], mode="nearest")  # [B, 1, 32, 32]
            sq = sq * mask
            loss = sq.sum() / mask.sum()
            
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
                
        avg = epoch_loss / len(dataloader)
        losses.append(avg)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")

        if epoch % config["saves"]["DiT_Save_every"] == 0:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses