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

    for param in vae.parameters():
        param.requires_grad = False

    train_size = int(0.7 * len(dataloader.dataset))
    test_size = len(dataloader.dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataloader.dataset, [train_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers
    )

    losses = []
    test_losses = []

    for epoch in range(start_epoch, epochs):
        dit.train()
        epoch_loss = 0.0
        step_count = 0

        for images, numbers, mask, _, _ in train_loader:
            step_count += 1
            images = images.to(device)
            numbers = numbers.float().to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

            noise = torch.randn_like(z).to(device)
            t = torch.randint(0, 1000, (z.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(z, noise, t)

            mask_px = mask.to(device)
            pred = dit(noisy_im, t, numbers, mask_px)

            sq = (pred - noise).pow(2)
            mask_lat = F.interpolate(
                mask_px.float().unsqueeze(1),
                size=sq.shape[-2:],
                mode="nearest"
            )
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
                      f"Step {step_count}/{len(train_loader)}  "
                      f"loss={avg_so_far:.6f}", flush=True)

        scheduler_lr.step()

        avg = epoch_loss / len(train_loader)
        losses.append(avg)

        dit.eval()
        test_loss = 0.0

        with torch.no_grad():
            for images, numbers, mask, _, _ in test_loader:
                images = images.to(device)
                numbers = numbers.float().to(device)

                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

                noise = torch.randn_like(z).to(device)
                t = torch.randint(0, 1000, (z.shape[0],)).to(device)

                noisy_im = scheduler.add_noise(z, noise, t)

                mask_px = mask.to(device)
                pred = dit(noisy_im, t, numbers, mask_px)

                sq = (pred - noise).pow(2)
                mask_lat = F.interpolate(
                    mask_px.float().unsqueeze(1),
                    size=sq.shape[-2:],
                    mode="nearest"
                )
                sq = sq * mask_lat
                loss = sq.sum() / (mask_lat.sum() * sq.shape[1])

                test_loss += loss.item()

        avg_test = test_loss / len(test_loader)
        test_losses.append(avg_test)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  "
              f"train_loss={avg:.6f}  test_loss={avg_test:.6f}")

        is_last = (epoch + 1) == epochs
        if (epoch + 1) % config["saves"]["DiT_Save_every"] == 0 or is_last:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses, test_losses