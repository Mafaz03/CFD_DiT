import torch
from torch.optim import AdamW
import torch.nn.functional as F

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)


# ---------------------------------------------------------------------
# No collate function needed anymore: wall_mask is (H, W), same shape
# as `mask` every time, so the default DataLoader collate stacks it
# into (B, H, W) automatically just like every other fixed-shape field.
# ---------------------------------------------------------------------


def no_slip_bc_loss(x0_pred_latent, vae, wall_mask, meta):
    """
    x0_pred_latent : (B, C_latent, h, w) estimated clean latent
    wall_mask      : (B, H, W) -- 1 where u=0,v=0 should hold, 0 elsewhere
    meta           : key into config["Stats"][meta] for un-normalization
    """
    decoded = vae.decode(x0_pred_latent)  # (B, 3, H, W), normalized (u, v, p)

    stats = config["Stats"][meta]
    u = decoded[:, 0:1] * stats["U_STD"] + stats["U_MEAN"]  # (B, 1, H, W)
    v = decoded[:, 1:2] * stats["V_STD"] + stats["V_MEAN"]

    wall_mask = wall_mask.unsqueeze(1).float()  # (B, 1, H, W)

    # In case decoded resolution ever differs from wall_mask's native
    # resolution, align them the same way `mask` is already aligned to
    # the latent resolution for the data loss below.
    if wall_mask.shape[-2:] != u.shape[-2:]:
        wall_mask = F.interpolate(wall_mask, size=u.shape[-2:], mode="nearest")

    n_valid = wall_mask.sum().clamp_min(1.0)
    loss = ((u ** 2 + v ** 2) * wall_mask).sum() / n_valid
    return loss


def train_pinn(start_epoch, epochs, dataloader, dit, vae, scheduler, device, acc_steps, meta, **kwargs):
    optimizer = AdamW(dit.parameters(), lr=config["Training"]["learning_rate"], weight_decay=0)

    bc_weight = config["Training"].get("bc_loss_weight", 0.0)
    use_snr_weighting = config["Training"].get("bc_snr_weighting", True)

    for param in vae.parameters():
        param.requires_grad = False

    losses = []
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_bc_loss = 0.0
        step_count = 0

        for images, numbers, mask, wall_mask, uv_mask in dataloader:
            step_count += 1
            images = images.to(device)
            numbers = numbers.float().to(device)
            wall_mask = wall_mask.to(device)
            # uv_mask left unused for now, same as before

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

            noise = torch.randn_like(z).to(device)
            t = torch.randint(0, 1000, (z.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(z, noise, t)
            pred = dit(noisy_im, t, numbers)

            # --- data (noise-matching) loss, masked to the domain ---
            sq = (pred - noise).pow(2)  # (B, C_latent, h, w)
            mask_ = mask.float().unsqueeze(1).to(device)
            mask_ = F.interpolate(mask_, size=sq.shape[-2:], mode="nearest")
            data_loss = (sq * mask_).sum() / mask_.sum()

            # --- BC loss: estimate x0 from the noise prediction, decode,
            #     mask to wall pixels, penalize u,v != 0 ---
            if bc_weight > 0:
                sqrt_alpha = scheduler.sqrt_alpha_cum_prod.to(device)[t].reshape(-1, 1, 1, 1)
                sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alpha_cum_prod.to(device)[t].reshape(-1, 1, 1, 1)
                x0_pred = (noisy_im - sqrt_one_minus_alpha * pred) / sqrt_alpha

                bc_loss = no_slip_bc_loss(x0_pred, vae, wall_mask, meta=meta)

                if use_snr_weighting:
                    # Down-weight the BC loss at high t, where x0_pred is a
                    # poor estimate of the clean sample (alpha_cum_prod -> 0
                    # as t -> T).
                    snr_w = (sqrt_alpha.reshape(-1) ** 2).mean()
                    bc_term = bc_weight * snr_w * bc_loss
                else:
                    bc_term = bc_weight * bc_loss
            else:
                bc_loss = torch.tensor(0.0, device=device)
                bc_term = torch.tensor(0.0, device=device)

            loss = data_loss + bc_term
            (loss / acc_steps).backward()

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_data_loss += data_loss.item()
            epoch_bc_loss += bc_loss.item()

            if step_count % 50 == 0:
                avg_so_far = epoch_loss / step_count
                print(f"[DiT] Epoch {epoch+1}/{epochs}  "
                      f"Step {step_count}/{len(dataloader)}  "
                      f"loss={avg_so_far:.6f} "
                      f"(data={epoch_data_loss/step_count:.6f}, "
                      f"bc={epoch_bc_loss/step_count:.6f})", flush=True)

        avg = epoch_loss / len(dataloader)
        losses.append(avg)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}  "
              f"(data={epoch_data_loss/len(dataloader):.6f}, "
              f"bc={epoch_bc_loss/len(dataloader):.6f})")

        if epoch % config["saves"]["DiT_Save_every"] == 0:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses