import torch
from torch.optim import AdamW
import torch.nn.functional as F

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

def to_unit_range(x, clip_min, clip_max):
    return 2.0 * (x - clip_min) / (clip_max - clip_min) - 1.0

def from_unit_range(x, clip_min, clip_max):
    return (x + 1.0) / 2.0 * (clip_max - clip_min) + clip_min

def no_slip_bc_loss(x0_pred_latent, vae, wall_mask, meta):
    """
    x0_pred_latent : (B, C_latent, h, w) estimated clean latent
    wall_mask      : (B, H, W) -- 1 where u=0,v=0 should hold, 0 elsewhere
    meta           : key into config["Stats"][meta] for un-normalization
    """
    decoded = vae.decode(x0_pred_latent)  # (B, 3, H, W), normalized (u, v, p)

    # stats = config["Stats"][meta]
    # u = decoded[:, 0:1] * stats["U_STD"] + stats["U_MEAN"]  # (B, 1, H, W)
    # v = decoded[:, 1:2] * stats["V_STD"] + stats["V_MEAN"]
    stats = config["Stats"][meta]
    u = from_unit_range(decoded[:, 0:1], stats["U_CLIP_MIN"], stats["U_CLIP_MAX"])
    v = from_unit_range(decoded[:, 1:2], stats["V_CLIP_MIN"], stats["V_CLIP_MAX"])

    wall_mask = wall_mask.unsqueeze(1).float()  # (B, 1, H, W)

    if wall_mask.shape[-2:] != u.shape[-2:]:
        wall_mask = F.interpolate(wall_mask, size=u.shape[-2:], mode="nearest")

    n_valid = wall_mask.sum().clamp_min(1.0)
    loss = ((u ** 2 + v ** 2) * wall_mask).sum() / n_valid
    return loss



def gradient_weight_map(feild, grad_weight=5.0, base_weight=1.0):
    """
    feild: (B, 3, H, W) ground-truth (u, v, p), pixel space, normalized.
    Returns (B, 1, H, W) weight map in [base_weight, base_weight+grad_weight].
    """
    uv = feild[:, 0:2]  # (B, 2, H, W) -- gradient computed from u, v only
    gy, gx = torch.gradient(uv, dim=(-2, -1))
    grad_mag = (gx.pow(2) + gy.pow(2)).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
    grad_mag = grad_mag / grad_mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return base_weight + (grad_weight * grad_mag)


def train_pinn(start_epoch, epochs, dataloader, dit, vae, scheduler, device, acc_steps, meta, **kwargs):
    optimizer = AdamW(dit.parameters(), lr=config["Training"]["learning_rate"], weight_decay=0)

    bc_weight = config["Training"].get("bc_loss_weight", 0.0)
    use_snr_weighting = config["Training"].get("bc_snr_weighting", True)
    grad_weight = config["Training"].get("grad_weight", 0.0)  # 0 = old behavior (no weighting)

    for param in vae.parameters():
        param.requires_grad = False

    # total optimizer steps, used to compute annealing progress below.
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * epochs
    global_step = start_epoch * steps_per_epoch

    # Only wire annealing in if the model actually has a Fourier embedding
    # with a `.progress` attribute (FourierDiT / MultiScaleFourierPositionEmbedding2D).
    # If not present, this is silently a no-op -- no crash for other model types.
    fourier_pos_embed = getattr(
        getattr(dit, "patch_embed_layer", None), "fourier_pos_embed", None
    )

    losses = []
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_bc_loss = 0.0
        step_count = 0

        for images, numbers, mask, wall_mask, _ in dataloader:
            step_count += 1
            global_step += 1
            images = images.to(device)
            numbers = numbers.float().to(device)
            wall_mask = wall_mask.to(device)

            # update annealing progress before this step's forward pass
            # if fourier_pos_embed is not None:
            #     fourier_pos_embed.progress = global_step / max(total_steps, 1)

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

            noise = torch.randn_like(z).to(device)
            t = torch.randint(0, 1000, (z.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(z, noise, t)
            pred = dit(noisy_im, t, numbers)

            # data (noise-matching) loss, masked to the domain
            sq = (pred - noise).pow(2)  # (B, C_latent, h, w)
            mask_ = mask.float().unsqueeze(1).to(device)
            mask_ = F.interpolate(mask_, size=sq.shape[-2:], mode="nearest")

            # fold in gradient-based weighting, downsampled to latent res 
            if grad_weight > 0:
                weight_map = gradient_weight_map(images, grad_weight=grad_weight)
                weight_map = F.interpolate(weight_map, size=sq.shape[-2:], mode="bilinear", align_corners=False)
                mask_ = mask_ * weight_map

            data_loss = (sq * mask_).sum() / mask_.sum().clamp_min(1.0)

            #  BC loss: estimate x0 from the noise prediction, decode,
            #  mask to wall pixels, penalize u,v != 0 
            if bc_weight > 0:
                sqrt_alpha = scheduler.sqrt_alpha_cum_prod.to(device)[t].reshape(-1, 1, 1, 1)
                sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alpha_cum_prod.to(device)[t].reshape(-1, 1, 1, 1)
                x0_pred = (noisy_im - sqrt_one_minus_alpha * pred) / sqrt_alpha

                bc_loss = no_slip_bc_loss(x0_pred, vae, wall_mask, meta=meta)

                if use_snr_weighting:
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
                progress_str = f" progress={fourier_pos_embed.progress:.2f}" if fourier_pos_embed is not None else ""
                print(f"[DiT] Epoch {epoch+1}/{epochs}  "
                      f"Step {step_count}/{len(dataloader)}  "
                      f"loss={avg_so_far:.6f} "
                      f"(data={epoch_data_loss/step_count:.6f}, "
                      f"bc={epoch_bc_loss/step_count:.6f}){progress_str}", flush=True)

        avg = epoch_loss / len(dataloader)
        losses.append(avg)

        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}  "
              f"(data={epoch_data_loss/len(dataloader):.6f}, "
              f"bc={epoch_bc_loss/len(dataloader):.6f})")

        if epoch % config["saves"]["DiT_Save_every"] == 0:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses