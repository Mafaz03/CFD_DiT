import torch
from torch.optim import AdamW
import torch.nn.functional as F

import json
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingLR


ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)


def to_unit_range(x, clip_min, clip_max):
    return 2.0 * (x - clip_min) / (clip_max - clip_min) - 1.0

def from_unit_range(x, clip_min, clip_max):
    return (x + 1.0) / 2.0 * (clip_max - clip_min) + clip_min



# Loss functions

def masked_mean_std(tgt, m, min_std=0.05):
    """Per-sample, per-channel mean / std of tgt inside mask m. tgt (b,C,H,W), m (b,1,H,W)."""
    n = m.sum((2, 3), keepdim=True).clamp_min(1)
    mean = (tgt * m).sum((2, 3), keepdim=True) / n
    std = (((tgt - mean) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt().clamp_min(min_std)
    return mean, std, n


def eps_loss_fn(pred, noise, mask_lat):
    """Masked MSE on the predicted noise (latent space), normalised per sample so every
    sample counts once regardless of domain size. Scalar."""
    denom = mask_lat.sum((1, 2, 3)).clamp_min(1) * pred.shape[1]
    return ((pred - noise).pow(2).sum((1, 2, 3)) / denom).mean()


def wall_loss_fn(d, wall, meta):
    """No-slip: u^2 + v^2 at wall pixels, physical units. d (b,3,H,W) decoded, wall (b,H,W). Returns (b,)."""
    s = config["Stats"][meta]
    u = from_unit_range(d[:, 0:1], s["U_CLIP_MIN"], s["U_CLIP_MAX"])
    v = from_unit_range(d[:, 1:2], s["V_CLIP_MIN"], s["V_CLIP_MAX"])
    wall = wall.unsqueeze(1).float()
    return ((u ** 2 + v ** 2) * wall).sum((1, 2, 3)) / wall.sum((1, 2, 3)).clamp_min(1)


def gradient_loss_fn(d, tgt, mask):
    """L1 error of finite-difference gradients vs ground truth, std-normalised, inside the domain. Returns (b,)."""
    m = mask.unsqueeze(1).float()
    mx = m[..., 1:] * m[..., :-1]
    my = m[:, :, 1:, :] * m[:, :, :-1, :]

    _, std, _ = masked_mean_std(tgt, m)
    std = std.detach()

    gx = ((d[..., 1:] - d[..., :-1]) - (tgt[..., 1:] - tgt[..., :-1])).abs() / std
    gy = ((d[:, :, 1:] - d[:, :, :-1]) - (tgt[:, :, 1:] - tgt[:, :, :-1])).abs() / std
    C = tgt.shape[1]
    return ((gx * mx).sum((1, 2, 3)) / (mx.sum((1, 2, 3)).clamp_min(1) * C)
            + (gy * my).sum((1, 2, 3)) / (my.sum((1, 2, 3)).clamp_min(1) * C))


def corr_loss_fn(d, tgt, mask):
    """1 - Pearson correlation inside the domain, per channel, averaged over channels. Returns (b,)."""
    m = mask.unsqueeze(1).float()
    d_mean, _, _ = masked_mean_std(d, m)
    t_mean, _, _ = masked_mean_std(tgt, m)
    dc = (d - d_mean) * m
    yc = (tgt - t_mean) * m
    corr = 1 - (dc * yc).sum((2, 3)) / ((dc.pow(2).sum((2, 3)) + 1e-8).sqrt() * (yc.pow(2).sum((2, 3)) + 1e-8).sqrt())
    return corr.mean(1)



def decode_losses(x0_pred, vae, wall, mask, tgt, meta):
    """Decode x0 once, evaluate all pixel-space losses on it. Each returned tensor is (b,)."""
    d = vae.decode(x0_pred)                                   # (b,3,H,W), normalised
    return (wall_loss_fn(d, wall, meta),
            gradient_loss_fn(d, tgt, mask),
            corr_loss_fn(d, tgt, mask))



# Helpers
def prepare_batch(images, numbers, mask, wall, vae, scheduler, device):
    images  = images.to(device)
    numbers = numbers.float().to(device)

    with torch.no_grad():
        mu, logvar = vae.encode(images)
        z = vae.reparameterize(mu, logvar)

    mask_px = mask.to(device).float()                                   # (B,H,W)
    cond = torch.stack([mask_px, wall.to(device).float()], dim=1)       # (B,2,H,W)
    boundary_mask = F.interpolate(mask_px.unsqueeze(1), size=z.shape[-2:], mode="nearest")

    noise = torch.randn_like(z) * boundary_mask
    t = torch.randint(0, 1000, (z.shape[0],)).to(device)
    noisy_im = scheduler.add_noise(z, noise, t)
    return images, numbers, z, cond, mask_px, boundary_mask, noise, t, noisy_im


@torch.no_grad()
def evaluate_eps(test_loader, dit, vae, scheduler, device):
    dit.eval()
    total = 0.0
    for images, numbers, mask, wall, _ in test_loader:
        _, numbers, z, cond, mask_px, boundary_mask, noise, t, noisy_im = prepare_batch(
            images, numbers, mask, wall, vae, scheduler, device)
        pred = dit(noisy_im, t, numbers, cond) * boundary_mask
        total += eps_loss_fn(pred, noise, boundary_mask).item()
    return total / len(test_loader)



def train_dit(start_epoch, epochs, dataloader, dit, vae, scheduler, device, acc_steps, **kwargs):
    meta = kwargs.get('meta', None)

    lam_bc   = config["Training"].get("bc_lambda", 1.0)
    lam_gr   = config["Training"].get("gr_lambda", 1.0)
    lam_corr = config["Training"].get("corr_lambda", 0.1)

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
        epoch_eps = epoch_bc = epoch_gr = epoch_cr = epoch_tot = 0.0
        step_count = 0

        for images, numbers, mask, wall_mask, _ in train_loader:
            step_count += 1
            images, numbers, z, cond, mask_px, boundary_mask, noise, t, noisy_im = prepare_batch(images, numbers, mask, wall_mask, vae, scheduler, device)

            pred = dit(noisy_im, t, numbers, cond) * boundary_mask    # output forced to 0 outside

            # latent-space eps loss
            eps_loss = eps_loss_fn(pred, noise, boundary_mask)


            # pixel-space losses on decoded x0 (low-noise samples only)
            ab   = scheduler.alpha_cum_prod.to(device)[t]
            keep = t < 500
            bc_term = torch.zeros((), device=device)
            gr_term = torch.zeros((), device=device)
            cr_term = torch.zeros((), device=device)
            if keep.any():
                x0 = scheduler.get_x0(noisy_im[keep], pred[keep], t[keep], clamp_range=(-4, 4))
                wall_e, grad_e, corr_e = decode_losses(x0, vae, wall_mask.to(device)[keep],
                                                       mask_px[keep], images[keep], meta)
                w = ab[keep]
                bc_term = lam_bc   * (w * wall_e).sum() / z.shape[0]
                gr_term = lam_gr   * (w * grad_e).sum() / z.shape[0]
                cr_term = lam_corr * (w * corr_e).sum() / z.shape[0]

            total = eps_loss + bc_term + gr_term + cr_term
            loss = total / acc_steps
            loss.backward()

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_eps += eps_loss.item()
            epoch_bc  += bc_term.item()
            epoch_gr  += gr_term.item()
            epoch_cr  += cr_term.item()
            epoch_tot += total.item()

        scheduler_lr.step()

        n_b = len(train_loader)
        avg_eps, avg_bc, avg_gr = epoch_eps / n_b, epoch_bc / n_b, epoch_gr / n_b
        avg_cr, avg_tot = epoch_cr / n_b, epoch_tot / n_b
        losses.append(avg_tot)

        avg_test = evaluate_eps(test_loader, dit, vae, scheduler, device)
        test_losses.append(avg_test)

        print(f"[DiT] epoch={epoch+1} total={avg_tot:.6f} | eps={avg_eps:.6f} bc={avg_bc:.6f} "
              f"gr={avg_gr:.6f} corr={avg_cr:.6f} | test_eps={avg_test:.6f}",
              flush=True)

        is_last = (epoch + 1) == epochs
        if (epoch + 1) % config["saves"]["DiT_Save_every"] == 0 or is_last:
            torch.save(dit.state_dict(), config["saves"]["DiT_Path"])

    return losses, test_losses