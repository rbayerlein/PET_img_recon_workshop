#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:53:22 2026

@author: rbayerlein

PSF overshoot / Gibbs-like ringing demo for PET reconstruction in 2D

Demonstrates:
1) blur from system PSF
2) MLEM without PSF modeling
3) MLEM with PSF modeling
4) overshoot / ringing near sharp edges when PSF modeling is pushed too far
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import radon, iradon

# ============================================================
# Parameters
# ============================================================
N = 128
angles = np.linspace(0, 180, 180, endpoint=False)
total_counts = 500000
sigma_psf = 2.0          # stronger PSF blur to make the effect obvious
iter_no_psf = 30
iter_psf_mild = 30
iter_psf_strong = 100

# ============================================================
# Helper functions
# ============================================================
def make_circular_mask(N):
    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2, (N - 1) / 2
    r = N / 2 - 1
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2)

def make_edge_phantom(N):
    """Create a phantom with sharp edges to highlight PSF overshoot."""
    img = np.zeros((N, N), dtype=np.float32)
    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2, (N - 1) / 2

    # Large hot disk
    mask1 = (xx - cx) ** 2 + (yy - cy) ** 2 <= 28 ** 2
    img[mask1] = 1.0

    # Small hot disk
    mask2 = (xx - (cx + 22)) ** 2 + (yy - (cy - 18)) ** 2 <= 5 ** 2
    img[mask2] = 1.0

    # Small hot disk in warm background
    mask3 = (xx - cx - 5) ** 2 + (yy - cy - 5) ** 2 <= 5 ** 2
    img[mask3] = 2.0
    
    return img

def poissonize_by_number(sino, total_counts, seed=None):
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0:
        return np.zeros_like(s, dtype=np.float32)
    lam = s * (total_counts / ssum)
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)

def H(x, sigma):
    return gaussian_filter(x, sigma=sigma)

# ============================================================
# Phantom and geometry
# ============================================================
mask = make_circular_mask(N)
act = make_edge_phantom(N)
act = np.where(mask, act, 0.0).astype(np.float32)

A = lambda x: radon(x, theta=angles, circle=True, preserve_range=True)
AT = lambda y: iradon(
    y,
    theta=angles,
    filter_name=None,
    circle=True,
    output_size=N,
    preserve_range=True
)

# ============================================================
# Simulate data with PSF blur
# ============================================================
act_blurred = H(act, sigma_psf)
act_blurred = np.where(mask, act_blurred, 0.0)

sino_psf = A(act_blurred)
y = poissonize_by_number(sino_psf, total_counts=total_counts, seed=1)

print("Simulated total counts:", int(y.sum()))

# ============================================================
# Sensitivity images
# ============================================================
ones_sino = np.ones_like(y, dtype=np.float32)

S_no_psf = AT(ones_sino)
S_no_psf[S_no_psf <= 0] = 1e-6
S_no_psf = np.where(mask, S_no_psf, 1.0)

S_psf = H(AT(ones_sino), sigma_psf)
S_psf[S_psf <= 0] = 1e-6
S_psf = np.where(mask, S_psf, 1.0)

# ============================================================
# MLEM reconstruction functions
# ============================================================
def mlem_no_psf(y, n_iter):
    x = np.full((N, N), 0.2, dtype=np.float32)
    eps = 1e-8

    for _ in range(n_iter):
        Ax = A(np.where(mask, x, 0.0))
        lam = np.maximum(Ax, eps)

        ratio = y / lam
        back = AT(ratio)

        update = back / S_no_psf
        x *= update
        x = np.where(mask, x, 0.0)
        x = np.clip(x, 0, None)

    return x

def mlem_psf(y, n_iter, sigma_psf):
    x = np.full((N, N), 0.2, dtype=np.float32)
    eps = 1e-8

    for _ in range(n_iter):
        Hx = H(np.where(mask, x, 0.0), sigma_psf)
        Ax = A(Hx)
        lam = np.maximum(Ax, eps)

        ratio = y / lam
        back = AT(ratio)
        back_psf = H(back, sigma_psf)

        update = back_psf / S_psf
        x *= update
        x = np.where(mask, x, 0.0)
        x = np.clip(x, 0, None)

    return x

# ============================================================
# Run reconstructions
# ============================================================
recon_no_psf = mlem_no_psf(y, iter_no_psf)
recon_psf_mild = mlem_psf(y, iter_psf_mild, sigma_psf)
recon_psf_strong = mlem_psf(y, iter_psf_strong, sigma_psf)

# ============================================================
# Normalize to truth mean inside object
# ============================================================
obj_mask = act > 0

truth_mean = act[obj_mask].mean()

def norm_to_truth(x):
    m = x[obj_mask].mean()
    if m <= 0:
        return x
    return x * (truth_mean / m)

recon_no_psf = norm_to_truth(recon_no_psf)
recon_psf_mild = norm_to_truth(recon_psf_mild)
recon_psf_strong = norm_to_truth(recon_psf_strong)

# ============================================================
# Profiles
# ============================================================
row = N // 2
prof_truth = act[row, :]
prof_blur = act_blurred[row, :]
prof_no_psf = recon_no_psf[row, :]
prof_psf_mild = recon_psf_mild[row, :]
prof_psf_strong = recon_psf_strong[row, :]

# ============================================================
# Plot images
# ============================================================
vmin, vmax = 0.0, 2.2

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

im0 = ax[0, 0].imshow(act, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Truth")
ax[0, 0].axis('off')
plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

im1 = ax[0, 1].imshow(act_blurred, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 1].set_title(f"Truth blurred by PSF (σ={sigma_psf})")
ax[0, 1].axis('off')
plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

im2 = ax[0, 2].imshow(sino_psf, cmap='inferno', aspect='auto')
ax[0, 2].set_title("Measured sinogram")
ax[0, 2].set_xlabel("Angle index")
ax[0, 2].set_ylabel("Detector bin")
plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

im3 = ax[1, 0].imshow(recon_no_psf, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 0].set_title(f"MLEM without PSF ({iter_no_psf} iter)")
ax[1, 0].axis('off')
plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

im4 = ax[1, 1].imshow(recon_psf_mild, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 1].set_title(f"MLEM with PSF ({iter_psf_mild} iter)")
ax[1, 1].axis('off')
plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

im5 = ax[1, 2].imshow(recon_psf_strong, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 2].set_title(f"MLEM with PSF ({iter_psf_strong} iter)")
ax[1, 2].axis('off')
plt.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

plt.suptitle("PSF Modeling: Edge Overshoot / Gibbs-like Ringing Demo")
plt.tight_layout()
plt.show()

# ============================================================
# Plot line profiles
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(prof_truth, label="Truth", linewidth=2)
plt.plot(prof_blur, label="Blurred truth", linewidth=2)
plt.plot(prof_no_psf, label=f"No PSF ({iter_no_psf} iter)", linewidth=2)
plt.plot(prof_psf_mild, label=f"PSF ({iter_psf_mild} iter)", linewidth=2)
plt.plot(prof_psf_strong, label=f"PSF ({iter_psf_strong} iter)", linewidth=2)

plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title("Central line profile: PSF overshoot near sharp edge")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()