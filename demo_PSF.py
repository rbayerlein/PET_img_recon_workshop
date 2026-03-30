#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 2026

@author: rbayerlein

PSF modeling demo for PET reconstruction in 2D

Demonstrates:
1) true object
2) blurred object due to system PSF
3) sinogram generation
4) MLEM reconstruction without PSF modeling
5) MLEM reconstruction with PSF modeling
6) line profiles for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import radon, iradon
from phantoms import make_circular_mask, make_disk_phantom

# ============================================================
# Parameters
# ============================================================
N = 128
angles = np.linspace(0, 180, 180, endpoint=False)
total_counts = 500_000
sigma_psf = 2.0   # PSF width in pixels

num_iter = 50
# ============================================================
# Helper functions
# ============================================================

def poissonize_by_number(sino, total_counts, seed=None):
    """Scale sinogram to desired total expected counts, then sample Poisson."""
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0:
        return np.zeros_like(s, dtype=np.float32)

    lam = s * (total_counts / ssum)
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)


def H(x, sigma):
    """Symmetric Gaussian PSF operator."""
    return gaussian_filter(x, sigma=sigma)


# ============================================================
# Build phantom
# ============================================================
mask = make_circular_mask(N)
act = make_disk_phantom(N)
act = np.where(mask, act, 0.0).astype(np.float32)

# ============================================================
# Forward / backward projectors
# ============================================================
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

sino_true = A(act)
sino_psf = A(act_blurred)

y = poissonize_by_number(sino_psf, total_counts=total_counts, seed=1)

print("Simulated total counts:", int(y.sum()))
print("Average counts per sinogram bin:", y.mean())

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
# MLEM without PSF modeling
# ============================================================
def mlem_no_psf(y, n_iter, x0=None):
    x = np.full((N, N), 0.2, dtype=np.float32) if x0 is None else x0.copy().astype(np.float32)
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


# ============================================================
# MLEM with PSF modeling
# ============================================================
def mlem_psf(y, n_iter, sigma_psf, x0=None):
    x = np.full((N, N), 0.2, dtype=np.float32) if x0 is None else x0.copy().astype(np.float32)
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
recon_no_psf = mlem_no_psf(y, num_iter)
recon_psf = mlem_psf(y, num_iter, sigma_psf)

# ============================================================
# Normalize reconstructions to truth mean for display
# ============================================================
mean_truth = act[mask].mean()

def norm_by_masked_mean(x):
    m = x[mask].mean()
    if m <= 0:
        return x
    return x * (mean_truth / m)

recon_no_psf_disp = norm_by_masked_mean(recon_no_psf)
recon_psf_disp = norm_by_masked_mean(recon_psf)

# ============================================================
# Line profiles
# ============================================================
row =50
prof_truth = act[row, :]
prof_blur = act_blurred[row, :]
prof_no_psf = recon_no_psf_disp[row, :]
prof_psf = recon_psf_disp[row, :]

# ============================================================
# Plots
# ============================================================
vmin, vmax = 0.0, 1.0

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

im0 = ax[0, 0].imshow(act, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Truth")
# ax[0, 0].axis('off')
plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

im1 = ax[0, 1].imshow(act_blurred, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 1].set_title(f"Truth blurred by PSF (σ={sigma_psf} px)")
ax[0, 1].axis('off')
plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

im2 = ax[0, 2].imshow(sino_psf, cmap='inferno', aspect='auto')
ax[0, 2].set_title("Sinogram from blurred object")
ax[0, 2].set_xlabel("Angle index")
ax[0, 2].set_ylabel("Detector bin")
plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

im3 = ax[1, 0].imshow(recon_no_psf_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 0].set_title(f"MLEM without PSF ({num_iter} iter)")
ax[1, 0].axis('off')
plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

im4 = ax[1, 1].imshow(recon_psf_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 1].set_title(f"MLEM with PSF ({num_iter} iter)")
ax[1, 1].axis('off')
plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

# Leave last panel for line profiles
ax[1, 2].plot(prof_truth, label='Truth', linewidth=2)
ax[1, 2].plot(prof_blur, label='Blurred truth', linewidth=2)
ax[1, 2].plot(prof_no_psf, label='MLEM no PSF', linewidth=2)
ax[1, 2].plot(prof_psf, label='MLEM with PSF', linewidth=2)
ax[1, 2].set_title("Central line profile")
ax[1, 2].set_xlabel("Pixel")
ax[1, 2].set_ylabel("Intensity")
ax[1, 2].grid(True)
ax[1, 2].legend()

plt.suptitle(f"PSF Modeling Demo in PET Reconstruction (N={N}, counts≈{int(y.sum())})")
plt.tight_layout()
plt.show()