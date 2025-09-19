# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 06:10:42 2025

@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon

### USER PARAMETERS ###
total_counts   = 55_000_000         # total expected counts in scan
randoms_frac   = 0.25               # X% randoms fraction
n_iter = 200

#######################

# ----------------------------
# Geometry (reuse your choices)
# ----------------------------
N0 = 400
pix_size_cm0 = 0.1
FOV_cm = N0 * pix_size_cm0

N = 100
pix_size_cm = FOV_cm / N
angles = np.linspace(0, 180, 60, endpoint=False)
mu_water_cm = 0.096

# ----------------------------
# Helpers
# ----------------------------
def poissonize_by_number(sino, total_counts, seed=None):
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0: 
        return np.zeros_like(s, np.float32)
    lam = s * (total_counts / ssum)
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)

def make_uniform_randoms(shape):
    r = np.ones(shape, dtype=np.float32)
    return r

def make_angled_randoms(shape):
    # example non-uniform randoms pattern (angle-dependent)
    H, W = shape
    cols = np.linspace(0, 1, W, dtype=np.float32)
    r = np.tile(0.5 + 0.5*np.sin(2*np.pi*cols), (H,1)).astype(np.float32)
    r -= r.min(); 
    return r + 1e-6

# ----------------------------
# Phantom & μ
# ----------------------------

# --- create phantom ---
img_hi = shepp_logan_phantom().astype(np.float32)

# μ-map: inside phantom = water, outside = 0
mu_map_hi = np.zeros_like(img_hi, dtype=np.float32).copy()
mu_map_hi[img_hi > 0] = mu_water_cm

# --- downsample phantom size to save time---
mu_map_hi = resize(mu_map_hi, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
mu_map    = resize(mu_map_hi, (N, N),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

img_hi = resize(img_hi, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
act    = resize(img_hi, (N, N),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

# Mask outside circle to satisfy radon(circle=True)
H, W = act.shape
cy, cx = (H-1)/2, (W-1)/2
r0 = min(H, W)/2 - 1
yy, xx = np.ogrid[:H, :W]
mask = ((yy-cy)**2 + (xx-cx)**2) <= r0**2
act    = np.where(mask, act,    0.0)
mu_map = np.where(mask, mu_map, 0.0)

# Matched projector/backprojector
A  = lambda x: radon(x, theta=angles, circle=True, preserve_range=True)
AT = lambda y: iradon(y, theta=angles, filter_name=None, circle=True,
                      output_size=N, preserve_range=True)

# Forward components
sino_act = A(act)
sino_mu  = A(mu_map) * pix_size_cm
T        = np.exp(-np.clip(sino_mu, 0, None))

# ----------------------------
# Build signal + randoms, then draw data
# ----------------------------

# Base (unnormalized) shapes
sig_shape = np.clip(sino_act * T, 0, None)

rnd_shape = make_uniform_randoms(sig_shape.shape)  
# --- OR ---
# rnd_shape = make_angled_randoms(sig_shape.shape)  

# Allocate counts between signal and randoms
sig_counts = total_counts * (1.0 - randoms_frac)
rnd_counts = total_counts * randoms_frac

y_sig = poissonize_by_number(sig_shape, sig_counts, seed=1)
y_rnd = poissonize_by_number(rnd_shape, rnd_counts, seed=2)
y     = y_sig + y_rnd

# ----------------------------
# Randoms estimation (delayed window)
# ----------------------------
# Assume a delayed acquisition k times longer than prompt frame
k = 2.0  # scale factor (longer delayed window → lower variance in estimate)
# In practice, delayed counts ~ Poisson(k * true_randoms); divide by k to estimate r
y_delayed = np.random.default_rng(3).poisson(k * y_rnd).astype(np.float32)
r_hat = y_delayed / k  # unbiased estimator of randoms sinogram, noisy

# ----------------------------
# Sensitivities
# ----------------------------
S_AC  = AT(T.astype(np.float32));  S_AC  = np.where(mask, S_AC,  1e-6)
S_NAC = AT(np.ones_like(y, np.float32)); S_NAC = np.where(mask, S_NAC, 1e-6)

# ----------------------------
# MLEM variants
# ----------------------------
def mlem(y, n_iter, S, T_sino, r_sino=None, x0=None,
         sinogram_valid=None, eps=1e-3, upd_clip=(0.2, 5.0)):
    # x init
    x = np.full((N, N), 0.2, np.float32) if x0 is None else x0.astype(np.float32)

    # randoms default
    r = 0.0 if r_sino is None else r_sino.astype(np.float32)

    # sinogram validity mask (where the model has support)
    if sinogram_valid is None:
        # project an all-ones image inside FOV to find support
        ones_img = np.where(mask, 1.0, 0.0).astype(np.float32)
        sino_support = A(ones_img)
        sinogram_valid = (T_sino > 1e-12) & (sino_support > 1e-12)
    sinogram_valid = sinogram_valid.astype(bool)

    # floor sensitivity INSIDE FOV; set arbitrary 1 outside to avoid /0
    S_safe = S.copy().astype(np.float32)
    S_safe[~mask] = 1.0
    S_safe = np.maximum(S_safe, eps)

    for _ in range(n_iter):
        Ax   = A(np.where(mask, x, 0.0)).astype(np.float32)
        lam  = T_sino * Ax + r
        lam  = np.maximum(lam, eps)  # floor expected counts

        # ratio only where valid; elsewhere 0 so it doesn't contribute
        ratio = np.zeros_like(y, dtype=np.float32)
        ratio[sinogram_valid] = y[sinogram_valid] / lam[sinogram_valid]

        # multiplicative EM update
        back = AT(T_sino * ratio).astype(np.float32)
        update = back / S_safe

        # # optional safety clip to prevent numerical blow-ups
        # if upd_clip is not None:
        #     lo, hi = upd_clip
        #     update = np.clip(update, lo, hi)

        x *= update
        x  = np.where(mask, x, 0.0)
        x  = np.clip(x, 0, None)

        # (optional) guard against NaN/Inf if something still goes wrong
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x


# 1) WRONG model (no randoms term): shows positive bias
print('running non-RC recon')
recon_noRC = mlem(y, n_iter, S_AC, T, r_sino=None)

# 2) Correct model with TRUE r (best case)
print('running RC recon with TRUE randoms')
recon_RC_true = mlem(y, n_iter, S_AC, T, r_sino=y_rnd)

# 3) Correct model with ESTIMATED r̂ (delayed window)
print('running RC recon with ESTIMATED randoms')
recon_RC_est = mlem(y, n_iter, S_AC, T, r_sino=r_hat)

# (Optional) NAC comparison
# recon_NAC_noRC = mlem(y, n_iter, S_NAC, np.ones_like(y))

# ---------------------------- Normalize Images
mean_img = act.mean()
def norm_by_mean(x):
    mean_recon = x.mean()
    return x * (mean_img/mean_recon)

# ----------------------------
# Display
# ----------------------------
vmin, vmax = 0, 1
fig, ax = plt.subplots(2, 3, figsize=(13, 8))

im0 = ax[0,0].imshow(act, cmap='gray', vmin=vmin, vmax=vmax); ax[0,0].set_title("Truth"); ax[0,0].axis('off'); plt.colorbar(im0, ax=ax[0,0])
im1 = ax[0,1].imshow(T,   cmap='viridis', aspect='auto');     ax[0,1].set_title("Transmission T"); plt.colorbar(im1, ax=ax[0,1])
im2 = ax[0,2].imshow(y_rnd, cmap='magma', aspect='auto');     ax[0,2].set_title("Randoms sinogram (true)"); plt.colorbar(im2, ax=ax[0,2])

im3 = ax[1,0].imshow(norm_by_mean(recon_noRC), cmap='gray', vmin=vmin, vmax=vmax); ax[1,0].set_title("MLEM w/ AC, NO RC (biased)"); ax[1,0].axis('off'); plt.colorbar(im3, ax=ax[1,0])
im4 = ax[1,1].imshow(norm_by_mean(recon_RC_true), cmap='gray', vmin=vmin, vmax=vmax); ax[1,1].set_title("MLEM w/ AC + TRUE r"); ax[1,1].axis('off'); plt.colorbar(im4, ax=ax[1,1])
im5 = ax[1,2].imshow(norm_by_mean(recon_RC_est), cmap='gray', vmin=vmin, vmax=vmax);  ax[1,2].set_title("MLEM w/ AC + EST. r̂"); ax[1,2].axis('off'); plt.colorbar(im5, ax=ax[1,2])

plt.suptitle(f"Randoms Demo: total={int(y.sum())}, RF={randoms_frac:.0%}, iters={n_iter}")
plt.tight_layout(); plt.show()

print("Observed randoms fraction in y:", y_rnd.sum()/y.sum())
