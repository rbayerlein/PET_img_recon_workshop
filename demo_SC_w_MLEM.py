# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:09:34 2025

@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
from scatter_models import make_scatter_blur

### USER PARAMETERS ###
total_counts   = 50_000_000         # total expected counts in scan
randoms_frac   = 0.15               # X% randoms fraction
scatter_frac   = 0.25               # X% scatter fraction

cnt_MC = 10_000_000

n_iter = 200

#######################

# ----------------------------
# Geometry
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
# Build signal + randoms + scatter
# ----------------------------

# Base (unnormalized) shapes
sig_shape = np.clip(sino_act * T, 0, None)

# Allocate counts between signal and randoms and scatters
sig_counts = total_counts * (1.0 - randoms_frac - scatter_frac)
rnd_counts = total_counts * randoms_frac
sct_counts = total_counts * scatter_frac

# -------- RANDOMS --------
rnd_shape = make_uniform_randoms(sig_shape.shape)  
# --- OR ---
# rnd_shape = make_angled_randoms(sig_shape.shape)  


# -------- SCATTERS --------
# Pick a scatter generator (e.g., blur model)
sct_shape = make_scatter_blur(sig_shape, sino_mu, scatter_frac, randoms=rnd_shape,
                           sigma_det=8.0, sigma_ang=1.0, definition="total")

# -------- TRUES + R + S --------
y_sig = poissonize_by_number(sig_shape, sig_counts, seed=1)
y_rnd = poissonize_by_number(rnd_shape, rnd_counts, seed=2)
y_sct = poissonize_by_number(sct_shape, sct_counts, seed=3)
y     = y_sig + y_rnd + y_sct
print("Prompt counts:", y.sum(), "scatter counts:", y_sct.sum(), "random counts:", y_rnd.sum())
print("Prompt minus delayed:", (y - y_rnd).sum())

# ----------------------------
# Randoms estimation (delayed window)
# ----------------------------
# Assume a delayed acquisition k times longer than prompt frame
k = 2.0  # scale factor (longer delayed window → lower variance in estimate)
# In practice, delayed counts ~ Poisson(k * true_randoms); divide by k to estimate r
y_delayed = np.random.default_rng(3).poisson(k * y_rnd).astype(np.float32)
r_hat = y_delayed / k  # unbiased estimator of randoms sinogram, noisy

# ----------------------------
# Scatter estimation
# ----------------------------
################################################################
################################################################
# assuming the sinograms originate from a Monte Carlo (MC) simulation
sig_count_MC = cnt_MC * (1.0 - randoms_frac - scatter_frac)
sct_count_MC = cnt_MC * scatter_frac
y_sig_MC = poissonize_by_number(sig_shape, sig_count_MC, seed=4)
y_sct_MC = poissonize_by_number(sct_shape, sct_count_MC, seed=5)
print("signal counts MC:", y_sig_MC.sum(), "scatter counts MC:", y_sct_MC.sum())

# ----------------------------
# Sensitivities
# ----------------------------
S_AC  = AT(T.astype(np.float32));  S_AC  = np.where(mask, S_AC,  1e-6)

# ----------------------------
# MLEM variants
# ----------------------------
def mlem(y, n_iter, S, T_sino, r_sino=None, s_sino=None):
    x = np.full(act.shape, 0.2, np.float32); eps = 1e-3
    r_sino = 0.0 if r_sino is None else r_sino
    s_sino = 0.0 if s_sino is None else s_sino
    for _ in range(n_iter):
        Ax  = A(x)
        lam = T_sino * Ax + r_sino + s_sino + eps
        ratio = y / lam
        x *= AT(T_sino * ratio) / (S + eps)
        x *= mask                                  # keep solution inside FOV
        x = np.clip(x, 0, None)
    return x

# 1) WRONG model (no sc term): shows positive bias
print('running non-SC recon')
recon_noSC = mlem(y, n_iter, S_AC, T, r_sino=y_rnd, s_sino=None)
# recon_noSC[recon_noSC>0.0001] = 0.0

# 2) Correct model with TRUE s (best case)
print('running SC recon with TRUE scatters')
recon_SC_true = mlem(y, n_iter, S_AC, T, r_sino=y_rnd, s_sino=y_sct)

# 3) Correct model with ESTIMATED s (per sinogram scaling)
print('running RC recon with ESTIMATED scatters')
alpha = (y - y_rnd).sum() / np.maximum((y_sig_MC + y_sct_MC).sum(), 1e-8)
print("Sinogram scale factor:", alpha)
y_sct_scaled = alpha * y_sct_MC
recon_SC_est = mlem(y, n_iter, S_AC, T, r_sino=y_rnd, s_sino=y_sct_scaled)


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
im2 = ax[0,1].imshow(y_sct, cmap='magma', aspect='auto');     ax[0,1].set_title("Scatter sinogram (true)"); plt.colorbar(im2, ax=ax[0,1])
im2 = ax[0,2].imshow(alpha*(y_sct_MC), cmap='magma', aspect='auto');     ax[0,2].set_title("Scaled scatter sinogram (MC)"); plt.colorbar(im2, ax=ax[0,2])

im3 = ax[1,0].imshow(norm_by_mean(recon_noSC), cmap='gray', vmin=vmin, vmax=vmax); ax[1,0].set_title("MLEM NO SC (biased)"); ax[1,0].axis('off'); plt.colorbar(im3, ax=ax[1,0])
im4 = ax[1,1].imshow(norm_by_mean(recon_SC_true), cmap='gray', vmin=vmin, vmax=vmax); ax[1,1].set_title("MLEM + TRUE s"); ax[1,1].axis('off'); plt.colorbar(im4, ax=ax[1,1])
im5 = ax[1,2].imshow(norm_by_mean(recon_SC_est), cmap='gray', vmin=vmin, vmax=vmax);     ax[1,2].set_title("MLEM + estimated S"); ax[1,1].axis('off'); plt.colorbar(im4, ax=ax[1,2])

plt.suptitle(f"Scatters Demo: total={int(y.sum())}, RF={randoms_frac:.0%}, SF={scatter_frac:.0%}, iters={n_iter}")
plt.tight_layout(); plt.show()

print("Observed randoms fraction in y:", y_rnd.sum()/y.sum())
print("Observed scatter fraction in y:", y_sct_scaled.sum()/y.sum())
