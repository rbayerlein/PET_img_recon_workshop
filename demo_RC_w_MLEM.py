# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 06:10:42 2025

@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
# from aux_functions import line_profile_at_height

### USER PARAMETERS ###
total_counts   = 10_000_000         # total expected counts in scan
randoms_frac   = 0.35               # X% randoms fraction
n_iter = 50
num_angles = 180
#######################

# ----------------------------
# Geometry (reuse your choices)
# ----------------------------
N0 = 400
pix_size_cm0 = 0.1
FOV_cm = N0 * pix_size_cm0

N = 100
pix_size_cm = FOV_cm / N
angles = np.linspace(0, 180, num_angles, endpoint=False)
mu_water_cm = 0.01


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
offset_value = 0
sl_offset = np.full((N0, N0), offset_value, np.float32)
img_hi = img_hi + sl_offset

# μ-map: inside phantom = water, outside = 0
mu_map_hi = np.zeros_like(img_hi, dtype=np.float32).copy()
mu_map_hi[img_hi > 0] = mu_water_cm

# --- downsample phantom size to save time---
mu_map_hi = resize(mu_map_hi, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
mu_map    = resize(mu_map_hi, (N, N),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

img_hi = resize(img_hi, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
act    = resize(img_hi, (N, N),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

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
S_AC  = AT(T.astype(np.float32));  
# S_AC  = np.where(mask, S_AC,  1e-6); 
S_AC[S_AC <= 0] = 1e-6
S_NAC = AT(np.ones_like(y, np.float32)); 
# S_NAC = np.where(mask, S_NAC, 1e-6); 
S_NAC[S_NAC <= 0] = 1e-6
T_NAC = np.ones_like(y, dtype=np.float32)       # transmission image without attenuation

# ----------------------------
# MLEM variants
# ----------------------------

def mlem(y, n_iter=20, s_tmp=S_AC, t_tmp=T, r_sino=None):
    rand = 0.0 if r_sino is None else r_sino.astype(np.float32)

    x = np.full(act.shape, 1, dtype=np.float32)  # init (non-negative)
    mask = np.zeros_like(act, dtype=bool)
    yy, xx = np.ogrid[:act.shape[0], :act.shape[1]]
    margin = 1
    r = act.shape[0]/2 - margin
    mask = (yy - r - margin)**2 + (xx - r- margin)**2 < r**2    # circular FOV
    eps = 0.0001
    for k in range(n_iter):
        # Ax = A(x)                                  # unattenuated projections
        Ax   = A(np.where(mask, x, 0.0)).astype(np.float32)
        lam = t_tmp * Ax + rand + eps          # expected counts with attenuation and randoms
        lam[lam==eps] = 1
        ratio = y / lam                            # y, your measured Poisson data
        prod_tmp = t_tmp * ratio
        bproj = AT(prod_tmp) 
        x *= bproj / (s_tmp + eps)         # attenuated backprojection
        x *= mask                                  # keep solution inside FOV
        x = np.clip(x, 0, None)                    # non-negativity
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
im1 = ax[0,1].imshow(y,   cmap='viridis', aspect='auto', vmin=vmin);     ax[0,1].set_title("Prompts sinogram"); plt.colorbar(im1, ax=ax[0,1])
im2 = ax[0,2].imshow(y_rnd, cmap='magma', aspect='auto');     ax[0,2].set_title("Randoms sinogram"); plt.colorbar(im2, ax=ax[0,2])
im3 = ax[1,0].imshow(norm_by_mean(recon_noRC), cmap='gray', vmin=vmin, vmax=vmax); ax[1,0].set_title("MLEM w/ AC, NO RC (biased)"); ax[1,0].axis('off'); plt.colorbar(im3, ax=ax[1,0])
im4 = ax[1,1].imshow(norm_by_mean(recon_RC_true), cmap='gray', vmin=vmin, vmax=vmax); ax[1,1].set_title("MLEM w/ AC + TRUE r"); ax[1,1].axis('off'); plt.colorbar(im4, ax=ax[1,1])
im5 = ax[1,2].imshow(norm_by_mean(recon_RC_est), cmap='gray', vmin=vmin, vmax=vmax);  ax[1,2].set_title("MLEM w/ AC + EST. r̂"); ax[1,2].axis('off'); plt.colorbar(im5, ax=ax[1,2])

plt.suptitle(f"Randoms Demo: total={int(y.sum())}, RF={randoms_frac:.0%}, iters={n_iter}")
plt.tight_layout(); plt.show()

print("Observed randoms fraction in y:", y_rnd.sum()/y.sum())

# profile_act = line_profile_at_height(norm_by_mean(act),  N/2, 20, 80)
# profile_nonRC = line_profile_at_height(norm_by_mean(recon_noRC), N/2, 20, 80)
# profile_True_RC = line_profile_at_height(norm_by_mean(recon_RC_true),  N/2, 20, 80)
# profile_RC = line_profile_at_height(norm_by_mean(recon_RC_est),  N/2, 20, 80)

# plt.plot(profile_act, label='ground truth')
# plt.plot(profile_nonRC, label='non RC')
# plt.plot(profile_RC, label='RC (estimator)')
# plt.ylim(-0.01,0.3)

# plt.legend()
# plt.tight_layout(); plt.show()