# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:36:31 2025

@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
# from scipy.ndimage import gaussian_filter

### USER PARAMETERS ###
mu_water_cm = 0.096           # linear att. coeff of water at 511 keV [1/cm]
num_iter = 200
total_counts = 100_000_000 # number of desired counts in the sinogram

#######################

# ----------------------------
pad = 16                      # padding for FOV margin
angles = np.linspace(0, 180, 60, endpoint=False)

# --- original settings (example) ---
N0 = 400
pix_size_cm0 = 0.1        # original pixel size [cm] for 440x440
FOV_cm = N0 * pix_size_cm0 # keep physical FOV constant

# --- choose downsampled size ---
N = 100                     # target resolution
pix_size_cm = FOV_cm / N    # doubles if N halves → preserves FOV


# ----------------------------
# Helpers
# ----------------------------

def poissonize_by_number(sino, total_counts, seed=None):
    """Scale sinogram to a given total number of expected counts, then sample Poisson."""
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0:
        return np.zeros_like(s, dtype=np.float32)
    lam = s * (total_counts / ssum)   # expected counts per bin
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)

# ----------------------------
# Ground truth activity & μ-map
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

# ----------------------------
# Forward projections
# ----------------------------
# Activity Radon (unattenuated emission line integrals)
sino_act = radon(act, theta=angles, circle=True, preserve_range=True)

# Path-integrated μ along each LOR: Radon(μ) * pixel_size
# (Radon sums pixel values; multiply by physical path per pixel to get ∫μ dl)
sino_mu = radon(mu_map, theta=angles, circle=True, preserve_range=True) * pix_size_cm

# Transmission factor T = exp(-∫μ dl)
T = np.exp(-np.clip(sino_mu, 0, None))

# ATTENUATED emission sinogram (pedagogical approximation)
sino_att = sino_act * T

# ----------------------------
# Simulate Poisson sinogram counts
# ----------------------------

y = poissonize_by_number(sino_att, total_counts, seed=1)
print("Simulated total events:", y.sum())
print("Average number of events per sinogram bin:", np.sum(y)/y.size)

# For AC: pre-correct projections by 1/T (avoid div-by-zero)
y_ac = y / np.clip(T, 1e-8, None)

# ----------------------------
# Reconstructions
# ---------------------------- MLEM
# Forward / adjoint
A  = lambda x: radon(x, theta=angles, circle=True, preserve_range=True)
AT = lambda y: iradon(y, theta=angles, filter_name=None, circle=True,
                      output_size=act.shape[0], preserve_range=True)

# Sensitivity image no attenuation
ones_sino = np.ones_like(y_ac, dtype=np.float32)
S_NAC = AT(ones_sino); S_NAC[S_NAC<=0] = 1e-6
# S_NAC  = np.where(mask, S_NAC,  1e-6)

# Sensitivity with attenuation
S = AT(T.astype(np.float32))
S[S <= 0] = 1e-6
# S  = np.where(mask, S,  1e-6)

# MLEM with attenuation
eps = 1e-8

# Optional additive terms (set to 0 if unused)
rand = 0.0            # randoms sinogram or array same shape as T
scat = 0.0            # scatter sinogram or array same shape as T

def mlem(y, n_iter=20, s_tmp=S, t_tmp=T):
    x = np.full(act.shape, 1, dtype=np.float32)  # init (non-negative)
    mask = np.zeros_like(act, dtype=bool)
    yy, xx = np.ogrid[:act.shape[0], :act.shape[1]]
    r = act.shape[0]/2 - 1
    mask = (yy - r)**2 + (xx - r)**2 <= r**2   # circular FOV
    for k in range(n_iter):
        Ax = A(x)                                  # unattenuated projections
        lam = t_tmp * Ax + rand + scat + eps           # expected counts with attenuation
        ratio = y / lam                            # y, your measured Poisson data
        x *= AT(t_tmp * ratio) / (s_tmp + eps)         # attenuated backprojection
        x *= mask                                  # keep solution inside FOV
        x = np.clip(x, 0, None)                    # non-negativity
        # x = gaussian_filter(x, sigma=0.6)
    return x

recon_with_ac = mlem(y, num_iter, S, T)

T_NAC = np.ones_like(y, dtype=np.float32)       # transmission image without attenuation
recon_no_ac = mlem(y, num_iter, S_NAC, T_NAC)

# ---------------------------- END OF RECONS

# ---------------------------- Normalize Images
mean_img = act.mean()
def norm_by_mean(x):
    mean_recon = x.mean()
    return x * (mean_img/mean_recon)

# ----------------------------
# Plots
# ----------------------------
# Fixed visualization window in [0,1] to match phantom
vmin, vmax = 0.0, 1.0

fig, ax = plt.subplots(2, 3, figsize=(12, 7))

im0 = ax[0,0].imshow(np.clip(act, vmin, vmax), cmap='gray', vmin=vmin, vmax=vmax)
ax[0,0].set_title("Activity (Truth)"); ax[0,0].axis('off')
plt.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04)

im1 = ax[0,1].imshow(mu_map, cmap='magma')
ax[0,1].set_title(r"Attenuation map $\mu$ [1/cm]"); ax[0,1].axis('off')
plt.colorbar(im1, ax=ax[0,1], fraction=0.046, pad=0.04)

# im2 = ax[0,2].imshow(sino_mu, cmap='gray', aspect='auto')
# ax[0,2].set_title("forward projected μ-map")
# ax[0,2].set_xlabel("Angle (index)"); ax[0,2].set_ylabel("Detector bin")
# plt.colorbar(im2, ax=ax[0,2], fraction=0.046, pad=0.04)

im2 = ax[0,2].imshow(T, cmap='viridis', aspect='auto')
ax[0,2].set_title("Transmission sinogram T = exp(-∫μ dl)")
ax[0,2].set_xlabel("Angle (index)"); ax[0,2].set_ylabel("Detector bin")
plt.colorbar(im2, ax=ax[0,2], fraction=0.046, pad=0.04)

im3 = ax[1,0].imshow(np.clip(sino_att, 0, None), cmap='inferno', aspect='auto')
ax[1,0].set_title("Emission sinogram with attenuation")
ax[1,0].set_xlabel("Angle (index)"); ax[1,0].set_ylabel("Detector bin")
plt.colorbar(im3, ax=ax[1,0], fraction=0.046, pad=0.04)

im4 = ax[1,1].imshow(norm_by_mean(recon_no_ac), cmap='gray', vmin=vmin, vmax=vmax)
ax[1,1].set_title("Reconstruction WITHOUT AC (MLEM)"); ax[1,1].axis('off')
plt.colorbar(im4, ax=ax[1,1], fraction=0.046, pad=0.04)

im5 = ax[1,2].imshow(norm_by_mean(recon_with_ac), cmap='gray', vmin=vmin, vmax=vmax)
ax[1,2].set_title("Reconstruction WITH AC (MLEM)"); ax[1,2].axis('off')
plt.colorbar(im5, ax=ax[1,2], fraction=0.046, pad=0.04)

plt.suptitle(f"PET Attenuation Demo (N={act.shape[0]}, pix={pix_size_cm} cm, counts≈{int(y.sum())})")
plt.tight_layout()
plt.show()
