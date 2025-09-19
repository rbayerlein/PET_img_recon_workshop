# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:12:20 2025

@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# -------- utilities --------
def fft2c(x):
    """Centered 2D FFT (k-space)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    """Centered 2D IFFT (image space)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def make_kgrid(Ny, Nx):
    """Return normalized frequency grids kx, ky in cycles/pixel ∈ [-0.5, 0.5)."""
    ky = (np.arange(Ny) - Ny//2) / Ny
    kx = (np.arange(Nx) - Nx//2) / Nx
    return np.meshgrid(kx, ky)  # kx, ky shape (Ny,Nx)

def circ_mask(Ny, Nx, k0, k1=None):
    """Circular (radial) mask in k-space. 
       k0 = outer radius for LP; 
       if k1 provided, pass-band is k1<=|k|<=k0 (band-pass).
       Radii are normalized (0..0.5)."""
    kx, ky = make_kgrid(Ny, Nx)
    kr = np.sqrt(kx**2 + ky**2)
    if k1 is None:
        return (kr <= k0).astype(np.float32)
    else:
        return ((kr >= k1) & (kr <= k0)).astype(np.float32)

def gauss_lowpass(Ny, Nx, sigma):
    """Gaussian LP in k-space: exp(-(kr/sigma)^2 / 2). sigma in normalized frequency."""
    kx, ky = make_kgrid(Ny, Nx)
    kr = np.sqrt(kx**2 + ky**2)
    return np.exp(-(kr**2) / (2.0 * sigma**2)).astype(np.float32)

def imshow_gray(ax, img, title, vmin=None, vmax=None):
    im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.axis('off')
    return im

# -------- data --------
N = 256
img = resize(shepp_logan_phantom().astype(np.float32), (N, N), order=3, anti_aliasing=True, preserve_range=True)

# k-space
X = fft2c(img)

# -------- filters (normalized frequency radii) --------
# Nyquist = 0.5 cycles/pixel. Pick intuitive cutoffs:
k_lp = 0.08              # low-pass cutoff (keep below 0.08)
k_hp = 0.12              # high-pass cutoff (remove below 0.12)
k_bp_lo, k_bp_hi = 0.08, 0.18  # band-pass ring
sigma_g = 0.10           # Gaussian LP sigma

Ny, Nx = img.shape
LP_mask   = circ_mask(Ny, Nx, k_lp)
HP_mask   = 1.0 - circ_mask(Ny, Nx, k_hp)
BP_mask   = circ_mask(Ny, Nx, k_bp_hi, k_bp_lo)
GAUSS_flt = gauss_lowpass(Ny, Nx, sigma=sigma_g)

# apply in k-space
X_lp   = X * LP_mask
X_hp   = X * HP_mask
X_bp   = X * BP_mask
X_glp  = X * GAUSS_flt

# back to image space (take real part; imaginary ~0 from numerical noise)
img_lp  = np.real(ifft2c(X_lp))
img_hp  = np.real(ifft2c(X_hp))
img_bp  = np.real(ifft2c(X_bp))
img_glp = np.real(ifft2c(X_glp))

# -------- plots --------
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# original + k-space magnitude (log)
imshow_gray(ax[0,0], img, "Image (spatial domain)", vmin=0, vmax=1)

# log-magnitude for visibility; add tiny epsilon to avoid log(0)
log_mag = np.log10(np.abs(X) + 1e-6)
im = ax[0,1].imshow(log_mag, cmap='magma')
ax[0,1].set_title("k-space |X(kx,ky)| (log)"); ax[0,1].axis('off')
plt.colorbar(im, ax=ax[0,1], fraction=0.046, pad=0.04)

# show LP mask as reference
im = ax[0,2].imshow(LP_mask, cmap='viridis')
ax[0,2].set_title(f"LP mask (k ≤ {k_lp:.2f})"); ax[0,2].axis('off')
plt.colorbar(im, ax=ax[0,2], fraction=0.046, pad=0.04)

# filtered images
imshow_gray(ax[1,0], img_lp,  f"Low-pass (k≤{k_lp:.2f})", vmin=0, vmax=1)
imshow_gray(ax[1,1], img_hp,  f"High-pass (k≥{k_hp:.2f})", vmin=np.percentile(img_hp, 1), vmax=np.percentile(img_hp, 99))
imshow_gray(ax[1,2], img_bp,  f"Band-pass ({k_bp_lo:.2f}–{k_bp_hi:.2f})", vmin=np.percentile(img_bp, 1), vmax=np.percentile(img_bp, 99))

plt.tight_layout()
plt.show()

# Optional: compare Gaussian LP vs hard LP
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
im = ax[0].imshow(GAUSS_flt, cmap='viridis'); ax[0].set_title(f"Gaussian LP in k (σ={sigma_g:.2f})"); ax[0].axis('off')
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
imshow_gray(ax[1], img_glp, "Gaussian LP (image)", vmin=0, vmax=1)
imshow_gray(ax[2], img_lp,  f"Hard LP (k≤{k_lp:.2f})", vmin=0, vmax=1)
plt.tight_layout(); plt.show()
