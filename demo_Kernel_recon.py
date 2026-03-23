#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:14:56 2026

@author: rbayerlein, OpenAI / ChatGPT

Kernelized PET reconstruction demo in 2D

Demonstrates:
1) Standard MLEM
2) MLEM + Gaussian post-smoothing
3) Kernelized MLEM: x = K @ alpha

Uses:
- brain-like structured phantom
- radon / iradon for projection model
- sparse local kernel matrix K

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from skimage.transform import radon, iradon

# ============================================================
# Parameters
# ============================================================
N = 64
angles = np.linspace(0, 180, 90, endpoint=False)
num_iter = 20
total_counts = 50_000

# Post-smoothing strength for comparison
sigma_post = 1.0

# Kernel parameters
kernel_radius = 2          # 5x5 neighborhood
sigma_spatial = 1.2
sigma_intensity = 0.10

# ============================================================
# Helper functions
# ============================================================
def make_circular_mask(N):
    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2.0, (N - 1) / 2.0
    r = N / 2.0 - 1.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2)


def poissonize_by_number(sino, total_counts, seed=None):
    """Scale expected sinogram to desired total counts, then draw Poisson."""
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0:
        return np.zeros_like(s, dtype=np.float32)
    lam = s * (total_counts / ssum)
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)


def make_brain_like_pet_phantom(N):
    """
    Create a structured, brain-like PET phantom with cortex, deep nuclei,
    ventricles, cerebellum, and asymmetric uptake variations.
    """
    y, x = np.mgrid[-1:1:complex(0, N), -1:1:complex(0, N)]

    # Elliptical head / brain masks
    outer = (x / 0.82) ** 2 + (y / 0.95) ** 2 <= 1.0
    inner = (x / 0.58) ** 2 + (y / 0.70) ** 2 <= 1.0
    cortex = outer & (~inner)

    img = np.zeros((N, N), dtype=np.float32)

    # Base uptake
    img[outer] = 0.18
    img[cortex] = 0.48

    # Cortical texture / gyral-like modulation
    theta = np.arctan2(y, x)
    radial = np.sqrt(x**2 + y**2)
    mod = 0.08 * np.sin(8 * theta) + 0.04 * np.sin(17 * theta + 0.6)
    img[cortex] += mod[cortex]

    # Frontal / parietal asymmetry
    img[(cortex) & (x < -0.15) & (y < 0.15)] += 0.05
    img[(cortex) & (x >  0.15) & (y > 0.00)] -= 0.03

    # Basal ganglia
    bg_left  = ((x + 0.18) / 0.10) ** 2 + ((y + 0.05) / 0.08) ** 2 <= 1.0
    bg_right = ((x - 0.18) / 0.10) ** 2 + ((y + 0.05) / 0.08) ** 2 <= 1.0
    img[bg_left | bg_right] = 0.95

    # Thalami
    th_left  = ((x + 0.10) / 0.08) ** 2 + ((y + 0.02) / 0.06) ** 2 <= 1.0
    th_right = ((x - 0.10) / 0.08) ** 2 + ((y + 0.02) / 0.06) ** 2 <= 1.0
    img[th_left | th_right] = 0.72

    # Ventricles
    vent_left  = ((x + 0.07) / 0.06) ** 2 + ((y - 0.02) / 0.10) ** 2 <= 1.0
    vent_right = ((x - 0.07) / 0.06) ** 2 + ((y - 0.02) / 0.10) ** 2 <= 1.0
    img[vent_left | vent_right] = 0.05

    # Temporal / occipital regions
    img[((x + 0.45) / 0.18) ** 2 + ((y + 0.10) / 0.12) ** 2 <= 1.0] += 0.15
    img[((x - 0.40) / 0.14) ** 2 + ((y + 0.18) / 0.12) ** 2 <= 1.0] += 0.15

    # Cerebellum
    cereb_left  = ((x + 0.22) / 0.20) ** 2 + ((y - 0.62) / 0.14) ** 2 <= 1.0
    cereb_right = ((x - 0.22) / 0.20) ** 2 + ((y - 0.62) / 0.14) ** 2 <= 1.0
    img[cereb_left | cereb_right] = 0.55

    # Brainstem-ish structure
    stem = ((x / 0.07) ** 2 + ((y - 0.43) / 0.12) ** 2 <= 1.0)
    img[stem] = 0.45

    img = gaussian_filter(img, sigma=0.4)
    img = np.clip(img, 0, None)

    # Normalize to [0,1]
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img.astype(np.float32)


def build_kernel_matrix(ref, mask, radius=2, sigma_spatial=1.2, sigma_intensity=0.10):
    """
    Build sparse local kernel matrix K.
    Each row corresponds to one image pixel and contains normalized weights
    over a local neighborhood based on spatial and feature similarity.
    """
    H, W = ref.shape
    n = H * W

    rows = []
    cols = []
    vals = []

    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            offsets.append((dy, dx, d2))

    def idx(r, c):
        return r * W + c

    for r in range(H):
        for c in range(W):
            i = idx(r, c)

            if not mask[r, c]:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                continue

            w_list = []
            j_list = []

            f_i = ref[r, c]

            for dy, dx, d2 in offsets:
                rr = r + dy
                cc = c + dx
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if not mask[rr, cc]:
                    continue

                f_j = ref[rr, cc]

                w_spatial = np.exp(-d2 / (2.0 * sigma_spatial ** 2))
                w_feat = np.exp(-((f_i - f_j) ** 2) / (2.0 * sigma_intensity ** 2))
                w = w_spatial * w_feat

                w_list.append(w)
                j_list.append(idx(rr, cc))

            w_list = np.array(w_list, dtype=np.float32)
            w_list /= (w_list.sum() + 1e-8)

            rows.extend([i] * len(j_list))
            cols.extend(j_list)
            vals.extend(w_list.tolist())

    K = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    return K


# ============================================================
# Build truth image and reference image
# ============================================================
mask = make_circular_mask(N)

act = make_brain_like_pet_phantom(N)
act = np.where(mask, act, 0.0).astype(np.float32)

# Reference image for the kernel
# For a workshop demo, a lightly smoothed truth works well as a stand-in
# for anatomical guidance or a denoised prior.
ref = gaussian_filter(act, sigma=0.8)
ref = np.where(mask, ref, 0.0).astype(np.float32)

# ============================================================
# Projectors
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
# Simulate measured sinogram
# ============================================================
sino_true = A(act)
y = poissonize_by_number(sino_true, total_counts=total_counts, seed=1)

print("Simulated total counts:", int(y.sum()))
print("Average counts per sinogram bin:", y.mean())

# ============================================================
# Sensitivity
# ============================================================
ones_sino = np.ones_like(y, dtype=np.float32)
S = AT(ones_sino)
S[S <= 0] = 1e-6
S = np.where(mask, S, 1.0).astype(np.float32)

# ============================================================
# Build sparse kernel matrix K
# ============================================================
print("Building sparse kernel matrix...")
K = build_kernel_matrix(
    ref,
    mask=mask,
    radius=kernel_radius,
    sigma_spatial=sigma_spatial,
    sigma_intensity=sigma_intensity
)
print("Kernel matrix built.")
print("K shape:", K.shape, "  nnz:", K.nnz)

S_vec = S.ravel().astype(np.float32)
KtS = (K.T @ S_vec).astype(np.float32)
KtS = np.maximum(KtS, 1e-6)

# ============================================================
# Standard MLEM
# ============================================================
def mlem_standard(y, n_iter, x0=None):
    x = np.full((N, N), 0.2, dtype=np.float32) if x0 is None else x0.copy().astype(np.float32)
    eps = 1e-8

    for _ in range(n_iter):
        x_masked = np.where(mask, x, 0.0)
        Ax = A(x_masked)
        lam = np.maximum(Ax, eps)

        ratio = y / lam
        back = AT(ratio)
        update = back / S

        x *= update
        x = np.where(mask, x, 0.0)
        x = np.clip(x, 0, None)

    return x


# ============================================================
# Kernelized MLEM
# ============================================================
def mlem_kernel(y, n_iter, K, KtS, alpha0=None):
    """
    Kernel EM with x = K @ alpha
    alpha is reconstructed, then mapped to x.
    """
    n = N * N
    alpha = np.full(n, 0.2, dtype=np.float32) if alpha0 is None else alpha0.copy().astype(np.float32)
    eps = 1e-8

    for _ in range(n_iter):
        x_vec = K @ alpha
        x = x_vec.reshape(N, N)
        x = np.where(mask, x, 0.0)

        Ax = A(x)
        lam = np.maximum(Ax, eps)

        ratio = y / lam
        back = AT(ratio).ravel().astype(np.float32)

        numer = (K.T @ back).astype(np.float32)
        update = numer / KtS

        alpha *= update
        alpha = np.clip(alpha, 0, None)

    x_final = (K @ alpha).reshape(N, N)
    x_final = np.where(mask, x_final, 0.0)
    return x_final, alpha


# ============================================================
# Run reconstructions
# ============================================================
print("Running standard MLEM...")
recon_mlem = mlem_standard(y, num_iter)

print("Running kernel MLEM...")
recon_kernel, alpha_hat = mlem_kernel(y, num_iter, K, KtS)

# Post-smoothed MLEM for comparison
recon_mlem_smooth = gaussian_filter(recon_mlem, sigma=sigma_post)
recon_mlem_smooth = np.where(mask, recon_mlem_smooth, 0.0)

# Post-smoothed MLEM using Kernel for comparison
# recon_mlem = recon_mlem.reshape(N*N,1)
# recon_mlem_smooth = (K @ recon_mlem).reshape(N,N)
# recon_mlem_smooth = np.where(mask, recon_mlem_smooth, 0.0)
# recon_mlem = recon_mlem.reshape(N,N)

# ============================================================
# Normalize for display
# ============================================================
mean_truth = act[mask].mean()

def norm_by_masked_mean(x):
    m = x[mask].mean()
    if m <= 0:
        return x
    return x * (mean_truth / m)

act_disp = act
ref_disp = ref
recon_mlem_disp = norm_by_masked_mean(recon_mlem)
recon_smooth_disp = norm_by_masked_mean(recon_mlem_smooth)
recon_kernel_disp = norm_by_masked_mean(recon_kernel)

# ============================================================
# Line profiles
# ============================================================
row = N // 2
prof_truth = act_disp[row, :]
prof_mlem = recon_mlem_disp[row, :]
prof_smooth = recon_smooth_disp[row, :]
prof_kernel = recon_kernel_disp[row, :]

# ============================================================
# Plot images
# ============================================================
vmin, vmax = 0.0, 1.0

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

im0 = ax[0, 0].imshow(act_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Truth (brain-like PET phantom)")
ax[0, 0].axis('off')
plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

im1 = ax[0, 1].imshow(ref_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[0, 1].set_title("Reference image for kernel")
ax[0, 1].axis('off')
plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

im2 = ax[0, 2].imshow(y, cmap='inferno', aspect='auto')
ax[0, 2].set_title("Measured sinogram")
ax[0, 2].set_xlabel("Angle index")
ax[0, 2].set_ylabel("Detector bin")
plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

im3 = ax[1, 0].imshow(recon_mlem_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 0].set_title(f"Standard MLEM ({num_iter} iter)")
ax[1, 0].axis('off')
plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

im4 = ax[1, 1].imshow(recon_smooth_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 1].set_title(f"MLEM + Gaussian smoothing (σ={sigma_post})")
ax[1, 1].axis('off')
plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

im5 = ax[1, 2].imshow(recon_kernel_disp, cmap='gray', vmin=vmin, vmax=vmax)
ax[1, 2].set_title("Kernelized MLEM")
ax[1, 2].axis('off')
plt.colorbar(im5, ax=ax[1, 2], fraction=0.046, pad=0.04)

plt.suptitle("Kernelized PET Reconstruction Demo")
plt.tight_layout()
plt.show()

# ============================================================
# Plot profiles
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(prof_truth, label="Truth", linewidth=2)
plt.plot(prof_mlem, label="Standard MLEM", linewidth=2)
plt.plot(prof_smooth, label="MLEM + smoothing", linewidth=2)
plt.plot(prof_kernel, label="Kernelized MLEM", linewidth=2)

plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title("Central line profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Show one kernel row as a local weight map
# ============================================================
# Pick a central cortical pixel
r0, c0 = N // 2 - 8, N // 2 + 18
idx0 = r0 * N + c0
kernel_row = K.getrow(idx0).toarray().reshape(N, N)

plt.figure(figsize=(5, 4))
plt.imshow(kernel_row, cmap='viridis')
plt.title(f"Kernel weights for one pixel at ({r0}, {c0})")
# plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

