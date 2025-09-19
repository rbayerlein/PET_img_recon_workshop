# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:34:49 2025

@author: Reimund Bayerlein
"""

import numpy as np, matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon

# --- original settings (example) ---
N0 = 400
pix_size_cm0 = 0.1        # original pixel size [cm] for 440x440
FOV_cm = N0 * pix_size_cm0 # keep physical FOV constant

# --- choose downsampled size ---
N = 100                     # target resolution
pix_size_cm = FOV_cm / N    # doubles if N halves → preserves FOV

# --- create phantom ---
img = shepp_logan_phantom().astype(np.float32)
img_hi = resize(img, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
img    = resize(img_hi, (N, N),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

num_angles = 60
angles = np.linspace(0, 180, num_angles, endpoint=False)
num_iter = 1
poisson_scale = 5e3
total_counts = 50_000_000

# Forward model (A x)
def A(x):   return radon(x, theta=angles, circle=True)
# Approximate adjoint (Aᵀ y) via unfiltered backprojection
def AT(y):  return iradon(y, theta=angles, filter_name=None, circle=True, output_size=img.shape[0])

# Poisson data (set expected total counts via scale)
def pois(sino, scale=1e5, seed=0):
    rng = np.random.default_rng(seed)
    s = sino - sino.min()
    s = s / s.max() * scale
    return rng.poisson(s).astype(np.float32)

def poissonize_by_number(sino, total_counts, seed=None):
    s = np.clip(sino, 0, None)
    ssum = s.sum()
    if ssum == 0: 
        return np.zeros_like(s, np.float32)
    lam = s * (total_counts / ssum)
    rng = np.random.default_rng(seed)
    return rng.poisson(lam).astype(np.float32)

y_true = A(img)
# y_proj_data = pois(y_true, poisson_scale, seed=1)
y_proj_data = poissonize_by_number(y_true, total_counts, seed=1)
total_events = np.sum(y_proj_data)
evts_per_bin = total_events/y_proj_data.size
print("Total events:", total_events)
print("Average number of events per sinogram bin:", evts_per_bin)

# attenuation model
mu_water_cm = 0.096           # linear att. coeff of water at 511 keV [1/cm]
# μ-map: inside phantom = water, outside = 0
mu_map = np.zeros_like(img, dtype=np.float32).copy()
mu_map[img > 0] = mu_water_cm

# Path-integrated μ along each LOR: Radon(μ) * pixel_size
# (Radon sums pixel values; multiply by physical path per pixel to get ∫μ dl)
sino_mu = radon(mu_map, theta=angles, circle=True, preserve_range=True) * pix_size_cm

# Transmission factor T = exp(-∫μ dl)
T = np.exp(-np.clip(sino_mu, 0, None))

# ATTENUATED emission sinogram (pedagogical approximation)
y_proj_data = y_proj_data * T

# Sensitivity with attenuation
sens = AT(T.astype(np.float32))
sens[sens <= 0] = 1e-6

eps = 1e-8
# poisson log likelihood function
def poisson_loglik(x, A, y, r=None, s=None):
    lam = A(x)
    if r is not None: lam = lam + r  #for ranodoms
    if s is not None: lam = lam + s  #for scatters
    lam = np.clip(lam, eps, None)                 # avoid log(0)
    return float(np.sum(y * np.log(lam) - lam))   # up to constant

loglik_hist = [] # for saving log likelihood values
def mlem(y, n_iter=20):
    x = np.full(img.shape, 1.0, dtype=np.float32)   # non-negative start
    mask = np.zeros_like(img, dtype=bool)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    r = img.shape[0]/2 - 1
    mask = (yy - r)**2 + (xx - r)**2 <= r**2   # circular FOV
    
    eps = 1e-6
    for k in range(n_iter):
        
        # forward projection
        Ax = A(x) + eps        

        # update term  
        ratio = y / np.clip(Ax, eps, None)   
        
        # sensitivity image
        
        # back projection of the ratio
        BP = AT(ratio)
        
        # weighted back projection = correction term
        BP_W = BP / sens
        
        # update the image
        x *= BP_W

        if k == n_iter-1:
            fig, ax = plt.subplots(2, 4, figsize=(8,4))
            print("plotting...")
            im0 = ax[0,0].imshow(Ax, cmap='gray')
            ax[0,0].set_title("FP"); ax[0,0].axis('off')       
            
            im1 = ax[0,1].imshow(y, cmap='gray')
            ax[0,1].set_title("measurement"); ax[0,1].axis('off')
            
            im2 = ax[0,2].imshow(ratio, cmap='gray')
            ax[0,2].set_title("ratio y/FP"); ax[0,2].axis('off')
            
            im3 = ax[0,3].imshow(sens, cmap='gray')
            ax[0,3].set_title("sens img"); ax[0,3].axis('off')
            
            im4 = ax[1,0].imshow(BP, cmap='gray')
            ax[1,0].set_title("BP"); ax[1,0].axis('off')
            
            im5 = ax[1,1].imshow(BP_W, cmap='gray')
            ax[1,1].set_title("BP/sens"); ax[1,1].axis('off')
            
            im6 = ax[1,2].imshow(x, cmap='gray')
            ax[1,2].set_title("image update"); ax[1,2].axis('off')

            im7 = ax[1,3].imshow(img, cmap='gray')
            ax[1,3].set_title("Truth"); ax[1,3].axis('off')
            
            plt.tight_layout()
            plt.show()

        x *= mask               # keep solution inside FOV
        x = np.clip(x, 0, None) # non-negativity constraint
        ll = poisson_loglik(x, A, y)   # (add r=rand_sino, s=scat_sino if simulated)
        loglik_hist.append(ll)
    return x

recon_exmpl  = mlem(y_proj_data, num_iter)

all_images = [img, recon_exmpl] 
global_min = min(im.min() for im in all_images)
global_max = max(im.max() for im in all_images)
def global_norm(x):
    return (x - global_min) / (global_max - global_min + 1e-8)

mean_img = img.mean()
mean_recon = recon_exmpl.mean()
def norm_by_mean(x):
    return x * (mean_img/mean_recon)

# fig, ax = plt.subplots(1, 2, figsize=(8,4))

# im0 = ax[0].imshow(img, cmap='gray')
# ax[0].set_title("Truth"); ax[0].axis('off')
# plt.colorbar(im0, ax=ax[0])   # attach colorbar to this axis

# #recon_exmpl = np.clip(recon_exmpl, 0, 1) # limit display range to [0:1]
# im1 = ax[1].imshow(norm_by_mean(recon_exmpl), cmap='gray')
# ax[1].set_title(f"MLEM ({num_iter} iterations)"); ax[1].axis('off')
# plt.colorbar(im1, ax=ax[1])   # attach colorbar to this axis

# plt.tight_layout()
# plt.show()


# plot log likelihood
# plt.figure(figsize=(6,4))
# plt.plot(range(1, len(loglik_hist)+1), loglik_hist, marker='o')
# plt.xlabel('Iteration number')
# plt.ylabel('Poisson log-likelihood')
# plt.title('Convergence of Iterative Reconstruction')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
