# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:30:14 2025

Demonstrator of Back Projection using a Shepp-Logan phantom
@author: Reimund Bayerlein
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon
from skimage.transform import radon, resize

### USER PARAMETERS ###
N_projections = 48
num_angles = 90

#######################

# Generate phantom
image = shepp_logan_phantom()
# image = np.pad(image, pad_width=20, mode='constant')  # Padding for clarity
N0 = 400
img_hi = resize(image, (N0, N0), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
image = resize(img_hi, (N_projections, N_projections),   order=3, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

# Define projection angles
angles = np.linspace(0., 180., num_angles, endpoint=False)

# Forward projection (Radon transform)
sinogram = radon(image, theta=angles)

# Simple unfiltered backprojection
reconstruction_bp = iradon(sinogram, theta=angles, filter_name=None)

# Filtered backprojection (standard reconstruction)
reconstruction_fbp = iradon(sinogram, theta=angles, filter_name='ramp')
# filter types: shepp-logan, ramp

# Plot reconstructions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img_hi, cmap='gray')
ax[0].set_title('Original Phantom')
ax[0].axis('off')

ax[1].imshow(reconstruction_bp, cmap='gray')
ax[1].set_title('Unfiltered Backprojection')
ax[1].axis('off')

ax[2].imshow(reconstruction_fbp, cmap='gray')
ax[2].set_title('Filtered Backprojection')
ax[2].axis('off')

plt.tight_layout()
plt.show()
