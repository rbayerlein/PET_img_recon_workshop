# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:13:57 2025

Demonstrator of Forward Projection using a Shepp-Logan phantom
@author: Reimund Bayerlein
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

### USER PARAMETERS ###
N_projections = 128
num_angles = 180

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

# Plot original phantom and its sinogram
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(img_hi, cmap='gray')
ax[0].set_title('Original Shepp-Logan Phantom')
ax[0].axis('off')

ax[1].imshow(sinogram, cmap='gray', aspect='auto', extent=(0, 180, 0, sinogram.shape[0]))
ax[1].set_title('Forward Projection (Sinogram)')
ax[1].set_xlabel('Projection angle (degrees)')
ax[1].set_ylabel('Detector position')

plt.tight_layout()
plt.show()
