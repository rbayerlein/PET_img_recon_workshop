#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:21:32 2026

@author: rbayerlein
"""

import numpy as np
from scipy.ndimage import gaussian_filter

#####################################################################

def make_disk_phantom(N):
    """Create a simple phantom with several hot disks."""
    img = np.zeros((N, N), dtype=np.float32)

    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2, (N - 1) / 2

    def add_disk(x0, y0, r, val):
        mask = (xx - x0)**2 + (yy - y0)**2 <= r**2
        img[mask] = val

    # Background object
    add_disk(cx, cy, 40, 0.2)

    # Hot lesions of different size
    add_disk(cx - 18, cy - 10, 8, 1.0)
    add_disk(cx + 15, cy - 12, 5, 1.0)
    add_disk(cx + 5,  cy + 18, 3, 1.0)

    return img

#####################################################################
def make_circular_mask(N):
    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2, (N - 1) / 2
    r = N / 2 - 1
    return ((xx - cx)**2 + (yy - cy)**2 <= r**2)



#####################################################################
def create_iq_phantom(
    N=128,
    background_value=1.0,
    lesion_to_background_ratio=4.0,
    lesion_diameters_px=(6, 10, 14, 18),
    cold_diameter_px=16,
    smooth_sigma=0.0,
):
    """
    Create a 2D image-quality phantom for PET demonstrations.

    Phantom contents:
    - circular warm background
    - multiple hot lesions of different sizes
    - one cold insert

    Parameters
    ----------
    N : int
        Image size (NxN).
    background_value : float
        Activity concentration in the warm background.
    lesion_to_background_ratio : float
        Hot lesion uptake divided by background uptake.
        Example: 4.0 means lesions have value 4 * background_value.
    lesion_diameters_px : tuple of int
        Diameters of hot lesions in pixels.
    cold_diameter_px : int
        Diameter of cold insert in pixels.
    smooth_sigma : float
        Optional Gaussian smoothing applied to phantom edges.
        Use 0 for perfectly sharp phantom.

    Returns
    -------
    img : ndarray, shape (N, N)
        Phantom image.
    info : dict
        Dictionary containing metadata:
        - 'mask_body'
        - 'hot_lesions': list of dicts with center/radius/value
        - 'cold_insert': dict with center/radius/value
        - 'background_value'
        - 'lesion_value'
    """

    img = np.zeros((N, N), dtype=np.float32)

    yy, xx = np.ogrid[:N, :N]
    cy, cx = (N - 1) / 2.0, (N - 1) / 2.0

    # Main body / warm background
    body_radius = int(0.42 * N)
    mask_body = (xx - cx) ** 2 + (yy - cy) ** 2 <= body_radius ** 2
    img[mask_body] = background_value

    lesion_value = background_value * lesion_to_background_ratio

    # Place hot lesions in upper half / lateral regions
    # Angles chosen so objects are well separated
    lesion_angles_deg = [-180, -135, -90, -45, 0, 45, 90]
    lesion_ring_radius = 0.23 * N

    hot_lesions = []
    n_les = len(lesion_diameters_px)

    for i in range(n_les):
        d = lesion_diameters_px[i]
        r = d / 2.0
        ang = np.deg2rad(lesion_angles_deg[i % len(lesion_angles_deg)])

        x0 = cx + lesion_ring_radius * np.cos(ang)
        y0 = cy + lesion_ring_radius * np.sin(ang)

        lesion_mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r ** 2
        img[lesion_mask] = lesion_value

        hot_lesions.append(
            {
                "center": (float(y0), float(x0)),
                "diameter_px": float(d),
                "radius_px": float(r),
                "value": float(lesion_value),
            }
        )

    # Cold insert in lower half
    cold_r = cold_diameter_px / 2.0
    cold_x = cx
    cold_y = cy + 0.22 * N
    cold_mask = (xx - cold_x) ** 2 + (yy - cold_y) ** 2 <= cold_r ** 2
    img[cold_mask] = 0.0

    cold_insert = {
        "center": (float(cold_y), float(cold_x)),
        "diameter_px": float(cold_diameter_px),
        "radius_px": float(cold_r),
        "value": 0.0,
    }

    # Zero outside body
    img[~mask_body] = 0.0

    # Optional smoothing
    if smooth_sigma > 0:
        img = gaussian_filter(img, sigma=smooth_sigma)
        img[~mask_body] = 0.0

    info = {
        "mask_body": mask_body,
        "hot_lesions": hot_lesions,
        "cold_insert": cold_insert,
        "background_value": float(background_value),
        "lesion_value": float(lesion_value),
    }

    return img.astype(np.float32), info