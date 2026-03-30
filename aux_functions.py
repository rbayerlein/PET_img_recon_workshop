#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:45:03 2026

@author: rbayerlein
"""

import numpy as np

def line_profile_at_height(img2d, y, x_start=0, x_end=None, method="nearest"):
    """
    Extract a horizontal line profile from a 2D image at a given height (row index).

    Parameters
    ----------
    img2d : array-like, shape (H, W)
        Input 2D image.
    y : float or int
        Height at which to extract the profile.
        - If int: uses that row index.
        - If float: uses interpolation (nearest or linear).
    x_start : int, optional
        Starting x index (inclusive). Default: 0.
    x_end : int or None, optional
        Ending x index (exclusive). Default: None (to image width).
    method : {"nearest", "linear"}, optional
        Interpolation method when y is not an integer. Default: "nearest".

    Returns
    -------
    profile : np.ndarray, shape (x_end-x_start,)
        The extracted line profile (float64).
    """
    img = np.asarray(img2d)
    if img.ndim != 2:
        raise ValueError(f"img2d must be 2D, got shape {img.shape}")

    H, W = img.shape
    if x_end is None:
        x_end = W

    # clamp x-range
    x_start = int(max(0, x_start))
    x_end = int(min(W, x_end))
    if x_end <= x_start:
        raise ValueError("x_end must be > x_start after clamping to image bounds")

    # clamp y-range
    if y < 0 or y > H - 1:
        raise ValueError(f"y must be in [0, {H-1}], got {y}")

    # integer y -> direct slice
    if isinstance(y, (int, np.integer)) or float(y).is_integer():
        yi = int(round(float(y)))
        return img[yi, x_start:x_end].astype(float)

    # fractional y -> interpolate between neighboring rows
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, H - 1)
    t = float(y) - y0

    row0 = img[y0, x_start:x_end].astype(float)
    row1 = img[y1, x_start:x_end].astype(float)

    if method == "nearest":
        return (row0 if t < 0.5 else row1)
    elif method == "linear":
        return (1.0 - t) * row0 + t * row1
    else:
        raise ValueError("method must be 'nearest' or 'linear'")
 