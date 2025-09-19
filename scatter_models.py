# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:08:47 2025

@author: Reimund Bayerlein
"""

# scatter_models.py
import numpy as np
from typing import Optional, Literal

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import scipy.signal as sig
    _HAS_SIG = True
except Exception:
    _HAS_SIG = False


Array = np.ndarray

def _thickness_weight(sino_mu: Array, kind: Literal["one_minus_exp","linear"] = "one_minus_exp") -> Array:
    """Map path integral ∫μdl (sino_mu) to a [0,1]-like weight: more thickness → more scatter."""
    u = np.clip(sino_mu, 0, None)
    if kind == "one_minus_exp":
        # 1 - exp(-∫μdl) saturates toward 1 for thick paths
        return 1.0 - np.exp(-u)
    elif kind == "linear":
        # crude linear scaling then normalize to [0,1] per-figure
        w = u.copy()
        m = w.max() or 1.0
        return w / m
    else:
        raise ValueError("Unknown kind for thickness weight.")


def _scale_to_scatter_fraction(
    primary: Array,
    scatter_unscaled: Array,
    randoms: Optional[Array] = None,
    sf: float = 0.35,
    definition: Literal["total","primary"] = "total",
    eps: float = 1e-12,
) -> Array:
    """
    Scale 'scatter_unscaled' so it achieves the desired scatter fraction 'sf'.

    definition = "total":  sf = S / (P + R + S)
    definition = "primary": sf = S / (P)

    Returns the scaled scatter.
    """
    P = float(np.clip(primary, 0, None).sum())
    R = float(np.clip(randoms, 0, None).sum()) if randoms is not None else 0.0
    S0 = float(np.clip(scatter_unscaled, 0, None).sum())

    if S0 <= eps:
        return np.zeros_like(primary, dtype=np.float32)

    if definition == "total":
        # Solve k*S0 / (P + R + k*S0) = sf  →  k = sf*(P+R) / (S0*(1-sf))
        k = sf * (P + R) / max(S0 * (1.0 - sf), eps)
    elif definition == "primary":
        # k*S0 / P = sf → k = sf*P / S0
        k = sf * P / max(S0, eps)
    else:
        raise ValueError("definition must be 'total' or 'primary'")

    return np.clip(scatter_unscaled * k, 0, None).astype(np.float32)


def make_scatter_blur(
    primary: Array,
    sino_mu: Array,
    sf: float = 0.35,
    randoms: Optional[Array] = None,
    sigma_det: float = 8.0,
    sigma_ang: float = 1.0,
    weight_kind: Literal["one_minus_exp","linear"] = "one_minus_exp",
    definition: Literal["total","primary"] = "total",
) -> Array:
    """
    Pedagogical scatter: broad 2D Gaussian blur of the primary, modulated by thickness.
    primary : expected true (non-random, non-scatter) attenuated sinogram, i.e. primary = T*(A x)
    sino_mu : path integral ∫μdl (same shape as primary)
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy.ndimage is required for make_scatter_blur()")

    base = gaussian_filter(np.clip(primary, 0, None), sigma=(sigma_det, sigma_ang))
    w = _thickness_weight(sino_mu, kind=weight_kind)
    s_unscaled = base * np.clip(w, 0, None)
    return _scale_to_scatter_fraction(primary, s_unscaled, randoms, sf, definition)


def _lorentz_kernel(L: int = 61, gamma: float = 10.0) -> Array:
    x = np.arange(L, dtype=np.float32) - (L // 2)
    k = 1.0 / (1.0 + (x / max(gamma, 1e-6)) ** 2)
    k /= k.sum()
    return k.astype(np.float32)


def make_scatter_tails(
    primary: Array,
    sino_mu: Array,
    sf: float = 0.35,
    randoms: Optional[Array] = None,
    kernel: Literal["lorentz","gauss"] = "lorentz",
    L: int = 61,
    gamma: float = 10.0,      # for lorentz
    sigma_det: float = 10.0,  # for gauss
    weight_kind: Literal["one_minus_exp","linear"] = "one_minus_exp",
    definition: Literal["total","primary"] = "total",
) -> Array:
    """
    Tail-kernel scatter: 1D convolution along detector bins for each angle (creates long tails).
    """
    if not _HAS_SIG and kernel == "lorentz":
        raise ImportError("scipy.signal is required for make_scatter_tails(kernel='lorentz').")
    if not _HAS_SCIPY and kernel == "gauss":
        raise ImportError("scipy.ndimage is required for make_scatter_tails(kernel='gauss').")

    prim = np.clip(primary, 0, None).astype(np.float32)
    H, W = prim.shape

    if kernel == "lorentz":
        k = _lorentz_kernel(L=L, gamma=gamma)
        s_unscaled = np.empty_like(prim, dtype=np.float32)
        for j in range(W):
            s_unscaled[:, j] = sig.convolve(prim[:, j], k, mode="same")
    else:  # "gauss"
        s_unscaled = gaussian_filter(prim, sigma=(sigma_det, 0.0))  # blur only along detector

    w = _thickness_weight(sino_mu, kind=weight_kind)
    s_unscaled *= np.clip(w, 0, None)
    return _scale_to_scatter_fraction(primary, s_unscaled, randoms, sf, definition)


def make_scatter_iterative_from_image(
    x_img: Array,
    A,                     # callable: image -> sinogram (Radon)
    T_sino: Array,         # transmission sinogram (same shape as A(x))
    sino_mu: Array,
    sf: float = 0.35,
    randoms: Optional[Array] = None,
    model: Literal["blur","tails"] = "blur",
    **kwargs,              # forwarded to model maker (sigma_det, kernel params, etc.)
) -> Array:
    """
    Iterative proxy (SSS-like): build primary from current image estimate, then model scatter.
    Call every m iterations if you like.
    """
    p_hat = np.clip(T_sino * np.clip(A(x_img), 0, None), 0, None)

    if model == "blur":
        return make_scatter_blur(p_hat, sino_mu, sf=sf, randoms=randoms, **kwargs)
    elif model == "tails":
        return make_scatter_tails(p_hat, sino_mu, sf=sf, randoms=randoms, **kwargs)
    else:
        raise ValueError("model must be 'blur' or 'tails'")
