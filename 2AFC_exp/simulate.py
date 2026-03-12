# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

"""
simulate.py

Synthetic image generation for a simple human-reader 2AFC experiment.

This MVP supports:
- homogeneous background
- Gaussian lesion
- Gaussian or Poisson noise
- optional Gaussian blur

Main public function
--------------------
generate_image_pair(config, rng) -> absent_img, present_img, meta
"""

from pathlib import Path
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError as exc:
    raise ImportError(
        "simulate.py requires scipy. Please install scipy in your Python environment."
    ) from exc


def make_background(image_size: int, background_level: float) -> np.ndarray:
    """
    Create a homogeneous 2D background image.

    Parameters
    ----------
    image_size : int
        Image will be image_size x image_size.
    background_level : float
        Constant background intensity.

    Returns
    -------
    np.ndarray
        2D float image.
    """
    return np.full((image_size, image_size), background_level, dtype=np.float64)


def make_gaussian_lesion(
    image_size: int,
    center_x: float,
    center_y: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    """
    Create a 2D Gaussian lesion image.

    Parameters
    ----------
    image_size : int
        Image will be image_size x image_size.
    center_x : float
        Lesion center x-coordinate in pixels.
    center_y : float
        Lesion center y-coordinate in pixels.
    sigma : float
        Gaussian sigma in pixels.
    amplitude : float
        Peak lesion amplitude.

    Returns
    -------
    np.ndarray
        2D lesion image.
    """
    y, x = np.meshgrid(
        np.arange(image_size, dtype=np.float64),
        np.arange(image_size, dtype=np.float64),
        indexing="ij",
    )

    lesion = amplitude * np.exp(
        -((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * sigma ** 2)
    )
    return lesion


def sample_lesion_center(config: dict, rng: np.random.Generator) -> tuple[float, float]:
    """
    Determine lesion center from config.

    Supports:
    - fixed center
    - random center (kept away from borders)

    Parameters
    ----------
    config : dict
        Global configuration dictionary.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    (float, float)
        center_x, center_y
    """
    image_cfg = config["image"]
    image_size = int(image_cfg["image_size"])
    center_mode = image_cfg.get("lesion_center_mode", "fixed")

    if center_mode == "fixed":
        return float(image_cfg["lesion_center_x"]), float(image_cfg["lesion_center_y"])

    if center_mode == "random":
        sigma = float(image_cfg["lesion_sigma"])
        margin = max(5, int(np.ceil(3 * sigma)))
        center_x = rng.integers(margin, image_size - margin)
        center_y = rng.integers(margin, image_size - margin)
        return float(center_x), float(center_y)

    raise ValueError(
        f"Unsupported lesion_center_mode: {center_mode!r}. "
        "Use 'fixed' or 'random'."
    )


def add_gaussian_noise(
    image: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    noise_std : float
        Standard deviation of Gaussian noise.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Noisy image.
    """
    noise = rng.normal(loc=0.0, scale=noise_std, size=image.shape)
    return image + noise


def add_poisson_noise(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply Poisson noise directly to the image.

    Notes
    -----
    This assumes the image values are nonnegative and interpretable as
    expected counts. For this MVP, that is acceptable as a simple model.

    Parameters
    ----------
    image : np.ndarray
        Input mean image. Must be nonnegative.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Poisson-noisy image as float64.
    """
    clipped = np.clip(image, a_min=0.0, a_max=None)
    noisy = rng.poisson(clipped)
    return noisy.astype(np.float64)


def apply_blur(image: np.ndarray, blur_sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    blur_sigma : float
        Sigma of Gaussian blur in pixels.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if blur_sigma is None or blur_sigma <= 0:
        return image
    return gaussian_filter(image, sigma=blur_sigma)


def finalize_image(image: np.ndarray) -> np.ndarray:
    """
    Final cleanup for simulated images.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Finalized image with nonnegative intensities.
    """
    return np.clip(image, a_min=0.0, a_max=None)


def simulate_from_mean(
    mean_image: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Turn a mean image into a simulated final image using the configured
    noise model and blur.

    Parameters
    ----------
    mean_image : np.ndarray
        Mean image before noise.
    config : dict
        Global configuration dictionary.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Final simulated image.
    """
    image_cfg = config["image"]
    noise_model = image_cfg.get("noise_model", "gaussian")
    blur_sigma = float(image_cfg.get("blur_sigma", 0.0))

    if noise_model == "gaussian":
        noise_std = float(image_cfg.get("noise_std", 0.0))
        sim_img = add_gaussian_noise(mean_image, noise_std=noise_std, rng=rng)

    elif noise_model == "poisson":
        sim_img = add_poisson_noise(mean_image, rng=rng)

    else:
        raise ValueError(
            f"Unsupported noise_model: {noise_model!r}. "
            "Use 'gaussian' or 'poisson'."
        )

    sim_img = apply_blur(sim_img, blur_sigma=blur_sigma)
    sim_img = finalize_image(sim_img)
    return sim_img


def generate_image_pair(
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate one absent/present image pair for a 2AFC trial.

    Parameters
    ----------
    config : dict
        Global configuration dictionary.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    absent_img : np.ndarray
        Simulated lesion-absent image.
    present_img : np.ndarray
        Simulated lesion-present image.
    meta : dict
        Metadata describing the generated pair.
    """
    image_cfg = config["image"]

    image_size = int(image_cfg["image_size"])
    background_type = image_cfg.get("background_type", "homogeneous")
    background_level = float(image_cfg["background_level"])

    lesion_shape = image_cfg.get("lesion_shape", "gaussian")
    lesion_sigma = float(image_cfg["lesion_sigma"])
    lesion_amplitude = float(image_cfg["lesion_amplitude"])

    noise_model = image_cfg.get("noise_model", "gaussian")
    noise_std = image_cfg.get("noise_std", np.nan)
    blur_sigma = float(image_cfg.get("blur_sigma", 0.0))

    if background_type != "homogeneous":
        raise ValueError(
            f"Unsupported background_type: {background_type!r}. "
            "This MVP currently supports only 'homogeneous'."
        )

    if lesion_shape != "gaussian":
        raise ValueError(
            f"Unsupported lesion_shape: {lesion_shape!r}. "
            "This MVP currently supports only 'gaussian'."
        )

    center_x, center_y = sample_lesion_center(config=config, rng=rng)

    # Mean background
    background = make_background(
        image_size=image_size,
        background_level=background_level,
    )

    # Lesion signal
    lesion = make_gaussian_lesion(
        image_size=image_size,
        center_x=center_x,
        center_y=center_y,
        sigma=lesion_sigma,
        amplitude=lesion_amplitude,
    )

    # Mean images before noise
    absent_mean = background.copy()
    present_mean = background + lesion

    # Final simulated images
    absent_img = simulate_from_mean(absent_mean, config=config, rng=rng)
    present_img = simulate_from_mean(present_mean, config=config, rng=rng)

    meta = {
        "lesion_center_x": center_x,
        "lesion_center_y": center_y,
        "lesion_sigma": lesion_sigma,
        "lesion_amplitude": lesion_amplitude,
        "background_type": background_type,
        "background_level": background_level,
        "noise_model": noise_model,
        "noise_std": float(noise_std) if noise_model == "gaussian" else np.nan,
        "blur_sigma": blur_sigma,
    }

    return absent_img, present_img, meta