#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

"""
dataset.py

Dataset construction for a simple human-reader 2AFC experiment.

This module provides:
- build_images(...): create or load the image set and image_table
- build_trials(...): pair images into 2AFC trials and randomize left/right order

Assumptions
-----------
1. For data_mode == "simulate", this module expects simulate.py to provide:

    generate_image_pair(config, rng) -> absent_img, present_img, meta

   where:
   - absent_img : 2D numpy array
   - present_img: 2D numpy array
   - meta       : dict with optional keys:
       lesion_center_x, lesion_center_y,
       lesion_sigma, lesion_amplitude,
       background_type, background_level,
       noise_model, noise_std, blur_sigma

2. For data_mode == "load", a loader is not implemented yet.

3. Images are kept in memory in a dictionary:
       images[image_id] = np.ndarray

4. image_table contains one row per image.
5. trial_table contains one row per 2AFC trial.
"""

from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


def _empty_image_record() -> dict:
    """
    Return a template dictionary for one row of image_table.
    """
    return {
        "image_id": None,
        "trial_id": None,
        "label": None,               # 0 absent, 1 present
        "role": None,                # "absent" or "present"
        "source": None,              # "simulated" or "loaded"
        "image_path": None,
        "lesion_center_x": np.nan,
        "lesion_center_y": np.nan,
        "lesion_sigma": np.nan,
        "lesion_amplitude": np.nan,
        "background_type": None,
        "background_level": np.nan,
        "noise_model": None,
        "noise_std": np.nan,
        "blur_sigma": np.nan,
    }


def _save_image_array(image: np.ndarray, image_id: int, output_dir: Path | None) -> str | None:
    """
    Save one image as .npy if an output directory is provided.

    Parameters
    ----------
    image : np.ndarray
        Image array to save.
    image_id : int
        Unique image ID.
    output_dir : Path | None
        Base output directory.

    Returns
    -------
    str | None
        File path as string if saved, otherwise None.
    """
    if output_dir is None:
        return None

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"img_{image_id:04d}.npy"
    np.save(image_path, image)
    return str(image_path)


def _validate_config_for_build_images(config: dict) -> None:
    """
    Perform minimal config validation for dataset generation.
    """
    if "experiment" not in config:
        raise KeyError("CONFIG must contain an 'experiment' section.")
    if "image" not in config:
        raise KeyError("CONFIG must contain an 'image' section.")

    n_trials = config["experiment"].get("n_trials", None)
    if n_trials is None or int(n_trials) <= 0:
        raise ValueError("CONFIG['experiment']['n_trials'] must be a positive integer.")

    data_mode = config["experiment"].get("data_mode", None)
    if data_mode not in ("simulate", "load"):
        raise ValueError("CONFIG['experiment']['data_mode'] must be 'simulate' or 'load'.")


def build_images(
    config: dict,
    rng: np.random.Generator,
    output_dir: str | Path | None = None,
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """
    Build the image dataset and the corresponding image_table.

    Parameters
    ----------
    config : dict
        Global experiment configuration.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    output_dir : str | Path | None
        Optional base output directory. If provided, images are saved as .npy.

    Returns
    -------
    images : dict[int, np.ndarray]
        Dictionary mapping image_id -> image array.
    image_table : pd.DataFrame
        One row per image with metadata.

    Notes
    -----
    For the MVP:
    - Each trial contributes exactly 2 images:
        1 absent image
        1 present image
    - Therefore total number of images = 2 * n_trials
    """
    _validate_config_for_build_images(config)

    if output_dir is not None:
        output_dir = Path(output_dir)

    data_mode = config["experiment"]["data_mode"]
    n_trials = int(config["experiment"]["n_trials"])

    images: Dict[int, np.ndarray] = {}
    records: list[dict] = []

    if data_mode == "simulate":
        # Delayed import so that dataset.py can still be inspected independently.
        try:
            from simulate import generate_image_pair
        except ImportError as exc:
            raise ImportError(
                "build_images() requires simulate.py with a function:\n"
                "    generate_image_pair(config, rng) -> absent_img, present_img, meta"
            ) from exc

        image_id = 0

        for trial_id in range(n_trials):
            absent_img, present_img, meta = generate_image_pair(config=config, rng=rng)

            if not isinstance(absent_img, np.ndarray) or not isinstance(present_img, np.ndarray):
                raise TypeError("generate_image_pair() must return numpy arrays for absent_img and present_img.")

            # ---- absent image record ----
            absent_record = _empty_image_record()
            absent_record["image_id"] = image_id
            absent_record["trial_id"] = trial_id
            absent_record["label"] = 0
            absent_record["role"] = "absent"
            absent_record["source"] = "simulated"
            absent_record["image_path"] = _save_image_array(absent_img, image_id, output_dir)

            absent_record["lesion_center_x"] = meta.get("lesion_center_x", np.nan)
            absent_record["lesion_center_y"] = meta.get("lesion_center_y", np.nan)
            absent_record["lesion_sigma"] = meta.get("lesion_sigma", np.nan)
            absent_record["lesion_amplitude"] = 0.0
            absent_record["background_type"] = meta.get("background_type", "homogeneous")
            absent_record["background_level"] = meta.get("background_level", np.nan)
            absent_record["noise_model"] = meta.get("noise_model", None)
            absent_record["noise_std"] = meta.get("noise_std", np.nan)
            absent_record["blur_sigma"] = meta.get("blur_sigma", np.nan)

            images[image_id] = absent_img
            records.append(absent_record)
            image_id += 1

            # ---- present image record ----
            present_record = _empty_image_record()
            present_record["image_id"] = image_id
            present_record["trial_id"] = trial_id
            present_record["label"] = 1
            present_record["role"] = "present"
            present_record["source"] = "simulated"
            present_record["image_path"] = _save_image_array(present_img, image_id, output_dir)

            present_record["lesion_center_x"] = meta.get("lesion_center_x", np.nan)
            present_record["lesion_center_y"] = meta.get("lesion_center_y", np.nan)
            present_record["lesion_sigma"] = meta.get("lesion_sigma", np.nan)
            present_record["lesion_amplitude"] = meta.get("lesion_amplitude", np.nan)
            present_record["background_type"] = meta.get("background_type", "homogeneous")
            present_record["background_level"] = meta.get("background_level", np.nan)
            present_record["noise_model"] = meta.get("noise_model", None)
            present_record["noise_std"] = meta.get("noise_std", np.nan)
            present_record["blur_sigma"] = meta.get("blur_sigma", np.nan)

            images[image_id] = present_img
            records.append(present_record)
            image_id += 1

    elif data_mode == "load":
        raise NotImplementedError(
            "data_mode='load' is not implemented yet. "
            "For now, use data_mode='simulate'."
        )

    image_table = pd.DataFrame.from_records(records)
    image_table = image_table.sort_values(by="image_id").reset_index(drop=True)

    return images, image_table


def build_trials(
    image_table: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build the 2AFC trial table from image_table.

    Parameters
    ----------
    image_table : pd.DataFrame
        Must contain, at minimum:
        - image_id
        - trial_id
        - label
        - role

    rng : numpy.random.Generator
        Random number generator used to randomize whether the present image
        appears on the left or right.

    Returns
    -------
    trial_table : pd.DataFrame
        One row per trial with columns:
        - trial_id
        - left_image_id
        - right_image_id
        - present_side
        - left_label
        - right_label
        - reader_choice
        - correct
        - reaction_time_sec

    Notes
    -----
    This function assumes exactly one absent image and one present image
    per trial_id.
    """
    required_cols = {"image_id", "trial_id", "label", "role"}
    missing_cols = required_cols - set(image_table.columns)
    if missing_cols:
        raise KeyError(f"image_table is missing required columns: {sorted(missing_cols)}")

    trial_records: list[dict] = []

    grouped = image_table.groupby("trial_id", sort=True)

    for trial_id, group in grouped:
        if len(group) != 2:
            raise ValueError(
                f"trial_id={trial_id} has {len(group)} images in image_table. "
                "Each trial must have exactly 2 images."
            )

        absent_rows = group[group["label"] == 0]
        present_rows = group[group["label"] == 1]

        if len(absent_rows) != 1 or len(present_rows) != 1:
            raise ValueError(
                f"trial_id={trial_id} must contain exactly one absent image (label=0) "
                f"and one present image (label=1)."
            )

        absent_image_id = int(absent_rows.iloc[0]["image_id"])
        present_image_id = int(present_rows.iloc[0]["image_id"])

        present_on_left = bool(rng.integers(0, 2))

        if present_on_left:
            left_image_id = present_image_id
            right_image_id = absent_image_id
            present_side = "left"
            left_label = 1
            right_label = 0
        else:
            left_image_id = absent_image_id
            right_image_id = present_image_id
            present_side = "right"
            left_label = 0
            right_label = 1

        trial_record = {
            "trial_id": int(trial_id),
            "left_image_id": left_image_id,
            "right_image_id": right_image_id,
            "present_side": present_side,
            "left_label": left_label,
            "right_label": right_label,
            "reader_choice": None,
            "correct": np.nan,
            "reaction_time_sec": np.nan,
        }

        trial_records.append(trial_record)

    trial_table = pd.DataFrame.from_records(trial_records)
    trial_table = trial_table.sort_values(by="trial_id").reset_index(drop=True)

    return trial_table