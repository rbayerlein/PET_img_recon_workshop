# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

"""
experiment.py

Run a human-reader 2AFC experiment trial by trial.

This module expects display.py to provide:

    show_trial(left_image, right_image, trial_index, n_trials, config)
        -> reader_choice, reaction_time_sec

where:
- reader_choice is "left" or "right"
- reaction_time_sec is a float
"""

import pandas as pd


def _validate_trial_table(trial_table: pd.DataFrame) -> None:
    """
    Check that trial_table contains the required columns.
    """
    required_cols = {
        "trial_id",
        "left_image_id",
        "right_image_id",
        "present_side",
        "reader_choice",
        "correct",
        "reaction_time_sec",
    }

    missing = required_cols - set(trial_table.columns)
    if missing:
        raise KeyError(f"trial_table is missing required columns: {sorted(missing)}")


def _validate_images(images: dict) -> None:
    """
    Basic validation for the in-memory image dictionary.
    """
    if not isinstance(images, dict):
        raise TypeError("images must be a dictionary mapping image_id -> image array.")

    if len(images) == 0:
        raise ValueError("images dictionary is empty.")


def run_experiment(
    images: dict,
    image_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Run the human-reader 2AFC experiment.

    Parameters
    ----------
    images : dict
        Dictionary mapping image_id -> numpy array.
    image_table : pd.DataFrame
        Image metadata table. Included for consistency and future extension.
    trial_table : pd.DataFrame
        Trial table with one row per 2AFC trial.
    config : dict
        Global configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Updated trial_table including:
        - reader_choice
        - reaction_time_sec

    Notes
    -----
    This function does not compute correctness yet.
    That is handled later in evaluate.py.
    """
    _validate_images(images)
    _validate_trial_table(trial_table)

    try:
        from display import show_trial
    except ImportError as exc:
        raise ImportError(
            "run_experiment() requires display.py with a function:\n"
            "    show_trial(left_image, right_image, trial_index, n_trials, config)\n"
            "that returns: (reader_choice, reaction_time_sec)"
        ) from exc

    n_trials = len(trial_table)

    print("Starting 2AFC experiment...")
    print("Instructions: click on the LEFT or RIGHT image to indicate which contains the lesion.")
    print()

    updated_trial_table = trial_table.copy()

    for row_idx in range(n_trials):
        row = updated_trial_table.iloc[row_idx]

        trial_id = int(row["trial_id"])
        left_image_id = int(row["left_image_id"])
        right_image_id = int(row["right_image_id"])

        if left_image_id not in images:
            raise KeyError(f"left_image_id={left_image_id} not found in images dictionary.")
        if right_image_id not in images:
            raise KeyError(f"right_image_id={right_image_id} not found in images dictionary.")

        left_image = images[left_image_id]
        right_image = images[right_image_id]

        reader_choice, reaction_time_sec = show_trial(
            left_image=left_image,
            right_image=right_image,
            trial_index=row_idx,
            n_trials=n_trials,
            config=config,
        )

        updated_trial_table.at[row_idx, "reader_choice"] = reader_choice
        updated_trial_table.at[row_idx, "reaction_time_sec"] = reaction_time_sec

        print(
            f"Trial {trial_id + 1:03d}/{n_trials:03d} | "
            f"choice = {reader_choice:>5s} | "
            f"RT = {reaction_time_sec:.3f} s"
        )

    print("\nExperiment finished.")
    return updated_trial_table