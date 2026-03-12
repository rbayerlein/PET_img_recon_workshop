# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

"""
evaluate.py

Evaluation functions for a human-reader 2AFC experiment.

Main public functions
---------------------
evaluate_reader_performance(trial_table) -> updated trial_table
summarize_results(trial_table, config) -> summary_metrics_dict

Notes
-----
For a 2AFC experiment, the primary performance measure is percent correct.

An AUC-like estimate can also be reported. Under standard signal detection
theory assumptions, 2AFC proportion correct is numerically equivalent to
ROC AUC for the same discrimination task, so we report:

    auc_estimate = proportion_correct

This should be interpreted as a derived summary, not as an independently
measured ROC from rating data.
"""

import numpy as np
import pandas as pd


def _validate_trial_table_for_evaluation(trial_table: pd.DataFrame) -> None:
    """
    Check that trial_table contains the required columns.
    """
    required_cols = {
        "trial_id",
        "present_side",
        "reader_choice",
        "reaction_time_sec",
    }

    missing = required_cols - set(trial_table.columns)
    if missing:
        raise KeyError(f"trial_table is missing required columns: {sorted(missing)}")


def evaluate_reader_performance(trial_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compare reader choices with ground truth and fill the 'correct' column.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Must contain at least:
        - trial_id
        - present_side
        - reader_choice
        - reaction_time_sec

    Returns
    -------
    pd.DataFrame
        Updated copy of trial_table with 'correct' filled:
        - 1 for correct
        - 0 for incorrect
        - NaN if the trial has no valid response yet
    """
    _validate_trial_table_for_evaluation(trial_table)

    updated = trial_table.copy()

    correct_values = []

    for _, row in updated.iterrows():
        present_side = row["present_side"]
        reader_choice = row["reader_choice"]

        if pd.isna(reader_choice) or reader_choice is None:
            correct_values.append(np.nan)
            continue

        if reader_choice not in ("left", "right"):
            correct_values.append(np.nan)
            continue

        correct_values.append(1 if reader_choice == present_side else 0)

    updated["correct"] = correct_values
    return updated


def summarize_results(trial_table: pd.DataFrame, config: dict | None = None) -> dict:
    """
    Compute summary metrics for the 2AFC experiment.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Trial table after responses have been collected and correctness
        has been evaluated.
    config : dict | None
        Global configuration dictionary. Included for compatibility with
        main.py and possible future extensions.

    Returns
    -------
    dict
        Summary metrics dictionary with keys:
        - n_trials_total
        - n_trials_completed
        - n_trials_correct
        - percent_correct
        - proportion_correct
        - mean_reaction_time_sec
        - median_reaction_time_sec
        - auc_estimate

    Notes
    -----
    percent_correct is based only on completed trials.

    auc_estimate is reported as:
        auc_estimate = proportion_correct

    This is appropriate for 2AFC under the standard interpretation that
    2AFC proportion correct corresponds to ROC AUC for the same task.
    """
    _validate_trial_table_for_evaluation(trial_table)

    n_trials_total = int(len(trial_table))

    valid_choice_mask = trial_table["reader_choice"].isin(["left", "right"])
    completed_trials = trial_table.loc[valid_choice_mask].copy()
    n_trials_completed = int(len(completed_trials))

    if n_trials_completed == 0:
        return {
            "n_trials_total": n_trials_total,
            "n_trials_completed": 0,
            "n_trials_correct": 0,
            "percent_correct": np.nan,
            "proportion_correct": np.nan,
            "mean_reaction_time_sec": np.nan,
            "median_reaction_time_sec": np.nan,
            "auc_estimate": np.nan,
        }

    if "correct" not in completed_trials.columns:
        raise KeyError(
            "trial_table must contain a 'correct' column before calling summarize_results(). "
            "Run evaluate_reader_performance(trial_table) first."
        )

    correct_numeric = pd.to_numeric(completed_trials["correct"], errors="coerce")
    rt_numeric = pd.to_numeric(completed_trials["reaction_time_sec"], errors="coerce")

    n_trials_correct = int(np.nansum(correct_numeric))
    proportion_correct = float(np.nanmean(correct_numeric))
    percent_correct = 100.0 * proportion_correct

    mean_reaction_time_sec = float(np.nanmean(rt_numeric))
    median_reaction_time_sec = float(np.nanmedian(rt_numeric))

    # For 2AFC, proportion correct is commonly interpreted as equivalent to AUC
    auc_estimate = proportion_correct

    summary = {
        "n_trials_total": n_trials_total,
        "n_trials_completed": n_trials_completed,
        "n_trials_correct": n_trials_correct,
        "percent_correct": percent_correct,
        "proportion_correct": proportion_correct,
        "mean_reaction_time_sec": mean_reaction_time_sec,
        "median_reaction_time_sec": median_reaction_time_sec,
        "auc_estimate": auc_estimate,
    }

    return summary