#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""
"""
main.py

Main entry point for a human-reader 2AFC PET-like experiment.

Workflow
--------
1. Read configuration from config.py
2. Generate or load images
3. Build 2AFC trial table
4. Run the human-reader experiment
5. Evaluate reader performance
6. Save results

Notes
-----
This MVP is centered on a human observer, not a numerical observer.
A numerical observer can be added later as an optional comparison branch.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import CONFIG
from dataset import build_images, build_trials
from experiment import run_experiment
from evaluate import evaluate_reader_performance, summarize_results


def ensure_output_dir(output_dir: str) -> Path:
    """
    Create the output directory if it does not already exist.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def print_config_summary(config: dict) -> None:
    """
    Print a short summary of the current run configuration.
    """
    print("\n=== Human Reader 2AFC Experiment ===")
    print(f"Data mode        : {config['experiment']['data_mode']}")
    print(f"Number of trials : {config['experiment']['n_trials']}")
    print(f"Random seed      : {config['experiment']['random_seed']}")
    print(f"Image size       : {config['image']['image_size']} x {config['image']['image_size']}")
    print(f"Display size     : {config['display']['display_size_px']} px")
    print("===================================\n")


def save_tables(
    image_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    summary_metrics: dict,
    output_dir: Path,
) -> None:
    """
    Save experiment outputs to CSV files.
    """
    image_csv = output_dir / "image_table.csv"
    trial_csv = output_dir / "trial_table.csv"
    summary_csv = output_dir / "summary_metrics.csv"

    image_table.to_csv(image_csv, index=False)
    trial_table.to_csv(trial_csv, index=False)
    pd.DataFrame([summary_metrics]).to_csv(summary_csv, index=False)

    print(f"Saved image table   : {image_csv}")
    print(f"Saved trial table   : {trial_csv}")
    print(f"Saved summary table : {summary_csv}")


def main() -> None:
    """
    Run the full human-reader 2AFC workflow.
    """
    config = CONFIG
    rng = np.random.default_rng(config["experiment"]["random_seed"])

    print_config_summary(config)

    output_dir = ensure_output_dir(config["output"]["output_dir"])


    # ------------------------------------------------------------------
    # 1) Build or load images
    # ------------------------------------------------------------------
    # Expected return:
    # images: dict[image_id] -> np.ndarray
    # image_table: pd.DataFrame with one row per image
    images, image_table = build_images(
        config=config,
        rng=rng,
        output_dir=output_dir,
    )

    print(f"Prepared {len(images)} images.")

    # ------------------------------------------------------------------
    # 2) Build 2AFC trials
    # ------------------------------------------------------------------
    # Expected return:
    # trial_table: pd.DataFrame with one row per trial
    trial_table = build_trials(
        image_table=image_table,
        rng=rng,
    )

    print(f"Prepared {len(trial_table)} trials.")

    # ------------------------------------------------------------------
    # 3) Run the human-reader experiment
    # ------------------------------------------------------------------
    # Expected behavior:
    # - show each trial
    # - collect reader choice ("left" or "right")
    # - collect reaction time
    # - write results into trial_table
    trial_table = run_experiment(
        images=images,
        image_table=image_table,
        trial_table=trial_table,
        config=config,
    )

    # ------------------------------------------------------------------
    # 4) Evaluate reader performance
    # ------------------------------------------------------------------
    print("Evaluating trials...")
    # Expected behavior:
    # - compare reader_choice to present_side
    # - fill "correct"
    # - compute summary metrics such as percent correct
    trial_table = evaluate_reader_performance(trial_table=trial_table)

    summary_metrics = summarize_results(
        trial_table=trial_table,
        config=config,
    )

    # ------------------------------------------------------------------
    # 5) Print final summary
    # ------------------------------------------------------------------
    print("\n=== Results ===")
    print(f"Trials completed     : {summary_metrics['n_trials_completed']}")
    print(f"2AFC percent correct : {summary_metrics['percent_correct']:.2f}%")
    print(f"Mean reaction time   : {summary_metrics['mean_reaction_time_sec']:.3f} s")
    if "auc_estimate" in summary_metrics:
        print(f"AUC estimate         : {summary_metrics['auc_estimate']:.4f}")
    print("===============\n")

    # ------------------------------------------------------------------
    # 6) Save outputs
    # ------------------------------------------------------------------
    if config["output"]["save_tables_csv"]:
        save_tables(
            image_table=image_table,
            trial_table=trial_table,
            summary_metrics=summary_metrics,
            output_dir=output_dir,
        )

    print("Experiment complete.")


if __name__ == "__main__":
    main()