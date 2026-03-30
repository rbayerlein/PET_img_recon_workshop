# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

"""
display.py

Interactive display for a human-reader 2AFC experiment.

Main public function
--------------------
show_trial(left_image, right_image, trial_index, n_trials, config)
    -> reader_choice, reaction_time_sec

reader_choice is:
- "left"
- "right"

reaction_time_sec is the elapsed time between image display and mouse click.
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def _compute_figsize_from_display_px(display_size_px: int, dpi: int = 100) -> tuple[float, float]:
    """
    Convert desired per-image display size in pixels into a matplotlib figure size.

    Parameters
    ----------
    display_size_px : int
        Approximate display width/height for each image in pixels.
    dpi : int
        Figure DPI.

    Returns
    -------
    (fig_width_in, fig_height_in) : tuple[float, float]
        Figure size in inches.
    """
    # Two images side by side, plus some extra room for margins/title
    fig_width_px = 2 * display_size_px + 120
    fig_height_px = display_size_px + 120

    return fig_width_px / dpi, fig_height_px / dpi


def _get_display_limits(config: dict, left_image: np.ndarray, right_image: np.ndarray) -> tuple[float | None, float | None]:
    """
    Get display intensity limits from config.

    If vmax is None, determine it from the current pair of images.
    """
    display_cfg = config["display"]
    vmin = display_cfg.get("vmin", None)
    vmax = display_cfg.get("vmax", None)

    if vmax is None:
        vmax = float(max(np.max(left_image), np.max(right_image)))

    return vmin, vmax


def show_trial(
    left_image: np.ndarray,
    right_image: np.ndarray,
    trial_index: int,
    n_trials: int,
    config: dict,
) -> tuple[str, float]:
    """
    Show one 2AFC trial and collect a mouse-click response.

    Parameters
    ----------
    left_image : np.ndarray
        Image shown on the left.
    right_image : np.ndarray
        Image shown on the right.
    trial_index : int
        Zero-based trial index.
    n_trials : int
        Total number of trials.
    config : dict
        Global configuration dictionary.

    Returns
    -------
    reader_choice : str
        "left" or "right"
    reaction_time_sec : float
        Time from image appearance to mouse click in seconds.

    Notes
    -----
    The user must click inside either the left or right image axes.
    Clicks outside the image axes are ignored.
    """
    display_cfg = config["display"]
    display_size_px = int(display_cfg.get("display_size_px", 400))
    cmap = display_cfg.get("colormap", "gray")
    interpolation = display_cfg.get("interpolation", "nearest")

    fig_w, fig_h = _compute_figsize_from_display_px(display_size_px)
    vmin, vmax = _get_display_limits(config, left_image, right_image)

    # Container for the response
    response = {
        "choice": None,
        "reaction_time_sec": None,
        "t0": None,
    }

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=100)
    ax_left, ax_right = axes

    ax_left.imshow(left_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax_right.imshow(right_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)

    ax_left.set_title("Left")
    ax_right.set_title("Right")

    ax_left.axis("off")
    ax_right.axis("off")

    fig.suptitle(
        f"Trial {trial_index + 1} / {n_trials}\nClick the image that contains the lesion",
        fontsize=12,
    )

    plt.tight_layout()

    def on_click(event):
        """
        Mouse-click callback. Records response if click occurred in left or right axes.
        """
        if event.inaxes is None:
            return

        if event.inaxes == ax_left:
            response["choice"] = "left"
        elif event.inaxes == ax_right:
            response["choice"] = "right"
        else:
            return

        response["reaction_time_sec"] = time.perf_counter() - response["t0"]
        plt.close(fig)

    # Start timing immediately before display becomes interactive
    response["t0"] = time.perf_counter()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)

    # Safety check in case window was closed without a valid click
    if response["choice"] is None or response["reaction_time_sec"] is None:
        raise RuntimeError(
            "Trial window was closed without a valid left/right click."
        )

    return response["choice"], float(response["reaction_time_sec"])