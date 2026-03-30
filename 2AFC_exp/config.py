#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2026

@author: rbayerlein
"""

CONFIG = {
    "experiment": {
        "n_trials": 20,
        "random_seed": 1234,
        "data_mode": "simulate",   # "simulate" or "load"
    },

    "image": {
        "image_size": 128,         # image is image_size x image_size
        "background_type": "homogeneous",
        "background_level": 2.0,
        "lesion_shape": "gaussian",
        "lesion_sigma": 3.0,
        "lesion_amplitude": 0.5,
        "lesion_center_mode": "fixed",   # "fixed" or "random"
        "lesion_center_x": 64,
        "lesion_center_y": 64,
        "noise_model": "gaussian",       # "gaussian" or "poisson"
        "noise_std": 1.0,
        "blur_sigma": 1.5,
    },

    "display": {
        "display_size_px": 400,
        "colormap": "binary",             # "gray", "binary"
        "vmin": 0.0,
        "vmax": None,
        "interpolation": "nearest",
    },

    "observer": {
        "observer_type": "template",
        "template_shape": "gaussian",
        "template_sigma": 3.0,
        "template_center_x": 64,
        "template_center_y": 64,
    },

    "output": {
        "save_example_figures": True,
        "save_roc_figure": True,
        "save_tables_csv": True,
        "output_dir": "outputs",
    },

    "input": {
        "image_dir": None,
        "labels_csv": None,
    }
}