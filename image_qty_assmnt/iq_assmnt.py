#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 2026

@author: rbayerlein


Interactive IQ assessment tool for 2D images

Features
--------
- Load an image
- Draw circular ROIs by mouse click
- ROI radius specified by user in pixels/voxels
- Label ROIs as hot / background / cold
- Save ROIs to JSON
- Enter ground-truth lesion-to-background ratio
- Compute and print image-quality metrics

Supported image inputs
----------------------
- .npy
- common image formats readable by matplotlib/Pillow (png, jpg, tif, etc.)

Metrics
-------
- For hot ROIs:
    * mean, std
    * contrast
    * CRC (if truth ratio provided)
    * CNR (using pooled background statistics)
- For background ROIs:
    * mean, std
    * CoV
    * background variability
    * SNR
- For cold ROIs:
    * mean, std
    * residual cold activity fraction
    * scatter clearance proxy = 1 - residual/background
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# Image loading
# ============================================================
def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        img = np.load(path)
    else:
        img = plt.imread(path)

        # If RGB/RGBA, convert to grayscale
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)

    img = np.asarray(img, dtype=np.float32)

    # If image has singleton dimensions, squeeze
    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}")

    return img


# ============================================================
# ROI helpers
# ============================================================
def make_circular_mask(shape, center, radius):
    H, W = shape
    cy, cx = center
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def roi_stats(img, roi):
    mask = make_circular_mask(img.shape, roi["center"], roi["radius"])
    vals = img[mask]
    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def pooled_background_stats(img, bg_rois):
    if len(bg_rois) == 0:
        return None

    all_vals = []
    roi_means = []

    for roi in bg_rois:
        mask = make_circular_mask(img.shape, roi["center"], roi["radius"])
        vals = img[mask]
        if vals.size > 0:
            all_vals.append(vals)
            roi_means.append(np.mean(vals))

    if len(all_vals) == 0:
        return None

    all_vals = np.concatenate(all_vals)
    roi_means = np.array(roi_means, dtype=np.float32)

    return {
        "n": int(all_vals.size),
        "mean": float(np.mean(all_vals)),
        "std": float(np.std(all_vals, ddof=1)) if all_vals.size > 1 else 0.0,
        "cov": float(np.std(all_vals, ddof=1) / np.mean(all_vals)) if np.mean(all_vals) != 0 else np.nan,
        "snr": float(np.mean(all_vals) / np.std(all_vals, ddof=1)) if np.std(all_vals, ddof=1) != 0 else np.nan,
        "roi_means": roi_means,
        "bv_percent": float(100.0 * np.std(roi_means, ddof=1) / np.mean(roi_means)) if (len(roi_means) > 1 and np.mean(roi_means) != 0) else np.nan,
    }


# ============================================================
# Metric computation
# ============================================================
def compute_metrics(img, rois, truth_ratio=None):
    hot_rois = [r for r in rois if r["label"] == "hot"]
    bg_rois = [r for r in rois if r["label"] == "background"]
    cold_rois = [r for r in rois if r["label"] == "cold"]

    bg_stats = pooled_background_stats(img, bg_rois)

    results = {
        "background": [],
        "hot": [],
        "cold": [],
        "summary": {}
    }

    # Background summary
    if bg_stats is not None:
        results["summary"]["background_mean"] = bg_stats["mean"]
        results["summary"]["background_std"] = bg_stats["std"]
        results["summary"]["background_cov"] = bg_stats["cov"]
        results["summary"]["background_snr"] = bg_stats["snr"]
        results["summary"]["background_variability_percent"] = bg_stats["bv_percent"]

    # Per-background ROI
    for i, roi in enumerate(bg_rois):
        s = roi_stats(img, roi)
        s["index"] = i + 1
        results["background"].append(s)

    # Per-hot ROI
    for i, roi in enumerate(hot_rois):
        s = roi_stats(img, roi)
        s["index"] = i + 1

        if bg_stats is not None and np.isfinite(bg_stats["mean"]) and bg_stats["mean"] != 0:
            contrast = s["mean"] / bg_stats["mean"]
            s["contrast"] = float(contrast)

            if truth_ratio is not None:
                true_contrast = truth_ratio - 1.0
                s["crc"] = float((contrast -1)/ (true_contrast)) if true_contrast != 0 else np.nan
            else:
                s["crc"] = np.nan

            if np.isfinite(bg_stats["std"]) and bg_stats["std"] != 0:
                s["cnr"] = float((s["mean"] - bg_stats["mean"]) / bg_stats["std"])
            else:
                s["cnr"] = np.nan
        else:
            s["contrast"] = np.nan
            s["crc"] = np.nan
            s["cnr"] = np.nan

        results["hot"].append(s)

    # Per-cold ROI
    for i, roi in enumerate(cold_rois):
        s = roi_stats(img, roi)
        s["index"] = i + 1

        if bg_stats is not None and np.isfinite(bg_stats["mean"]) and bg_stats["mean"] != 0:
            residual_frac = s["mean"] / bg_stats["mean"]
            s["residual_cold_fraction"] = float(residual_frac)
            s["scatter_clearance_proxy"] = float(1.0 - residual_frac)
        else:
            s["residual_cold_fraction"] = np.nan
            s["scatter_clearance_proxy"] = np.nan

        results["cold"].append(s)

    return results


# ============================================================
# Printing
# ============================================================
def print_results(results, truth_ratio=None):
    print("\n" + "=" * 70)
    print("IMAGE QUALITY ASSESSMENT RESULTS")
    print("=" * 70)

    if truth_ratio is not None:
        print(f"Ground-truth lesion-to-background ratio: {truth_ratio:.4f}")

    print("\n--- Background summary ---")
    summary = results["summary"]
    if len(summary) == 0:
        print("No background ROIs available.")
    else:
        print(f"Background mean                : {summary.get('background_mean', np.nan):.6f}")
        print(f"Background std                 : {summary.get('background_std', np.nan):.6f}")
        print(f"Coefficient of variation (CoV): {summary.get('background_cov', np.nan):.6f}")
        print(f"Background SNR                : {summary.get('background_snr', np.nan):.6f}")
        print(f"Background variability [%]    : {summary.get('background_variability_percent', np.nan):.6f}")

    print("\n--- Hot ROIs ---")
    if len(results["hot"]) == 0:
        print("No hot ROIs available.")
    else:
        for s in results["hot"]:
            print(f"\nHot ROI {s['index']}:")
            print(f"  mean      : {s['mean']:.6f}")
            print(f"  std       : {s['std']:.6f}")
            print(f"  contrast  : {s['contrast']:.6f}")
            print(f"  CRC       : {s['crc']:.6f}")
            print(f"  CNR       : {s['cnr']:.6f}")

    print("\n--- Cold ROIs ---")
    if len(results["cold"]) == 0:
        print("No cold ROIs available.")
    else:
        for s in results["cold"]:
            print(f"\nCold ROI {s['index']}:")
            print(f"  mean                     : {s['mean']:.6f}")
            print(f"  std                      : {s['std']:.6f}")
            print(f"  residual cold fraction   : {s['residual_cold_fraction']:.6f}")
            print(f"  scatter clearance proxy  : {s['scatter_clearance_proxy']:.6f}")

    print("\n" + "=" * 70 + "\n")


# ============================================================
# ROI editor
# ============================================================
class ROIEditor:
    def __init__(self, img, roi_radius_px, cold_roi_radius_px):
        self.img = img
        self.roi_radius_px = float(roi_radius_px)
        self.cold_roi_radius_px = float(cold_roi_radius_px)
        self.rois = []
        self.current_label = "hot"

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(img, cmap="gray")
        self.ax.set_title(self._title_text())
        self.ax.set_xlabel("Click to place ROI center")
        self._draw_help_text()

        # self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title_text(self):
        return (
            f"ROI label mode: {self.current_label.upper()} | "
            f"radius = {self.roi_radius_px:.1f} px | "
            f"Keys: h=hot, b=background, c=cold, u=undo, q=finish"
        )

    def _draw_help_text(self):
        txt = (
            "Controls:\n"
            " h = hot ROI\n"
            " b = background ROI\n"
            " c = cold ROI\n"
            " u = undo last ROI\n"
            " q = finish and close"
        )
        self.ax.text(
            0.88, 0.95, txt,
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    def redraw(self):
        self.ax.clear()
        self.im = self.ax.imshow(self.img, cmap="gray")
        self.ax.set_title(self._title_text())
        self.ax.set_xlabel("Click to place ROI center")
        self._draw_help_text()

        color_map = {
            "hot": "red",
            "background": "lime",
            "cold": "cyan",
        }

        for i, roi in enumerate(self.rois):
            cy, cx = roi["center"]
            radius = roi["radius"]
            color = color_map[roi["label"]]

            circ = Circle((cx, cy), radius, fill=False, edgecolor=color, linewidth=2)
            self.ax.add_patch(circ)
            self.ax.text(cx, cy, str(i + 1), color=color, fontsize=10, ha="center", va="center")

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        cx = float(event.xdata)
        cy = float(event.ydata)
        if self.current_label == "cold":
            this_radius = self.cold_roi_radius_px
        else:
            this_radius = self.roi_radius_px
        
        roi = {
            "label": self.current_label,
            "center": (cy, cx),
            "radius": this_radius
        }
        self.rois.append(roi)
        self.redraw()

    def on_key(self, event):
        if event.key == "h":
            self.current_label = "hot"
        elif event.key == "b":
            self.current_label = "background"
        elif event.key == "c":
            self.current_label = "cold"
        elif event.key == "u":
            if len(self.rois) > 0:
                self.rois.pop()
        elif event.key == "q":
            plt.close(self.fig)
            return

        self.redraw()

    def run(self):
        self.redraw()
        plt.show(block=True)
        return self.rois


# ============================================================
# Save / load ROI JSON
# ============================================================
def save_rois_json(rois, output_path):
    serializable = []
    for r in rois:
        serializable.append({
            "label": r["label"],
            "center": [float(r["center"][0]), float(r["center"][1])],
            "radius": float(r["radius"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# ============================================================
# Main
# ============================================================
def main():
    print("Interactive 2D image quality assessment")
    print("---------------------------------------")

    image_path = input("Enter image path (.npy, .png, .jpg, .tif, ...): ").strip()
    img = load_image(image_path)

    print(f"Loaded image with shape: {img.shape}")
    print(f"Image min/max: {img.min():.6f} / {img.max():.6f}")

    radius = float(input("Enter circular ROI radius in pixels/voxels: ").strip())
    radius_cold = float(input("Enter circular ROI radius in COLD region in pixels/voxels: ").strip())

    ratio_str = input(
        "Enter ground-truth lesion-to-background ratio "
        "(e.g. 4 for 4:1), or press Enter to skip CRC: "
    ).strip()
    
    print("Draw ROIs in the figure window. Press 'q' when finished.")

    print("\nOpen figure controls:")
    print("  h = hot ROI")
    print("  b = background ROI")
    print("  c = cold ROI")
    print("  u = undo last ROI")
    print("  q = finish\n")

    editor = ROIEditor(img, roi_radius_px=radius, cold_roi_radius_px=radius_cold)
    rois = editor.run()

    if len(rois) == 0:
        print("No ROIs were drawn. Exiting.")
        return

    # output_json = input("Enter output ROI JSON filename [default: rois.json]: ").strip()
    # if output_json == "":
    #     output_json = "rois.json"

    # save_rois_json(rois, output_json)
    # print(f"Saved {len(rois)} ROIs to: {output_json}")

    truth_ratio = None
    if ratio_str != "":
        truth_ratio = float(ratio_str)

    results = compute_metrics(img, rois, truth_ratio=truth_ratio)
    print_results(results, truth_ratio=truth_ratio)


if __name__ == "__main__":
    main()