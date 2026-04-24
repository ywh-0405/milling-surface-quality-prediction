"""Synthetic data generation for demos and smoke testing."""

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_samples=60, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n_samples):
        spindle = rng.uniform(2500, 8000)
        feed = rng.uniform(0.02, 0.12)
        axial = rng.uniform(0.2, 2.0)
        radial = rng.uniform(0.5, 5.0)
        flutes = 4

        vib_x_rms = feed * axial * 0.9 + rng.uniform(0.01, 0.04)
        vib_y_rms = feed * radial * 0.4 + rng.uniform(0.01, 0.03)
        vib_z_rms = feed * axial * radial * 0.12 + rng.uniform(0.005, 0.02)

        row = {
            "sample_id": f"S{idx+1:03d}",
            "spindle_speed_rpm": spindle,
            "feed_per_tooth_mm": feed,
            "axial_depth_mm": axial,
            "radial_depth_mm": radial,
            "tool_flutes": flutes,
            "vib_x_rms": vib_x_rms,
            "vib_x_std": vib_x_rms * 0.85,
            "vib_x_peak": vib_x_rms * 2.2,
            "vib_x_pp": vib_x_rms * 4.0,
            "vib_y_rms": vib_y_rms,
            "vib_y_std": vib_y_rms * 0.88,
            "vib_y_peak": vib_y_rms * 2.1,
            "vib_y_pp": vib_y_rms * 3.8,
            "vib_z_rms": vib_z_rms,
            "vib_z_std": vib_z_rms * 0.9,
            "vib_z_peak": vib_z_rms * 2.0,
            "vib_z_pp": vib_z_rms * 3.6,
        }
        base_ra = 18.0 * (feed ** 1.2) * (axial ** 0.22) * (radial ** 0.15) * (4000.0 / spindle) ** 0.35
        row["Ra_um"] = float(max(0.08, base_ra + 0.6 * vib_z_rms + rng.normal(0, 0.03)))
        raw_bins = np.array([row["Ra_um"] / (i**1.25) for i in range(1, 9)], dtype=float)
        raw_bins += rng.uniform(0.0, 0.03, size=8)
        raw_bins = raw_bins / raw_bins.sum()
        for i, value in enumerate(raw_bins, start=1):
            row[f"freq_bin_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

