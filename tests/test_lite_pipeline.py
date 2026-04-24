import pandas as pd

from src.models_lite import train_lite_models


def test_train_lite_models_returns_metrics():
    features = pd.DataFrame(
        [
            {
                "sample_id": f"S{i}",
                "spindle_speed_rpm": 4000 + i * 150,
                "feed_per_tooth_mm": 0.03 + i * 0.002,
                "axial_depth_mm": 0.8 + i * 0.05,
                "radial_depth_mm": 1.5 + i * 0.08,
                "tool_flutes": 4,
                "vib_x_rms": 0.12 + i * 0.005,
                "vib_x_std": 0.10 + i * 0.004,
                "vib_x_peak": 0.22 + i * 0.006,
                "vib_x_pp": 0.42 + i * 0.01,
                "vib_y_rms": 0.10 + i * 0.004,
                "vib_y_std": 0.09 + i * 0.003,
                "vib_y_peak": 0.19 + i * 0.005,
                "vib_y_pp": 0.38 + i * 0.008,
                "vib_z_rms": 0.07 + i * 0.003,
                "vib_z_std": 0.06 + i * 0.002,
                "vib_z_peak": 0.15 + i * 0.004,
                "vib_z_pp": 0.31 + i * 0.006,
                "spindle_freq_hz": (4000 + i * 150) / 60.0,
                "tooth_pass_freq_hz": (4000 + i * 150) * 4 / 60.0,
                "feed_rate_mm_min": (4000 + i * 150) * 4 * (0.03 + i * 0.002),
            }
            for i in range(8)
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "sample_id": f"S{i}",
                "Ra_um": 0.7 + i * 0.05,
                "freq_bin_1": 0.20,
                "freq_bin_2": 0.18,
                "freq_bin_3": 0.15,
                "freq_bin_4": 0.13,
                "freq_bin_5": 0.11,
                "freq_bin_6": 0.09,
                "freq_bin_7": 0.08,
                "freq_bin_8": 0.06,
            }
            for i in range(8)
        ]
    )

    result = train_lite_models(features, targets)

    assert "ra_model" in result
    assert "freq_model" in result
    assert "metrics" in result
    assert "ra_mae" in result["metrics"]

