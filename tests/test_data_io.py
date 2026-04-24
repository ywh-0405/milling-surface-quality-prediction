from pathlib import Path

import pandas as pd

from src.data_io import prepare_summary_table


def test_prepare_summary_table_writes_features_and_targets(tmp_path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "S001",
                "spindle_speed_rpm": 6000,
                "feed_per_tooth_mm": 0.05,
                "axial_depth_mm": 1.0,
                "radial_depth_mm": 2.0,
                "tool_flutes": 4,
                "vib_x_rms": 0.10,
                "vib_x_std": 0.09,
                "vib_x_peak": 0.22,
                "vib_x_pp": 0.41,
                "vib_y_rms": 0.08,
                "vib_y_std": 0.07,
                "vib_y_peak": 0.19,
                "vib_y_pp": 0.36,
                "vib_z_rms": 0.06,
                "vib_z_std": 0.05,
                "vib_z_peak": 0.15,
                "vib_z_pp": 0.30,
                "Ra_um": 0.85,
                "freq_bin_1": 0.21,
                "freq_bin_2": 0.18,
                "freq_bin_3": 0.15,
                "freq_bin_4": 0.13,
                "freq_bin_5": 0.11,
                "freq_bin_6": 0.09,
                "freq_bin_7": 0.08,
                "freq_bin_8": 0.05,
            }
        ]
    )
    src = tmp_path / "input.csv"
    out_dir = tmp_path / "processed"
    df.to_csv(src, index=False)

    result = prepare_summary_table(src, out_dir)

    assert (out_dir / "features.csv").exists()
    assert (out_dir / "targets.csv").exists()
    assert "spindle_freq_hz" in result["features"].columns
    assert list(result["targets"].columns) == ["sample_id", "Ra_um"] + [
        f"freq_bin_{i}" for i in range(1, 9)
    ]

