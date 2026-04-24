"""Column definitions shared by data preparation, training, and prediction."""

BASE_COLUMNS = [
    "sample_id",
    "spindle_speed_rpm",
    "feed_per_tooth_mm",
    "axial_depth_mm",
    "radial_depth_mm",
    "tool_flutes",
]

VIBRATION_COLUMNS = [
    "vib_x_rms",
    "vib_x_std",
    "vib_x_peak",
    "vib_x_pp",
    "vib_y_rms",
    "vib_y_std",
    "vib_y_peak",
    "vib_y_pp",
    "vib_z_rms",
    "vib_z_std",
    "vib_z_peak",
    "vib_z_pp",
]

REQUIRED_INPUT_COLUMNS = BASE_COLUMNS[1:] + VIBRATION_COLUMNS
DERIVED_COLUMNS = ["spindle_freq_hz", "tooth_pass_freq_hz", "feed_rate_mm_min"]
TARGET_COLUMNS = ["Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)]


def derive_process_columns(row):
    spindle_speed = float(row["spindle_speed_rpm"])
    feed_per_tooth = float(row["feed_per_tooth_mm"])
    flutes = float(row["tool_flutes"])
    return {
        "spindle_freq_hz": spindle_speed / 60.0,
        "tooth_pass_freq_hz": spindle_speed * flutes / 60.0,
        "feed_rate_mm_min": spindle_speed * flutes * feed_per_tooth,
    }

