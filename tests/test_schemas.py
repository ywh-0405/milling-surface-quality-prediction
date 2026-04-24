from src.schemas import REQUIRED_INPUT_COLUMNS, TARGET_COLUMNS, derive_process_columns


def test_schema_lists_are_stable():
    assert "spindle_speed_rpm" in REQUIRED_INPUT_COLUMNS
    assert "tool_flutes" in REQUIRED_INPUT_COLUMNS
    assert TARGET_COLUMNS == ["Ra_um"] + [f"freq_bin_{i}" for i in range(1, 9)]


def test_derive_process_columns():
    row = {
        "spindle_speed_rpm": 6000.0,
        "feed_per_tooth_mm": 0.05,
        "tool_flutes": 4,
    }
    derived = derive_process_columns(row)
    assert round(derived["spindle_freq_hz"], 4) == 100.0
    assert round(derived["tooth_pass_freq_hz"], 4) == 400.0
    assert round(derived["feed_rate_mm_min"], 4) == 1200.0

