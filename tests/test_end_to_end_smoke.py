from src.synthetic_data import generate_synthetic_dataset


def test_synthetic_dataset_can_be_generated():
    df = generate_synthetic_dataset(20)
    assert len(df) == 20
    assert "Ra_um" in df.columns
    assert "freq_bin_8" in df.columns
