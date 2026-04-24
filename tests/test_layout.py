from pathlib import Path


def test_project_directories_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").is_dir()
    assert (root / "data" / "raw").is_dir()
    assert (root / "data" / "processed").is_dir()
    assert (root / "data" / "synthetic").is_dir()
    assert (root / "outputs" / "lite").is_dir()
    assert (root / "outputs" / "research").is_dir()
    assert (root / "examples").is_dir()

