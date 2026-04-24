from pathlib import Path


def test_root_readme_mentions_core_sections():
    content = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(
        encoding="utf-8"
    )
    assert "项目背景" in content
    assert "输入与输出" in content
    assert "快速开始" in content
    assert "未来如何接入真实实验数据" in content
