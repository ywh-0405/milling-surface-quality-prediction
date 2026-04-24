"""Shared configuration values for the milling prediction project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LITE_OUTPUT_DIR = OUTPUTS_DIR / "lite"
RESEARCH_OUTPUT_DIR = OUTPUTS_DIR / "research"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

DEFAULT_TOOL_FLUTES = 4
DEFAULT_FREQ_BIN_COUNT = 8

