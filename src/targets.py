"""Target builders for current and future surface-quality tasks."""

from src.schemas import TARGET_COLUMNS


def build_ra_targets(df):
    return df[["sample_id", "Ra_um"]].copy()


def build_frequency_targets(df):
    return df[["sample_id"] + [column for column in TARGET_COLUMNS if column != "Ra_um"]].copy()


def build_profile_targets(df):
    raise NotImplementedError(
        "Profile-sequence targets are reserved for a future version when real surface profiles are available."
    )

