"""Unified prediction interface for trained models."""

from pathlib import Path
import json

import pandas as pd


class SurfaceQualityPredictor:
    def __init__(self, feature_columns, ra_model=None, freq_model=None):
        self.feature_columns = feature_columns
        self.ra_model = ra_model
        self.freq_model = freq_model

    def predict_from_dataframe(self, df):
        X = df[self.feature_columns]
        result = {}
        if self.ra_model is not None:
            result["Ra_um"] = self.ra_model.predict(X).tolist()
        if self.freq_model is not None:
            result["frequency_bins"] = self.freq_model.predict(X).tolist()
        return result

    def save_metadata(self, output_path):
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"feature_columns": self.feature_columns}, f, ensure_ascii=False, indent=2)


def predict_csv(input_csv, predictor):
    df = pd.read_csv(input_csv)
    return predictor.predict_from_dataframe(df)

