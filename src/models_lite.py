"""Lightweight models intended for easy local execution."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_lite_models(features, targets):
    merged = features.merge(targets, on="sample_id")
    freq_columns = [f"freq_bin_{i}" for i in range(1, 9)]
    X = merged.drop(columns=["sample_id", "Ra_um"] + freq_columns)
    y_ra = merged["Ra_um"]
    y_freq = merged[freq_columns]

    ra_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), random_state=42, max_iter=2000)),
        ]
    )
    freq_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MultiOutputRegressor(
                    MLPRegressor(hidden_layer_sizes=(48, 24), random_state=42, max_iter=2000)
                ),
            ),
        ]
    )

    ra_model.fit(X, y_ra)
    freq_model.fit(X, y_freq)

    ra_pred = ra_model.predict(X)
    freq_pred = freq_model.predict(X)
    return {
        "ra_model": ra_model,
        "freq_model": freq_model,
        "feature_columns": list(X.columns),
        "metrics": {
            "ra_mae": float(mean_absolute_error(y_ra, ra_pred)),
            "ra_rmse": float(np.sqrt(mean_squared_error(y_ra, ra_pred))),
            "ra_r2": float(r2_score(y_ra, ra_pred)),
            "freq_mae": float(mean_absolute_error(y_freq, freq_pred)),
        },
    }
