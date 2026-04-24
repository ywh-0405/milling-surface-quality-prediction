"""Evaluation helpers for trained milling surface-quality models."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_ra(true_values, pred_values):
    return {
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
        "r2": float(r2_score(true_values, pred_values)),
    }
