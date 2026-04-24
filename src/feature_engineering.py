"""Feature helpers for future raw vibration signal processing."""

import numpy as np


def basic_vibration_statistics(signal):
    signal = np.asarray(signal, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(signal**2))),
        "std": float(np.std(signal)),
        "peak": float(np.max(np.abs(signal))),
        "pp": float(np.ptp(signal)),
    }

