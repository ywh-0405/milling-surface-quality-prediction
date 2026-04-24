"""Research-grade model definitions."""

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    nn = None


def build_research_model(n_features, n_freq_bins=8):
    if nn is None:
        raise ImportError("PyTorch is required for research mode")

    class ResearchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.ra_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.freq_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_freq_bins),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            return self.ra_head(encoded), self.freq_head(encoded)

    return ResearchNet()

