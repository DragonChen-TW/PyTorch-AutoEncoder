import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x.view(-1, 28 * 28))
        out = self.decoder(out)

        return out

model = AutoEncoder()
x = torch.randn(64, 28 * 28)
print(model(x).shape)