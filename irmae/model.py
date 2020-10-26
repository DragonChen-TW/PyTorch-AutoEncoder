import torch
from torch import nn
import torch.nn.functional as F

class IRMAE(nn.Module):
    def __init__(self, len_w=8, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )

        self.w_layers = nn.Sequential(*[
            nn.Linear(latent_dim, latent_dim) for _ in range(len_w)
        ])

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
        out = self.w_layers(out)
        out = self.decoder(out)

        return out

if __name__ == '__main__':
    model = IRMAE()
    print(model)
    x = torch.randn(64, 28 * 28)
    print(model(x).shape)
