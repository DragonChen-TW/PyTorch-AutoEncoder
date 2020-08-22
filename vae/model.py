import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_size=10):
        super().__init__()
        self.latent_size = latent_size

        self.encode_fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.Linear(256, 32),
            nn.ReLU(True),
        )

        self.mu_fc = nn.Linear(32, latent_size)
        self.logvar_fc = nn.Linear(32, latent_size)

        self.decode_fc = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
        )

    def encode(self, x):
        x = self.encode_fc(x.view(-1, 784))
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def decode(self, z):
        out = self.decode_fc(z)
        out = torch.sigmoid(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))

        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

if __name__ == '__main__':
    x = torch.randn(16, 1, 28, 28)
    model = VAE()
    out, _, _ = model(x)
    print(out.shape)