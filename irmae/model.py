import torch
from torch import nn
import torch.nn.functional as F

class Unflatten(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.dims = (c, h, w)
    
    def __call__(self, x):
        return x.view(-1, *self.dims)

class ConvAutoEncoder(nn.Module):
    # 1.57 M
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1), # 24, 14, 14
            nn.ReLU(True),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1), # 48, 7, 7
            nn.ReLU(True),
            nn.Conv2d(48, 24, kernel_size=1), # 24, 7, 7
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(24 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24 * 7 * 7),
            Unflatten(24, 7, 7),
            nn.ConvTranspose2d(24, 48, kernel_size=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out

class Unflatten(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.dims = (c, h, w)
    
    def __call__(self, x):
        return x.view(-1, *self.dims)
class IRMAE(nn.Module):
    # 1.57 M
    def __init__(self, len_w=4, latent_dim=128, mode='normal'):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1), # 24, 14, 14
            nn.ReLU(True),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1), # 48, 7, 7
            nn.ReLU(True),
            nn.Conv2d(48, 24, kernel_size=1), # 24, 7, 7
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(24 * 7 * 7, latent_dim),
        )
        
        if mode == 'normal':
            w_layers = [nn.Linear(latent_dim, latent_dim) for _ in range(len_w)]
        self.w_layers = nn.Sequential(*w_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24 * 7 * 7),
            Unflatten(24, 7, 7),
            nn.ConvTranspose2d(24, 48, kernel_size=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.w_layers(out)
        out = self.decoder(out)

        return out

if __name__ == '__main__':
    model = IRMAE()
    print(model)
    x = torch.randn(64, 1, 28, 28)
    print(model(x).shape)
