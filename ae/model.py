import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, mu, logvar=None, z=None):
        if logvar:
            # reparameterize
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
        else:
            z = mu

        # decode
        out = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))

        out = self.decode(mu, logvar)
        return out, mu, logvar
