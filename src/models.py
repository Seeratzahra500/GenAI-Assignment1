import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_msssim import ssim

class ConvAE(nn.Module):
    """Encoder 128->64->32->16->8, linear bottleneck, mirrored decoder. No skip connections."""
    def __init__(self, base=32, bottleneck=256, dropout=0.1):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        layers, c_in = [], 3
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in = c
        self.enc = nn.Sequential(*layers)
        self.last = ch[-1]
        self.to_z = nn.Linear(self.last * 64, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.from_z = nn.Linear(bottleneck, self.last * 64)
        layers = []
        for c_in, c_out in zip(ch[::-1][:-1], ch[::-1][1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(ch[0], 3, 4, 2, 1), nn.Sigmoid()]
        self.dec = nn.Sequential(*layers)

    def encode(self, x):
        return self.drop(self.to_z(self.enc(x).flatten(1)))

    def decode(self, z):
        return self.dec(F.relu(self.from_z(z)).view(-1, self.last, 8, 8))

    def forward(self, x):
        return self.decode(self.encode(x))

def restoration_loss(pred, target, alpha=0.8):
    return alpha * F.l1_loss(pred, target) + (1 - alpha) * (1 - ssim(pred, target, data_range=1.0))
