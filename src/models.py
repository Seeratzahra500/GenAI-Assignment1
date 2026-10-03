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

class ConvAE2(nn.Module):
    """Conv bottleneck: 128->64->32->16->8, then 1x1 conv to latent_ch (latent = 8x8xlatent_ch).
    skip_ch>0 adds one narrow skip at 32x32 (for the skip-connection ablation)."""
    def __init__(self, base=32, latent_ch=64, dropout=0.1, skip_ch=0):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.e1, self.e2 = down(3, ch[0]), down(ch[0], ch[1])
        self.e3, self.e4 = down(ch[1], ch[2]), down(ch[2], ch[3])
        self.to_z = nn.Conv2d(ch[3], latent_ch, 1)
        self.drop = nn.Dropout2d(dropout)
        self.from_z = nn.Sequential(nn.Conv2d(latent_ch, ch[3], 1), nn.ReLU(True))
        self.d4, self.d3 = up(ch[3], ch[2]), up(ch[2], ch[1])
        self.skip = nn.Conv2d(ch[1], skip_ch, 1) if skip_ch else None
        self.d2 = up(ch[1] + skip_ch, ch[0])
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1), nn.BatchNorm2d(ch[0]),
                                nn.ReLU(True), nn.Conv2d(ch[0], 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        h2 = self.e2(self.e1(x))
        z = self.drop(self.to_z(self.e4(self.e3(h2))))
        y = self.d3(self.d4(self.from_z(z)))
        if self.skip is not None:
            y = torch.cat([y, self.skip(h2)], 1)
        return self.d1(self.d2(y))
