import torch, torch.nn as nn, torch.nn.functional as F, optuna, wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pytorch_msssim import ssim
from src.fs2k import get_splits, FS2KDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"

def down(i, o, norm=True):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False),
                         nn.InstanceNorm2d(o) if norm else nn.Identity(), nn.LeakyReLU(0.2, True))

class Up(nn.Module):
    def __init__(self, i, o, drop=0.0):
        super().__init__()
        L = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.InstanceNorm2d(o), nn.ReLU(True)]
        if drop: L.append(nn.Dropout(drop))
        self.b = nn.Sequential(*L)
    def forward(self, x, skip):
        return torch.cat([self.b(x), skip], 1)

def style_map(e, like):
    return e[:, :, None, None].expand(-1, -1, like.shape[2], like.shape[3])

class UNetG(nn.Module):
    """U-Net 128->1x1. Style embedding is concatenated to the input as extra channels AND added at the bottleneck."""
    def __init__(self, base=64, emb_dim=16, dropout=0.3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(3, emb_dim)
        self.d1, self.d2, self.d3 = down(3 + emb_dim, b, False), down(b, 2*b), down(2*b, 4*b)
        self.d4, self.d5, self.d6 = down(4*b, 8*b), down(8*b, 8*b), down(8*b, 8*b)
        self.d7 = down(8*b, 8*b, False)
        self.zp = nn.Linear(emb_dim, 8*b)
        self.u1, self.u2, self.u3 = Up(8*b, 8*b, dropout), Up(16*b, 8*b, dropout), Up(16*b, 8*b, dropout)
        self.u4, self.u5, self.u6 = Up(16*b, 4*b), Up(8*b, 2*b), Up(4*b, b)
        self.out = nn.Sequential(nn.ConvTranspose2d(2*b, 3, 4, 2, 1), nn.Tanh())
    def forward(self, x, style):
        e = self.emb(style)
        x1 = self.d1(torch.cat([x, style_map(e, x)], 1))
        x2 = self.d2(x1); x3 = self.d3(x2); x4 = self.d4(x3); x5 = self.d5(x4); x6 = self.d6(x5); x7 = self.d7(x6)
        y = x7 + self.zp(e)[:, :, None, None]
        y = self.u1(y, x6); y = self.u2(y, x5); y = self.u3(y, x4)
        y = self.u4(y, x3); y = self.u5(y, x2); y = self.u6(y, x1)
        return self.out(y)

class PatchD(nn.Module):
    """Conditional PatchGAN: input = photo + sketch + style-embedding map; output = 14x14 patch logits."""
    def __init__(self, base=64, emb_dim=16):
        super().__init__()
        b = base
        self.emb = nn.Embedding(3, emb_dim)
        def c(i, o, s, n=True):
            return [nn.Conv2d(i, o, 4, s, 1, bias=False)] + ([nn.InstanceNorm2d(o)] if n else []) + [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*c(6 + emb_dim, b, 2, False), *c(b, 2*b, 2), *c(2*b, 4*b, 2), *c(4*b, 8*b, 1),
                                 nn.Conv2d(8*b, 1, 4, 1, 1))
    def forward(self, photo, sketch, style):
        return self.net(torch.cat([photo, sketch, style_map(self.emb(style), photo)], 1))

@torch.no_grad()
def evaluate(G, loader):
    G.eval(); S, P, L = [], [], []
    for p, s, y in loader:
        p, s, y = p.to(dev), s.to(dev), y.to(dev)
        a, b = (G(p, y) + 1) / 2, (s + 1) / 2
        S.append(ssim(a, b, data_range=1.0, size_average=False).cpu())
        mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        P.append((10 * torch.log10(1 / mse)).cpu()); L.append((a - b).abs().flatten(1).mean(1).cpu())
    return torch.cat(S).mean().item(), torch.cat(P).mean().item(), torch.cat(L).mean().item()

@torch.no_grad()
def sample_grid(G, fp, fs):
    """Rows: photos, ground-truth sketches, then the generated sketch in style 1, 2, 3."""
    G.eval(); rows = [fp, fs] + [G(fp, torch.full((len(fp),), st, device=dev)) for st in range(3)]
    return make_grid(torch.cat(rows), nrow=len(fp), normalize=True, value_range=(-1, 1))

def train_gan(cfg, epochs, trial=None, run=None, ckpt=None, img_every=10):
    d = get_splits()
    tl = DataLoader(FS2KDataset(d["train"], augment=True), batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=2, drop_last=True)
    vl = DataLoader(FS2KDataset(d["val"]), batch_size=32, num_workers=2)
    fp, fs, _ = next(iter(DataLoader(FS2KDataset(d["val"]), batch_size=4)))   # same 4 val photos every time
    fp, fs = fp.to(dev), fs.to(dev)
    G = UNetG(cfg["base"], cfg["emb_dim"], cfg["dropout"]).to(dev)
    D = PatchD(cfg["base"], cfg["emb_dim"]).to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); best = -9.0
    for ep in range(epochs):
        G.train(); D.train(); acc = torch.zeros(4); n = 0
        for p, s, y in tl:
            p, s, y = p.to(dev), s.to(dev), y.to(dev)
            fake = G(p, y)
            dr, df_ = D(p, s, y), D(p, fake.detach(), y)
            l_dr, l_df = bce(dr, torch.ones_like(dr)), bce(df_, torch.zeros_like(df_))
            oD.zero_grad(); (0.5 * (l_dr + l_df)).backward(); oD.step()
            dg = D(p, fake, y)
            l_adv, l_l1 = bce(dg, torch.ones_like(dg)), F.l1_loss(fake, s)
            oG.zero_grad(); (l_adv + cfg["lam_l1"] * l_l1).backward(); oG.step()
            acc += torch.tensor([l_dr.item(), l_df.item(), l_adv.item(), l_l1.item()]); n += 1
        acc /= n; S, P, L = evaluate(G, vl); sc = S - L
        m = dict(d_real=acc[0].item(), d_fake=acc[1].item(), g_adv=acc[2].item(), g_l1=acc[3].item(),
                 val_ssim=S, val_psnr=P, val_l1=L, val_score=sc)
        print(f"ep {ep}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()))
        if run:
            run.log({"epoch": ep, **m})
            if img_every and (ep % img_every == 0 or ep == epochs - 1):
                run.log({"samples": wandb.Image(sample_grid(G, fp, fs).permute(1, 2, 0).cpu().numpy()), "epoch": ep})
        if ckpt and sc > best: torch.save(G.state_dict(), ckpt)
        best = max(best, sc)
        if trial:
            trial.report(sc, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return G, best

def run_gan_study(n_trials=10, epochs=12):
    def objective(trial):
        cfg = dict(lr_g=trial.suggest_float("lr_g", 1e-4, 1e-3, log=True),
                   lr_d=trial.suggest_float("lr_d", 1e-4, 1e-3, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
                   base=trial.suggest_categorical("base", [32, 48, 64]),
                   dropout=trial.suggest_float("dropout", 0.0, 0.5),
                   emb_dim=trial.suggest_categorical("emb_dim", [8, 16, 32]),
                   lam_l1=trial.suggest_float("lam_l1", 10, 200, log=True))
        run = wandb.init(project="genai-a1-task4", name=f"gan-optuna-t{trial.number}", config=cfg,
                         group="optuna", reinit="finish_previous")
        try: return train_gan(cfg, epochs, trial=trial, run=run, img_every=0)[1]
        finally: run.finish()
    study = optuna.create_study(study_name="task4_gan", storage="sqlite:////kaggle/working/optuna_task4_gan.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3), load_if_exists=True)
    if len(study.trials) == 0:   # trial 0 = the assignment's suggested starting point
        study.enqueue_trial({"lr_g": 2e-4, "lr_d": 2e-4, "batch_size": 16, "base": 64, "dropout": 0.3, "emb_dim": 16, "lam_l1": 100})
    study.optimize(objective, n_trials=n_trials)
    return study
