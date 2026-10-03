import json, numpy as np, torch, optuna, wandb
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from src.data import *
from src.models import ConvAE, ConvAE2, restoration_loss

def build_model(cfg):
    if "latent_ch" in cfg:
        return ConvAE2(cfg["base"], cfg["latent_ch"], cfg["dropout"], cfg.get("skip_ch", 0))
    return ConvAE(cfg["base"], cfg["bottleneck"], cfg["dropout"])

ROOT = "/kaggle/working/GenAI-Assignment1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}

def get_data():
    if not _cache:
        imgs = load_split("trainval")
        tr, _ = train_val_indices(len(imgs))
        _cache.update(imgs=imgs, tr=tr, val=json.load(open(f"{ROOT}/manifests/val_manifest.json")))
    return _cache["imgs"], _cache["tr"], _cache["val"]

def psnr_batch(p, t):
    mse = ((p - t) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); S, P = [], []
    for xc, x, _ in loader:
        out = model(xc.to(dev)); x = x.to(dev)
        S.append(ssim(out, x, data_range=1.0, size_average=False).cpu())
        P.append(psnr_batch(out, x).cpu())
    return torch.cat(S).mean().item(), torch.cat(P).mean().item()

def score(s, p):
    """Optuna objective: SSIM (structure) + PSNR/40 (pixel fidelity), both roughly 0-1."""
    return s + p / 40

def train(cfg, epochs, trial=None, run=None, ckpt=None):
    imgs, tr, val_m = get_data()
    tl = DataLoader(PetDataset(imgs, tr), batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=2, drop_last=True)
    vl = DataLoader(PetDataset(imgs, manifest=val_m), batch_size=128, num_workers=2)
    model = build_model(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = -1.0
    for ep in range(epochs):
        model.train(); tot = 0
        for xc, x, _ in tl:
            xc, x = xc.to(dev), x.to(dev)
            loss = restoration_loss(model(xc), x, cfg["alpha"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        s, p = evaluate(model, vl); sc = score(s, p)
        print(f"ep {ep}: loss {tot/len(tl):.4f} ssim {s:.4f} psnr {p:.2f} score {sc:.4f}")
        if run:
            run.log({"epoch": ep, "train_loss": tot / len(tl), "val_ssim": s, "val_psnr": p, "val_score": sc})
        if ckpt and sc > best:
            torch.save(model.state_dict(), ckpt)
        best = max(best, sc)
        if trial:
            trial.report(sc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return model, best

def run_study(n_trials=15, epochs=5):
    def objective(trial):
        cfg = dict(
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            bottleneck=trial.suggest_categorical("bottleneck", [128, 256, 512]),
            base=trial.suggest_categorical("base", [16, 32, 48]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            alpha=trial.suggest_float("alpha", 0.5, 0.95))
        run = wandb.init(project="genai-a1-task1", name=f"optuna-t{trial.number}",
                         config=cfg, group="optuna", reinit=True)
        try:
            return train(cfg, epochs, trial=trial, run=run)[1]
        finally:
            run.finish()
    study = optuna.create_study(
        study_name="task1_ae", storage="sqlite:////kaggle/working/optuna_task1.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study
