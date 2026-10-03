import torch, torch.nn as nn, torch.nn.functional as F, optuna, wandb
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from src.data import *
from src.models import ConvAE2
from src.train_ae import get_data, psnr_batch, dev
from src.task2_clf import BalancedCorruptionDataset, CorruptionClassifier

CK = "/kaggle/working/ckpts"
EXPERTS = ["salt", "blur", "occlusion"]

class SoftMoE(nn.Module):
    """w = softmax(gate(x)/tau) over [clean-identity, salt, blur, occlusion]; y = sum_k w_k * branch_k(x)."""
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.gate = CorruptionClassifier(base=16, dropout=0.09091248360355031)
        self.experts = nn.ModuleList([ConvAE2(base=64, latent_ch=32, dropout=0.02, skip_ch=0) for _ in EXPERTS])
    def load_task2(self):
        self.gate.load_state_dict(torch.load(f"{CK}/task2_clf_best.pt", map_location="cpu"))
        for e, c in zip(self.experts, EXPERTS):
            e.load_state_dict(torch.load(f"{CK}/task2_spec_{c}.pt", map_location="cpu"))
        return self
    def forward_all(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, 1)
        outs = [x] + [e(x) for e in self.experts]
        y = sum(w[:, i, None, None, None] * o for i, o in enumerate(outs))
        return y, w, logits
    def forward(self, x):
        return self.forward_all(x)[0]

def set_stage(m, stage):
    """warmup: experts frozen (and BatchNorm kept in eval); joint: everything trainable."""
    for p in m.experts.parameters(): p.requires_grad = (stage == "joint")
    m.train()
    if stage == "warmup": m.experts.eval()

def moe_loss(y_hat, w, logits, clean, label, cfg):
    l1 = F.l1_loss(y_hat, clean)
    ss = 1 - ssim(y_hat, clean, data_range=1.0)
    ce = F.cross_entropy(logits / cfg["tau"], label)
    bal = ((w.mean(0) - 0.25) ** 2).sum()
    return cfg["lam_rec"] * l1 + (1 - cfg["lam_rec"]) * ss + cfg["lam_ce"] * ce + cfg["lam_bal"] * bal

@torch.no_grad()
def eval_moe(model, loader):
    model.eval(); S, P, Y, W = [], [], [], []
    for xc, x, y in loader:
        out, w, _ = model.forward_all(xc.to(dev)); x = x.to(dev)
        S.append(ssim(out, x, data_range=1.0, size_average=False).cpu()); P.append(psnr_batch(out, x).cpu())
        Y.append(y); W.append(w.cpu())
    S, P, Y, W = map(torch.cat, (S, P, Y, W)); nc = Y != 0
    r = dict(ssim=S.mean().item(), ssim_nc=S[nc].mean().item(), psnr_nc=P[nc].mean().item(),
             gate_acc=(W.argmax(1) == Y).float().mean().item(), usage_min=W.mean(0).min().item())
    r["obj"] = r["ssim"] + r["psnr_nc"] / 40
    return r

def train_moe(cfg, warm_epochs, joint_epochs, trial=None, run=None, ckpt=None):
    imgs, tr, val_m = get_data()
    tl = DataLoader(BalancedCorruptionDataset(imgs, tr), batch_size=32, shuffle=False, num_workers=2, drop_last=True)
    vl = DataLoader(PetDataset(imgs, manifest=val_m), batch_size=64, num_workers=2)
    model = SoftMoE(cfg["tau"]).load_task2().to(dev)
    best, step = -1.0, 0
    for stage, n in [("warmup", warm_epochs), ("joint", joint_epochs)]:
        set_stage(model, stage)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=cfg["gate_lr"] if stage == "warmup" else cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n, 1)) if stage == "joint" else None
        for ep in range(n):
            set_stage(model, stage); tot = 0
            for xc, x, y in tl:
                xc, x, y = xc.to(dev), x.to(dev), y.to(dev)
                yh, w, lg = model.forward_all(xc)
                loss = moe_loss(yh, w, lg, x, y, cfg)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if sched: sched.step()
            r = eval_moe(model, vl)
            print(f"{stage} ep {ep}: loss {tot/len(tl):.4f} ssim {r['ssim']:.4f} ssim_nc {r['ssim_nc']:.4f} "
                  f"psnr_nc {r['psnr_nc']:.2f} gate_acc {r['gate_acc']:.4f} usage_min {r['usage_min']:.3f} obj {r['obj']:.4f}")
            if run: run.log({"step": step, "stage": stage, "train_loss": tot / len(tl), **{f"val_{k}": v for k, v in r.items()}})
            if ckpt and r["obj"] > best: torch.save(model.state_dict(), ckpt)
            best = max(best, r["obj"])
            if trial:
                trial.report(r["obj"], step)
                if r["usage_min"] < 0.05: raise optuna.TrialPruned()   # routing collapse
                if trial.should_prune(): raise optuna.TrialPruned()
            step += 1
    return model, best

def run_moe_study(n_trials=6, warm=1, joint=3):
    def objective(trial):
        cfg = dict(lr=trial.suggest_float("lr", 1e-5, 1e-4, log=True),
                   tau=trial.suggest_float("tau", 0.5, 2.0),
                   lam_ce=trial.suggest_float("lam_ce", 0.01, 0.5, log=True),
                   lam_bal=trial.suggest_float("lam_bal", 1e-3, 1.0, log=True),
                   lam_rec=trial.suggest_float("lam_rec", 0.5, 0.95), gate_lr=1e-4)
        run = wandb.init(project="genai-a1-task3", name=f"moe-optuna-t{trial.number}", config=cfg,
                         group="moe-optuna", reinit="finish_previous")
        try: return train_moe(cfg, warm, joint, trial=trial, run=run)[1]
        finally: run.finish()
    study = optuna.create_study(study_name="task3_moe", storage="sqlite:////kaggle/working/optuna_task3_moe.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1), load_if_exists=True)
    if len(study.trials) == 0:   # trial 0 = the assignment's suggested starting values
        study.enqueue_trial({"lr": 5e-5, "tau": 1.0, "lam_ce": 0.1, "lam_bal": 0.01, "lam_rec": 0.8})
    study.optimize(objective, n_trials=n_trials)
    return study
