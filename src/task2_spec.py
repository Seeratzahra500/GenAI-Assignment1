import numpy as np, torch, optuna, wandb
from torch.utils.data import Dataset, DataLoader
from src.data import *
from src.models import restoration_loss
from src.train_ae import get_data, evaluate, score, build_model, dev

class MixedCorruptionDataset(Dataset):
    """Class = classes[i % len(classes)], random image, fresh random corruption each load."""
    def __init__(self, images, indices, classes):
        self.images, self.indices, self.classes = images, indices, classes
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        rng = np.random.default_rng()
        c = self.classes[i % len(self.classes)]
        clean = self.images[self.indices[int(rng.integers(len(self.indices)))]].float() / 255
        return apply_corruption(clean, sample_params(c, rng)), clean, CLASSES.index(c)

def train_spec(cfg, epochs, classes, trial=None, run=None, ckpt=None):
    imgs, tr, val_m = get_data()
    vm = [e for e in val_m if e["params"]["type"] in classes]
    tl = DataLoader(MixedCorruptionDataset(imgs, tr, classes), batch_size=cfg["batch_size"],
                    shuffle=False, num_workers=2, drop_last=True)
    vl = DataLoader(PetDataset(imgs, manifest=vm), batch_size=128, num_workers=2)
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
        if run: run.log({"epoch": ep, "train_loss": tot / len(tl), "val_ssim": s, "val_psnr": p, "val_score": sc})
        if ckpt and sc > best: torch.save(model.state_dict(), ckpt)
        best = max(best, sc)
        if trial:
            trial.report(sc, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return model, best

def run_spec_study(n_trials=10, epochs=5):
    def objective(trial):
        cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [32, 64]),
                   latent_ch=trial.suggest_categorical("latent_ch", [32, 64, 128]),
                   base=trial.suggest_categorical("base", [32, 48, 64]),
                   alpha=trial.suggest_float("alpha", 0.5, 0.95), dropout=0.02)
        run = wandb.init(project="genai-a1-task2", name=f"spec-optuna-t{trial.number}", config=cfg,
                         group="spec-optuna", reinit="finish_previous")
        try: return train_spec(cfg, epochs, ["salt", "blur", "occlusion"], trial=trial, run=run)[1]
        finally: run.finish()
    study = optuna.create_study(study_name="task2_spec", storage="sqlite:////kaggle/working/optuna_task2_spec.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2), load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study
