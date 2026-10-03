import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, optuna, wandb
from torch.utils.data import Dataset, DataLoader
from src.data import *
from src.train_ae import get_data, dev

class BalancedCorruptionDataset(Dataset):
    """Class = i % 4, image random, so any shuffle=False batch (size % 4 == 0) is exactly balanced.
    fixed_class: always use that class (for the Task 2 specialists)."""
    def __init__(self, images, indices, fixed_class=None):
        self.images, self.indices, self.fixed = images, indices, fixed_class
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        rng = np.random.default_rng()
        c = self.fixed if self.fixed is not None else CLASSES[i % 4]
        clean = self.images[self.indices[int(rng.integers(len(self.indices)))]].float() / 255
        p = sample_params(c, rng)
        return apply_corruption(clean, p), clean, CLASSES.index(c)

class CorruptionClassifier(nn.Module):
    """4 conv blocks (channels base, 2b, 4b, 8b) -> global avg pool -> dropout -> linear(4). Returns logits."""
    def __init__(self, base=32, dropout=0.3):
        super().__init__()
        ch, layers, c_in = [base, base * 2, base * 4, base * 8], [], 3
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(True),
                       nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(ch[-1], 4)
    def forward(self, x):
        return self.fc(self.drop(self.features(x).mean((2, 3))))

@torch.no_grad()
def predict(model, loader):
    model.eval(); Y, P, L = [], [], []
    for xc, _, y in loader:
        lg = model(xc.to(dev)); Y.append(y); L.append(lg.cpu()); P.append(lg.argmax(1).cpu())
    return torch.cat(Y).numpy(), torch.cat(P).numpy(), torch.cat(L)

def train_clf(cfg, epochs, trial=None, run=None, ckpt=None):
    imgs, tr, val_m = get_data()
    tl = DataLoader(BalancedCorruptionDataset(imgs, tr), batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=2, drop_last=True)
    vl = DataLoader(PetDataset(imgs, manifest=val_m), batch_size=128, num_workers=2)
    model = CorruptionClassifier(cfg["base"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = -1.0
    for ep in range(epochs):
        model.train(); tot = 0
        for xc, _, y in tl:
            loss = F.cross_entropy(model(xc.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        y, p, _ = predict(model, vl); acc = float((y == p).mean())
        print(f"ep {ep}: loss {tot/len(tl):.4f} val_acc {acc:.4f}")
        if run: run.log({"epoch": ep, "train_loss": tot / len(tl), "val_acc": acc})
        if ckpt and acc > best: torch.save(model.state_dict(), ckpt)
        best = max(best, acc)
        if trial:
            trial.report(acc, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
    return model, best

def run_clf_study(n_trials=8, epochs=3):
    def objective(trial):
        cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [32, 64]),
                   base=trial.suggest_categorical("base", [16, 32, 48]),
                   dropout=trial.suggest_float("dropout", 0.0, 0.5),
                   weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        run = wandb.init(project="genai-a1-task2", name=f"clf-optuna-t{trial.number}", config=cfg,
                         group="clf-optuna", reinit="finish_previous")
        try: return train_clf(cfg, epochs, trial=trial, run=run)[1]
        finally: run.finish()
    study = optuna.create_study(study_name="task2_clf", storage="sqlite:////kaggle/working/optuna_task2_clf.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1), load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study
