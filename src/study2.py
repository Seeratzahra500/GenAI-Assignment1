import optuna, wandb
from src.train_ae import train

def run_study_v2(n_trials=20, epochs=8):
    def objective(trial):
        cfg = dict(
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            bottleneck=trial.suggest_categorical("bottleneck", [256, 512, 1024, 2048]),
            base=trial.suggest_categorical("base", [16, 32, 48]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            alpha=trial.suggest_float("alpha", 0.5, 0.95))
        run = wandb.init(project="genai-a1-task1", name=f"optuna2-t{trial.number}",
                         config=cfg, group="optuna-v2", reinit="finish_previous")
        try:
            return train(cfg, epochs, trial=trial, run=run)[1]
        finally:
            run.finish()
    study = optuna.create_study(
        study_name="task1_ae_v2", storage="sqlite:////kaggle/working/optuna_task1.db",
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study
