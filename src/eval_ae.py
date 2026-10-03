import json, os, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from src.data import *
from src.models import ConvAE
from src.train_ae import psnr_batch, ROOT

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEV = ["none", "low", "medium", "high"]

def load_model(path, cfg):
    m = ConvAE(cfg["base"], cfg["bottleneck"], cfg["dropout"]).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()

@torch.no_grad()
def eval_test(model):
    man = json.load(open(f"{ROOT}/manifests/test_manifest.json"))
    ds = PetDataset(load_split("test"), manifest=man)
    S, P, S0, P0 = [], [], [], []
    for xc, x, _ in DataLoader(ds, batch_size=256, num_workers=2):
        xc, x = xc.to(dev), x.to(dev); out = model(xc)
        S.append(ssim(out, x, data_range=1.0, size_average=False).cpu())
        P.append(psnr_batch(out, x).cpu())
        S0.append(ssim(xc, x, data_range=1.0, size_average=False).cpu())
        P0.append(psnr_batch(xc, x).cpu())
    df = pd.DataFrame({
        "i": range(len(man)), "idx": [e["idx"] for e in man],
        "type": [e["params"]["type"] for e in man],
        "severity": [e["params"]["severity"] for e in man],
        "ssim": torch.cat(S).numpy(), "psnr": torch.cat(P).numpy(),
        "in_ssim": torch.cat(S0).numpy(), "in_psnr": torch.cat(P0).numpy()})
    df.loc[df.type == "clean", "in_psnr"] = np.nan  # identical images give a meaningless 100 dB
    return df, ds

def summarize(df):
    cols = ["ssim", "psnr", "in_ssim", "in_psnr"]
    g = df.groupby(["type", "severity"])
    t = g[cols].mean(); t["n"] = g.size()
    order = [("clean", "none")] + [(c, s) for c in ["salt", "blur", "occlusion"] for s in SEV[1:]]
    return t.loc[order].round(4), df.groupby("type")[cols].mean().round(4)

def pick_examples(df):
    rows, k = [], 0
    for t, sevs in [("clean", ["none"] * 3), ("salt", SEV[1:]), ("blur", SEV[1:]), ("occlusion", SEV[1:])]:
        for s in sevs:
            cand = df[(df.type == t) & (df.severity == s)]
            rows.append(int(cand.iloc[3 + 11 * k].i)); k += 1
    return rows

def worst_cases(df):
    return df.loc[df.groupby("type").ssim.idxmin()].i.tolist()

def plot_rows(model, ds, rows, df, path):
    xs = [ds[int(r)] for r in rows]
    xc = torch.stack([a[0] for a in xs]); x = torch.stack([a[1] for a in xs])
    with torch.no_grad():
        out = model(xc.to(dev)).cpu().clamp(0, 1)
    err = (out - x).abs().mean(1)
    fig, ax = plt.subplots(len(rows), 4, figsize=(9, 2.3 * len(rows)))
    for k, r in enumerate(rows):
        row = df.iloc[int(r)]
        ims = [x[k].permute(1, 2, 0), xc[k].permute(1, 2, 0), out[k].permute(1, 2, 0), err[k]]
        names = ["clean target", f"input ({row.type}/{row.severity})", f"restored, SSIM {row.ssim:.3f}", "abs error"]
        for j in range(4):
            if j == 3: ax[k, j].imshow(ims[j], cmap="inferno", vmin=0, vmax=0.5)
            else: ax[k, j].imshow(ims[j])
            ax[k, j].set_title(names[j], fontsize=8); ax[k, j].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.show()

def export_onnx(model, path, ds, n_check=64):
    import onnx, onnxruntime as ort
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = model.cpu().eval()
    torch.onnx.export(m, torch.zeros(1, 3, 128, 128), path, input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(path)
    idx = np.linspace(0, len(ds) - 1, n_check).astype(int)
    x = torch.stack([ds[int(i)][0] for i in idx])
    with torch.no_grad():
        ref = m(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    print("max abs diff:", np.abs(out - ref).max(), "| allclose(atol=1e-4):", np.allclose(out, ref, atol=1e-4))
    print("size MB:", os.path.getsize(path) / 1e6)
