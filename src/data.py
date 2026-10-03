import json, math, os
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

SIZE, SEED = 128, 42
CLASSES = ["clean", "salt", "blur", "occlusion"]

def load_split(split, root="/kaggle/working/data", cache="/kaggle/working/cache"):
    """uint8 tensor N x 3 x 128 x 128 for 'trainval' or 'test' (RGB, resized, cached)."""
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/{split}_{SIZE}.pt"
    if os.path.exists(path):
        return torch.load(path)
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root=root, split=split, download=False)
    out = torch.empty(len(ds), 3, SIZE, SIZE, dtype=torch.uint8)
    for i in range(len(ds)):
        img = ds[i][0].convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        out[i] = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    torch.save(out, path)
    return out

def train_val_indices(n, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = int(0.8 * n)
    return perm[:k], perm[k:]

def _occ_rects(rng, n, total, equal=False):
    fr = np.full(n, total / n) if (equal or n == 1) else rng.dirichlet(np.ones(n)) * total
    rects = []
    for f in fr:
        area, ar = f * SIZE * SIZE, rng.uniform(0.5, 2.0)
        w = int(min(SIZE, max(1, round(math.sqrt(area * ar)))))
        h = int(min(SIZE, max(1, round(area / w))))
        rects.append([int(rng.integers(0, SIZE - w + 1)), int(rng.integers(0, SIZE - h + 1)), w, h])
    return rects

def sample_params(ctype, rng, level=None):
    """level=None: random training-style severity. level=0/1/2: fixed test severity."""
    if ctype == "clean":
        return {"type": "clean"}
    if ctype == "salt":
        p = rng.uniform(0.02, 0.15) if level is None else [0.03, 0.08, 0.15][level]
        return {"type": "salt", "p": float(p), "seed": int(rng.integers(2**31))}
    if ctype == "blur":
        if level is None:
            k, s = int(rng.choice([3, 5, 7])), float(rng.uniform(0.5, 2.5))
        else:
            k, s = [(3, 0.7), (5, 1.5), (7, 2.5)][level]
        return {"type": "blur", "k": k, "sigma": float(s)}
    if level is None:
        n, frac = int(rng.integers(1, 4)), float(rng.uniform(0.10, 0.35))
    else:
        n, frac = level + 1, [0.10, 0.20, 0.35][level]
    return {"type": "occlusion", "target_frac": frac,
            "rects": _occ_rects(rng, n, frac, equal=level is not None)}

def severity_of(p):
    def b(v, lo, hi): return ["low", "medium", "high"][min(2, int(3 * (v - lo) / (hi - lo)))]
    t = p["type"]
    if t == "clean": return "none"
    if t == "salt": return b(p["p"], 0.02, 0.15)
    if t == "blur": return b(p["sigma"], 0.5, 2.5)
    return b(p["target_frac"], 0.10, 0.35)

def apply_corruption(x, p):
    """x: float 3xHxW in [0,1]. Fully deterministic given p."""
    t = p["type"]
    if t == "clean":
        return x
    if t == "salt":
        r = np.random.default_rng(p["seed"])
        m = torch.from_numpy(r.random((SIZE, SIZE)) < p["p"])
        v = torch.from_numpy(r.random((SIZE, SIZE)) < 0.5).float()
        x = x.clone()
        x[:, m] = v[m]
        return x
    if t == "blur":
        return TF.gaussian_blur(x, [p["k"], p["k"]], [p["sigma"], p["sigma"]])
    x = x.clone()
    for a, b, w, h in p["rects"]:
        x[:, b:b + h, a:a + w] = 0
    return x

class PetDataset(Dataset):
    """Train: pass indices -> new random corruption on every load.
    Val/test: pass manifest -> fixed corruption. Returns (corrupted, clean, label)."""
    def __init__(self, images, indices=None, manifest=None):
        self.images, self.indices, self.manifest = images, indices, manifest
    def __len__(self):
        return len(self.manifest) if self.manifest is not None else len(self.indices)
    def __getitem__(self, i):
        if self.manifest is not None:
            e = self.manifest[i]
            clean, p = self.images[e["idx"]].float() / 255, e["params"]
        else:
            clean = self.images[self.indices[i]].float() / 255
            rng = np.random.default_rng()
            p = sample_params(CLASSES[int(rng.integers(4))], rng)
        return apply_corruption(clean, p), clean, CLASSES.index(p["type"])

def build_manifests(out_dir="/kaggle/working/GenAI-Assignment1/manifests"):
    imgs = load_split("trainval")
    _, val_idx = train_val_indices(len(imgs))
    val = []
    for j, idx in enumerate(val_idx):
        rng = np.random.default_rng(SEED * 1_000_000 + j)
        p = sample_params(CLASSES[int(rng.integers(4))], rng)
        p["severity"] = severity_of(p)
        val.append({"idx": idx, "params": p})
    test = []
    for idx in range(len(load_split("test"))):
        test.append({"idx": idx, "params": {"type": "clean", "severity": "none"}})
        for ci, c in enumerate(CLASSES[1:]):
            for lv in range(3):
                rng = np.random.default_rng((SEED + 1) * 1_000_000 + idx * 10 + ci * 3 + lv + 1)
                p = sample_params(c, rng, level=lv)
                p["severity"] = ["low", "medium", "high"][lv]
                test.append({"idx": idx, "params": p})
    os.makedirs(out_dir, exist_ok=True)
    json.dump(val, open(f"{out_dir}/val_manifest.json", "w"))
    json.dump(test, open(f"{out_dir}/test_manifest.json", "w"))
    return val, test
