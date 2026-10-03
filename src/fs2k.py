import json, os, random, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

ROOT = "/kaggle/working/data/FS2K"
SIZE, SEED = 128, 42
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]

def _find(base):
    for ext in (".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"):
        if os.path.exists(base + ext):
            return base + ext
    return base + ".jpg"   # missing: the pairing check reports it

def paths(name):
    d, f = name.split("/")
    sd, sf = d.replace("photo", "sketch"), f.replace("image", "sketch")
    return _find(f"{ROOT}/photo/{d}/{f}"), _find(f"{ROOT}/sketch/{sd}/{sf}")

def _load(entries, cache):
    if os.path.exists(cache):
        return torch.load(cache)
    P = torch.empty(len(entries), 3, SIZE, SIZE, dtype=torch.uint8); S = torch.empty_like(P)
    for i, e in enumerate(entries):
        pp, sp = paths(e["image_name"])
        for out, p in ((P, pp), (S, sp)):
            img = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            out[i] = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    y = torch.tensor([e["style"] for e in entries])
    torch.save((P, S, y), cache)
    return P, S, y

def get_splits(cache_dir="/kaggle/working/cache"):
    """Official train/test; 15% of official train held out as validation, stratified by style, seed 42."""
    from sklearn.model_selection import train_test_split
    os.makedirs(cache_dir, exist_ok=True)
    tr = json.load(open(f"{ROOT}/anno_train.json")); te = json.load(open(f"{ROOT}/anno_test.json"))
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.15,
                                  stratify=[e["style"] for e in tr], random_state=SEED)
    P, S, y = _load(tr, f"{cache_dir}/fs2k_trainfull.pt")
    return {"train": (P[i_tr], S[i_tr], y[i_tr]), "val": (P[i_va], S[i_va], y[i_va]),
            "test": _load(te, f"{cache_dir}/fs2k_test.pt")}

class FS2KDataset(Dataset):
    """Returns (photo, sketch, style) in [-1, 1]. Spatial augmentation is applied identically to both."""
    def __init__(self, data, augment=False):
        self.P, self.S, self.y = data; self.augment = augment
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        p, s = self.P[i].float() / 255, self.S[i].float() / 255
        if self.augment:
            if random.random() < 0.5:
                p, s = TF.hflip(p), TF.hflip(s)
            c = random.randint(104, SIZE); t = random.randint(0, SIZE - c); l = random.randint(0, SIZE - c)
            p = TF.resized_crop(p, t, l, c, c, [SIZE, SIZE], antialias=True)
            s = TF.resized_crop(s, t, l, c, c, [SIZE, SIZE], antialias=True)
        return p * 2 - 1, s * 2 - 1, int(self.y[i])
