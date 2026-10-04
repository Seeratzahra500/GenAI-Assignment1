"""Runtime corruptions, mirroring src/data.py used for training and the test manifests.

Images are float32 arrays, shape (H, W, 3), values in [0, 1].
"""
import math

import cv2
import numpy as np

SIZE = 128
CORRUPTIONS = ("none", "salt", "blur", "occlusion")
SEVERITIES = ("low", "medium", "high")

_SALT_P = {"low": 0.03, "medium": 0.08, "high": 0.15}
_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}  # (kernel, sigma)
_OCC = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}  # (rectangles, area fraction)


def _occlusion_rects(rng, n, total):
    rects = []
    area = total / n * SIZE * SIZE
    for _ in range(n):
        ratio = rng.uniform(0.5, 2.0)
        w = int(min(SIZE, max(1, round(math.sqrt(area * ratio)))))
        h = int(min(SIZE, max(1, round(area / w))))
        rects.append([int(rng.integers(0, SIZE - w + 1)), int(rng.integers(0, SIZE - h + 1)), w, h])
    return rects


def make_params(kind, severity, rng):
    if kind == "none":
        return {"type": "none"}
    if kind == "salt":
        return {"type": "salt", "severity": severity, "p": _SALT_P[severity], "noise_seed": int(rng.integers(2**31))}
    if kind == "blur":
        k, s = _BLUR[severity]
        return {"type": "blur", "severity": severity, "kernel": k, "sigma": s}
    n, frac = _OCC[severity]
    return {"type": "occlusion", "severity": severity, "target_area": frac, "rects": _occlusion_rects(rng, n, frac)}


def apply_corruption(img, params):
    kind = params["type"]
    if kind == "none":
        return img
    if kind == "salt":
        r = np.random.default_rng(params["noise_seed"])
        mask = r.random((SIZE, SIZE)) < params["p"]
        value = (r.random((SIZE, SIZE)) < 0.5).astype(np.float32)  # white or black, equal probability
        out = img.copy()
        out[mask] = value[mask][:, None]
        return out
    if kind == "blur":
        k = params["kernel"]
        return cv2.GaussianBlur(img, (k, k), params["sigma"])
    out = img.copy()
    for x, y, w, h in params["rects"]:
        out[y:y + h, x:x + w] = 0.0
    return out


def describe(params, corrupted, clean):
    """Settings shown in the UI (adds the measured occluded area, which is a little below the
    target when rectangles overlap)."""
    d = dict(params)
    if params["type"] == "occlusion":
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        for x, y, w, h in params["rects"]:
            mask[y:y + h, x:x + w] = True
        d["measured_area"] = round(float(mask.mean()), 4)
    return d
