"""FastAPI backend: validates uploads, applies runtime corruptions, runs the ONNX models and returns
images (base64 PNG) together with routing information and timings."""
import base64
import io
import math
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.metrics import structural_similarity

from corruptions import CORRUPTIONS, SEVERITIES, SIZE, apply_corruption, describe, make_params
from registry import ModelRegistry

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
CLASSES = ["clean", "salt", "blur", "occlusion"]          # output order of the Task 2 classifier
BRANCHES = ["identity", "salt", "blur", "occlusion"]      # output order of the Task 3 gate
EXPERT_LABEL = {"salt": "Salt-and-pepper specialist", "blur": "Blur specialist", "occlusion": "Occlusion specialist"}
SAMPLE_KINDS = ("pets", "faces")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
registry = ModelRegistry(os.getenv("MODELS_DIR", "/models"))
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", Path(__file__).parent / "samples"))


@asynccontextmanager
async def lifespan(_app):
    registry.load()
    yield


app = FastAPI(title="Generative AI Assignment 1 API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])
router = APIRouter(prefix="/api")


# ----------------------------------------------------------------------------- helpers
def read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 8 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, "Unsupported image type. Use JPEG, PNG or WebP.")
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "The image has more than 25 million pixels.")
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "The file is not a readable image.")


def to_array(img: Image.Image, crop: bool = False) -> np.ndarray:
    """RGB image -> float32 (128, 128, 3) in [0, 1]. Training resized without cropping; `crop`
    (centre square) is used for face photos so webcam frames are not stretched."""
    if crop:
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_nchw(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def to_uri(arr: np.ndarray, scale: int = 1) -> str:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
    if scale > 1:
        img = img.resize((SIZE * scale, SIZE * scale), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


def ssim(a, b):
    return float(structural_similarity(b, a, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False, win_size=11))


def quality(inp, out, ref):
    """PSNR / SSIM against the clean reference (only known when the server applied the corruption)."""
    if ref is None:
        return None
    return {"psnr": round(psnr(out, ref), 2), "ssim": round(ssim(out, ref), 4),
            "input_psnr": round(psnr(inp, ref), 2), "input_ssim": round(ssim(inp, ref), 4)}


def prepare(file: UploadFile, corruption: str, severity: str, seed: Optional[int]):
    """Returns (model input, clean reference or None, settings dict)."""
    if corruption not in CORRUPTIONS:
        raise HTTPException(422, f"corruption must be one of {CORRUPTIONS}")
    if severity not in SEVERITIES:
        raise HTTPException(422, f"severity must be one of {SEVERITIES}")
    clean = to_array(read_image(file))
    if corruption == "none":
        return clean, None, {"type": "none"}
    seed = random.randrange(2**31) if seed is None else seed
    params = make_params(corruption, severity, np.random.default_rng(seed))
    params["seed"] = seed
    corrupted = apply_corruption(clean, params)
    return corrupted, clean, describe(params, corrupted, clean)


def run_image_model(key, x_hwc):
    (out,), ms = registry.run(key, {"input": to_nchw(x_hwc)})
    return np.clip(out[0].transpose(1, 2, 0), 0, 1), ms


# ----------------------------------------------------------------------------- endpoints
@router.get("/health")
def health():
    status = registry.status()
    return {"status": "ok" if all(m["loaded"] for m in status.values()) else "degraded",
            "models": status, "loaded": sum(m["loaded"] for m in status.values()), "total": len(status)}


@router.get("/samples")
def samples():
    out = {}
    for kind in SAMPLE_KINDS:
        folder = SAMPLES_DIR / kind
        out[kind] = sorted(p.name for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    return out


@router.get("/samples/{kind}/{name}")
def sample_file(kind: str, name: str):
    path = SAMPLES_DIR / kind / Path(name).name
    if kind not in SAMPLE_KINDS or not path.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(path)


@router.post("/universal")
def universal(file: UploadFile = File(...), corruption: str = Form("none"),
              severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    inp, ref, settings = prepare(file, corruption, severity, seed)
    out, ms = run_image_model("universal", inp)
    return {"input_image": to_uri(inp), "clean_image": to_uri(ref) if ref is not None else None,
            "output_image": to_uri(out), "settings": settings, "inference_ms": round(ms, 2),
            "metrics": quality(inp, out, ref)}


@router.post("/hard")
def hard_routed(file: UploadFile = File(...), corruption: str = Form("none"),
                severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    inp, ref, settings = prepare(file, corruption, severity, seed)
    (probs,), t_cls = registry.run("classifier", {"input": to_nchw(inp)})
    p = probs[0]
    idx = int(p.argmax())
    predicted = CLASSES[idx]
    if idx == 0:   # clean input: identity bypass, no expert is run
        out, t_exp, selected = inp, 0.0, "Identity bypass (no restoration)"
    else:
        out, t_exp = run_image_model(f"spec_{predicted}", inp)
        selected = EXPERT_LABEL[predicted]
    truth = None if corruption == "none" else corruption
    return {"input_image": to_uri(inp), "clean_image": to_uri(ref) if ref is not None else None,
            "output_image": to_uri(out), "settings": settings,
            "probabilities": {c: round(float(p[i]), 5) for i, c in enumerate(CLASSES)},
            "predicted": predicted, "selected_expert": selected, "true_label": truth,
            "routing_correct": None if truth is None else truth == predicted,
            "classifier_ms": round(t_cls, 2), "expert_ms": round(t_exp, 2),
            "inference_ms": round(t_cls + t_exp, 2), "metrics": quality(inp, out, ref)}


@router.post("/soft")
def soft_mixture(file: UploadFile = File(...), corruption: str = Form("none"),
                 severity: str = Form("medium"), seed: Optional[int] = Form(None)):
    inp, ref, settings = prepare(file, corruption, severity, seed)
    (restored, weights), ms = registry.run("soft_moe", {"input": to_nchw(inp)})
    w = weights[0]
    out = np.clip(restored[0].transpose(1, 2, 0), 0, 1)
    ranked = sorted(zip(BRANCHES, w), key=lambda t: -t[1])
    return {"input_image": to_uri(inp), "clean_image": to_uri(ref) if ref is not None else None,
            "output_image": to_uri(out), "settings": settings,
            "weights": {b: round(float(w[i]), 5) for i, b in enumerate(BRANCHES)},
            "dominant": ranked[0][0], "contributors": [b for b, v in ranked if v >= 0.05],
            "inference_ms": round(ms, 2), "metrics": quality(inp, out, ref)}


@router.post("/sketch")
def sketch(file: UploadFile = File(...), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3.")
    photo = to_array(read_image(file), crop=True)
    x = np.ascontiguousarray((photo * 2 - 1).transpose(2, 0, 1)[None], dtype=np.float32)
    (out,), ms = registry.run("sketch", {"photo": x, "style": np.array([style - 1], dtype=np.int64)})
    result = (out[0].transpose(1, 2, 0) + 1) / 2
    return {"photo_image": to_uri(photo, 4), "sketch_image": to_uri(result, 4), "style": style,
            "inference_ms": round(ms, 2), "model_resolution": SIZE, "display_upscale": 4}


app.include_router(router)
