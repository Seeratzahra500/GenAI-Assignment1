"""Download the ONNX models from the public Hugging Face repo into MODELS_DIR (skips files already present)."""
import os
import sys
import urllib.request
from pathlib import Path

from registry import MODEL_FILES

repo = os.getenv("MODELS_REPO", "").strip()
dest = Path(os.getenv("MODELS_DIR", "/models"))
dest.mkdir(parents=True, exist_ok=True)

for fname in MODEL_FILES.values():
    target = dest / fname
    if target.exists() and target.stat().st_size > 0:
        print(f"[models] {fname}: present")
        continue
    if not repo or repo.startswith("REPLACE"):
        print(f"[models] {fname}: missing and MODELS_REPO is not set (edit .env)", file=sys.stderr)
        continue
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    tmp = target.with_suffix(".part")
    try:
        print(f"[models] downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[models] {fname}: download failed ({exc})", file=sys.stderr)
