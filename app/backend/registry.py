"""ONNX Runtime sessions for the seven exported models."""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException

MODEL_FILES = {
    "universal": "task1_universal_conv.onnx",
    "classifier": "task2_classifier.onnx",
    "spec_salt": "task2_spec_salt.onnx",
    "spec_blur": "task2_spec_blur.onnx",
    "spec_occlusion": "task2_spec_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "sketch": "task4_generator.onnx",
}


class ModelRegistry:
    def __init__(self, models_dir):
        self.dir = Path(models_dir)
        self.sessions = {}
        self.errors = {}

    def load(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        for key, fname in MODEL_FILES.items():
            path = self.dir / fname
            if not path.exists():
                self.errors[key] = f"{fname} not found in {self.dir}"
                continue
            try:
                self.sessions[key] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            except Exception as exc:  # corrupted or incompatible file
                self.errors[key] = f"{fname}: {exc}"

    def status(self):
        return {k: {"file": f, "loaded": k in self.sessions, "error": self.errors.get(k)}
                for k, f in MODEL_FILES.items()}

    def run(self, key, feeds):
        """Returns (list of output arrays, milliseconds spent inside the ONNX Runtime call)."""
        sess = self.sessions.get(key)
        if sess is None:
            raise HTTPException(503, f"Model '{key}' is not loaded: {self.errors.get(key, 'unknown error')}. "
                                     "Run the model download step and restart the backend.")
        t0 = time.perf_counter()
        out = sess.run(None, feeds)
        return out, (time.perf_counter() - t0) * 1000.0
