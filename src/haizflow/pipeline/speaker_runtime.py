"""Bounded ONNX inference owned by the isolated speaker worker, not Qt."""

from __future__ import annotations

import os
import sys
from pathlib import Path


class SpeakerSession:
    def __init__(self, model: Path, report=None):
        import onnxruntime as ort

        self.ort = ort
        self.model = model
        self.report = report
        self.device = "cpu"
        self.fallback_reason = ""
        self.options = ort.SessionOptions()
        # On macOS with CoreML available, accelerate inference through CoreMLExecutionProvider (~16x speedup).
        # Otherwise, bounded CPU inference with bounded threads.
        available = ort.get_available_providers() if hasattr(ort, "get_available_providers") else ["CPUExecutionProvider"]
        use_coreml = (
            sys.platform == "darwin"
            and "CoreMLExecutionProvider" in available
            and os.environ.get("HAIZFLOW_DISABLE_COREML") != "1"
        )
        if use_coreml:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            self.device = "coreml"
            self.options.intra_op_num_threads = min(4, os.cpu_count() or 2)
            self.options.inter_op_num_threads = 1
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"
            self.options.intra_op_num_threads = 2
            self.options.inter_op_num_threads = 1
        self.options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(model), sess_options=self.options, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(model), sess_options=self.options, providers=["CPUExecutionProvider"])
            self.device = "cpu"
        if report:
            report(self.device, "")

    def run(self, features):
        return self.session.run(None, {"feats": features})


class VoiceProfileSession:
    """Small q8 ECAPA classifier; bounded CPU inference without Torch or VRAM."""

    def __init__(self, model: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 2) if sys.platform == "darwin" else 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # The q8 graph yields NaN logits once ORT's extended fusions are applied
        # (seen with onnxruntime 1.31 on Apple silicon); basic folding is exact.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    def male_probability(self, pcm):
        import numpy as np

        # The graph contains its own mel normalization. Input is linear mono
        # 16kHz float PCM, not standardized samples/WeSpeaker fbank. Separated
        # float WAVs may exceed nominal full scale: preserve that shape instead
        # of clipping or applying different gains to neighboring speaker turns.
        audio = np.asarray(pcm, dtype=np.float32) / 32768.0
        logits = np.asarray(self.session.run(None, {"input_values": audio[None]})[0]).reshape(-1)
        if logits.shape != (2,) or not np.isfinite(logits).all():
            raise RuntimeError("Voice profile model returned invalid logits.")
        probabilities = np.exp(logits - logits.max())
        return float(probabilities[0] / probabilities.sum())
