"""Whisper Turbo on the Apple GPU (Metal) through mlx-whisper.

``MlxWhisperPipeline`` stands in for WhisperX's ``FasterWhisperPipeline`` so
the existing recognition flow in ``transcribe.py`` (VAD chunking, pause
splitting, per-sentence language checks, alignment and timing refinement)
is unchanged; only decoding moves from CTranslate2/CUDA to MLX.

WhisperX's Pyannote VAD still decides the speech chunks, on the CPU. Unlike
WhisperX's batched decode (no timestamps), MLX returns timed phrases inside
each chunk; they are kept, so Chinese cues follow the speech instead of
spanning a whole VAD chunk. MLX has no beam search, so decoding is greedy
with Whisper's temperature fallback, and its known hallucinations on
music, screams and credits are filtered here.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000
# Subtitle-credit and channel phrases Whisper invents over non-speech audio.
_HALLUCINATION_RE = re.compile(
    r"字幕|独播|獨播|YoYo|Television|订阅|訂閱|点赞|點讚|转发|打赏|明镜|請不吝|请不吝|Amara|subtitles?\b",
    re.IGNORECASE,
)


_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")


def _plausible(segment, language: str = "") -> bool:
    """Reject decodes that faster-whisper's thresholds would also distrust."""
    text = segment.text.strip()
    if not text or _HALLUCINATION_RE.search(text):
        return False
    # Over screams or music a Chinese decode drifts into another script
    # ("최정!", "I read."). Keep real code-switching longer than a few words.
    if language == "zh" and not _HAN_RE.search(text) and (
        _HANGUL_RE.search(text) or len(text.split()) <= 3
    ):
        return False
    if segment.avg_logprob < -1.0 or getattr(segment, "compression_ratio", 0.0) > 2.4:
        return False
    return not (segment.no_speech_prob > 0.6 and segment.avg_logprob < -0.7)


def _load(model_dir: str):
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    # mlx_whisper.transcribe() reuses this cached instance for the same path.
    return ModelHolder.get_model(model_dir, mx.float16)


class _MlxDecoder:
    """The subset of faster-whisper's ``WhisperModel`` used by transcribe.py."""

    # MLX keeps short timed phrases, where Whisper's language ID is noisy: a
    # scream inside Chinese speech reads as Korean/English at ~0.6. Only
    # leave the video's primary language on a confident detection.
    SWITCH_CONFIDENCE = 0.9

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = _load(model_dir)
        self.primary_language = ""

    def detect_language(self, audio=None, language_detection_threshold=None, **_unused):
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

        clip = np.asarray(audio, dtype=np.float32)
        mel = log_mel_spectrogram(clip, n_mels=self.model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _tokens, probabilities = self.model.detect_language(mel)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        language, probability = ranked[0]
        primary = self.primary_language
        if primary and language != primary and probability < self.SWITCH_CONFIDENCE:
            language, probability = primary, float(probabilities.get(primary, 0.0)) or 1.0 - probability
            # Report the primary language as confident so it is kept as-is.
            probability = max(probability, self.SWITCH_CONFIDENCE)
        return language, float(probability), [(code, float(value)) for code, value in ranked]

    def transcribe(self, audio, language=None, beam_size=None, word_timestamps=False,
                   condition_on_previous_text=False, vad_filter=False, **_unused):
        import mlx_whisper

        clip = np.asarray(audio, dtype=np.float32)
        result = mlx_whisper.transcribe(
            clip, path_or_hf_repo=self.model_dir, language=language, verbose=None,
            word_timestamps=bool(word_timestamps),
            condition_on_previous_text=bool(condition_on_previous_text),
        )
        segments = [
            SimpleNamespace(
                text=str(item.get("text") or ""),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                avg_logprob=float(item.get("avg_logprob", -10.0)),
                no_speech_prob=float(item.get("no_speech_prob", 1.0)),
                compression_ratio=float(item.get("compression_ratio", 0.0)),
                words=[
                    SimpleNamespace(word=str(word.get("word") or ""), start=float(word["start"]),
                                    end=float(word["end"]), probability=float(word.get("probability", 0.0)))
                    for word in item.get("words") or []
                ] if word_timestamps else None,
            )
            for item in result.get("segments", [])
        ]
        info = SimpleNamespace(language=result.get("language") or language, language_probability=1.0)
        return iter(segments), info


class MlxWhisperPipeline:
    """Drop-in for WhisperX's pipeline: VAD chunks, one decoded text per chunk."""

    def __init__(self, model_dir: Path, vad_model_path: Path):
        import torch
        from whisperx.vads import Pyannote

        self._vad_params = {"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
        self.vad_model = Pyannote(torch.device("cpu"), token=None, model_fp=str(vad_model_path),
                                  **self._vad_params)
        self.model = _MlxDecoder(str(model_dir))

    def transcribe(self, audio, batch_size=None, language=None, chunk_size=30, **_unused):
        audio = np.asarray(audio, dtype=np.float32)
        scores = self.vad_model({"waveform": self.vad_model.preprocess_audio(audio), "sample_rate": SAMPLE_RATE})
        chunks = self.vad_model.merge_chunks(
            scores, chunk_size, onset=self._vad_params["vad_onset"], offset=self._vad_params["vad_offset"],
        )
        if not language:
            self.model.primary_language = ""  # A warm model may hold the previous video's.
            language = self.model.detect_language(audio[: 30 * SAMPLE_RATE])[0]
            # A full-file call establishes the video's primary language.
            self.model.primary_language = language
        segments = []
        for chunk in chunks:
            start, end = float(chunk["start"]), float(chunk["end"])
            clip = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            if not len(clip):
                continue
            decoded, _info = self.model.transcribe(clip, language=language)
            phrases = []
            for item in decoded:
                # A looping decode repeats one phrase; keep its first copy.
                if not _plausible(item, language) or (phrases and phrases[-1].text.strip() == item.text.strip()):
                    continue
                phrases.append(item)
            for index, item in enumerate(phrases):
                # Phrase times are relative to the chunk; keep them inside it
                # and close each phrase where the next begins.
                phrase_start = min(end, start + max(0.0, item.start))
                phrase_end = min(end, start + max(item.end, item.start))
                if index + 1 == len(phrases):
                    phrase_end = max(phrase_end, min(end, phrase_start + 0.25))
                if phrase_end - phrase_start < 0.05:
                    continue
                segments.append({
                    "text": item.text.strip(),
                    "start": round(phrase_start, 3),
                    "end": round(phrase_end, 3),
                    "avg_logprob": item.avg_logprob,
                })
        return {"segments": segments, "language": language}
