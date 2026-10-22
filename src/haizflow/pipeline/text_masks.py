"""Find where and when burned-in text is on screen, for timed source masks.

The subtitle-region detector samples a few frames for one stable box. Videos
that move their captions, or add labels elsewhere, then leak text outside that
box or mask it while nothing is shown. This pass runs the same local text
detector on frames sampled at a fixed rate and groups the boxes into screen
rows, each with the time windows in which text occupies it.

Only the detector model runs here, through onnxruntime in the core process, so
no recognition model or optional engine is required. Text that moves while it
is visible (labels on a product held in hand) is left untouched; captions and
overlays are static and are masked.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLES_PER_SECOND = 4
CACHE_VERSION = 1
# The detector input; both sides must be multiples of 32.
DETECTOR_LONG_SIDE = 864
PROBABILITY_THRESHOLD = 0.3
# DB-style detectors predict a shrunken text kernel; grow it back to glyphs.
UNCLIP_RATIO = 0.75
MIN_BOX_HEIGHT_PERCENT = 0.6
MIN_BOX_WIDTH_PERCENT = 1.5
# Masked lines after unclipping: smaller is fine print on objects, taller is a
# block of several merged lines (packaging, documents), not an overlay.
CAPTION_HEIGHT_PERCENT = (3.0, 7.0)
# Edge-to-edge lines are table rows on printed material, not captions.
MAX_CAPTION_WIDTH_PERCENT = 90.0
# A frame with more detected lines than this shows printed text (a box, a
# chart). Only the main caption row is masked there. Neighbouring frames are
# not included: that would cut short the captions of the scene before it.
DENSE_FRAME_BOXES = 8
# Characters closer than this are one box; wider gaps separate labels on a row.
COLUMN_GAP_PERCENT = 3.0
# Missing at most this many consecutive samples keeps one time window.
MAX_GAP_SAMPLES = 2
# Between two samples, a row that shifts this much without changing width is
# moving content rather than a caption whose text changed.
MAX_STEP_SHIFT_PERCENT = (3.0, 1.5)
# Text can appear or vanish between two samples; cover that unseen edge.
EDGE_PAD_MS = 120
# Consecutive boxes overlapping this much show the same caption.
SAME_CAPTION_IOU = 0.6
# Recogniser input height, widest accepted line image, and the mean character
# confidence below which a line is treated as unreadable.
REC_HEIGHT = 48
REC_MAX_WIDTH = 1600
MIN_RECOGNITION_CONFIDENCE = 0.5
# Moments, as fractions of a line's window, at which it is read. Captions are
# replaced by the next sentence within a window; labels stay or build up.
READ_FRACTIONS = (0.15, 0.5, 0.85)
CACHE_VERSION_TEXTS = 2
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _detector_size(source_width: int, source_height: int) -> tuple[int, int]:
    scale = DETECTOR_LONG_SIDE / max(source_width, source_height)
    return (max(32, round(source_width * scale / 32) * 32),
            max(32, round(source_height * scale / 32) * 32))


def _runs(flags: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """Half-open index ranges where ``flags`` is set, joining gaps under ``min_gap``."""
    indices = np.flatnonzero(flags)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > min_gap)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def boxes_from_probability(probability: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Return text boxes as ``(x, y, width, height)`` percentages of the frame.

    Captions are horizontal lines, so rows of text pixels are split first and
    each row is split again at wide column gaps. This needs neither OpenCV nor
    polygon clipping, which the core environment does not ship.
    """
    height, width = probability.shape
    mask = probability > PROBABILITY_THRESHOLD
    column_gap = max(1, round(width * COLUMN_GAP_PERCENT / 100))
    boxes = []
    for top, bottom in _runs(mask.any(axis=1)):
        for left, right in _runs(mask[top:bottom].any(axis=0), column_gap):
            box_height = bottom - top
            pad = UNCLIP_RATIO * box_height
            x0, x1 = max(0.0, left - pad), min(float(width), right + pad)
            y0, y1 = max(0.0, top - pad), min(float(height), bottom + pad)
            box = (100 * x0 / width, 100 * y0 / height, 100 * (x1 - x0) / width, 100 * (y1 - y0) / height)
            if (100 * box_height / height >= MIN_BOX_HEIGHT_PERCENT
                    and 100 * (right - left) / width >= MIN_BOX_WIDTH_PERCENT):
                boxes.append(box)
    return boxes


def _row_overlap(a, b) -> float:
    """Vertical overlap as a fraction of the shorter box; 0.5+ is the same row."""
    overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return overlap / max(1e-6, min(a[3], b[3]))


def _union(boxes):
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _moves(previous, current) -> bool:
    shift_x = abs((previous[0] + previous[2] / 2) - (current[0] + current[2] / 2))
    shift_y = abs((previous[1] + previous[3] / 2) - (current[1] + current[3] / 2))
    return (shift_y > MAX_STEP_SHIFT_PERCENT[1]
            or (shift_x > MAX_STEP_SHIFT_PERCENT[0] and abs(previous[2] - current[2]) < 2.0))


def group_text_layers(samples: list[list[tuple[float, float, float, float]]],
                      samples_per_second: float = SAMPLES_PER_SECOND) -> list[dict]:
    """Group per-sample boxes into screen rows with their visible time windows."""
    dense = {index for index, boxes in enumerate(samples) if len(boxes) > DENSE_FRAME_BOXES}
    samples = [[box for box in boxes if CAPTION_HEIGHT_PERCENT[0] <= box[3] <= CAPTION_HEIGHT_PERCENT[1]
                and box[2] <= MAX_CAPTION_WIDTH_PERCENT] for boxes in samples]
    rows: list[dict] = []
    for index, boxes in enumerate(samples):
        for box in boxes:
            # Match the row's latest box, not its union: a growing union would
            # swallow neighbouring rows one caption at a time.
            scored = [(_row_overlap(row["last"], box), row) for row in rows]
            score, row = max(scored, key=lambda item: item[0], default=(0.0, None))
            if row is None or score < 0.5:
                row = {"samples": {}}
                rows.append(row)
            row["last"] = box
            row["samples"].setdefault(index, []).append(box)
    main = max(rows, key=lambda row: len(row["samples"]), default=None)
    step = 1000 / samples_per_second
    layers = []
    for row in rows:
        frames = {index: _union(boxes) for index, boxes in row["samples"].items()
                  if row is main or index not in dense}
        moving = set()
        for index in frames:
            if index - 1 in frames and _moves(frames[index - 1], frames[index]):
                moving.update((index - 1, index))
        present = np.zeros(len(samples), dtype=bool)
        present[[index for index in frames if index not in moving]] = True
        for start, end in _runs(present, MAX_GAP_SAMPLES + 1):
            kept = [index for index in range(start, end) if index in frames and index not in moving]
            # A row's run keeps its whole window for reading and classifying,
            # since captions are told apart by being replaced within it. Masks
            # follow each caption: a new segment starts when the text box
            # changes, so a long or two-line caption does not enlarge the mask
            # over a short one.
            segments = [[kept[0]]]
            for index in kept[1:]:
                if _iou(_union([frames[item] for item in segments[-1]]), frames[index]) >= SAME_CAPTION_IOU:
                    segments[-1].append(index)
                else:
                    segments.append([index])
            layers.append({
                "box": _union([frames[index] for index in kept]),
                **_window_ms(kept[0], kept[-1], step),
                "segments": [{"box": _union([frames[index] for index in segment]),
                              **_window_ms(segment[0], segment[-1], step)} for segment in segments],
            })
    return [_block_layer(block) for block in _blocks(layers)]


def _window_ms(first: int, last: int, step: float) -> dict:
    # A sample stands for the half step either side of it.
    return {"start_ms": max(0, round((first - 0.5) * step) - EDGE_PAD_MS),
            "end_ms": round((last + 0.5) * step) + EDGE_PAD_MS}


def _iou(a, b) -> float:
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / (a[2] * a[3] + b[2] * b[3] - shared)


def _joins(a, b) -> bool:
    """Lines shown together and stacked closely are one caption or label block."""
    overlap_ms = min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"])
    # Against the longer line: a brief detection beside a long caption is not
    # part of it, while the lines of one label appear and vanish together.
    longer_ms = max(a["end_ms"] - a["start_ms"], b["end_ms"] - b["start_ms"])
    (ax, ay, aw, ah), (bx, by, bw, bh) = a["box"], b["box"]
    vertical_gap = max(ay, by) - min(ay + ah, by + bh)
    horizontal_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    return overlap_ms >= 0.5 * longer_ms and vertical_gap <= min(ah, bh) and horizontal_overlap > 0


def _blocks(lines: list[dict]) -> list[list[dict]]:
    """Join lines into blocks. The detector can miss one line of a block for a
    while; masking the whole block for its whole lifetime covers that gap."""
    parent = list(range(len(lines)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            if _joins(lines[first], lines[second]):
                parent[root(first)] = root(second)
    blocks: dict[int, list[dict]] = {}
    for index, line in enumerate(lines):
        blocks.setdefault(root(index), []).append(line)
    return sorted(blocks.values(), key=lambda block: min(line["start_ms"] for line in block))


def _region(box) -> dict:
    x, y, width, height = box
    return {"x_percent": round(x, 2), "y_percent": round(y, 2),
            "width_percent": round(width, 2), "height_percent": round(height, 2)}


def _block_layer(block: list[dict]) -> dict:
    lines = sorted(block, key=lambda line: line["box"][1])
    region = _region(_union([line["box"] for line in lines]))
    window = {"start_ms": min(line["start_ms"] for line in lines), "end_ms": max(line["end_ms"] for line in lines)}
    # A single line is masked caption by caption. A stacked block is masked as
    # a whole for its lifetime, which also covers a line detected late.
    masks = ([{"region": _region(segment["box"]), "intervals": [{"start_ms": segment["start_ms"],
                                                                 "end_ms": segment["end_ms"]}]}
              for segment in lines[0]["segments"]]
             if len(lines) == 1 else [{"region": region, "intervals": [window]}])
    return {
        "region": region,
        "intervals": [window],
        "masks": masks,
        "lines": [{"region": _region(line["box"]), "start_ms": line["start_ms"], "end_ms": line["end_ms"]}
                  for line in lines],
    }


def _source_state(video_path: str) -> dict:
    state = os.stat(video_path)
    return {"path": os.path.abspath(video_path), "size": state.st_size, "mtime_ns": state.st_mtime_ns}


def _line_key(line: dict) -> str:
    return f"{line['start_ms']}:{line['region']['y_percent']}"


def _with_texts(layers: list[dict], texts: dict[str, list[str]]) -> list[dict]:
    """Attach readings; a block reads its lines top to bottom, latest reading as its text."""
    for layer in layers:
        for line in layer["lines"]:
            line["readings"] = list(texts.get(_line_key(line)) or [""] * len(READ_FRACTIONS))
            line["text"] = line["readings"][-1]
        layer["readings"] = ["\n".join(line["readings"][index] for line in layer["lines"] if line["readings"][index])
                             for index in range(len(READ_FRACTIONS))]
        layer["text"] = layer["readings"][-1]
    return layers


def _read_cache(cache_path: str | Path, video_path: str) -> dict | None:
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if payload.get("version") == CACHE_VERSION and payload.get("source") == _source_state(video_path):
            return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def _write_cache(cache_path: str | Path, payload: dict) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, cache_path)


def cached_text_layers(cache_path: str | Path, video_path: str) -> list[dict] | None:
    """Read layers for the current source file without running detection."""
    payload = _read_cache(cache_path, video_path)
    if payload is None:
        return None
    try:
        layers = group_text_layers([[tuple(box) for box in boxes] for boxes in payload["samples"]],
                                   payload["samples_per_second"])
    except (KeyError, TypeError, ValueError):
        return None
    texts = payload.get("texts") if payload.get("texts_version") == CACHE_VERSION_TEXTS else None
    return _with_texts(layers, texts or {})


def _decode(probabilities: np.ndarray, characters: list[str]) -> str:
    """Greedy CTC decoding; index 0 is the blank."""
    best = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    kept = [position for position, index in enumerate(best)
            if index and (position == 0 or index != best[position - 1])]
    if not kept or float(np.mean(confidence[kept])) < MIN_RECOGNITION_CONFIDENCE:
        return ""
    return "".join(characters[best[position]] for position in kept if best[position] < len(characters)).strip()


def _line_image(frame: np.ndarray, region: dict) -> np.ndarray | None:
    """Crop one line back to its glyph band; the mask region carries margins."""
    frame_height, frame_width = frame.shape[:2]
    height = frame_height * region["height_percent"] / 100
    glyph = height / (1 + 2 * UNCLIP_RATIO)
    centre = frame_height * (region["y_percent"] + region["height_percent"] / 2) / 100
    top, bottom = max(0, round(centre - 0.85 * glyph)), min(frame_height, round(centre + 0.85 * glyph))
    left = max(0, round(frame_width * region["x_percent"] / 100))
    right = min(frame_width, round(frame_width * (region["x_percent"] + region["width_percent"]) / 100))
    if bottom - top < 4 or right - left < 4:
        return None
    return frame[top:bottom, left:right]


def _recognition_tensor(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    height, width = image.shape[:2]
    target_width = max(16, min(REC_MAX_WIDTH, round(REC_HEIGHT * width / height)))
    resized = np.asarray(Image.fromarray(image).resize((target_width, REC_HEIGHT), Image.BILINEAR),
                         dtype=np.float32)
    # The recogniser was trained on BGR input normalised to [-1, 1].
    return ((resized[:, :, ::-1] / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def _recognize_lines(video_path: str, lines: list[dict], video_id: str) -> dict[str, list[str]]:
    import onnxruntime

    from haizflow.config import MODELS_DIR
    from haizflow.core.model_integrity import verify_subtitle_ocr_models
    from haizflow.pipeline.process_registry import check_cancellation
    from haizflow.utils.ffmpeg import _binary, get_video_dimensions

    model_dir = verify_subtitle_ocr_models(Path(MODELS_DIR) / "subtitle-ocr")
    session = onnxruntime.InferenceSession(str(model_dir / "subtitle-rec.onnx"), providers=["CPUExecutionProvider"])
    characters = ["", *session.get_modelmeta().custom_metadata_map["character"].splitlines(), " "]
    input_name = session.get_inputs()[0].name
    width, height = get_video_dimensions(video_path)
    frame_bytes = width * height * 3
    texts = {}
    for line in lines:
        readings = []
        for fraction in READ_FRACTIONS:
            check_cancellation(video_id)
            moment = (line["start_ms"] + fraction * (line["end_ms"] - line["start_ms"])) / 1000
            raw = subprocess.run(
                [_binary("ffmpeg"), "-v", "error", "-ss", f"{moment:.3f}", "-i", video_path, "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            image = (_line_image(np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(height, width, 3),
                                 line["region"]) if len(raw) >= frame_bytes else None)
            readings.append("" if image is None else _decode(
                session.run(None, {input_name: _recognition_tensor(image)})[0][0], characters))
        texts[_line_key(line)] = readings
    return texts


def detect_text_layers(
    video_path: str,
    cache_path: str | Path,
    video_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Detect timed text rows for ``video_path``, cached beside the project."""
    payload = _read_cache(cache_path, video_path)
    if payload is None:
        payload = _scan(video_path, video_id, progress_callback)
        _write_cache(cache_path, payload)
    layers = cached_text_layers(cache_path, video_path) or []
    texts = dict(payload.get("texts") or {}) if payload.get("texts_version") == CACHE_VERSION_TEXTS else {}
    missing = [line for layer in layers for line in layer["lines"] if _line_key(line) not in texts]
    if missing:
        texts.update(_recognize_lines(video_path, missing, video_id))
        _write_cache(cache_path, {**payload, "texts": texts, "texts_version": CACHE_VERSION_TEXTS})
    return _with_texts(layers, texts)


def _scan(video_path: str, video_id: str, progress_callback) -> dict:
    import onnxruntime

    from haizflow.config import MODELS_DIR
    from haizflow.core.model_integrity import verify_subtitle_ocr_models
    from haizflow.pipeline.process_registry import check_cancellation
    from haizflow.utils.ffmpeg import _binary, get_video_dimensions, get_video_duration

    source_width, source_height = get_video_dimensions(video_path)
    duration = get_video_duration(video_path)
    if source_width <= 0 or source_height <= 0 or duration <= 0:
        raise RuntimeError("Unable to inspect the source video for on-screen text.")
    model_dir = verify_subtitle_ocr_models(Path(MODELS_DIR) / "subtitle-ocr")
    session = onnxruntime.InferenceSession(str(model_dir / "subtitle-det.onnx"),
                                           providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    width, height = _detector_size(source_width, source_height)
    frame_bytes = width * height * 3
    total = max(1, int(duration * SAMPLES_PER_SECOND))
    process = subprocess.Popen(
        [_binary("ffmpeg"), "-v", "error", "-i", video_path, "-an", "-vf",
         f"fps={SAMPLES_PER_SECOND},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    samples = []
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            check_cancellation(video_id)
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(np.float32) / 255
            tensor = ((image - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
            samples.append(boxes_from_probability(session.run(None, {input_name: tensor})[0][0, 0]))
            if progress_callback:
                progress_callback(min(len(samples), total), total)
    finally:
        process.kill()
        process.wait()
    if not samples:
        raise RuntimeError("On-screen text scan did not produce any video frames.")
    return {"version": CACHE_VERSION, "source": _source_state(video_path),
            "samples_per_second": SAMPLES_PER_SECOND,
            "samples": [[[round(value, 2) for value in box] for box in boxes] for boxes in samples]}
