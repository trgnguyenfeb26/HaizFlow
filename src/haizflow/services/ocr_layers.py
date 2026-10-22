"""Timed source-caption treatments in the saved editor document."""

import json
from pathlib import Path

from haizflow.schemas.editor import EditorClip, EditorTrack
from haizflow.services.ocr_regions import effective_region, normalize_region

PRIMARY_TRACK = "ocr-source"
PRIMARY_CLIP = "ocr-source-region"
EXTRA_LAYERS_ENABLED = True
# Burned-in captions usually appear slightly before and linger after speech.
SUBTITLE_PADDING_MS = 250
# Merge short pauses so the mask does not flicker between adjacent captions.
SUBTITLE_MERGE_GAP_MS = 400


def detected_region(video):
    from haizflow.services import video_store

    path = str((getattr(video, "files", {}) or {}).get("ocr_region") or
               Path(video_store.get_video_dir(video.video_id)) / "temp" / "original_subtitle_region.json")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload.get("region", payload) if isinstance(payload, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def ensure_primary(document) -> bool:
    changed = False
    if not any(track.track_id == PRIMARY_TRACK for track in document.tracks):
        document.tracks.append(EditorTrack(
            track_id=PRIMARY_TRACK, kind="ocr", name="Vùng nhận diện", order=-1,
        ))
        changed = True
    if not any(clip.clip_id == PRIMARY_CLIP for clip in document.clips):
        document.clips.append(EditorClip(
            clip_id=PRIMARY_CLIP, track_id=PRIMARY_TRACK, kind="ocr",
            name="Vùng nhận diện", duration_ms=max(1, document.sequence.duration_ms),
            metadata={"primary_ocr": True},
        ))
        changed = True
    return changed


def text_layers_path(video) -> Path:
    from haizflow.services import video_store

    return Path(video_store.get_video_dir(video.video_id)) / "cache" / "manual" / "ocr" / "text-layers.json"


def detected_text_layers(video) -> list[dict] | None:
    """Timed on-screen text rows, or ``None`` until the source has been scanned."""
    from haizflow.pipeline.text_masks import cached_text_layers

    source = str((getattr(video, "files", {}) or {}).get("video_input") or "")
    if not source or not getattr(video, "video_id", ""):
        return None
    return cached_text_layers(text_layers_path(video), source)


def _to_sequence(document, windows) -> list[dict]:
    """Detection runs on the source file; masks live in sequence time."""
    if not document.sequence.edit_decisions:
        return [dict(item) for item in windows]
    result = []
    for decision in document.sequence.edit_decisions:
        shift = decision.sequence_start_ms - decision.source_start_ms
        for item in windows:
            start = max(item["start_ms"], decision.source_start_ms)
            end = min(item["end_ms"], decision.source_end_ms)
            if end > start:
                result.append({"start_ms": start + shift, "end_ms": end + shift})
    return result


def _clip_windows(windows, clip) -> list[dict]:
    start, end = clip.start_ms, clip.start_ms + clip.duration_ms
    return [{"start_ms": max(start, item["start_ms"]), "end_ms": min(end, item["end_ms"])}
            for item in windows if min(end, item["end_ms"]) > max(start, item["start_ms"])]


def _merged(windows) -> list[dict]:
    merged: list[dict] = []
    for item in sorted(windows, key=lambda value: value["start_ms"]):
        if merged and item["start_ms"] <= merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], item["end_ms"])
        else:
            merged.append(dict(item))
    return merged


def primary_intervals(video, document, clip) -> list[dict] | None:
    """When the primary region masks: detected text if scanned, else subtitle cues."""
    text_layers = detected_text_layers(video) if video is not None else None
    if text_layers is not None:
        windows = _to_sequence(document, [item for layer in text_layers for item in layer["intervals"]])
        return _merged(_clip_windows(windows, clip))
    return subtitle_intervals(document, clip)


def _same_row(first: dict, second: dict) -> bool:
    overlap = (min(first["y_percent"] + first["height_percent"], second["y_percent"] + second["height_percent"])
               - max(first["y_percent"], second["y_percent"]))
    return overlap >= 0.5 * min(first["height_percent"], second["height_percent"])


def masks_for(video, blocks: list[dict], region: dict | None) -> list[dict]:
    """Where and when to mask detected text: ``[{"region", "intervals"}]``.

    Each caption gets its own detected box. When the user has drawn the
    detected region by hand, text on that row is masked with their box's height
    and position instead, still only while it is shown; a caption wider than
    the box widens it so no characters show. Text elsewhere keeps its box.
    """
    masks = [mask for block in blocks for mask in (block.get("masks")
                                                   or [{"region": block["region"], "intervals": block["intervals"]}])]
    if not region or not getattr(video, "original_subtitle_region_override", None):
        return masks
    result = []
    for mask in masks:
        if not _same_row(mask["region"], region):
            result.append(mask)
            continue
        detected = mask["region"]
        left = min(region["x_percent"], detected["x_percent"])
        right = max(region["x_percent"] + region["width_percent"], detected["x_percent"] + detected["width_percent"])
        result.append({"region": {**region, "x_percent": left, "width_percent": right - left},
                       "intervals": mask["intervals"]})
    return result


def _primary_treatments(video, document, item) -> list[dict]:
    """Split the primary treatment into timed masks over the detected text."""
    from haizflow.services import editor_documents

    text_layers = detected_text_layers(video)
    if text_layers is None:
        return [item]
    clip = editor_documents.clip_by_id(document, item["clip_id"])
    result = []
    for mask in masks_for(video, text_layers, item["region"]):
        windows = _clip_windows(_to_sequence(document, mask["intervals"]), clip)
        if windows:
            result.append({**item, "region": mask["region"], "intervals": windows,
                           "start_ms": windows[0]["start_ms"],
                           "duration_ms": windows[-1]["end_ms"] - windows[0]["start_ms"]})
    return result


def subtitle_intervals(document, clip) -> list[dict] | None:
    """Times inside ``clip`` where a subtitle cue is shown, padded and merged.

    The detector samples a few frames for one stable box and records no
    timing, so the primary region follows the subtitle track instead. ``None``
    means there are no cues yet and the region keeps covering its whole clip.
    """
    clip_start = clip.start_ms
    clip_end = clip.start_ms + clip.duration_ms
    cues = sorted(
        (item.start_ms, item.start_ms + item.duration_ms) for item in document.clips
        if item.track_id == "subtitles" and item.enabled and item.duration_ms > 0
    )
    if not cues:
        return None
    merged: list[list[int]] = []
    for start, end in cues:
        start = max(clip_start, start - SUBTITLE_PADDING_MS)
        end = min(clip_end, end + SUBTITLE_PADDING_MS)
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= SUBTITLE_MERGE_GAP_MS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start_ms": start, "end_ms": end} for start, end in merged]


def layers(video, document, detected=None) -> list[dict]:
    """Topmost first, matching the panel and timeline track order."""
    tracks = {track.track_id: track for track in document.tracks if track.kind == "ocr"}
    primary_region = effective_region(video, detected)
    result = []
    for clip in sorted(document.clips, key=lambda clip: (
        tracks[clip.track_id].order if clip.track_id in tracks else 0, clip.start_ms, clip.clip_id,
    )):
        if clip.kind != "ocr" or clip.track_id not in tracks:
            continue
        primary = bool(clip.metadata.get("primary_ocr"))
        if not primary and not EXTRA_LAYERS_ENABLED:
            continue
        region = primary_region if primary else clip.metadata.get("region", {})
        try:
            region = normalize_region(region) if region else {}
        except ValueError:
            region = {}
        intervals = primary_intervals(video, document, clip) if primary and EXTRA_LAYERS_ENABLED else None
        result.append({
            "clip_id": clip.clip_id, "track_id": clip.track_id,
            "name": clip.name, "primary": primary, "region": region,
            "intervals": intervals,
            "visible": tracks[clip.track_id].visible,
            "pending": bool(clip.metadata.get("pending", False)),
            "mode": str(getattr(video, "original_subtitle_removal_mode", "patch"))
                if primary else str(clip.metadata.get("removal_mode", "blur")),
            "start_ms": clip.start_ms if EXTRA_LAYERS_ENABLED else 0,
            "duration_ms": clip.duration_ms if EXTRA_LAYERS_ENABLED else document.sequence.duration_ms,
            "enabled": (clip.enabled and tracks[clip.track_id].visible
                and (bool(getattr(video, "remove_original_subtitles", True)) if primary else True))
                if EXTRA_LAYERS_ENABLED else bool(getattr(video, "remove_original_subtitles", True)),
        })
    return result


def render_region(video, detected=None) -> dict | None:
    """Keep OCR layout geometry separate from the ordered render treatments."""
    from haizflow.services import editor_documents

    primary = effective_region(video, detected)
    if not EXTRA_LAYERS_ENABLED:
        return primary if getattr(video, "remove_original_subtitles", True) else None
    if getattr(video, "project_type", "single") != "manual":
        return primary if getattr(video, "remove_original_subtitles", True) else None
    document = editor_documents.load(video.video_id)
    if document is None:
        return primary if getattr(video, "remove_original_subtitles", True) else None
    ensure_primary(document)
    treatments = [treatment for item in reversed(layers(video, document, detected))
                  if item["enabled"] and item["region"] and item["duration_ms"] > 0 and item["intervals"] != []
                  for treatment in (_primary_treatments(video, document, item) if item["primary"] else [item])]
    return {**(primary or {}), "treatment_layers": treatments} if treatments else None


def render_time_offset(sequence: dict, source_start: float) -> float:
    """A direct trim proxy reads source time but treatments live in sequence time."""
    decisions = sequence.get("edit_decisions", []) if sequence else []
    if len(decisions) == 1:
        return (float(decisions[0].get("sequence_start_ms", 0))
                - float(decisions[0].get("source_start_ms", 0))) / 1000 + source_start
    return source_start
