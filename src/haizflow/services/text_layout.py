"""Put the translation where the source video shows its own text.

Detected on-screen text blocks are split into two kinds. Narration captions
repeat what is spoken: the Vietnamese subtitle cue is moved onto the caption it
replaces, so it covers the masked source text instead of sitting elsewhere.
Labels (titles, callouts, product notes) are not spoken and have no subtitle:
they are translated and added as text clips over the masked label, timed to it.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

from haizflow.schemas.editor import EditorClip, EditorTransform

AUTO_LABEL = "auto_text_label"
AUTO_POSITION = "auto_text_position"
LABEL_TRACK = "overlays"
# Share of a single static line's Han characters found in the overlapping
# speech above which it is a narration caption. Transcripts may be traditional
# while captions are simplified, so shared characters still differ.
NARRATION_MATCH = 0.75
# The recogniser reads check and cross emoji as these characters, and a cross
# directly after Chinese text sometimes as a Latin X.
_MARKS = re.compile(r"[×√✓✗✔✘❌✅]|(?<=[㐀-鿿])[Xx]")
_HAN = re.compile(r"[㐀-鿿]")
# Visible glyph height of a detected line relative to its mask box, which adds
# margins around the text the detector found.
_GLYPH_SHARE = 0.6


def _han(text: str) -> list[str]:
    return _HAN.findall(text or "")


def _overlap_ms(start_a, end_a, start_b, end_b) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _window(block: dict) -> tuple[int, int]:
    return block["intervals"][0]["start_ms"], block["intervals"][-1]["end_ms"]


def _cues(document) -> list:
    return sorted((clip for clip in document.clips if clip.track_id == "subtitles" and clip.kind == "subtitle"),
                  key=lambda clip: clip.start_ms)


def narration_score(block: dict, cues) -> float:
    """Fraction of the block's Han characters that occur in the speech it overlaps."""
    characters = _han(block.get("text", ""))
    if not characters:
        return 0.0
    start, end = _window(block)
    spoken = set()
    for cue in cues:
        if _overlap_ms(start, end, cue.start_ms, cue.start_ms + cue.duration_ms):
            spoken.update(_han(str((cue.metadata.get("segment_payload") or {}).get("source_text") or "")))
    return sum(character in spoken for character in characters) / len(characters)


def _similarity(first: list[str], second: list[str]) -> float:
    """Shared characters (with multiplicity) over the longer reading."""
    remaining = list(second)
    shared = 0
    for character in first:
        if character in remaining:
            remaining.remove(character)
            shared += 1
    return shared / max(1, len(first), len(second))


def _change(readings: list[str]) -> str:
    """'replaced' when a later reading is another sentence, 'grows' when text is
    added to what was there, else 'static'. Small OCR differences are 'static'."""
    texts = [_han(reading) for reading in readings if _han(reading)]
    change = "static"
    for earlier, later in zip(texts, texts[1:]):
        if _similarity(earlier, later) >= 0.8:
            continue
        if len(later) > len(earlier) and _similarity(earlier, later) * len(later) >= 0.8 * len(earlier):
            change = "grows"
        else:
            return "replaced"
    return change


def _mostly_han(text: str) -> bool:
    visible = [character for character in _MARKS.sub("", text) if not character.isspace()]
    return bool(visible) and len(_han(text)) >= 0.5 * len(visible)


def classify(blocks: list[dict], cues) -> tuple[list[dict], list[dict]]:
    """Split blocks into narration captions and labels that need their own text.

    Spoken words alone cannot tell them apart: the narrator often reads the
    labels aloud. Captions are replaced sentence by sentence within a block's
    lifetime, while labels stay put or build up as items are added.
    """
    narration, labels = [], []
    for block in blocks:
        if not _mostly_han(block.get("text", "")):
            continue  # Unreadable, numbers or printed material: mask only.
        change = _change(block.get("readings") or [block["text"]])
        spoken = len(block["lines"]) == 1 and narration_score(block, cues) >= NARRATION_MATCH
        (narration if change == "replaced" or (change == "static" and spoken) else labels).append(block)
    return narration, labels


def _centre_y(block: dict) -> float:
    region = block["region"]
    return round(region["y_percent"] + region["height_percent"] / 2, 2)


def position_subtitles(document, narration: list[dict]) -> int:
    """Move each cue onto the narration caption it overlaps the longest."""
    moved = 0
    for cue in _cues(document):
        automatic = bool(cue.metadata.get(AUTO_POSITION))
        if "position_y_percent" in cue.style_override and not automatic:
            continue  # The user placed this cue; keep it.
        end = cue.start_ms + cue.duration_ms
        best = max(narration, key=lambda block: _overlap_ms(*_window(block), cue.start_ms, end), default=None)
        if best is None or not _overlap_ms(*_window(best), cue.start_ms, end):
            if automatic:
                cue.style_override = {key: value for key, value in cue.style_override.items()
                                      if key != "position_y_percent"}
                cue.metadata = {key: value for key, value in cue.metadata.items() if key != AUTO_POSITION}
            continue
        cue.style_override = {**cue.style_override, "position_y_percent": _centre_y(best)}
        cue.metadata = {**cue.metadata, AUTO_POSITION: True}
        moved += 1
    return moved


def label_source(block: dict) -> str:
    """The label text to translate, without the recognised check/cross marks."""
    return "\n".join(line for line in (_MARKS.sub("", value).strip() for value in block["text"].splitlines()) if line)


def label_clips(document, labels: list[dict], translations: dict[str, str], reference_height: int) -> int:
    """Replace earlier automatic label clips with one text clip per label."""
    font_family = next((style.font_family for style in document.styles
                        if style.style_id == document.default_subtitle_style_id), "Bangers")
    document.clips = [clip for clip in document.clips if not clip.metadata.get(AUTO_LABEL)]
    added = 0
    for index, block in enumerate(labels, start=1):
        source = label_source(block)
        text = translations.get(source, "").strip()
        if not text:
            continue
        start, end = _window(block)
        document.clips.append(EditorClip(
            clip_id=f"auto-label-{index}-{start}",
            track_id=LABEL_TRACK,
            kind="text",
            name=text,
            start_ms=start,
            duration_ms=end - start,
            transform=EditorTransform(position_x_percent=_centre_x(block), position_y_percent=_centre_y(block)),
            style_override=label_style(block, font_family, reference_height),
            metadata={AUTO_LABEL: True, "source_text": source},
        ))
        added += 1
    return added


def _centre_x(block: dict) -> float:
    region = block["region"]
    return round(region["x_percent"] + region["width_percent"] / 2, 2)


def label_style(block: dict, font_family: str, reference_height: int) -> dict:
    """Yellow outlined text sized to the source label's own glyphs."""
    line_height = min(line["region"]["height_percent"] for line in block["lines"])
    return {
        "target_type": "text_overlay",
        "font_family": font_family,
        "font_size": max(12.0, round(reference_height * line_height / 100 * _GLYPH_SHARE, 1)),
        "text_color": "#FFE14D",
        "outline_color": "#000000",
        "outline_width": 3.0,
        "alignment": "center",
        "max_width_percent": min(92.0, max(30.0, block["region"]["width_percent"] * 1.4)),
        "max_lines": max(1, len(block["lines"]) + 1),
    }


def batch_render_layout(video, region: dict | None, srt_path: str, transcript_path: str,
                        removal_mode: str, font_family: str, translate=None, progress_callback=None) -> dict:
    """Mask, caption placement and label overlays for the automatic pipeline.

    Returns ``{"region": payload, "cue_regions": {srt index: source region},
    "signature": str}``. Without a scan the plain detected region is kept, so a
    failure here never blocks a render. Label translations are cached beside
    the scan, so a re-render does not call the translator again.
    """
    import srt

    from haizflow.pipeline.text_masks import detect_text_layers
    from haizflow.pipeline.text_overlay_sprite import render_text_overlay_sprite
    from haizflow.schemas.editor import EditorTextStyle
    from haizflow.services.ocr_layers import text_layers_path
    from haizflow.utils.ffmpeg import get_video_dimensions

    source = video.files["video_input"]
    blocks = detect_text_layers(source, text_layers_path(video), video.video_id, progress_callback)
    with open(transcript_path, encoding="utf-8") as stream:
        segments = json.load(stream)
    cues = [SimpleNamespace(start_ms=round(float(item["start"]) * 1000),
                            duration_ms=round((float(item["end"]) - float(item["start"])) * 1000),
                            metadata={"segment_payload": {"source_text": item.get("source_text") or ""}})
            for item in segments if isinstance(item, dict)]
    narration, labels = classify(blocks, cues)

    cue_regions = {}
    with open(srt_path, encoding="utf-8") as stream:
        for subtitle in srt.parse(stream.read()):
            start = round(subtitle.start.total_seconds() * 1000)
            end = round(subtitle.end.total_seconds() * 1000)
            best = max(narration, key=lambda block: _overlap_ms(*_window(block), start, end), default=None)
            if best is not None and _overlap_ms(*_window(best), start, end):
                cue_regions[subtitle.index] = best["region"]

    cache = text_layers_path(video).with_name("label-translations.json")
    try:
        translations = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        translations = {}
    missing = [text for text in (label_source(block) for block in labels) if text and text not in translations]
    if missing:
        try:
            translations.update((translate or translate_labels)(video, missing))
            cache.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            if "cancelled" in str(exc).lower():
                raise
            # Masks and caption placement do not depend on the translator.
            from haizflow.services.video_store import log_to_video

            log_to_video(video.video_id, f"On-screen label translation failed: {exc}", level="WARNING",
                         component="OCR")

    width, height = get_video_dimensions(source)
    overlays = []
    for block in labels:
        text = translations.get(label_source(block), "").strip()
        if not text:
            continue
        style = EditorTextStyle(style_id="auto-label", **label_style(block, font_family, height))
        sprite = render_text_overlay_sprite(text, style, width, height)
        overlays.append({"path": sprite["path"], "text": text, "x_percent": _centre_x(block),
                         "y_percent": _centre_y(block), "intervals": block["intervals"]})
    from haizflow.services.ocr_layers import masks_for

    payload = {**(region or {}),
               "treatment_layers": [{"region": mask["region"], "mode": removal_mode, "intervals": mask["intervals"],
                                     "start_ms": mask["intervals"][0]["start_ms"],
                                     "duration_ms": mask["intervals"][-1]["end_ms"] - mask["intervals"][0]["start_ms"]}
                                    for mask in masks_for(video, blocks, region)],
               "text_overlays": overlays}
    signature = json.dumps({"layers": payload["treatment_layers"], "cues": cue_regions,
                            "labels": [(item["text"], item["x_percent"], item["y_percent"], item["intervals"])
                                       for item in overlays]}, sort_keys=True, ensure_ascii=False)
    return {"region": payload, "cue_regions": cue_regions, "signature": signature}


def apply(video, translate=None) -> dict:
    """Detect, classify and lay out translated text for ``video``'s editor document.

    Labels keep the text of an earlier automatic clip with the same source, so
    a re-run neither calls the translator again nor discards the user's edits.
    """
    from haizflow.services import editor_documents
    from haizflow.services.ocr_layers import detected_text_layers
    from haizflow.utils.ffmpeg import get_video_dimensions

    blocks = detected_text_layers(video)
    if blocks is None:
        return {"positioned": 0, "labels": 0}
    document = editor_documents.ensure(video)
    narration, labels = classify(blocks, _cues(document))
    known = {str(clip.metadata.get("source_text") or ""): clip.name
             for clip in document.clips if clip.metadata.get(AUTO_LABEL)}
    sources = [label_source(block) for block in labels]
    missing = [source for source in sources if source and source not in known]
    translations = {**known, **((translate or translate_labels)(video, missing) if missing else {})}
    _width, height = get_video_dimensions(video.files["video_input"])
    result = {}

    def layout(changed):
        if not any(track.track_id == LABEL_TRACK for track in changed.tracks):
            return
        result["positioned"] = position_subtitles(changed, narration)
        result["labels"] = label_clips(changed, labels, translations, height)

    editor_documents.mutate(video, layout)
    return {"positioned": result.get("positioned", 0), "labels": result.get("labels", 0)}


def translate_labels(video, texts: list[str]) -> dict[str, str]:
    """Translate label texts with the project's own translation settings."""
    from haizflow.services.translation import translate_segments

    texts = [text for text in dict.fromkeys(texts) if text]
    if not texts:
        return {}
    # Labels are on screen for seconds, not spoken: give each a generous slot
    # so speech-length budgets do not reject a faithful translation.
    segments = [{"text": text.replace("\n", " "), "start": index * 10.0, "end": index * 10.0 + 8.0, "language": "zh"}
                for index, text in enumerate(texts)]
    with tempfile.TemporaryDirectory(prefix="label-translation-") as directory:
        source = Path(directory) / "labels.json"
        target = Path(directory) / "labels.vi.json"
        source.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        translate_segments(
            str(source), str(target), video.video_id,
            target_language=str(getattr(video, "target_language", "vi") or "vi"),
            source_language="zh",
            provider=str(getattr(video, "translator_provider", "hymt2") or "hymt2"),
            translation_model=str(getattr(video, "translation_model", "auto") or "auto"),
        )
        translated = json.loads(target.read_text(encoding="utf-8"))
    return {text: str(item.get("text") or "").strip() for text, item in zip(texts, translated)}
