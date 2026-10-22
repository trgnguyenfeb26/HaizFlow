from haizflow.schemas.editor import EditorClip, EditorDocument, EditorSequence, EditorTrack
from haizflow.services import text_layout


def block(text, start, end, y, *, readings=None, lines=1):
    region = {"x_percent": 20.0, "y_percent": y, "width_percent": 60.0, "height_percent": 5.0 * lines}
    return {"region": region, "intervals": [{"start_ms": start, "end_ms": end}], "text": text,
            "readings": readings or [text, text, text],
            "lines": [{"region": {**region, "height_percent": 5.0}} for _ in range(lines)]}


def cue(clip_id, start, end, source_text, **kwargs):
    return EditorClip(clip_id=clip_id, track_id="subtitles", kind="subtitle", name="Câu tiếng Việt",
                      start_ms=start, duration_ms=end - start,
                      metadata={"segment_payload": {"source_text": source_text}}, **kwargs)


def document(*clips):
    return EditorDocument(video_id="layout", sequence=EditorSequence(duration_ms=10000),
                          tracks=[EditorTrack(track_id="subtitles", kind="subtitle", name="Phụ đề"),
                                  EditorTrack(track_id="overlays", kind="overlay", name="Lớp phủ")],
                          clips=list(clips))


def test_replaced_text_is_narration_and_static_or_growing_text_is_a_label():
    cues = [cue("a", 0, 6000, "不要用这种直梗的新手最好选U型梗")]
    captions = block("不要用这种直梗的", 0, 3000, 76.0,
                     readings=["假睫毛选的时候注意了", "不要用这种直梗的", "不要用这种直梗的"])
    growing = block("直梗的U型梗√", 0, 3000, 58.0, readings=["直梗的×", "直梗的×", "直梗的XU型梗√"])
    spoken_caption = block("新手最好选U型梗", 3000, 6000, 82.0)
    panel = block("U型梗弧度圆滑\n不磨眼睛√", 3000, 6000, 33.0, lines=2)
    printed = block("右眼(10-12mm)", 3000, 6000, 26.0)
    narration, labels = text_layout.classify([captions, growing, spoken_caption, panel, printed], cues)
    # The narrator reads the growing label aloud, yet it is still a label.
    assert [item["region"]["y_percent"] for item in narration] == [76.0, 82.0]
    assert [item["region"]["y_percent"] for item in labels] == [58.0, 33.0]


def test_cues_move_onto_the_caption_they_overlap_most_but_keep_user_placement():
    moved = cue("moved", 0, 4000, "")
    placed = cue("placed", 4000, 8000, "", style_override={"position_y_percent": 40.0})
    value = document(moved, placed)
    narration = [block("甲", 0, 1000, 80.0), block("乙", 1000, 4500, 70.0)]
    assert text_layout.position_subtitles(value, narration) == 1
    assert moved.style_override["position_y_percent"] == 72.5 and moved.metadata[text_layout.AUTO_POSITION]
    assert placed.style_override == {"position_y_percent": 40.0}
    # A later run without a matching caption returns an automatic cue to the default.
    assert text_layout.position_subtitles(value, []) == 0
    assert "position_y_percent" not in moved.style_override


def test_labels_become_timed_text_clips_over_the_masked_text():
    value = document()
    label = block("U型梗弧度圆滑\n不磨眼睛√", 2000, 5000, 33.0, lines=2)
    source = text_layout.label_source(label)
    assert source == "U型梗弧度圆滑\n不磨眼睛"
    assert text_layout.label_clips(value, [label], {source: "Gân chữ U bo tròn\nkhông cộm mắt"}, 1280) == 1
    clip = value.clips[0]
    assert (clip.kind, clip.track_id, clip.start_ms, clip.duration_ms) == ("text", "overlays", 2000, 3000)
    assert (clip.transform.position_x_percent, clip.transform.position_y_percent) == (50.0, 38.0)
    assert clip.style_override["font_size"] == 38.4  # glyphs fill 60 % of a 5 %, 1280 px line box
    # Re-running replaces earlier automatic labels instead of stacking them.
    text_layout.label_clips(value, [label], {source: "Bản dịch mới"}, 1280)
    assert [clip.name for clip in value.clips] == ["Bản dịch mới"]


def test_batch_layout_masks_blocks_places_cues_and_draws_labels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from haizflow.pipeline import text_masks, text_overlay_sprite
    from haizflow.services import ocr_layers
    from haizflow.utils import ffmpeg

    caption = block("不会像普通假睫毛一样", 0, 3000, 48.0,
                    readings=["它能够完美贴合", "不会像普通假睫毛一样", "直愣愣的"])
    panel = block("U型梗弧度圆滑\n不磨眼睛√", 0, 3000, 33.0, lines=2)
    (tmp_path / "vi.srt").write_text("1\n00:00:00,500 --> 00:00:02,500\nCâu một\n", encoding="utf-8")
    (tmp_path / "vi.json").write_text('[{"start": 0.5, "end": 2.5, "source_text": "它能够完美贴合"}]', encoding="utf-8")
    sprite = tmp_path / "sprite.png"
    video = SimpleNamespace(video_id="batch", files={"video_input": str(tmp_path / "in.mp4")})
    monkeypatch.setattr(ocr_layers, "text_layers_path", lambda _video: tmp_path / "text-layers.json")
    with patch.object(text_masks, "detect_text_layers", return_value=[caption, panel]), \
            patch.object(ffmpeg, "get_video_dimensions", return_value=(720, 1280)), \
            patch.object(text_overlay_sprite, "render_text_overlay_sprite", return_value={"path": str(sprite)}):
        result = text_layout.batch_render_layout(
            video, {"x_percent": 10, "y_percent": 75, "width_percent": 80, "height_percent": 5},
            str(tmp_path / "vi.srt"), str(tmp_path / "vi.json"), "blur", "Bangers",
            translate=lambda _video, texts: {text: "Gân chữ U" for text in texts})
    payload = result["region"]
    assert payload["y_percent"] == 75  # the detected region still lays out subtitles
    assert [layer["region"]["y_percent"] for layer in payload["treatment_layers"]] == [48.0, 33.0]
    assert result["cue_regions"] == {1: caption["region"]}
    assert [(item["text"], item["y_percent"]) for item in payload["text_overlays"]] == [("Gân chữ U", 38.0)]
    # A second render reuses the cached translation instead of calling the translator.
    with patch.object(text_masks, "detect_text_layers", return_value=[caption, panel]), \
            patch.object(ffmpeg, "get_video_dimensions", return_value=(720, 1280)), \
            patch.object(text_overlay_sprite, "render_text_overlay_sprite", return_value={"path": str(sprite)}):
        again = text_layout.batch_render_layout(
            video, None, str(tmp_path / "vi.srt"), str(tmp_path / "vi.json"), "blur", "Bangers",
            translate=lambda *_: (_ for _ in ()).throw(AssertionError("translator called")))
    assert again["region"]["text_overlays"][0]["text"] == "Gân chữ U"


def test_label_overlays_join_the_mask_chain_through_a_movie_source(tmp_path):
    from haizflow.pipeline.render import _ordered_subtitle_removal_prefix

    sprite = tmp_path / "label.png"
    sprite.write_bytes(b"png")
    payload = {"treatment_layers": [], "text_overlays": [
        {"path": str(sprite), "x_percent": 50, "y_percent": 38, "intervals": [{"start_ms": 1000, "end_ms": 2000}]}]}
    prefix = _ordered_subtitle_removal_prefix(payload, 720, 1280, "blur")
    assert prefix.startswith("movie='") and "\:/" in prefix  # the drive colon is escaped
    assert "enable='gte(t,1.000000)*lt(t,2.000000)'" in prefix
    assert prefix.endswith("[labelled_0]null[source_without_original];")
