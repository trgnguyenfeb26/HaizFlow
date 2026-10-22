from types import SimpleNamespace

import numpy as np

from haizflow.pipeline import text_masks
from haizflow.pipeline.render import _ordered_subtitle_removal_prefix
from haizflow.schemas.editor import EditorDocument, EditorSequence
from haizflow.services import editor_documents, ocr_layers

CAPTION = (20.0, 76.0, 60.0, 4.0)
LABEL = (30.0, 40.0, 40.0, 4.0)


def test_rows_and_wide_gaps_become_separate_unclipped_boxes():
    probability = np.zeros((100, 200), dtype=np.float32)
    probability[50:52, 20:60] = 1   # two labels on one row, far apart
    probability[50:52, 120:160] = 1
    probability[80:82, 40:160] = 1
    boxes = text_masks.boxes_from_probability(probability)
    assert len(boxes) == 3
    left, right, bottom = sorted(boxes, key=lambda box: (box[1], box[0]))
    # A 2 px kernel grows 1.5 px each side: 48.5 % .. 53.5 % of the height.
    assert left[1] == 48.5 and left[3] == 5.0 and right[0] > left[0] + left[2]
    assert bottom[1] == 78.5


def test_rows_get_their_own_windows_and_boxes():
    samples = [[CAPTION]] * 8 + [[CAPTION, LABEL]] * 4 + [[]] * 8 + [[(25.0, 76.5, 50.0, 4.0)]] * 4
    layers = text_masks.group_text_layers(samples)
    windows = [(layer["intervals"][0]["start_ms"], layer["intervals"][0]["end_ms"], layer["region"]["y_percent"])
               for layer in layers]
    pad = text_masks.EDGE_PAD_MS
    assert windows == [(0, 2875 + pad, 76.0), (1875 - pad, 2875 + pad, 40.0), (4875 - pad, 5875 + pad, 76.5)]
    # The later, narrower caption is not widened to the earlier one.
    assert layers[-1]["region"]["x_percent"] == 25.0


def test_short_detection_gaps_do_not_split_a_caption():
    samples = [[CAPTION], [CAPTION], [], [], [CAPTION]]
    assert len(text_masks.group_text_layers(samples)) == 1


def test_moving_text_and_printed_material_are_not_masked():
    sliding = [[(10.0 + 5 * index, 40.0, 30.0, 4.0)] for index in range(6)]
    assert text_masks.group_text_layers(sliding) == []
    fine_print = [[(10.0, 20.0, 30.0, 2.0)]] * 4
    assert text_masks.group_text_layers(fine_print) == []
    crowded = [[CAPTION] + [(5.0, 10.0 + 5 * row, 30.0, 4.0) for row in range(9)]] * 4
    layers = text_masks.group_text_layers(crowded)
    assert [layer["region"]["y_percent"] for layer in layers] == [76.0]


def test_scan_is_cached_per_source_and_regrouped_on_read(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    cache = tmp_path / "text.json"
    assert text_masks.cached_text_layers(cache, str(source)) is None
    payload = {"version": text_masks.CACHE_VERSION, "source": text_masks._source_state(str(source)),
               "samples_per_second": 4, "samples": [[list(CAPTION)]] * 4}
    cache.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert text_masks.cached_text_layers(cache, str(source))[0]["region"]["y_percent"] == 76.0
    source.write_bytes(b"replaced video")
    assert text_masks.cached_text_layers(cache, str(source)) is None


def test_primary_region_masks_each_detected_row_only_while_shown(monkeypatch):
    document = EditorDocument(video_id="text-mask", sequence=EditorSequence(duration_ms=4000))
    ocr_layers.ensure_primary(document)
    detected = [{"region": {"x_percent": 20, "y_percent": 76, "width_percent": 60, "height_percent": 4},
                 "intervals": [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 3000, "end_ms": 5000}]},
                {"region": {"x_percent": 30, "y_percent": 40, "width_percent": 40, "height_percent": 4},
                 "intervals": [{"start_ms": 500, "end_ms": 1500}]}]
    monkeypatch.setattr(ocr_layers, "detected_text_layers", lambda _video: detected)
    monkeypatch.setattr(editor_documents, "load", lambda _video_id: document)
    video = SimpleNamespace(video_id="text-mask", project_type="manual", remove_original_subtitles=True,
                            original_subtitle_removal_mode="blur", original_subtitle_region_override={})
    primary = next(item for item in ocr_layers.layers(video, document) if item["primary"])
    assert primary["intervals"] == [{"start_ms": 0, "end_ms": 1500}, {"start_ms": 3000, "end_ms": 4000}]
    payload = ocr_layers.render_region(video, {"x_percent": 10, "y_percent": 80, "width_percent": 80,
                                               "height_percent": 5})
    treatments = payload["treatment_layers"]
    assert [item["region"]["y_percent"] for item in treatments] == [76, 40]
    assert treatments[0]["intervals"][-1] == {"start_ms": 3000, "end_ms": 4000}
    prefix = _ordered_subtitle_removal_prefix(payload, 128, 96, "blur")
    assert "gte(t,3.000000)*lt(t,4.000000)" in prefix and "gte(t,0.500000)*lt(t,1.500000)" in prefix


def test_each_caption_on_a_row_gets_its_own_mask_box():
    wide, short = (10.0, 76.0, 80.0, 4.0), (40.0, 76.5, 20.0, 3.5)
    layers = text_masks.group_text_layers([[wide]] * 4 + [[short]] * 4)
    assert len(layers) == 1  # one row, read and classified as a whole
    assert [mask["region"]["width_percent"] for mask in layers[0]["masks"]] == [80.0, 20.0]
    assert layers[0]["masks"][1]["intervals"][0]["start_ms"] == 875 - text_masks.EDGE_PAD_MS


def test_a_brief_detection_beside_a_long_caption_is_not_merged_into_it():
    caption = [(10.0, 76.0, 80.0, 4.0)]
    samples = [caption] * 40
    samples[20] = caption + [(25.0, 70.0, 60.0, 4.0)]
    layers = text_masks.group_text_layers(samples)
    assert sorted(layer["region"]["y_percent"] for layer in layers) == [70.0, 76.0]


def test_a_hand_drawn_region_masks_its_row_while_other_text_keeps_detected_boxes():
    blocks = text_masks.group_text_layers([[CAPTION, LABEL]] * 4 + [[(40.0, 76.5, 20.0, 3.5)]] * 4)
    drawn = {"x_percent": 15, "y_percent": 75, "width_percent": 70, "height_percent": 7}
    assert len(ocr_layers.masks_for(SimpleNamespace(original_subtitle_region_override={}), blocks, drawn)) == 3
    narrow = {"x_percent": 30, "y_percent": 75, "width_percent": 40, "height_percent": 7}
    masks = ocr_layers.masks_for(SimpleNamespace(original_subtitle_region_override=narrow), blocks, narrow)
    caption_masks = [mask for mask in masks if mask["region"]["y_percent"] == 75]
    # The drawn height and position are kept; a wider caption widens the box.
    assert [mask["region"] for mask in caption_masks] == [
        {"x_percent": 20.0, "y_percent": 75, "width_percent": 60.0, "height_percent": 7},
        {"x_percent": 30, "y_percent": 75, "width_percent": 40, "height_percent": 7}]
    assert [mask["region"]["y_percent"] for mask in masks if mask not in caption_masks] == [40.0]
