import datetime
import json
import os
import re
import tempfile

import srt

from haizflow.services.video_store import log_to_video


def split_text_by_length(text: str, max_chars: int) -> str:
    """Wrap one subtitle cue into at most two readable lines."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added_length = len(word) + (1 if current else 0)
        if current and current_length + added_length > max_chars:
            lines.append(" ".join(current))
            current, current_length = [word], len(word)
        else:
            current.append(word)
            current_length += added_length
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# Words a cue should not end on: they open a phrase that continues in the next cue.
_DANGLING_WORDS = frozenset({
    "và", "của", "là", "thì", "mà", "để", "cho", "với", "những", "các", "một", "này", "như",
    "nhưng", "hoặc", "hay", "rằng", "bị", "đã", "sẽ", "đang", "rất", "cũng", "vì", "nên", "nếu",
    "khi", "trong", "ngoài", "trên", "dưới", "từ", "tại", "ở", "bởi", "do", "theo", "về", "đến",
    "mỗi", "từng", "cả", "còn", "vẫn", "chỉ", "lại", "đều", "tới", "sang", "qua", "giữa",
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for", "with", "but",
})


def _best_chunks(words: list[str], line_limit: int, cue_limit: int, max_words: int) -> list[list[str]]:
    """Split words into cues at natural pauses with evenly filled, never orphaned, cues.

    A greedy split cut mid-phrase and left a one-word cue at the end. Choosing the
    cheapest set of breaks instead keeps clauses together and balances cue lengths.
    """
    count = len(words)
    target = cue_limit * 0.75
    best = [0.0] + [float("inf")] * count
    previous = [0] * (count + 1)
    for end in range(1, count + 1):
        for start in range(end - 1, max(-1, end - max_words - 1), -1):
            chunk = words[start:end]
            text = " ".join(chunk)
            if start < end - 1 and (len(text) > cue_limit or split_text_by_length(text, line_limit).count("\n") >= 2):
                break  # Longer chunks only get longer.
            cost = 1.0 + 4.0 * ((target - len(text)) / cue_limit) ** 2
            last = chunk[-1].lower().strip(".,!?;:\"'…)")
            if end < count:
                if re.search(r"[.!?…]$", chunk[-1]):
                    cost -= 1.5
                elif re.search(r"[;:]$", chunk[-1]):
                    cost -= 1.2
                elif chunk[-1].endswith(","):
                    cost -= 1.0
                elif last in _DANGLING_WORDS:
                    cost += 2.5
            if len(chunk) < 3 and not re.search(r"[.!?…;:,]$", chunk[-1]):
                cost += 2.0  # A one- or two-word cue flashes by too fast to read.
            if best[start] + cost < best[end]:
                best[end], previous[end] = best[start] + cost, start
    chunks: list[list[str]] = []
    end = count
    while end > 0:
        chunks.append(words[previous[end]:end])
        end = previous[end]
    return chunks[::-1]


def split_segment_into_cues(segment: dict, max_chars_per_line: int) -> list[dict]:
    """Split a transcript block into short sequential cues for social-video captions."""
    text = " ".join(segment.get("text", "").split())
    start = float(segment.get("start", 0))
    end = max(start, float(segment.get("end", start)))
    words = text.split()
    if not words:
        return []

    line_limit = max(12, max_chars_per_line)
    cue_limit = min(42, line_limit * 2)
    max_words = max(3, min(8, cue_limit // 5))
    chunks = _best_chunks(words, line_limit, cue_limit, max_words)

    total_weight = sum(max(1, len(" ".join(chunk))) for chunk in chunks)
    duration = max(0.3, end - start)
    cursor = start
    cues: list[dict] = []
    for index, chunk in enumerate(chunks):
        weight = max(1, len(" ".join(chunk)))
        cue_end = end if index == len(chunks) - 1 else cursor + duration * weight / total_weight
        cues.append({
            "start": round(cursor, 3),
            "end": round(max(cursor + 0.1, cue_end), 3),
            "text": split_text_by_length(" ".join(chunk), line_limit),
        })
        cursor = cue_end
    return cues


def generate_srt(
    segments_json_path: str,
    output_srt_path: str,
    max_chars_per_line: int,
    video_id: str,
    *,
    preserve_segment_boundaries: bool = False,
    voice_parts_dir: str = "",
):
    """Materialize subtitle cues without changing the document clock.

    Automatic projects retain their compact pre-split SRT cues.  Manual
    projects preserve one cue per editable segment so the ASS renderer can
    partition one word clock for the whole spoken clip.  Otherwise changing
    font size can expose a different visual phrase split over stale SRT cue
    boundaries and make karaoke appear to run at another speed.
    """
    log_to_video(video_id, f"Compiling sequential SRT cues (Max characters per line: {max_chars_per_line})...")
    with open(segments_json_path, "r", encoding="utf-8") as file:
        segments = json.load(file)
    if voice_parts_dir:
        from haizflow.pipeline.speech_timing import voiced_subtitle_segments

        segments = voiced_subtitle_segments(segments, [
            os.path.join(voice_parts_dir, f"voice_{index:04d}.mp3")
            for index in range(1, len(segments) + 1)
        ])

    subtitles = []
    for segment in segments:
        if preserve_segment_boundaries:
            text = " ".join(str(segment.get("text") or "").split())
            start = float(segment.get("start", 0) or 0)
            end = max(start + 0.1, float(segment.get("end", start) or start))
            cues = [{"start": start, "end": end, "text": text}] if text else []
        else:
            cues = split_segment_into_cues(segment, max_chars_per_line)
        for cue in cues:
            subtitles.append(srt.Subtitle(
                index=len(subtitles) + 1,
                start=datetime.timedelta(seconds=cue["start"]),
                end=datetime.timedelta(seconds=cue["end"]),
                content=cue["text"],
            ))
    if not subtitles:
        raise RuntimeError("Subtitle generation produced no timed cues.")

    content = srt.compose(subtitles)
    try:
        with open(output_srt_path, "r", encoding="utf-8") as existing:
            if existing.read() == content:
                # Resume recompiles speech-aligned captions before checking
                # the render cache. Replacing identical bytes changes mtime
                # and wrongly invalidates an otherwise complete long render.
                log_to_video(video_id, f"Reusing unchanged subtitle cues: {output_srt_path}")
                return
    except (OSError, UnicodeError):
        pass
    output_directory = os.path.dirname(os.path.abspath(output_srt_path))
    os.makedirs(output_directory, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(
        prefix=".subtitles-",
        suffix=".srt.tmp",
        dir=output_directory,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_srt_path)
    except Exception:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise
    log_to_video(video_id, f"Saved {len(subtitles)} sequential subtitle cues to: {output_srt_path}")


def parse_srt_to_segments(srt_path: str) -> list:
    """Parse an input SRT into normalized timestamped text segments."""
    with open(srt_path, "r", encoding="utf-8") as file:
        subtitles = list(srt.parse(file.read()))
    return [{
        "start": round(sub.start.total_seconds(), 3),
        "end": round(sub.end.total_seconds(), 3),
        "text": " ".join(sub.content.replace("\n", " ").split()).strip(),
    } for sub in subtitles]
